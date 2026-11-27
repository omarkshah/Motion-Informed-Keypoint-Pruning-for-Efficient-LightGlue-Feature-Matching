import argparse
import numpy as np
import sys
import time 
import torch
import csv
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from camera import ROS_DEFAULT
from features import FeatureMatches, LightGlueMatcher, SuperPointExtractor
from frame import ImageSequence
from geometry import relative_transform
from pruner import MotionPruner

def feature_stats(fa, fb, sa, sb, matches):
    original = len(fa) + len(fb)
    kept = len(sa) + len(sb)

    return dict(keypoints_a=len(fa), keypoints_b=len(fb), kept_a=len(sa), kept_b=len(sb), matches=len(matches),
                pruned_pct=100.0 * (original - kept) / original if original else 0.0, 
                layers=matches.num_layers, empty_input=int(not len(sa) or not len(sb)), empty_output=int(not len(matches)))

def select_pairs(seq, gaps, count, seed, motion):
    rng = np.random.default_rng(seed)
    pairs = []
    n = len(seq)

    for gap in sorted(gaps):
        if gap >= n:
            continue

        candidates = [(i, i + gap) for i in range(n - gap)]

        if motion == "translation":
            candidates.sort(
                key=lambda p: get_pair_info(seq, *p)["translation_m"]
            )

        elif motion == "rotation":
            candidates.sort(
                key=lambda p: get_pair_info(seq, *p)["rotation_deg"]
            )

        else:
            rng.shuffle(candidates)

        # Sample across the range of camera motions
        groups = np.array_split(
            np.arange(len(candidates)),
            min(count, len(candidates))
        )

        for group in groups:
            pairs.append(candidates[int(rng.choice(group))])

    return pairs

def run_pipeline(extractor, matcher, pruner, a, b, intrinsics):

    device = extractor.device

    fa, fb = extractor.extract(a.data), extractor.extract(b.data)

    start = time.perf_counter()
    if pruner is None:
        sa, sb = fa, fb
        after_pruning = start
    else:
        ra, rb = pruner.prune_image_pair(fa, fb, a, b, intrinsics, intrinsics)
        sa, sb = ra.features, rb.features
        after_pruning = time.perf_counter()

    matches = matcher.match_features(sa, sb)
    end = time.perf_counter()
    return dict(pruning_ms=(after_pruning - start) * 1000.0,
                matching_ms=(end - after_pruning) * 1000.0,
                total_ms=(end - start) * 1000.0,
                **feature_stats(fa, fb, sa, sb, matches))

def get_pair_info(seq, i, j):
    a, b = seq.images[i], seq.images[j]
    R, _ = relative_transform(a, b)
    return dict(idx_a=i, idx_b=j, gap=j-i, timestamp_a=a.timestamp, timestamp_b=b.timestamp,
                dt_s=b.timestamp-a.timestamp, translation_m=float(np.linalg.norm(a.t-b.t)),
                rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))))

def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():

    parser = argparse.ArgumentParser(description="Speed test for the feature matching pipeline")
    parser.add_argument("--seq", type=Path, default=ROOT / "rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--gaps", type=int, nargs="+", default=[1, 10, 30, 100])
    parser.add_argument("--pairs-per-gap", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-kp", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "experiment_1")
    parser.add_argument("--margin", type=float, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--motion", choices=["random", "translation", "rotation"], default="random")
    args = parser.parse_args()


    # Validate arguments
    if args.pairs_per_gap < 1 or args.max_kp < 1:
        parser.error("Pair and keypoint counts must be positive")

    if min(args.gaps) < 1 or len(set(args.gaps)) != len(args.gaps):
        parser.error("Gaps must be positive and distinct")

    if not np.isfinite(args.margin) or args.margin < 0:
        parser.error("Margin must be finite and non-negative")

    if args.repeats < 1 or args.seed < 0:
        parser.error("Invalid repeats or seed")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")

    if not all((args.seq / f).is_file()
        for f in ("rgb.txt", "depth.txt", "groundtruth.txt")):
            parser.error(f"Invalid dataset: {args.seq}")

    seq = ImageSequence(args.seq, intrinsics=ROS_DEFAULT)
    pairs = select_pairs(seq, args.gaps, args.pairs_per_gap, args.seed, args.motion)

    args.out.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)

    extractor = SuperPointExtractor(device=args.device, max_num_keypoints=args.max_kp)
    matcher = LightGlueMatcher(device=extractor.device)
    pruner = MotionPruner(margin=args.margin)

    trials, pair_rows = [], []
    timings = [ "pruning_ms", "matching_ms", "total_ms"]

    for p, (i, j) in enumerate(pairs):
        a, b = seq.images[i], seq.images[j]
        info = get_pair_info(seq, i, j)

        _ = a.data
        _ = b.data

        # Load and time depth images
        start = time.perf_counter()
        _ = a.depth
        _ = b.depth
        depth_ms = (time.perf_counter() - start) * 1000.0

        methods = {"Standard": None, "Motion-pruned": pruner}

        # Run the pipeline multiple times for both the standard and the motion-pruned lightglue
        per_method = {name: [] for name in methods}
        for repeat in range(args.repeats):
            for name, method in methods.items():
                r = run_pipeline(extractor, matcher, method, a, b, seq.intrinsics)
                per_method[name].append(r)
                trials.append(dict(**info, method=name, repeat=repeat, depth_load_ms=depth_ms, **r))

        # Median result
        med = {}
        for name, rows in per_method.items():
            keys = rows[0].keys()
            row = dict(**info, method=name, depth_load_ms=depth_ms)
            for k in keys:
                row[k] = float(np.median([r[k] for r in rows]))
            med[name] = row
            pair_rows.append(row)

        # compare med and calculate speedup
        base = med["Standard"]
        reduced = med["Motion-pruned"]

        reduced["speedup"] = (
            base["total_ms"] / reduced["total_ms"]
        )

        reduced["matching_speedup"] = (
            base["matching_ms"] / reduced["matching_ms"]
        )

        print(
            f"Pair {i} -> {j}: "
            f"Pruned {reduced['pruned_pct']:.1f}% | "
            f"Matching speedup {reduced['matching_speedup']:.2f}x | "
            f"Net speedup {reduced['speedup']:.2f}x"
        )

        a.clear_cache()
        b.clear_cache()

    # Save individual trials and per-pair median results
    write_csv(args.out / "trials.csv", trials)
    write_csv(args.out / "per_pair.csv", pair_rows)

    metrics = timings + ["pruned_pct", "matches", "layers"]
    methods = ["Standard", "Motion-pruned"]

    # Calculate average results for each method and frame gap
    summary_by_gap = []

    for gap in sorted({r["gap"] for r in pair_rows}):
        for method in methods:
            rows = [
                r for r in pair_rows
                if r["gap"] == gap and r["method"] == method
            ]

            summary = dict(gap=gap, method=method, n_pairs=len(rows))

            for metric in metrics:
                values = [r[metric] for r in rows]
                summary[metric + "_mean"] = float(np.mean(values))
                summary[metric + "_std"] = float(np.std(values))

            summary_by_gap.append(summary)

    write_csv(args.out / "summary_by_gap.csv", summary_by_gap)

    # Calculate average results across all image pairs
    overall = []

    for method in methods:
        rows = [r for r in pair_rows if r["method"] == method]

        summary = dict(method=method, n_pairs=len(rows))

        for metric in metrics:
            summary[metric + "_mean"] = float(
                np.mean([r[metric] for r in rows])
            )

        overall.append(summary)

        print(f"{method}: {summary['total_ms_mean']:.2f} ms")

    write_csv(args.out / "summary_overall.csv", overall)

    # Report the median speedup across all image pairs
    speedups = [
        r["speedup"] for r in pair_rows
        if r["method"] == "Motion-pruned"
    ]

    print(f"Median speedup: {np.median(speedups):.2f}x")
    plot_results(summary_by_gap, overall, args.out)
 
def plot_results(summary_by_gap, overall, out):
    methods = ["Standard", "Motion-pruned"]
    gaps = sorted({r["gap"] for r in summary_by_gap})

    # Plot average pipeline runtime for each frame gap
    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(gaps))
    width = 0.35

    for i, method in enumerate(methods):
        rows = [r for r in summary_by_gap if r["method"] == method]

        means = [r["total_ms_mean"] for r in rows]
        stds = [r["total_ms_std"] for r in rows]

        ax.bar(x + i * width, means, width,
               yerr=stds, capsize=3, label=method)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(gaps)
    ax.set_xlabel("Frame gap")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Pipeline runtime by frame gap")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "runtime_by_gap.png", dpi=300)
    plt.close(fig)

    # Plot the average runtime of each pipeline stage
    fig, ax = plt.subplots(figsize=(6, 4))

    pruning = np.array([r["pruning_ms_mean"] for r in overall])
    matching = np.array([r["matching_ms_mean"] for r in overall])

    ax.bar(methods, pruning, label="Motion pruning")
    ax.bar(methods, matching, bottom=pruning, label="LightGlue")

    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Post-extraction runtime breakdown")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "runtime_breakdown.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()