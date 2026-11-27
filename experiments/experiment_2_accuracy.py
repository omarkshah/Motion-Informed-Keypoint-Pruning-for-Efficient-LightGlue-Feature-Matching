import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Reuse the same pair selection and CSV writer as Experiment 1.
from experiment_1_speed import ROOT, select_pairs, get_pair_info, write_csv
from camera import ROS_DEFAULT
from features import SuperPointExtractor, LightGlueMatcher
from frame import ImageSequence
from pruner import MotionPruner


def main():
    parser = argparse.ArgumentParser(description="Compare LightGlue match counts")
    parser.add_argument("--seq", type=Path, default=ROOT / "rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--gaps", type=int, nargs="+", default=[1, 10, 30, 100])
    parser.add_argument("--pairs-per-gap", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-kp", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--margin", type=float, default=20)
    parser.add_argument("--motion", choices=["random", "translation", "rotation"], default="random")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "experiment_2")
    args = parser.parse_args()

    if args.pairs_per_gap < 1 or args.max_kp < 1 or min(args.gaps) < 1:
        parser.error("Pair counts, keypoint counts and gaps must be positive")
    if len(set(args.gaps)) != len(args.gaps) or args.seed < 0:
        parser.error("Gaps must be distinct and seed must be non-negative")

    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)

    seq = ImageSequence(args.seq, intrinsics=ROS_DEFAULT)
    pairs = select_pairs(seq, args.gaps, args.pairs_per_gap, args.seed, args.motion)
    if not pairs:
        parser.error("No image pairs found for the requested gaps")

    extractor = SuperPointExtractor(device=args.device, max_num_keypoints=args.max_kp)
    matcher = LightGlueMatcher(device=extractor.device)
    pruner = MotionPruner(margin=args.margin)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, j in pairs:
        a, b = seq.images[i], seq.images[j]

        fa = extractor.extract(a.data)
        fb = extractor.extract(b.data)

        # Standard LightGlue
        standard_match = len(matcher.match_features(fa, fb)) if len(fa) and len(fb) else 0

        # Motion-pruned LightGlue
        ra, rb = pruner.prune_image_pair(fa, fb, a, b, seq.intrinsics, seq.intrinsics)
        sa, sb = ra.features, rb.features

        pruned_match = 0
        if len(sa) and len(sb):
            pruned_match = len(matcher.match_features(sa, sb))

        match_pct = 100.0 * pruned_match / standard_match if standard_match else None
        original_kp = len(fa) + len(fb)
        removed_kp = original_kp - len(sa) - len(sb)

        rows.append(dict(
            **get_pair_info(seq, i, j),
            standard_matches=standard_match,
            pruned_matches=pruned_match,
            match_difference=pruned_match - standard_match,
            match_count_pct=match_pct,
            pruned_keypoints_pct=100.0 * removed_kp / original_kp if original_kp else 0.0,
        ))

        pct_text = f"{match_pct:.1f}%" if match_pct is not None else "N/A"
        print(f"Pair {i} -> {j}: Standard {standard_match} | Pruned {pruned_match} | "
              f"Match count ratio {pct_text}")

        a.clear_cache()
        b.clear_cache()

    write_csv(args.out / "per_pair.csv", rows)

    # Summarize the average match count for each frame gap.
    summary = []
    gaps = sorted({row["gap"] for row in rows})

    for gap in gaps + ["Overall"]:
        group = rows if gap == "Overall" else [r for r in rows if r["gap"] == gap]
        standard_counts = [r["standard_matches"] for r in group]
        pruned_counts = [r["pruned_matches"] for r in group]
        pruned_counts = [r["pruned_matches"] for r in group]

        standard_mean = float(np.mean(standard_counts))
        pruned_mean = float(np.mean(pruned_counts))

        summary.append(dict(
            gap=gap,
            n_pairs=len(group),
            standard_mean=standard_mean,
            pruned_mean=pruned_mean,
            mean_difference=pruned_mean - standard_mean,
            match_count_pct=100.0 * pruned_mean / standard_mean if standard_mean else None,
        ))

        print(f"{gap}: Standard mean {standard_mean:.1f} | Pruned mean {pruned_mean:.1f}")

    write_csv(args.out / "summary_by_gap.csv", summary[:-1])
    write_csv(args.out / "summary_overall.csv", summary[-1:])

    # Compare average match counts by frame gap.
    x = np.arange(len(gaps))
    width = 0.35
    by_gap = summary[:-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [r["standard_mean"] for r in by_gap], width, label="Standard")
    ax.bar(x + width / 2, [r["pruned_mean"] for r in by_gap], width, label="Motion-pruned")
    ax.set_xticks(x)
    ax.set_xticklabels(gaps)
    ax.set_xlabel("Frame gap")
    ax.set_ylabel("Average number of matches")
    ax.set_title("Match count by frame gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "matches_by_gap.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
