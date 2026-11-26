"""Benchmark standard LightGlue vs motion-pruned LightGlue on one TUM pair.

Run from the repository root:
    python tests/test_pruning_speed.py --idx 100 --gap 30 --repeats 20

This is a runnable benchmark, not a unit test with a required speedup.
Times are synchronized wall-clock milliseconds, after warm-up.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from camera import ROS_DEFAULT
from features import FeatureMatches, LightGlueMatcher, SuperPointExtractor
from frame import ImageSequence
from pruner import MotionPruner


def sync(device):
    """Wait for asynchronous GPU work before taking a timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn, device):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter() - start) * 1000


def match_or_empty(matcher, f0, f1):
    # No possible correspondences if either feature set is empty.
    if len(f0) == 0 or len(f1) == 0:
        return FeatureMatches(
            kpts0=np.empty((0, 2)), kpts1=np.empty((0, 2)),
            scores=np.empty(0), indices=np.empty((0, 2), dtype=np.int64),
            num_layers=0,
        )
    return matcher.match_features(f0, f1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", type=Path, default=ROOT / "rgbd_dataset_freiburg1_xyz")
    p.add_argument("--idx", type=int, default=100)
    p.add_argument("--gap", type=int, default=30)
    p.add_argument("--max-kp", type=int, default=2048)
    p.add_argument("--margin", type=float, default=20.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.max_kp < 1 or args.gap < 1:
        p.error("repeats, max-kp and gap must be positive; warmup must be non-negative")
    if not np.isfinite(args.margin) or args.margin < 0:
        p.error("margin must be finite and non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        p.error("MPS is not available")

    seq = ImageSequence(args.seq, intrinsics=ROS_DEFAULT)
    j = args.idx + args.gap
    if not 0 <= args.idx < j < len(seq):
        p.error(f"frame indices out of range (sequence has {len(seq)} frames)")
    a, b = seq.images[args.idx], seq.images[j]

    extractor = SuperPointExtractor(device=args.device, max_num_keypoints=args.max_kp)
    matcher = LightGlueMatcher(device=extractor.device)
    pruner = MotionPruner(margin=args.margin)
    device = extractor.device

    # Same RGB images and feature sets for both methods. I/O is outside timing.
    img_a, img_b = a.data, b.data
    with torch.inference_mode():
        f0 = extractor.extract(img_a)
        f1 = extractor.extract(img_b)
        if not len(f0) or not len(f1):
            raise SystemExit("SuperPoint returned no features; choose another pair.")

        # Depth is extra input for our method. Measure initial loading/conversion
        # separately; warm-up and steady-state timings use the cached maps.
        _, depth_load_ms = timed(lambda: (a.depth, b.depth), device)

        def baseline():
            return timed(lambda: matcher.match_features(f0, f1), device)

        def motion_pruned():
            # Time the actual combined path, rather than summing independent medians.
            sync(device)
            start = time.perf_counter()
            r0, r1 = pruner.prune_image_pair(
                f0, f1, a, b, seq.intrinsics, seq.intrinsics
            )
            sync(device)
            after_prune = time.perf_counter()
            matches = match_or_empty(matcher, r0.features, r1.features)
            sync(device)
            end = time.perf_counter()
            return (r0, r1, matches, (after_prune - start) * 1000,
                    (end - after_prune) * 1000, (end - start) * 1000)

        # Warm up BOTH paths, including pruning/tensor slicing.
        for _ in range(args.warmup):
            baseline()
            motion_pruned()

        full_times, prune_times, reduced_times, total_times = [], [], [], []
        for i in range(args.repeats):
            # Alternate order to reduce systematic warm-cache/thermal bias.
            if i % 2 == 0:
                full, full_ms = baseline()
                reduced = motion_pruned()
            else:
                reduced = motion_pruned()
                full, full_ms = baseline()
            r0, r1, pruned, prune_ms, reduced_ms, total_ms = reduced
            full_times.append(full_ms)
            prune_times.append(prune_ms)
            reduced_times.append(reduced_ms)
            total_times.append(total_ms)

    full_ms = median(full_times)
    prune_ms = median(prune_times)
    reduced_ms = median(reduced_times)
    total_ms = median(total_times)
    # Positive means motion pruning improves runtime; negative means it is slower.
    saved_ms = full_ms - total_ms
    reduction = 100 * saved_ms / full_ms if full_ms else float("nan")
    speedup = full_ms / total_ms if total_ms else float("inf")
    paired_savings = [x - y for x, y in zip(full_times, total_times)]

    print(f"\nDevice: {device} | Frames: {args.idx} -> {j} | margin: {args.margin:g} px")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"RGB-D sequence: {args.seq}")
    print(f"Frame separation: {b.timestamp - a.timestamp:.3f} s")
    print(f"Warm-up: {args.warmup} | Repeats: {args.repeats} | Times: median ms")
    print("\n" + f"{'Metric':<30}{'Standard':>14}{'Motion-pruned':>17}")
    print("-" * 61)
    print(f"{'Keypoints A':<30}{len(f0):>14d}{len(r0.features):>17d}")
    print(f"{'Keypoints B':<30}{len(f1):>14d}{len(r1.features):>17d}")
    print(f"{'Pruning overhead (ms)':<30}{0.0:>14.3f}{prune_ms:>17.3f}")
    print(f"{'Matching (ms)':<30}{full_ms:>14.3f}{reduced_ms:>17.3f}")
    print(f"{'Total after SuperPoint (ms)':<30}{full_ms:>14.3f}{total_ms:>17.3f}")
    print(f"{'Matches':<30}{len(full):>14d}{len(pruned):>17d}")
    print(f"{'LightGlue layers':<30}{full.num_layers:>14d}{pruned.num_layers:>17d}")
    print(f"\nPruned keypoints: A {r0.n_pruned}/{r0.n_total}, B {r1.n_pruned}/{r1.n_total}")
    print(f"Matcher-only time saved: {full_ms - reduced_ms:+.3f} ms")
    print(f"Net time saved (median comparison): {saved_ms:+.3f} ms")
    print(f"Net runtime reduction: {reduction:+.2f}% | Speedup: {speedup:.3f}x")
    print(f"Median paired time saved: {median(paired_savings):+.3f} ms")
    print(f"Standard range: {min(full_times):.3f} - {max(full_times):.3f} ms")
    print(f"Pruned total range: {min(total_times):.3f} - {max(total_times):.3f} ms")
    print(f"Pruned faster in {sum(t > 0 for t in paired_savings)}/{args.repeats} trials")
    print(f"Initial depth loading/conversion (both maps): {depth_load_ms:.3f} ms (separate)")
    print("Timing excludes SuperPoint, RGB I/O, and pose acquisition; depth sampling is included.")
    print("Motion prior: TUM ground truth. Match counts are not accuracy measurements.")
    if not len(r0.features) or not len(r1.features):
        print("NOTE: one pruned feature set is empty; LightGlue was skipped on the pruned path.")
    elif not r0.n_pruned and not r1.n_pruned:
        print("NOTE: no keypoints were pruned; try a larger frame gap or another sequence.")


if __name__ == "__main__":
    main()
