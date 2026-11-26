"""End-to-end check: SuperPoint -> MotionPruner (pruner.py) -> LightGlue, compared with plain LightGlue.

Saves / shows one figure with three panels (each A | B):
    1. pruning     green = kept, red = pruned (out of view), yellow = kept because no depth
                   (cyan dots on B = where A's keypoints are predicted to land, with --show-pred)
    2. full LightGlue matches        green = consistent with GT epipolar geometry, red = not
    3. pruned LightGlue matches      same colouring

Also prints the key sanity number: how many of full LightGlue's correct matches had an endpoint
that pruning removed (should be ~0; if not, increase --margin or check the geometry).

Run from the repo root:
    python src/test_pruning_pipeline.py
    python src/test_pruning_pipeline.py --idx 100 --gap 45 --margin 10 --show-pred --save results/pipeline.png
    python src/test_pruning_pipeline.py --random                       # any two frames
    python src/test_pruning_pipeline.py --random --min-rot 15          # force a big viewpoint change
    python src/test_pruning_pipeline.py --random --seed 7              # reproduce a random pick
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from camera import FR1_RGB, ROS_DEFAULT
from features import LightGlueMatcher, SuperPointExtractor
from frame import ImageSequence
from geometry import relative_transform
from features import FeatureMatches
from pruner import MotionPruner, remap_matched_indices

ROOT = Path(__file__).resolve().parent.parent

GREEN, RED, YELLOW, CYAN, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 0), (255, 255, 255)


# ---------------------------------------------------------------- GT epipolar check

def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def epipolar_inliers(kA, kB, a, b, thresh):
    """Sampson error of matches (raw pixels in A, B) against the GT relative pose, in FR1-undistorted px."""
    if len(kA) == 0:
        return np.zeros(0, bool), np.zeros(0)
    R, t = relative_transform(b, a)          # X_A = R @ X_B - t  (geometry.py convention)
    t = -t                                   # -> X_A = R @ X_B + t
    Kinv = np.linalg.inv(FR1_RGB.K)
    F = Kinv.T @ skew(t) @ R @ Kinv          # x_A^T F x_B = 0
    uA, uB = FR1_RGB.undistort_points(kA), FR1_RGB.undistort_points(kB)
    hA = np.column_stack([uA, np.ones(len(uA))])
    hB = np.column_stack([uB, np.ones(len(uB))])
    Fb, Fta = hB @ F.T, hA @ F
    num = np.sum(hA * Fb, axis=1) ** 2
    den = Fb[:, 0] ** 2 + Fb[:, 1] ** 2 + Fta[:, 0] ** 2 + Fta[:, 1] ** 2
    err = np.sqrt(num / np.maximum(den, 1e-12))
    return err < thresh, err


# ---------------------------------------------------------------- timing

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeats):
    out, times = None, []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return out, 1000 * float(np.median(times))


# ---------------------------------------------------------------- pruner.py helpers

def n_kept(r):
    return len(r.kept_indices)


def remap_matches(m, ra, rb, fa, fb):
    """LightGlue matches on pruned features -> FeatureMatches indexing the ORIGINAL fa / fb."""
    idx = remap_matched_indices(m.indices, ra, rb).astype(np.int64) if len(m) else np.zeros((0, 2), np.int64)
    return FeatureMatches(kpts0=fa.keypoints[idx[:, 0]], kpts1=fb.keypoints[idx[:, 1]],
                          scores=m.scores, indices=idx, num_layers=m.num_layers)


def check_pruned_features(r, f, side):
    """Clear failure messages for the usual filter_features mistakes, before LightGlue crashes on them."""
    pf = r.features
    if pf.raw_superpoint is None or not isinstance(pf.raw_superpoint, dict):
        raise AssertionError(
            f"{side}: pruned features have no raw_superpoint (LightGlue's input). In pruner.py "
            "filter_features, pass image_size before filtered_raw:\n"
            "    return FeatureKeypoints(..., features.descriptors[keep_mask], features.image_size, filtered_raw)")
    if tuple(pf.image_size) != tuple(f.image_size):
        raise AssertionError(f"{side}: pruned image_size {pf.image_size} != original {f.image_size}")
    k = n_kept(r)
    assert r.keep_mask.shape == (len(f),) and r.keep_mask.dtype == np.bool_, f"{side}: bad keep_mask"
    assert len(pf) == k, f"{side}: {len(pf)} pruned keypoints but keep_mask keeps {k}"
    for name in ("keypoints", "keypoint_scores", "descriptors"):
        assert pf.raw_superpoint[name].shape[1] == k, f"{side}: raw {name} has {pf.raw_superpoint[name].shape[1]}, expected {k}"
    np.testing.assert_array_equal(pf.keypoints, f.keypoints[r.keep_mask])


# ---------------------------------------------------------------- drawing

def side_by_side(img_a, img_b):
    h, w = img_a.shape[:2]
    canvas = np.zeros((max(h, img_b.shape[0]), w + img_b.shape[1], 3), np.uint8)
    canvas[:h, :w] = img_a
    canvas[:img_b.shape[0], w:] = img_b
    return canvas, w


def label(canvas, lines):
    for i, text in enumerate(lines):
        y = 24 + 24 * i
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(canvas, (6, y - th - 6), (14 + tw, y + 6), (0, 0, 0), -1)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)


def draw_pruning(a, b, fa, fb, ra, rb, margin, show_pred):
    canvas, off = side_by_side(a.data, b.data)
    for f, r, dx in ((fa, ra, 0), (fb, rb, off)):
        unknown = r.reprojection.unknown
        for (x, y), keep, unk in zip(f.keypoints, r.keep_mask, unknown):
            colour = RED if not keep else YELLOW if unk else GREEN
            cv2.circle(canvas, (int(round(x)) + dx, int(round(y))), 2, colour, -1, cv2.LINE_AA)
    if show_pred:  # where A's keypoints are predicted to land in B
        for x, y in ra.reprojection.pixels[np.isfinite(ra.reprojection.pixels).all(axis=1)]:
            cv2.circle(canvas, (int(round(x)) + off, int(round(y))), 1, CYAN, -1)
    label(canvas, [f"Pruning (margin {margin:g}px): "
                   f"A kept {n_kept(ra)}/{ra.n_total}   B kept {n_kept(rb)}/{rb.n_total}",
                   "green kept   red pruned   yellow kept (no depth)"])
    return canvas


def draw_matches(a, b, m, inl, title):
    canvas, off = side_by_side(a.data, b.data)
    for p0, p1, ok in zip(m.kpts0, m.kpts1, inl):
        colour = GREEN if ok else RED
        q0 = (int(round(p0[0])), int(round(p0[1])))
        q1 = (int(round(p1[0])) + off, int(round(p1[1])))
        cv2.line(canvas, q0, q1, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, q0, 2, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, q1, 2, colour, -1, cv2.LINE_AA)
    label(canvas, [title])
    return canvas


# ---------------------------------------------------------------- pair selection

def pose_difference(a, b):
    """(baseline in metres, rotation in degrees) between two frames, from ground truth only."""
    R, t = relative_transform(a, b)
    rot = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(t)), float(rot)


def pick_random_pair(seq, seed=None, min_baseline=0.0, min_rot=0.0, max_tries=2000):
    """Two distinct random frames (i < j) whose GT motion meets the minimums. Uses poses only, no image loading."""
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 31))
    rng = np.random.default_rng(seed)
    n = len(seq)
    best, best_score = None, -1.0
    for _ in range(max_tries):
        i, j = sorted(rng.choice(n, size=2, replace=False))
        base, rot = pose_difference(seq.images[i], seq.images[j])
        if base >= min_baseline and rot >= min_rot:
            print(f"Random pair (seed {seed}): frames {i} and {j}")
            return int(i), int(j)
        score = min(base / max(min_baseline, 1e-9), rot / max(min_rot, 1e-9))
        if score > best_score:
            best, best_score = (int(i), int(j)), score
    print(f"WARNING: no pair met --min-baseline {min_baseline} m / --min-rot {min_rot} deg in {max_tries} tries "
          f"(this sequence may not move that much); using the closest one found. Seed {seed}.")
    return best


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq", default=str(ROOT / "rgbd_dataset_freiburg1_xyz"))
    p.add_argument("--idx", type=int, default=100)
    p.add_argument("--gap", type=int, default=30)
    p.add_argument("--max-kp", type=int, default=2048)
    p.add_argument("--margin", type=float, default=20.0)
    p.add_argument("--thresh", type=float, default=3.0, help="epipolar inlier threshold (px)")
    p.add_argument("--repeats", type=int, default=5, help="timing repeats (median)")
    p.add_argument("--show-pred", action="store_true", help="draw predicted A->B positions on B")
    p.add_argument("--save", default=str(ROOT / "results" / "pruning_pipeline.png"))
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--random", action="store_true",
                   help="pick two random frames from the whole sequence instead of --idx/--gap")
    p.add_argument("--seed", type=int, default=None, help="seed for --random (printed, so a pair can be rerun)")
    p.add_argument("--min-baseline", type=float, default=0.0, help="--random: minimum GT baseline (m)")
    p.add_argument("--min-rot", type=float, default=0.0, help="--random: minimum GT rotation (deg)")
    args = p.parse_args()

    seq = ImageSequence(args.seq, intrinsics=ROS_DEFAULT)
    if args.random:
        i, j = pick_random_pair(seq, args.seed, args.min_baseline, args.min_rot)
    else:
        i, j = args.idx, args.idx + args.gap
        if not (0 <= i < len(seq) and 0 <= j < len(seq)):
            raise SystemExit(f"Frame indices out of range (sequence has {len(seq)} frames)")
    a, b = seq.images[i], seq.images[j]

    baseline, rot = pose_difference(a, b)
    print(f"\nA [{i}] {a.name}\nB [{j}] {b.name}   (frame gap {j - i}, dt {b.timestamp - a.timestamp:.2f} s)")
    print(f"Rerun this pair: --idx {i} --gap {j - i}")
    print(f"GT motion: baseline {100 * baseline:.1f} cm, rotation {rot:.1f} deg")
    if baseline < 0.02:
        print("WARNING: baseline < 2 cm, the epipolar colouring is unreliable for this pair")

    extractor = SuperPointExtractor(max_num_keypoints=args.max_kp)
    matcher = LightGlueMatcher(device=extractor.device)
    pruner = MotionPruner(margin=args.margin)
    dev = extractor.device
    print(f"Device: {dev}\n")

    # 1. SuperPoint
    fa, fb = extractor.extract(a.data), extractor.extract(b.data)
    for _ in range(3):                                   # warm-up (CUDA init, kernel selection)
        matcher.match_features(fa, fb)

    # 2. Motion pruning
    (ra, rb), t_prune = timed(lambda: pruner.prune_image_pair(fa, fb, a, b, seq.intrinsics, seq.intrinsics),
                             dev, args.repeats)
    check_pruned_features(ra, fa, "A")
    check_pruned_features(rb, fb, "B")
    if n_kept(ra) == 0 or n_kept(rb) == 0:
        raise SystemExit("Pruning removed every keypoint on one side; LightGlue can't run. "
                         "Increase --margin or pick another pair.")

    # 3. LightGlue: full vs pruned
    m_full, t_full = timed(lambda: matcher.match_features(fa, fb), dev, args.repeats)
    m_sub, t_sub = timed(lambda: matcher.match_features(ra.features, rb.features), dev, args.repeats)
    m_pruned = remap_matches(m_sub, ra, rb, fa, fb)

    # ---- structural checks
    if len(m_pruned):
        assert len(ra.features) and len(rb.features)
        assert ra.keep_mask[m_pruned.indices[:, 0]].all() and rb.keep_mask[m_pruned.indices[:, 1]].all()
        np.testing.assert_allclose(m_pruned.kpts0, fa.keypoints[m_pruned.indices[:, 0]])
        np.testing.assert_allclose(m_pruned.kpts1, fb.keypoints[m_pruned.indices[:, 1]])
    print("Structure checks: PASS (pruned tensors consistent, indices map back to original features)\n")

    # ---- accuracy vs GT
    inl_full, _ = epipolar_inliers(m_full.kpts0, m_full.kpts1, a, b, args.thresh)
    inl_pruned, _ = epipolar_inliers(m_pruned.kpts0, m_pruned.kpts1, a, b, args.thresh)

    # correct full-LightGlue matches whose A or B keypoint pruning threw away
    lost = inl_full & ~(ra.keep_mask[m_full.indices[:, 0]] & rb.keep_mask[m_full.indices[:, 1]]) \
        if len(m_full) else np.zeros(0, bool)

    def pct(x, n):
        return f"{100 * x / max(n, 1):5.1f}%"

    print(f"{'':16}{'keypoints A/B':>15}{'LG ms':>9}{'total ms':>10}{'matches':>9}{'inliers':>9}{'prec':>8}{'layers':>8}")
    print(f"{'full':16}{f'{len(fa)}/{len(fb)}':>15}{t_full:9.1f}{t_full:10.1f}{len(m_full):9d}"
          f"{int(inl_full.sum()):9d}{pct(inl_full.sum(), len(m_full)):>8}{m_full.num_layers:8d}")
    print(f"{'motion-pruned':16}{f'{n_kept(ra)}/{n_kept(rb)}':>15}{t_sub:9.1f}{t_prune + t_sub:10.1f}"
          f"{len(m_pruned):9d}{int(inl_pruned.sum()):9d}{pct(inl_pruned.sum(), len(m_pruned)):>8}"
          f"{m_pruned.num_layers:8d}")
    print(f"\nPruning time: {t_prune:.2f} ms   kept A {pct(n_kept(ra), ra.n_total)}  B {pct(n_kept(rb), rb.n_total)}")
    for name, r in (("A", ra), ("B", rb)):
        rep = r.reprojection
        print(f"  {name}: no depth {int(rep.unknown.sum()):4d}   behind {int((rep.has_depth & rep.behind).sum()):4d}   "
              f"out of view {int(rep.out_of_view.sum()):4d}")
    print(f"Correct full matches lost to pruning: {int(lost.sum())}/{int(inl_full.sum())} "
          f"({pct(lost.sum(), inl_full.sum())})   <- should be ~0")

    # ---- figure
    fig = np.vstack([
        draw_pruning(a, b, fa, fb, ra, rb, args.margin, args.show_pred),
        draw_matches(a, b, m_full, inl_full,
                     f"Full LightGlue: {len(m_full)} matches, {int(inl_full.sum())} GT-consistent, {t_full:.1f} ms"),
        draw_matches(a, b, m_pruned, inl_pruned,
                     f"Motion-pruned: {len(m_pruned)} matches, {int(inl_pruned.sum())} GT-consistent, "
                     f"{t_sub:.1f} ms (+{t_prune:.1f} prune)"),
    ])
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(args.save, fig):
            raise OSError(f"Could not write {args.save}")
        print(f"\nSaved {args.save}")
    if not args.no_show:
        scale = min(1.0, 1000 / fig.shape[0])
        cv2.imshow("SuperPoint -> MotionPruner -> LightGlue", cv2.resize(fig, None, fx=scale, fy=scale))
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())