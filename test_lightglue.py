"""Match two TUM frames with SuperPoint + LightGlue and visualise the result.

Because every frame has a ground-truth pose, we can also check the matches: a correct
match must lie on the epipolar line predicted by the GT relative pose. Matches are drawn
green if their Sampson error is under --thresh pixels, red otherwise.

Usage:
    python test_lightglue.py rgbd_dataset_freiburg1_xyz
    python test_lightglue.py rgbd_dataset_freiburg1_xyz --i 0 --gap 30 --show
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from src.features import LightGlueMatcher, SuperPointExtractor
from src.frame import ImageSequence, relative_pose  # rename if your loader file is called something else

# TUM freiburg1 intrinsics + distortion (from the TUM dataset's camera calibration page)
K_FR1 = np.array([[517.3, 0.0, 318.6],
                  [0.0, 516.5, 255.3],
                  [0.0, 0.0, 1.0]])
DIST_FR1 = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])


# ---------------------------------------------------------------- geometry checks

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def undistort(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Distorted pixel coords -> undistorted pixel coords."""
    if len(pts) == 0:
        return pts
    return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K).reshape(-1, 2)


def sampson_error(F: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Sampson distance (pixels) for x0^T F x1 = 0. x0, x1: (K, 2) pixel coords."""
    h0 = np.hstack([x0, np.ones((len(x0), 1))])
    h1 = np.hstack([x1, np.ones((len(x1), 1))])
    Fx1 = h1 @ F.T      # rows: F @ x1
    Ftx0 = h0 @ F       # rows: F^T @ x0
    num = np.sum(h0 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx0[:, 0] ** 2 + Ftx0[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ---------------------------------------------------------------- drawing

def draw_matches(img0, img1, all0, all1, k0, k1, colors, header: list[str]) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    gap = 10
    canvas = np.full((max(h0, h1), w0 + gap + w1, 3), 255, np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0 + gap:] = img1
    off = np.array([w0 + gap, 0])

    for p in all0:
        cv2.circle(canvas, tuple(np.round(p).astype(int)), 1, (200, 200, 200), -1, cv2.LINE_AA)
    for p in all1:
        cv2.circle(canvas, tuple(np.round(p + off).astype(int)), 1, (200, 200, 200), -1, cv2.LINE_AA)

    for p0, p1, c in zip(k0, k1, colors):
        a = tuple(np.round(p0).astype(int))
        b = tuple(np.round(p1 + off).astype(int))
        cv2.line(canvas, a, b, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 2, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 2, c, -1, cv2.LINE_AA)

    for i, line in enumerate(header):
        y = 22 + 22 * i
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def score_colors(scores: np.ndarray) -> list[tuple[int, int, int]]:
    """Red (low confidence) -> green (high confidence), BGR."""
    return [(0, int(255 * s), int(255 * (1 - s))) for s in np.clip(scores, 0, 1)]


# ---------------------------------------------------------------- main

def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="TUM sequence directory (contains rgb.txt, groundtruth.txt)")
    ap.add_argument("--i", type=int, default=0, help="index of first frame")
    ap.add_argument("--j", type=int, default=None, help="index of second frame (default: i + gap)")
    ap.add_argument("--gap", type=int, default=10, help="frame gap if --j not given (~30 fps)")
    ap.add_argument("--max-keypoints", type=int, default=2048)
    ap.add_argument("--thresh", type=float, default=2.0, help="Sampson error inlier threshold (px)")
    ap.add_argument("--no-undistort", action="store_true", help="skip undistortion in the GT check")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--out", default="matches.png", help="where to save the visualisation")
    ap.add_argument("--show", action="store_true", help="also open a window")
    args = ap.parse_args()

    seq = ImageSequence(args.root)
    j = args.j if args.j is not None else args.i + args.gap
    if not (0 <= args.i < len(seq) and 0 <= j < len(seq)):
        raise SystemExit(f"Frame indices must be in [0, {len(seq) - 1}], got {args.i}, {j}")
    a, b = seq.images[args.i], seq.images[j]

    device = torch.device("cpu") if args.cpu else None
    extractor = SuperPointExtractor(max_num_keypoints=args.max_keypoints, device=device)
    matcher = LightGlueMatcher(device=device)
    dev = extractor.device
    print(f"Device: {dev}")
    print(f"Frame A: [{args.i}] {a.name}")
    print(f"Frame B: [{j}] {b.name}  (dt = {b.timestamp - a.timestamp:.3f} s)")

    # warm-up so the timings below aren't dominated by CUDA init / kernel compilation
    f0 = extractor.extract(a.data)
    matcher.match(f0, f0)

    sync(dev); t0 = time.perf_counter()
    f0 = extractor.extract(a.data)
    f1 = extractor.extract(b.data)
    sync(dev); t1 = time.perf_counter()
    m = matcher.match(f0, f1)
    sync(dev); t2 = time.perf_counter()

    print(f"\nSuperPoint: {len(f0)} / {len(f1)} keypoints  ({(t1 - t0) * 500:.1f} ms per image)")
    print(f"LightGlue:  {len(m)} matches  ({(t2 - t1) * 1000:.1f} ms, {m.num_layers} layers)")
    if len(m):
        print(f"Match score: mean {m.scores.mean():.3f}, median {np.median(m.scores):.3f}")

    header = [f"A[{args.i}] <-> B[{j}]   kpts {len(f0)}/{len(f1)}   matches {len(m)}"]

    # ---- ground-truth epipolar check
    R_gt, p_gt = relative_pose(a, b)   # x_A = R_gt x_B + p_gt
    baseline = np.linalg.norm(p_gt)
    print(f"\nGT relative motion: rotation {angle_deg(R_gt):.2f} deg, baseline {baseline * 100:.1f} cm")

    colors = score_colors(m.scores)
    if len(m) >= 8 and baseline > 0.01:
        if args.no_undistort:
            u0, u1 = m.kpts0.astype(np.float64), m.kpts1.astype(np.float64)
        else:
            u0, u1 = undistort(m.kpts0, K_FR1, DIST_FR1), undistort(m.kpts1, K_FR1, DIST_FR1)

        E_gt = skew(p_gt) @ R_gt                      # x_A^T E x_B = 0 (normalised coords)
        K_inv = np.linalg.inv(K_FR1)
        F_gt = K_inv.T @ E_gt @ K_inv                 # x_A^T F x_B = 0 (pixel coords)
        err = sampson_error(F_gt, u0, u1)
        inl = err < args.thresh

        print(f"Epipolar error vs GT: median {np.median(err):.2f} px, "
              f"{inl.sum()}/{len(m)} ({100 * inl.mean():.1f}%) under {args.thresh} px")

        # independent check: estimate the pose from the matches alone and compare with GT
        E_est, mask = cv2.findEssentialMat(u1, u0, K_FR1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E_est is not None and E_est.shape == (3, 3):
            _, R_est, t_est, _ = cv2.recoverPose(E_est, u1, u0, K_FR1, mask=mask)  # x_A = R x_B + t
            print(f"RANSAC pose: {int(mask.sum())} inliers, "
                  f"rotation error {angle_deg(R_est.T @ R_gt):.2f} deg, "
                  f"translation direction error {vec_angle_deg(t_est.ravel(), p_gt):.1f} deg")

        colors = [(0, 200, 0) if ok else (0, 0, 255) for ok in inl]
        header.append(f"GT epipolar: {inl.sum()}/{len(m)} inliers (<{args.thresh}px), "
                      f"median {np.median(err):.2f}px   green=ok red=bad")
    else:
        why = "too few matches" if len(m) < 8 else "baseline < 1 cm, epipolar check is meaningless"
        print(f"Skipping GT check ({why}); colouring by match score instead.")
        header.append("coloured by LightGlue score (red=low, green=high)")

    vis = draw_matches(a.data, b.data, f0.keypoints, f1.keypoints, m.kpts0, m.kpts1, colors, header)
    cv2.imwrite(args.out, vis)
    print(f"\nSaved visualisation to {args.out}")

    if args.show:
        cv2.imshow("SuperPoint + LightGlue", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
