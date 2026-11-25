"""Check that depth is metrically correct using the motion-capture ground truth.

Idea: if depth (metres) and poses (metres) are both right, then a pixel back-projected from
frame a, moved to frame b with the ground-truth pose, must land on the depth that frame b
actually measured there. With an unknown depth scale s (true = s * stored depth) the rigid
relation is   s * (d_b * ray_b - R_ab @ (d_a * ray_a)) = t_ab,
so s can be solved by least squares over thousands of pixels. Expect s ~= 1.0.
(If TUM's pre-scaling of fr1 depth had NOT been applied, s would come out near 1.035.)

Run:  python test_depth_scale.py
"""
import numpy as np

from frame import ImageSequence, relative_pose
from camera import ROS_DEFAULT  # TUM recommends these intrinsics for the registered depth


def estimate_depth_scale(a, b, intr=ROS_DEFAULT, stride=6, gate=0.10):
    """Least-squares depth scale s from one frame pair (a -> b), plus diagnostics.

    Returns (s, n_points, median_abs_residual_m) or None if too few usable pixels.
    """
    h, w = a.depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    u, v = u.ravel().astype(np.float64), v.ravel().astype(np.float64)
    d_a = a.depth[v.astype(int), u.astype(int)].astype(np.float64)
    ok = np.isfinite(d_a)
    u, v, d_a = u[ok], v[ok], d_a[ok]

    ray_a = np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], axis=1)
    X_a = ray_a * d_a[:, None]

    # a -> b pose from the ground truth.  relative_pose gives b expressed in a's frame
    R_ba, p_ba = relative_pose(a, b)          # R_B_A, position of b in a
    R_ab = R_ba.T                              # rotates a-camera coords into b-camera coords
    t_ab = -R_ab @ p_ba                        # X_b = R_ab @ X_a + t_ab
    X_b = X_a @ R_ab.T + t_ab

    # project into b, sample b's measured depth there
    z = X_b[:, 2]
    in_front = z > 0.1
    u_b = intr.fx * X_b[:, 0] / np.where(in_front, z, 1) + intr.cx
    v_b = intr.fy * X_b[:, 1] / np.where(in_front, z, 1) + intr.cy
    d_b = b.depth_at(np.stack([u_b, v_b], axis=1)).astype(np.float64)

    # keep pixels visible in b whose measured depth roughly agrees (rejects occlusions / bad matches)
    keep = in_front & np.isfinite(d_b) & (np.abs(d_b - z) < gate)
    if keep.sum() < 500:
        return None

    ray_b = np.stack([(u_b - intr.cx) / intr.fx, (v_b - intr.cy) / intr.fy, np.ones_like(u_b)], axis=1)
    vec = d_b[keep, None] * ray_b[keep] - (d_a[keep, None] * ray_a[keep]) @ R_ab.T   # s * vec = t_ab
    s = float((vec @ t_ab).sum() / (vec * vec).sum())
    resid = np.abs(d_b[keep] - z[keep])
    return s, int(keep.sum()), float(np.median(resid))


if __name__ == "__main__":
    seq = ImageSequence("rgbd_dataset_freiburg1_xyz")
    imgs = seq.images

    scales = []
    gap = 20
    for i in range(0, len(imgs) - gap, 10):
        a, b = imgs[i], imgs[i + gap]
        baseline = np.linalg.norm(relative_pose(a, b)[1])
        if baseline < 0.15:        # need real motion; with tiny motion the estimate is noise
            continue
        out = estimate_depth_scale(a, b)
        if out is None:
            continue
        s, n, resid = out
        scales.append(s)
        print(f"frames {i:3d}->{i + gap:3d}  baseline {baseline:.2f} m  points {n:5d}  "
              f"median |depth residual| {resid * 100:.1f} cm  scale {s:.3f}")

    if scales:
        print(f"\nmedian scale over {len(scales)} pairs: {np.median(scales):.3f}  (expect ~1.00)")
    else:
        print("no pair had enough motion/overlap; try a larger `gap` or a sequence with more motion")