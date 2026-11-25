import sys

import cv2
import numpy as np

from frame import ImageSequence
from features import SuperPointExtractor


# ADDED: helpers for the visual depth check (everything down to `if __name__` is new)
def pick_samples(depths, n_valid=5, n_invalid=2, seed=0):
    """Indices of a few keypoints with valid depth plus a few without (fixed seed -> same picks each run)."""
    rng = np.random.default_rng(seed)
    valid_idx = np.flatnonzero(np.isfinite(depths))
    invalid_idx = np.flatnonzero(~np.isfinite(depths))

    picks = list(rng.choice(valid_idx, size=min(n_valid, len(valid_idx)), replace=False))
    picks += list(rng.choice(invalid_idx, size=min(n_invalid, len(invalid_idx)), replace=False))
    return np.array(picks, dtype=int)


def colorize_depth(depth):
    """HxW float depth (NaN = invalid) -> HxWx3 BGR image. Blue = near, red = far, black = no depth."""
    valid = np.isfinite(depth)
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return vis, (0.0, 0.0)

    lo, hi = float(depth[valid].min()), float(depth[valid].max())
    scaled = ((depth[valid] - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    vis[valid] = cv2.applyColorMap(scaled.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    return vis, (lo, hi)


def draw_label(canvas, text, x, y, color):
    """Text with a dark outline so it stays readable on any background."""
    h, w = canvas.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x = int(np.clip(x + 8, 0, w - tw - 1))
    y = int(np.clip(y - 8, th + 1, h - 1))
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def visualize_depth_samples(img, points, depths, picks, out_path="depth_samples.png", show=False):
    """Left: RGB with the picked pixels labelled by depth. Right: colourised depth map with the same pixels.

    Green marker = depth found, red marker = no depth at that pixel.
    """
    rgb_vis = img.data.copy()
    depth_vis, (lo, hi) = colorize_depth(img.depth)

    for n, i in enumerate(picks):
        x, y = points[i]
        centre = (int(round(x)), int(round(y)))
        z = depths[i]
        color = (0, 255, 0) if np.isfinite(z) else (0, 0, 255)  # BGR

        label = f"#{n} ({centre[0]},{centre[1]}) " + (f"{z:.2f} m" if np.isfinite(z) else "no depth")

        for canvas in (rgb_vis, depth_vis):
            cv2.circle(canvas, centre, 6, (255, 255, 255), 3)
            cv2.circle(canvas, centre, 6, color, 2)
        draw_label(rgb_vis, label, *centre, color)
        draw_label(depth_vis, f"#{n}", *centre, (255, 255, 255))

    draw_label(depth_vis, f"depth {lo:.2f}-{hi:.2f} m  (blue near, red far, black none)", -8, 24, (255, 255, 255))

    combined = np.hstack([rgb_vis, depth_vis])
    cv2.imwrite(out_path, combined)
    print(f"Saved visualization to {out_path}")

    if show:
        cv2.imshow("RGB (left) | depth (right), press any key to close", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    seq = ImageSequence("rgbd_dataset_freiburg1_xyz")
    img = seq.images[0]

    print("RGB shape:", img.data.shape)
    print("Depth shape:", img.depth.shape)
    print("Depth type:", img.depth.dtype)

    print("Min depth:", np.nanmin(img.depth))
    print("Max depth:", np.nanmax(img.depth))

    extractor = SuperPointExtractor()
    features = extractor.extract(img.data)

    Z = img.depth_at(features.keypoints)

    print("Total keypoints:", len(features.keypoints))
    print("Keypoint depth shape:", Z.shape)
    print("Keypoints with depth:", np.isfinite(Z).sum())
    print("Keypoints without depth:", np.isnan(Z).sum())

    # ADDED: visual check on a few picked keypoints
    picks = pick_samples(Z)

    print("\nPicked pixels:")
    for n, i in enumerate(picks):
        x, y = features.keypoints[i]
        print(f"  #{n}: ({x:.1f}, {y:.1f}) -> depth {Z[i]:.3f} m")

        # depth_at must agree with indexing the depth map directly at the nearest pixel
        direct = img.depth[int(round(y)), int(round(x))]
        assert (np.isnan(direct) and np.isnan(Z[i])) or direct == Z[i], f"depth_at mismatch at #{n}"

    # ADDED: the raw 16-bit value should be 5000 * metres for valid pixels
    for i in picks:
        x, y = features.keypoints[i]
        raw = img.depth_raw[int(round(y)), int(round(x))]
        if np.isfinite(Z[i]):
            assert abs(raw / 5000.0 - Z[i]) < 1e-4, "raw value / 5000 should equal depth in metres"
        else:
            assert raw == 0, "NaN depth should correspond to a raw value of 0"
    print("depth_at / raw-value checks passed")

    visualize_depth_samples(img, features.keypoints, Z, picks, show="--show" in sys.argv)