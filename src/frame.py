"""Load a TUM RGB-D sequence as a list of posed images.

Usage:
    seq = ImageSequence("rgbd_dataset_freiburg1_xyz")
    images = seq.images          # list[Image], sorted by timestamp
    img = images[0]
    img.data                     # HxWx3 uint8 BGR array (loaded on first access)
    img.R, img.t                 # camera -> world pose
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from camera import Intrinsics, ROS_DEFAULT
from tum_associate import read_file_list, associate

# 16 bit depth values
DEPTH_SCALE = 5000.0

@dataclass
class Image:
    """One RGB frame plus its depth map and ground-truth camera pose (camera -> world)."""

    name: str            # path relative to the sequence root, e.g. "rgb/1305031102.175304.png"
    timestamp: float
    path: Path
    depth_path: Path
    R: np.ndarray = field(repr=False)   # (3, 3) rotation
    t: np.ndarray = field(repr=False)   # (3,)   translation

    @cached_property
    def data(self) -> np.ndarray:
        img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {self.path}")
        return img

    @property
    def rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)

    @property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)

    @cached_property
    def depth_raw(self) -> np.ndarray:
        """16-bit depth exactly as stored (5000 units=1m, 0=invalid)."""
        # preserve the original 16 bit depth values
        raw = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)

        if raw is None:
            raise FileNotFoundError(f"Could not read depth image: {self.depth_path}")
        if raw.ndim != 2 or raw.dtype != np.uint16:
            raise ValueError(
                f"Expected a single-channel uint16 depth PNG, got shape {raw.shape} dtype {raw.dtype}: {self.depth_path}"
            )
        return raw

    @cached_property
    def depth(self) -> np.ndarray:
        """convert depth in metres. Invalid pixels (stored as 0) are NaN."""
        depth = self.depth_raw.astype(np.float32) / DEPTH_SCALE
        depth[self.depth_raw == 0] = np.nan
        return depth


    def depth_at(self, points: np.ndarray) -> np.ndarray:
        """Depth in metres at (N, 2) pixel coordinates (x, y), sampled at the nearest pixel.
        Returns an (N,) array, NaN where the point is outside the image or the pixel has no depth measurement.
        """
        points = np.asarray(points)
        h, w = self.depth.shape

        # nearest pixel
        x = np.rint(points[:, 0]).astype(np.int64)
        y = np.rint(points[:, 1]).astype(np.int64)

        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        out = np.full(len(points), np.nan, dtype=np.float32)
        out[inside] = self.depth[y[inside], x[inside]]
        return out

    def clear_cache(self) -> None:
        """Free the loaded pixel and depth arrays."""
        for key in ("data", "depth_raw", "depth"):
            self.__dict__.pop(key, None)


class ImageSequence:
    """A TUM RGB-D sequence containing RGB frames, registered depth maps, and ground-truth camera poses.

    Args:
        root: Dataset directory containing rgb.txt, depth.txt and groundtruth.txt.
        max_dt: Maximum allowed timestamp difference between RGB, depth and ground-truth samples.
        intrinsics: Camera calibration parameters. Defaults to ROS_DEFAULT for registered TUM RGB-D images.
    """

    def __init__(self, root: str | Path, max_dt: float = 0.02, intrinsics: Intrinsics = ROS_DEFAULT):
        self.root = Path(root)
        self.intrinsics = intrinsics

        if max_dt <= 0:
            raise ValueError("max_dt must be positive")
        self.max_dt = max_dt

        self._rgb_list = read_file_list(self.root / "rgb.txt")
        self._gt_list = read_file_list(self.root / "groundtruth.txt")
        self._depth_list = read_file_list(self.root / "depth.txt")

        self.images: list[Image] = self._build_images()

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def _build_images(self) -> list[Image]:
        # associate rgb frames with ground truth poses, dropping frames with no close ground truth
        gt_matches = dict(associate(self._rgb_list, self._gt_list, 0.0, self.max_dt))
        depth_matches = dict(associate(self._rgb_list, self._depth_list, 0.0, self.max_dt))

        images = []
        gt_diffs = [] # TODO: for testing
        depth_diffs = [] # TODO: for testing

        # keep rgb frames that have a corresponding depth image and groundtruth pose sorted chronologically
        for rgb_ts in sorted(gt_matches.keys() & depth_matches.keys()):
            gt_ts = gt_matches[rgb_ts]
            depth_ts = depth_matches[rgb_ts]
            gt_diffs.append(abs(rgb_ts - gt_ts)) # for testing
            depth_diffs.append(abs(rgb_ts - depth_ts)) # for testing

            name = self._rgb_list[rgb_ts][0]
            tx, ty, tz, qx, qy, qz, qw = map(float, self._gt_list[gt_ts])
            depth_name = self._depth_list[depth_ts][0]

            images.append(Image(
                name=name,
                timestamp=rgb_ts,
                path=self.root / name,
                depth_path=self.root / depth_name,
                R=quaternion_to_matrix(np.array([qx, qy, qz, qw])),
                t=np.array([tx, ty, tz], dtype=np.float64),
            ))

        dropped = len(self._rgb_list) - len(images)
        print(f"Kept {len(images)} frames, dropped {dropped} (no GT or depth within {self.max_dt}s)")

        if gt_diffs:
            print(f"Maximum timestamp difference: {max(gt_diffs) * 1000:.2f} ms")
            print(f"Average timestamp difference: {np.mean(gt_diffs) * 1000:.2f} ms")
        
        if depth_diffs:
            print(f"Maximum depth timestamp difference: {max(depth_diffs) * 1000:.2f} ms")
            print(f"Average depth timestamp difference: {np.mean(depth_diffs) * 1000:.2f} ms")

        return images

def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Invalid zero-length quaternion")

    # Normalize the quaternion
    x, y, z, w = q / norm

    # Return a 3x3 rotation matrix
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x**2 + y**2)],
    ])


def relative_pose(a: Image, b: Image) -> tuple[np.ndarray, np.ndarray]:

    # rotation from camera b to camera a
    R_B_A = a.R.T @ b.R


    # position of camera b relative to camera a, in camera a's reference frame
    p_B_A = a.R.T @ (b.t - a.t)

    return R_B_A, p_B_A
