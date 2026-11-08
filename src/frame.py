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

from tum_associate import read_file_list, associate


@dataclass
class Image:
    """One RGB frame plus its ground-truth camera pose (camera -> world)."""

    name: str            # path relative to the sequence root, e.g. "rgb/1305031102.175304.png"
    timestamp: float
    path: Path
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


class ImageSequence:
    """A TUM RGB-D sequence: every RGB frame that has a ground-truth pose.

    Args:
        root: dataset directory containing rgb.txt and groundtruth.txt.
        max_dt: max time gap (s) between an RGB frame and a ground-truth sample;
            frames with no ground truth this close are dropped.
    """

    def __init__(self, root: str | Path, max_dt: float = 0.02):
        self.root = Path(root)

        if max_dt <= 0:
            raise ValueError("max_dt must be positive")
        self.max_dt = max_dt

        self._rgb_list = read_file_list(self.root / "rgb.txt")
        self._gt_list = read_file_list(self.root / "groundtruth.txt")

        self.images: list[Image] = self._build_images()

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def _build_images(self) -> list[Image]:
        # associate rgb frames with ground truth poses, dropping frames with no close ground truth
        matches = associate(self._rgb_list, self._gt_list, 0.0, self.max_dt)

        images = []
        time_diffs = [] # TODO: for testing

        for rgb_ts, gt_ts in matches:
            time_diffs.append(abs(rgb_ts - gt_ts)) # for testing

            name = self._rgb_list[rgb_ts][0]
            tx, ty, tz, qx, qy, qz, qw = map(float, self._gt_list[gt_ts])

            images.append(Image(
                name=name,
                timestamp=rgb_ts,
                path=self.root / name,
                R=quaternion_to_matrix(np.array([qx, qy, qz, qw])),
                t=np.array([tx, ty, tz], dtype=np.float64),
            ))

        dropped = len(self._rgb_list) - len(images)
        print(f"Kept {len(images)} frames, dropped {dropped} (no GT within {self.max_dt}s)")

        if time_diffs:
            print(f"Maximum timestamp difference: {max(time_diffs) * 1000:.2f} ms")
            print(f"Average timestamp difference: {np.mean(time_diffs) * 1000:.2f} ms")
        
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
