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
        interpolate: linearly interpolate the pose between the two bracketing
            ground-truth samples instead of taking the nearest one.
    """

    def __init__(self, root: str | Path, max_dt: float = 0.02, interpolate: bool = True):
        self.root = Path(root)
        self.max_dt = max_dt
        self.interpolate = interpolate

        self._groundtruth = _load_groundtruth(self.root / "groundtruth.txt")
        self.images: list[Image] = self._build_images(_load_rgb_index(self.root / "rgb.txt"))

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def _build_images(self, rgb_index: list[tuple[float, str]]) -> list[Image]:
        images = []
    
        for timestamp, name in rgb_index:

            pose = self.pose_at(timestamp)
            if pose is not None:
                R, t = pose
                images.append(Image(name, timestamp, self.root / name, R, t))

        return images

    def pose_at(self, timestamp: float) -> tuple[np.ndarray, np.ndarray] | None:
        """(R, t) at `timestamp`, or None if no ground truth within max_dt."""
        gt = self._groundtruth
        times = gt[:, 0]

        i = np.searchsorted(times, timestamp)   # first index with times[i] >= timestamp
        lo, hi = max(i - 1, 0), min(i, len(times) - 1)
        nearest = lo if abs(times[lo] - timestamp) <= abs(times[hi] - timestamp) else hi
        if abs(times[nearest] - timestamp) > self.max_dt:
            return None

        bracketed = lo != hi and times[lo] <= timestamp <= times[hi]
        if not (self.interpolate and bracketed):
            row = gt[nearest]
            return quaternion_to_matrix(row[4:8]), row[1:4].copy()

        a, b = gt[lo], gt[hi]
        alpha = (timestamp - a[0]) / (b[0] - a[0])
        t = (1 - alpha) * a[1:4] + alpha * b[1:4]
        qa, qb = a[4:8], b[4:8]
        if np.dot(qa, qb) < 0:          # q and -q are the same rotation; take the short way
            qb = -qb
        q = (1 - alpha) * qa + alpha * qb   # nlerp: fine for ~10 ms gaps
        return quaternion_to_matrix(q), t


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    # Normalize the quaternion
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)

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


def _read_rows(path: Path) -> list[list[str]]:

    # Reads and returns rows of the datafile
    # Ignores commented out lines
    # Each row is a list of strings, split by whitespace
    with open(path) as f:
        return [line.split() for line in f if line.strip() and not line.startswith("#")]


def _load_rgb_index(path: Path) -> list[tuple[float, str]]:
    
    return sorted((float(ts), name) for ts, name in _read_rows(path))


def _load_groundtruth(path: Path) -> np.ndarray:
    """groundtruth.txt rows "ts tx ty tz qx qy qz qw" -> (N, 8) array sorted by time."""
    gt = np.array(_read_rows(path), dtype=np.float64)
    return gt[np.argsort(gt[:, 0])]
