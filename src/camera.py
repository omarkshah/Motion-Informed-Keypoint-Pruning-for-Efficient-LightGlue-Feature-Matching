from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics plus OpenCV-style radial/tangential distortion (k1, k2, p1, p2, k3).

    TUM calls these d0..d4, which is OpenCV's (k1, k2, p1, p2, k3) order.
    NOTE: TUM recommends the ROS default parameters (ROS_DEFAULT below, no undistortion) when
    working with the pre-registered depth images, because undistorting those is not trivial.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    distortions: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def K(self) -> np.ndarray:
        """3x3 camera matrix. skew parameter assumed to be 0"""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)

        # TODO: property creates a new NumPy array each time, which is wasteful. Could cache it
        # esp not want to invert the same camera matrix for every individual keypoint

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array(self.distortions, dtype=np.float64)

    # for RGB-only experiments; not for ROS
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) distorted pixel coordinates -> (N, 2) undistorted pixel coordinates (same K)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)

        if len(pts) == 0:
            return pts

        # correct for distortion, then reproject to pixel coordinates
        out = cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist_coeffs, P=self.K)
        return out.reshape(-1, 2)


# TUM freiburg1 RGB calibration
FR1_RGB = Intrinsics(
    fx=517.3, fy=516.5, cx=318.6, cy=255.3,
    distortions=(0.2624, -0.9531, -0.0054, 0.0026, 1.1633),
)

# TUM's recommended intrinsics for the pre-registered depth images (and the colour image they line up with)
ROS_DEFAULT = Intrinsics(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
