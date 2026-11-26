"""Geometric reprojection between two camera views.

Coordinate convention:
    +X: right
    +Y: down
    +Z: forward

All 3D points and translations are expressed in metres. All 2D coordinates are expressed in pixels.

R and t transform source-camera coordinates into target-camera coordinates:
    P_target = R @ P_source + t

This module assumes an ideal pinhole camera model without lens distortion (e.g. ROS_DEFAULT).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camera import Intrinsics
from frame import Image

@dataclass
class ViewReprojection:
    pixels: np.ndarray          # (N, 2), target pixels
    target_depths: np.ndarray   # (N,), target Z coordinates
    has_depth: np.ndarray       # (N,), valid source depth
    in_front: np.ndarray        # (N,), safely in front
    behind: np.ndarray          # (N,), behind target camera
    in_image: np.ndarray        # (N,), inside actual image
    within_margin: np.ndarray   # (N,), inside expanded image

    @property
    def potentially_visible(self) -> np.ndarray:
        return (
            self.has_depth
            & self.in_front
            & self.in_image
        )

    @property
    def out_of_view(self) -> np.ndarray: 
        valid_projection = np.isfinite(self.pixels).all(axis=1)

        # points that are either behind the camera or in front but outside the expanded image
        return self.has_depth & (self.behind | 
                                 (self.in_front & valid_projection & ~self.within_margin))

    @property
    def unknown(self) -> np.ndarray:
        valid_projection = np.isfinite(self.pixels).all(axis=1)

        known = self.has_depth & (self.behind | (self.in_front & valid_projection))

        return ~known

    @property
    def keep_mask(self) -> np.ndarray:
        # TODO: could move it to motion_pruning later

        # Retain everything not explicitly out of view
        return ~self.out_of_view

    def __len__(self) -> int:
        return len(self.pixels)

def backproject_keypoints(keypoints: np.ndarray, depths: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Back-project 2D pixels into source-camera coordinates. Invalid source measurements produce NaN coordinates."""

    # keypoints are pixel coordinates and depth in metres
    keypoints = np.asarray(keypoints, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)

    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("Expected keypoints of shape (N, 2)")

    if depths.shape != (len(keypoints),):
        raise ValueError("Expected depths of shape (N,)")

    valid = (np.isfinite(keypoints).all(axis=1) & np.isfinite(depths) & (depths > 0))

    points = np.full((len(keypoints), 3),np.nan, dtype=np.float64)

    z = depths[valid]

    points[valid, 0] = ((keypoints[valid, 0] - intrinsics.cx)*z / intrinsics.fx)
    points[valid, 1] = ((keypoints[valid, 1] - intrinsics.cy)*z / intrinsics.fy)
    points[valid, 2] = z

    # in metres, in source-camera coordinates
    return points

def relative_transform(src_image, dest_image):
    """ Compute the relative transform (R, t) from the source camera (A) to the destination camera (B)"""

    R_WA = src_image.R # Rotation to the world frame from the source camera frame
    t_A = src_image.t  # Position of the source camera relative to the world origin

    R_WB = dest_image.R # Rotation to the world frame from the dest camera frame
    t_B = dest_image.t  # Position of the dest camera relative to the world origin

    if R_WA.shape != (3,3) or R_WB.shape != (3,3):
        raise ValueError("Expected camera rotations of shape (3, 3)")
    if t_A.shape != (3,) or t_B.shape != (3,):
        raise ValueError("Expected camera translations of shape (3,)")

    # relative rotation from source to destination
    R_BA = R_WB.T @ R_WA

    # relation translation from source to destination in the destination camera (B) reference frame
    t_BA = R_WB.T @ (t_A - t_B)

    return R_BA, t_BA

def transform_points(points: np.ndarray, R_BA: np.ndarray, t_BA: np.ndarray) -> np.ndarray:
    """Transform points from source to target camera coordinates."""

    points = np.asarray(points, dtype=np.float64)
    R_BA = np.asarray(R_BA, dtype=np.float64)
    t_BA = np.asarray(t_BA, dtype=np.float64)

    return points @ R_BA.T + t_BA

def project_points(points: np.ndarray, intrinsics: Intrinsics, eps: float = 1e-6) -> np.ndarray:
    """Project target-camera 3D points into target-image pixels. 
    Points behind the camera, near its camera plane, or with invalid coordinates are not projected."""
    
    points = np.asarray(points, dtype=np.float64)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = (np.isfinite(points).all(axis=1) & (points[:, 2] > eps))

    X = points[valid, 0]
    Y = points[valid, 1]
    Z = points[valid, 2]

    pixels[valid, 0] = (intrinsics.fx*X/Z + intrinsics.cx)
    pixels[valid, 1] = (intrinsics.fy*Y/Z + intrinsics.cy)

    # return predicted pixels
    return pixels

def reproject_keypoints(
    keypoints: np.ndarray,
    depths: np.ndarray,
    src_intrinsics: Intrinsics,
    dest_intrinsics: Intrinsics,
    src_image: Image,
    dest_image: Image,
    target_size: tuple[int, int],
    margin: float = 20.0,
    eps: float = 1e-6,
) -> ViewReprojection:
    """Predict source keypoint locations in the target view."""

    # margin in pixels
    if margin < 0:
        raise ValueError("margin must be non-negative")

    if eps <= 0:
        raise ValueError("eps must be positive")

    width, height = target_size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid target image size")

    # Compute the relative transform from the source camera (A) to teh destination camera (B)
    R_BA, t_BA = relative_transform(src_image, dest_image)

    # 3D point in the src camera's coordinate frame
    points_src = backproject_keypoints(keypoints, depths, src_intrinsics)
    has_depth = np.isfinite(points_src).all(axis=1)

    # 3D point in the target camera's coordinate frame
    points_dest = transform_points(points_src, R_BA, t_BA)

    target_depths = points_dest[:, 2]
    finite = np.isfinite(points_dest).all(axis=1)

    # determine whether the point is in front or behind
    in_front = finite & (target_depths > eps)
    behind = finite & (target_depths < -eps)

    # projecting the 3D point into the target camera's image plane
    pixels = project_points(points_dest, dest_intrinsics, eps)
    u = pixels[:, 0]
    v = pixels[:, 1]

    valid_projection = np.isfinite(pixels).all(axis=1)

    # test strict target image boundaries
    in_image = (in_front & valid_projection & (u >= 0) & (u < width) & (v >= 0) & (v < height))

    # test expanded boundaries for safe pruning
    within_margin = (in_front & valid_projection & (u >= -margin) & (u < width + margin) 
                     & (v >= -margin) & (v < height + margin))

    # preserve all results
    return ViewReprojection(
        pixels=pixels,
        target_depths=target_depths,
        has_depth=has_depth,
        in_front=in_front,
        behind=behind,
        in_image=in_image,
        within_margin=within_margin,
    )