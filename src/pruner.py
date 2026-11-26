import numpy as np
import torch

from dataclasses import dataclass

from features import FeatureKeypoints
from geometry import ViewReprojection, reproject_keypoints
from frame import Image
from camera import Intrinsics, ROS_DEFAULT

@dataclass
class PruningResult:

    keep_mask: np.ndarray
    reprojection: np.ndarray
    features: FeatureKeypoints

    @property
    def kept_indices(self):
        return np.flatnonzero(self.keep_mask)

    @property
    def pruned_indices(self):
        return np.flatnonzero(~self.keep_mask)

    @property
    def n_total(self):
        return len(self.keep_mask)

    @property
    def n_kept(self):
        return len(self.kept_indices)

    @property
    def n_pruned(self):
        return len(self.pruned_indices)

class MotionPruner:
    """Prune keypoitns prediceted to be outside the other camera's view"""

    def __init__(self, margin = 0, eps = 1e-6):

        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Margin must be finite or non-negative")

        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("Epsilon must be finite and positive")

        self.margin = margin
        self.eps = eps

    def compute_keep_mask(self, projection: ViewReprojection):
        return ~projection.out_of_view

    def filter_features(self, features: FeatureKeypoints, keep_mask: np.ndarray):
        keep_mask = np.asarray(keep_mask)
        n = len(features)

        if keep_mask.shape != (n,) or keep_mask.dtype != np.bool_:
            raise ValueError(f"Expected keep_mask of shape ({n},) and dtype bool")

        if (features.keypoints.shape != (n, 2)) or features.scores.shape != (n,) or features.descriptors.ndim != 2 or features.descriptors.shape[0] != n:
            raise ValueError("Features and keep_mask must have the same number of keypoints")

        raw = features.raw_superpoint
        filtered_raw = None

        if raw is not None:
            filtered_raw = raw.copy()

            for name in ("keypoints", "descriptors", "keypoint_scores"):

                tensor = raw[name]
                if tensor.ndim < 2 or tuple(tensor.shape[:2]) != (1, n):
                    raise ValueError(f"Unexpected shape for raw {name}")

                mask = torch.as_tensor(keep_mask, device=tensor.device)
                filtered_raw[name] = tensor[:, mask]

        return FeatureKeypoints(keypoints=features.keypoints[keep_mask], scores=features.scores[keep_mask], descriptors=features.descriptors[keep_mask], image_size=features.image_size, raw_superpoint=filtered_raw)

    def prune(self, features : FeatureKeypoints, src_image: Image, dest_image: Image, src_intrinsics: Intrinsics, dest_intrinsics: Intrinsics):
        """Prune keypoints predicted to be outside the other camera's view"""

        if src_intrinsics is None:
            src_intrinsics = ROS_DEFAULT
        if dest_intrinsics is None: 
            dest_intrinsics = src_intrinsics

        for intrinsics in (src_intrinsics, dest_intrinsics):
            if not np.isfinite([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]).all() or intrinsics.fx <= 0 or intrinsics.fy <= 0:
                raise ValueError("Invalid intrinsics")

        source_size = tuple(src_image.data.shape[:2][::-1])
        if (tuple(features.image_size) != source_size) or src_image.depth.shape != src_image.data.shape[:2]:
            raise ValueError(f"Feature image size {features.image_size} does not match source image size {source_size}")

        target_size = tuple(dest_image.data.shape[:2][::-1])

        points = np.asarray(features.keypoints)
        if points.shape != (len(features), 2):
            raise ValueError("Expected keypoints to be an (N, 2) array")

        w,h = source_size

        # Only extract depths for points that are finite and within the source image bounds
        valid = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h))
        depths = np.full(len(points), np.nan, dtype=np.float64)
        depths[valid] = src_image.depth_at(points[valid])

        projection = reproject_keypoints(points, depths, src_intrinsics, dest_intrinsics, src_image, dest_image, target_size, self.margin, self.eps)


        keep_mask = self.compute_keep_mask(projection)

        # Return the 
        return PruningResult(features=self.filter_features(features, keep_mask), keep_mask=keep_mask, reprojection=projection)

    def prune_image_pair(self, f_a: FeatureKeypoints, f_b: FeatureKeypoints, img_a: Image, img_b: Image, intrinsics_a: Intrinsics = None, intrinsics_b: Intrinsics = None):
        """Prune keypoints predicted to be outside the other camera's view for both images"""

        if intrinsics_a is None:
            intrinsics_a = ROS_DEFAULT
        if intrinsics_b is None:
            intrinsics_b = intrinsics_a

        results_0 = self.prune(f_a, img_a, img_b, intrinsics_a, intrinsics_b) # pruning keypoints in image A based on their predicted visibility in image B
        results_1 = self.prune(f_b, img_b, img_a, intrinsics_b, intrinsics_a) # pruning keypoints in image B based on their predicted visibility in image A

        return results_0, results_1

def remap_matched_indices(indices, result0, result1):

    indices = np.asarray(indices)

    return np.column_stack((result0.kept_indices[indices[:, 0]], result1.kept_indices[indices[:, 1]]))
