import numpy as np

from frame import ImageSequence, relative_pose
from features import SuperPointExtractor
from geometry import reproject_keypoints


seq = ImageSequence("rgbd_dataset_freiburg1_xyz")

extractor = SuperPointExtractor()

a = seq.images[0]
b = seq.images[20]

features_a = extractor.extract(a.data)
features_b = extractor.extract(b.data)

# Depth values for source keypoints.
depths_a = a.depth_at(features_a.keypoints)

# Transformation from source A to target B.
R_ab, t_ab = relative_pose(b, a)

# Predict where source features appear in B.
reprojection = reproject_keypoints(
    keypoints=features_a.keypoints,
    depths=depths_a,
    src_intrinsics=seq.intrinsics,
    dest_intrinsics=seq.intrinsics,
    R=R_ab,
    t=t_ab,
    target_size=features_b.image_size,
    margin=20
)

print("Total features:", len(reprojection))

print(
    "Valid source depth:",
    reprojection.has_depth.sum()
)

print(
    "In front of target:",
    reprojection.in_front.sum()
)

print(
    "Behind target camera:",
    reprojection.behind.sum()
)

print(
    "Inside target image:",
    reprojection.in_image.sum()
)

print(
    "Unknown geometry:",
    reprojection.unknown.sum()
)

print(
    "Pruning candidates:",
    reprojection.out_of_view.sum()
)

print(
    "Retained features:",
    reprojection.keep_mask.sum()
)

assert len(features_a) == len(reprojection)

assert (
    features_a.image_size
    == (a.data.shape[1], a.data.shape[0])
)

assert a.depth.shape == a.data.shape[:2]