from frame import ImageSequence
from features import SuperPointExtractor
import numpy as np

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