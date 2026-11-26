import argparse
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

from camera import ROS_DEFAULT
from visualization import visualize_results
from features import FeatureMatches, LightGlueMatcher, SuperPointExtractor
from frame import ImageSequence
from geometry import relative_transform
from pruner import MotionPruner


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=Path, default=ROOT / "rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--idx", type=int, default=100, help="First frame index")
    parser.add_argument("--gap", type=int, default=100, help="Gap to second frame")
    parser.add_argument("--margin", type=float, default=20.0, help="Pruning margin (pixels)")
    parser.add_argument("--max-kp", type=int, default=2048)
    parser.add_argument("--max-lines", type=int, default=150, help="Maximum match lines per panel")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)

    args = parser.parse_args()

    # Step 1: Load two RGB-D frames and their ground-truth camera poses
    img_seq = ImageSequence(args.seq, intrinsics=ROS_DEFAULT)

    i, j = args.idx, args.idx + args.gap
    if j >= len(img_seq):
        parser.error(f"Frame indices out of range (sequence has {len(img_seq)} frames)")    

    a,b = img_seq.images[i], img_seq.images[j]
    R, t = relative_transform(a, b)

    rotation = np.degrees(np.arccos((np.trace(R) - 1) / 2))

    print(f"\nFrames {i} and {j} | dt={b.timestamp - a.timestamp:.3f}s")
    print(f"Camera motion: {np.linalg.norm(t):.3f} m, {rotation:.2f} deg")
    print(f"Pruning margin: {args.margin:g} px")

    # Step 2: Initialize SuperPoint, Pruner, and LightGlue matcher
    extractor = SuperPointExtractor(device=args.device, max_num_keypoints=args.max_kp)
    matcher = LightGlueMatcher(device=extractor.device)
    pruner = MotionPruner(margin=args.margin)

    print(f"Device: {extractor.device}")

    # Step 3: Extract keypoints (features) from SuperPoint
    features_a = extractor.extract(a.data)
    features_b = extractor.extract(b.data)

    if a.data.shape != b.data.shape:
        raise ValueError("Demo expects both images to have the same shape")

    _ = a.depth, b.depth # load the depth maps (these are a cached property)

    # Step 4: Prune keypoints predicted to be outside the other camera's view
    results_a, results_b = pruner.prune_image_pair(features_a, features_b, a, b, img_seq.intrinsics, img_seq.intrinsics)

    # Step 5: Run LightGlue on the retained features
    matches = None
    if len(features_a) == 0 or len(features_b) == 0:
        print("No features to match.")
        matches = FeatureMatches(kpts0=np.empty((0, 2)), kpts1=np.empty((0, 2)), score=np.empty(0), indices=np.empty((0,2), dtype=np.int64), num_layers=0)

    matches =  matcher.match_features(results_a.features, results_b.features)

    visualize_results(a.data, b.data, features_a, features_b, results_a, results_b, matches, matches, i, j, 0.0, 0.0, 0.0, max_lines=args.max_lines)

if __name__ == "__main__":
    main()