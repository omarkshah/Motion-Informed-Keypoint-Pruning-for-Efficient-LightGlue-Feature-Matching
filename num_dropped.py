
from pathlib import Path
import sys

# Add src directory to Python import path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from frame import ImageSequence


def main():
    # Dataset path
    dataset_path = ROOT / "rgbd_dataset_freiburg1_xyz"

    # Maximum allowed timestamp difference (seconds)
    max_dt = 0.02

    # Load sequence
    seq = ImageSequence(dataset_path, max_dt=max_dt)

    # Statistics
    total_rgb = len(seq._rgb_list)
    matched = len(seq.images)
    dropped = total_rgb - matched

    matched_percentage = matched / total_rgb * 100 if total_rgb else 0
    dropped_percentage = dropped / total_rgb * 100 if total_rgb else 0

    print("\n========== SEQUENCE STATISTICS ==========")
    print(f"Dataset:                    {dataset_path.name}")
    print(f"Max timestamp difference:   {max_dt * 1000:.2f} ms")

    print(f"\nTotal RGB images:            {total_rgb}")
    print(f"Images with ground truth:   {matched}")
    print(f"Images without ground truth:{dropped}")

    print(f"\nMatched percentage:         {matched_percentage:.2f}%")
    print(f"Dropped percentage:         {dropped_percentage:.2f}%")
    print("=========================================")


if __name__ == "__main__":
    main()
