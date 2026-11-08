from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

from tum_associate import read_file_list, associate


@dataclass
class Frame:
    name: str            # e.g. "rgb/1305031102.175304.png"
    timestamp: float
    image_path: Path
    t: np.ndarray        # (3,)  translation, camera -> world
    R: np.ndarray        # (3,3) rotation,    camera -> world


def quaternion_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    """TUM order is (qx, qy, qz, qw). Returns a 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Invalid zero-length quaternion")

    q /= norm
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


class TumSequence:
    def __init__(self, root, max_diff=0.02):
        self.root = Path(root)
        self.max_diff = max_diff

        # rgb.txt: "timestamp filename"              -> {ts: [filename]}
        # groundtruth.txt: "ts tx ty tz qx qy qz qw" -> {ts: [tx, ty, tz, qx, qy, qz, qw]}
        self._rgb_list = read_file_list(self.root / "rgb.txt")
        self._gt_list = read_file_list(self.root / "groundtruth.txt")

        self.frames = self.load_data()

        # lookup by full relative name ("rgb/123.png") or bare filename ("123.png")
        self._by_name = {}
        for f in self.frames:
            self._by_name[f.name] = f
            self._by_name[Path(f.name).name] = f

    def _get(self, image_name) -> Frame:
        frame = self._by_name.get(image_name) or self._by_name.get(Path(image_name).name)

        if frame is None:
            raise FileNotFoundError(
                f"RGB image not found (or dropped for having no groundtruth): {image_name}"
            )

        return frame

    def get_timestamp(self, image_name) -> float:
        return self._get(image_name).timestamp

    def get_image(self, image_name, flags=cv2.IMREAD_COLOR):
        # openCV loads colour images in BGR order not RGB
        path = self._get(image_name).image_path
        image = cv2.imread(str(path), flags)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        return image

    def get_ground_truth(self, image_name):
        """Returns (translation (3,), rotation (3,3))."""
        frame = self._get(image_name)
        return frame.t, frame.R

    # ---------- cleanup ----------
    def load_data(self):
        """Keep only RGB frames that have a groundtruth pose within max_diff."""
        # one-to-one matches: [(rgb_ts, gt_ts), ...], sorted by rgb_ts
        matches = associate(self._rgb_list, self._gt_list, 0.0, self.max_diff)

        frames = []
        for rgb_ts, gt_ts in matches:
            name = self._rgb_list[rgb_ts][0]
            tx, ty, tz, qx, qy, qz, qw = map(float, self._gt_list[gt_ts])

            frames.append(Frame(
                name=name,
                timestamp=rgb_ts,
                image_path=self.root / name,
                t=np.array([tx, ty, tz], dtype=np.float64),
                R=quaternion_to_matrix(qx, qy, qz, qw),
            ))

        dropped = len(self._rgb_list) - len(frames)
        print(f"Kept {len(frames)} frames, dropped {dropped} (no GT within {self.max_diff}s)")
        return frames


def relative_pose(f1: Frame, f2: Frame):
    """Pose of camera 2 expressed in camera 1's frame (what the pruner needs)."""
    # t - position of camera in world frame, R - rotation of camera in world frame
    cam2_orientation_rel_cam1 = f1.R.T @ f2.R
    cam2_position_rel_cam1 = f1.R.T @ (f2.t - f1.t)
    return cam2_orientation_rel_cam1, cam2_position_rel_cam1

def transform_1_to_2(f1: Frame, f2: Frame):
    """Transform 3D points from camera 1 coordinates to camera 2 coordinates."""

    cam1_orientation_rel_cam2 = f2.R.T @ f1.R
    cam1_position_rel_cam2 = f2.R.T @ (f1.t - f2.t)

    return cam1_orientation_rel_cam2, cam1_position_rel_cam2