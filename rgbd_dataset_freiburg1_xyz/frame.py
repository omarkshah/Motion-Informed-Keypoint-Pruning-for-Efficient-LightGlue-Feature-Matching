from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2


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
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _read_tum_file(path):
    """Skip '#' comment lines, return list of token lists."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(line.split())
    return rows


class TumSequence:
    def __init__(self, root, max_diff=0.02, interpolate=True):
        self.root = Path(root)
        self.max_diff = max_diff
        self.interpolate = interpolate

        # rgb.txt: "timestamp filename"
        self.rgb = {}   # name -> timestamp
        for ts, name in _read_tum_file(self.root / "rgb.txt"):
            self.rgb[name] = float(ts)
        # also allow lookup by bare filename
        self._basename = {Path(n).name: n for n in self.rgb}

        # groundtruth.txt: "ts tx ty tz qx qy qz qw"
        gt = np.array(_read_tum_file(self.root / "groundtruth.txt"), dtype=np.float64)
        self.gt = gt[np.argsort(gt[:, 0])]

        self.frames = self.cleanup_data()

    # ---------- lookups ----------
    def _resolve(self, image_name):
        if image_name in self.rgb:
            return image_name
        return self._basename[Path(image_name).name]

    def get_timestamp(self, image_name) -> float:
        return self.rgb[self._resolve(image_name)]

    def get_image(self, image_name, flags=cv2.IMREAD_COLOR):
        return cv2.imread(str(self.root / self._resolve(image_name)), flags)

    # ---------- ground truth ----------
    def _pose_at(self, ts):
        """Return (t, R) at time ts, or None if no groundtruth within max_diff."""
        times = self.gt[:, 0]
        i = np.searchsorted(times, ts)          # first index with times[i] >= ts
        lo, hi = max(i - 1, 0), min(i, len(times) - 1)

        # nearest neighbour check against tolerance
        nearest = lo if abs(times[lo] - ts) <= abs(times[hi] - ts) else hi
        if abs(times[nearest] - ts) > self.max_diff:
            return None

        # no interpolation possible at the edges, or if disabled
        if not self.interpolate or lo == hi or not (times[lo] <= ts <= times[hi]):
            r = self.gt[nearest]
            return r[1:4].copy(), quaternion_to_matrix(*r[4:8])

        # linear interpolation between the two bracketing poses
        a, b = self.gt[lo], self.gt[hi]
        alpha = (ts - a[0]) / (b[0] - a[0])
        t = (1 - alpha) * a[1:4] + alpha * b[1:4]
        qa, qb = a[4:8], b[4:8]
        if np.dot(qa, qb) < 0:                  # q and -q are the same rotation
            qb = -qb
        q = (1 - alpha) * qa + alpha * qb       # nlerp; fine for ~10 ms gaps
        return t, quaternion_to_matrix(*q)

    def get_ground_truth(self, image_name):
        """Returns (translation (3,), rotation (3,3)), or None if unmatched."""
        return self._pose_at(self.get_timestamp(image_name))

    # ---------- cleanup ----------
    def cleanup_data(self):
        """Keep only RGB frames that have a groundtruth pose within max_diff."""
        frames, dropped = [], 0
        for name, ts in sorted(self.rgb.items(), key=lambda kv: kv[1]):
            pose = self._pose_at(ts)
            if pose is None:
                dropped += 1
                continue
            t, R = pose
            frames.append(Frame(name, ts, self.root / name, t, R))
        print(f"Kept {len(frames)} frames, dropped {dropped} (no GT within {self.max_diff}s)")
        return frames


def relative_pose(f1: Frame, f2: Frame):
    """Pose of camera 2 expressed in camera 1's frame (what the pruner needs)."""
    R_rel = f1.R.T @ f2.R
    t_rel = f1.R.T @ (f2.t - f1.t)
    return R_rel, t_rel