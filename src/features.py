"""SuperPoint keypoints + LightGlue matching, wrapped around the official cvg/LightGlue package.

Usage:
    extractor = SuperPointExtractor(max_num_keypoints=2048)
    matcher = LightGlueMatcher()

    f0 = extractor.extract(img0.data)    # HxWx3 BGR uint8 (or HxW gray)
    f1 = extractor.extract(img1.data)
    m = matcher.match(f0, f1)

    m.kpts0, m.kpts1                     # (K, 2) matched pixel coords (x, y) in each image
    m.scores                             # (K,)  match confidence in [0, 1]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """HxW gray or HxWx3 BGR uint8 (OpenCV convention) -> (C, H, W) float RGB in [0, 1]."""
    if image.ndim == 2:
        t = torch.from_numpy(np.ascontiguousarray(image))[None]
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = np.ascontiguousarray(image[..., ::-1])  # BGR -> RGB
        t = torch.from_numpy(rgb).permute(2, 0, 1)
    else:
        raise ValueError(f"Expected HxW or HxWx3 image, got shape {image.shape}")

    if image.dtype == np.uint8:
        t = t.float().div(255.0)
    else:
        t = t.float()
    return t.to(device)


@dataclass
class Features:
    """SuperPoint output for one image."""

    keypoints: np.ndarray      # (N, 2) pixel coords (x, y)
    scores: np.ndarray         # (N,)   detection scores
    descriptors: np.ndarray    # (N, 256)
    image_size: tuple[int, int]  # (W, H)
    raw: dict = field(repr=False)  # batched torch tensors, exactly what LightGlue expects

    def __len__(self) -> int:
        return len(self.keypoints)


@dataclass
class Matches:
    """LightGlue matches between two Features."""

    kpts0: np.ndarray    # (K, 2) matched keypoints in image 0
    kpts1: np.ndarray    # (K, 2) matched keypoints in image 1
    scores: np.ndarray   # (K,)   match confidence
    indices: np.ndarray  # (K, 2) indices into features0.keypoints / features1.keypoints
    num_layers: int      # transformer layers run before LightGlue's adaptive early exit

    def __len__(self) -> int:
        return len(self.scores)


class SuperPointExtractor:
    """SuperPoint keypoint detector + descriptor.

    Args:
        max_num_keypoints: keep at most this many keypoints (highest scores). None = no limit.
        detection_threshold: minimum keypoint score.
        nms_radius: non-maximum suppression radius in pixels.
        resize: resize the long side to this many pixels before detection. None keeps the
            native resolution (TUM is 640x480, so no resize is needed). Keypoints are always
            returned in the original image's pixel coordinates.
    """

    def __init__(
        self,
        max_num_keypoints: int | None = 2048,
        detection_threshold: float = 0.0005,
        nms_radius: int = 4,
        resize: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.device = torch.device(device) if device is not None else default_device()
        self.resize = resize
        self.model = SuperPoint(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            nms_radius=nms_radius,
        ).eval().to(self.device)

    @torch.inference_mode()
    def extract(self, image: np.ndarray) -> Features:
        raw = self.model.extract(to_tensor(image, self.device), resize=self.resize)
        feats = rbd(raw)  # drop batch dim for the numpy copies
        w, h = feats["image_size"].tolist()
        return Features(
            keypoints=feats["keypoints"].cpu().numpy(),
            scores=feats["keypoint_scores"].cpu().numpy(),
            descriptors=feats["descriptors"].cpu().numpy(),
            image_size=(int(w), int(h)),
            raw=raw,
        )


class LightGlueMatcher:
    """LightGlue matcher for SuperPoint features.

    Args:
        depth_confidence: early-exit confidence (lower = faster, -1 disables).
        width_confidence: point-pruning confidence (lower = faster, -1 disables).
        filter_threshold: minimum match score to keep a match.
    """

    def __init__(
        self,
        features: str = "superpoint",
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        device: torch.device | str | None = None,
    ):
        self.device = torch.device(device) if device is not None else default_device()
        self.model = LightGlue(
            features=features,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
            filter_threshold=filter_threshold,
        ).eval().to(self.device)

    @torch.inference_mode()
    def match(self, f0: Features, f1: Features) -> Matches:
        out = rbd(self.model({"image0": f0.raw, "image1": f1.raw}))
        idx = out["matches"].cpu().numpy().astype(np.int64)
        stop = out.get("stop", -1)
        return Matches(
            kpts0=f0.keypoints[idx[:, 0]],
            kpts1=f1.keypoints[idx[:, 1]],
            scores=out["scores"].cpu().numpy(),
            indices=idx,
            num_layers=int(stop),
        )
