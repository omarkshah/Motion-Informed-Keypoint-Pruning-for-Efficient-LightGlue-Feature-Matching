import numpy as np
import torch
from lightglue import SuperPoint
from lightglue.utils import rbd
from dataclasses import dataclass

torch.set_grad_enabled(False) # disable gradient computation for inference

@dataclass
class FeatureKeypoints:
    keypoints: np.ndarray # Nx2 array of keypoint coordinates
    scores: np.ndarray # N array of confidence scores for the keypoints
    descriptors: np.ndarray # N x 256 array of descriptors
    image_size: tuple[int, int] # (width, height) of the image as it may be resized
    raw_superpoint: dict = None # the raw output from the SuperPoint model, which is the expected input to LightGlue

    def __len__(self):
        return len(self.keypoints)

@dataclass
class FeatureMatches:
    
    kpts0: np.ndarray    # Kx2 matched keypoints in image 0
    kpts1: np.ndarray    # Kx2 matched keypoints in image 1
    scores: np.ndarray   # K vector denoting the match confidence
    indices: np.ndarray  # Kx2 indices into the original keypoitn arrays denoting the matched pairs
    num_layers: int      # number of transformer layers run before LightGlue's adaptive early exit

    def __len__(self) -> int:
        return len(self.scores)

class SuperPointExtractor:
    def __init__(self, device=None, max_num_keypoints=2048, resize=None):
        self.device = torch.device(device) if device is not None else get_device()
        self.max_num_keypoints = max_num_keypoints
        self.resize = resize
        self.superpoint = SuperPoint(max_num_keypoints=self.max_num_keypoints, resize=self.resize).eval().to(self.device)

    def extract(self, image):
        """
        Extracts keypoints and descriptors from the input image using SuperPoint

        Input:
        - image: np.ndarray of shape (H, W) or (H, W, 3), the input image

        Output:
        - FeatureKeypoints object for the image
        """

        raw_superpoint = self.superpoint.extract(to_tensor(image, self.device), resize=self.resize)
        features = rbd(raw_superpoint)
        w, h = features["image_size"].tolist()

        feature_keypoints = FeatureKeypoints(features["keypoints"].cpu().numpy(), features["keypoint_scores"].cpu().numpy(), features["descriptors"].cpu().numpy(), (w,h), raw_superpoint)

        return feature_keypoints

class LightGlue:
    def __init__(self, features = "superpoint",
                    input_dim = 256,
                    descriptor_dim = 256,
                    depth_confidence = 0.95,
                    width_confidence = 0.99,
                    filter_threshold = 0.1,
                    device = None):
        self.device = torch.device(device) if device is not None else default_device()
        
        self.model = LightGlue(
            features=features,
            input_dim=input_dim,
            descriptor_dim=descriptor_dim,
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
            filter_threshold=filter_threshold,
        ).eval().to(self.device)

    def match_features(self, f0: Features, f1: Features) -> Matches:
        out = rbd(self.model({"image0": f0.raw_superpoint, "image1": f1.raw_superpoint}))
    
        idx = out["matches"].cpu().numpy().astype(np.int64)
        stop = out.get("stop", -1)

        return Matches(
            kpts0=f0.keypoints[idx[:, 0]],
            kpts1=f1.keypoints[idx[:, 1]],
            scores=out["scores"].cpu().numpy(),
            indices=idx,
            num_layers=int(stop),
        )



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def to_tensor(image, device):
    """
    Converts a numpy image to a torch tensor

    Input:
    - image: np.ndarray of shape (H, W) or (H, W, 3), the input image
    - device: torch.device, the device to move the tensor to

    Output:
    - torch.Tensor of shape (C, H, W), the input image as a tensor
    """

    tensor = None

    if image.ndim == 2:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0) # (1, H, W)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = np.ascontiguousarray(image[..., ::-1]) # BGR to RGB conversion because OpenCV images are in BGR format
        tensor = torch.from_numpy(image).permute(2,0,1) # Convert to torch tensor of shape (C, H, W)
    else:
        raise ValueError(f"Excpected 2D grayscale or 3D RGB image, got shape {image.shape}")

    if image.dtype == np.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()

    return tensor.to(device)

