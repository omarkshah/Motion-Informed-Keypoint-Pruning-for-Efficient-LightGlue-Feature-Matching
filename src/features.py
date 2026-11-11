import numpy as np
import torch
from lightglue import SuperPoint
from lightglue.utils import rbd

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

class FeatureKeypoints:
    keypoints: np.ndarray # Nx2 array of keypoint coordinates
    scores: np.ndarray # N array of confidence scores for the keypoints
    descriptors: np.ndarray # N x 256 array of descriptors
    image_size: tuple[int, int] # (width, height) of the image as it may be resized

    def __len__(self):
        return len(self.keypoints)

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

        raw_superpoint = self.superpoint.extract(image, resize=self.resize)
        features = rbd(raw_superpoint)
        w, h = features["image_size"].tolist()

        feature_keypoints = FeatureKeypoints(features["keypoints"].cpu().numpy(), features["scores"].cpu().numpy(), features["descriptors"].cpu().numpy(), (w,h))

        return feature_keypoints

    

