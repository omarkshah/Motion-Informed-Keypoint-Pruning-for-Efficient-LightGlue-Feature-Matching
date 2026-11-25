from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class ViewReprojection:

    pixels: np.ndarray # Nx2 array of predicted pixel coordinates
    depth: np.ndarray  # Nx1 array of the depth values in the target image
    has_depth: np.ndarray # Nx1 boolean array
    in_image: np.ndarray # Nx1 boolean array of the prediced pixels lands in the target image

    @property
    def visible(self):
        return self.has_depth & self.in_image

    @property
    def out_of_view(self):

    def __len__(self):
        return len(self.pixels)




