import cv2
import numpy as np

def make_dark(image, factor=0.2):
    """
    Artificially darken an image by scaling its pixel intensities.
    """
    # Ensure image is float for scaling, then clip and convert back to uint8
    darkened = image.astype(np.float32) * factor
    return np.clip(darkened, 0, 255).astype(np.uint8)

def make_blurry(image, kernel_size=(15, 15)):
    """
    Artificially blur an image using a Gaussian kernel.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)
