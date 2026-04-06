"""
Synthetic Data Degradation Utilities

Provides controlled degradation functions that simulate real-world image
corruption. Each function is parameterised so that the benchmark generator
can sweep across severity levels.

Degradation models:
  - Low-light: Multiplicative intensity scaling I_d = alpha * I_0
  - Gaussian blur: Isotropic Gaussian convolution
  - Motion blur: Directional linear PSF convolution
  - Combined: Compound degradation (dark + blur)
  - Gaussian noise: Additive zero-mean Gaussian noise
"""

import cv2
import numpy as np


def make_dark(image, factor=0.2):
    """
    Simulate low-light conditions via multiplicative intensity scaling.

    Args:
        image: Input image (uint8)
        factor: Scaling factor in (0, 1). Lower = darker.
                Typical range: 0.10 (very dark) to 0.40 (moderately dark)

    Returns:
        Darkened image (uint8)
    """
    darkened = image.astype(np.float32) * factor
    return np.clip(darkened, 0, 255).astype(np.uint8)


def make_blurry(image, kernel_size=(15, 15)):
    """
    Simulate defocus/Gaussian blur via isotropic Gaussian convolution.

    Args:
        image: Input image (uint8)
        kernel_size: Gaussian kernel size (must be odd). Larger = more blur.

    Returns:
        Blurred image (uint8)
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def make_motion_blur(image, kernel_size=15, angle=0):
    """
    Simulate directional motion blur via a linear PSF.

    This models the effect of camera or subject motion during exposure.
    The PSF is a line segment at the specified angle.

    Args:
        image: Input image (uint8)
        kernel_size: Length of the motion blur kernel in pixels.
        angle: Direction of motion in degrees (0=horizontal, 90=vertical).

    Returns:
        Motion-blurred image (uint8)
    """
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # Draw line at the specified angle through the center
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))

    for i in range(kernel_size):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            k[y, x] = 1.0

    k = k / np.sum(k)  # normalise to unit sum
    return cv2.filter2D(image, -1, k)


def add_gaussian_noise(image, sigma=25):
    """
    Add zero-mean Gaussian noise to an image.

    Args:
        image: Input image (uint8)
        sigma: Standard deviation of the noise. Typical range: 10-50.

    Returns:
        Noisy image (uint8)
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_combined(image, dark_factor=0.2, blur_kernel=15, blur_angle=0):
    """
    Apply compound degradation: low-light + motion blur.

    This simulates the common real-world scenario of photographing in
    dim conditions with camera shake.

    Args:
        image: Input image (uint8)
        dark_factor: Low-light scaling factor
        blur_kernel: Motion blur kernel size
        blur_angle: Motion blur angle in degrees

    Returns:
        Degraded image (uint8)
    """
    dark = make_dark(image, factor=dark_factor)
    combined = make_motion_blur(dark, kernel_size=blur_kernel, angle=blur_angle)
    return combined
