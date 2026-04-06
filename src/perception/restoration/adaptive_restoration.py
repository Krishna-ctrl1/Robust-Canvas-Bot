"""
Adaptive Degradation-Aware Restoration Router (ADRR)

A novel contribution that analyzes input images to detect which degradations
are present and selectively applies only the relevant restoration steps.

Unlike blind restoration pipelines that unconditionally apply all enhancement
stages (which can degrade already-clean images), ADRR uses lightweight
statistical image analysis to make per-image routing decisions.

Detection methods:
  - Low-light: Mean brightness + 10th-percentile intensity analysis
  - Motion blur: Laplacian variance (focus measure)
  - Combined: Both degradations can be detected and corrected simultaneously

Reference:
  Pertuz et al. (2013), "Analysis of focus measure operators for shape-from-focus",
  Pattern Recognition, 46(5), 1415-1432. (Laplacian variance as focus measure)
"""

import cv2
import numpy as np
from .zero_dce import enhance_lowlight
from .deblur import enhance_blur


class AdaptiveRestorationRouter:
    """
    Analyzes an input image and applies only the restoration steps that
    are needed, based on detected degradation signatures.

    Thresholds are calibrated on a synthetic benchmark covering multiple
    degradation types and severity levels.
    """

    # --- Calibrated thresholds ---
    # Low-light detection: image is considered dark if mean brightness
    # is below this AND the 10th percentile is near-black.
    BRIGHTNESS_MEAN_THRESHOLD = 90      # [0-255] mean pixel intensity
    BRIGHTNESS_P10_THRESHOLD = 40       # [0-255] 10th percentile intensity

    # Blur detection: Laplacian variance measures "focus". Sharp images have
    # high variance (many strong edges); blurry images have low variance.
    # Lowered to 30 to be conservative and avoid deblurring clean images or isotropic Gaussian blur unnecessarily.
    LAPLACIAN_VARIANCE_THRESHOLD = 30  # variance of Laplacian response

    def __init__(self):
        pass

    @staticmethod
    def detect_low_light(image):
        """
        Detect whether an image suffers from low-light conditions.

        Uses two complementary statistics:
          1. Mean brightness of the grayscale image
          2. 10th percentile intensity (captures whether shadows are crushed)

        Both must fall below thresholds to trigger low-light restoration.
        This avoids false positives on images that are simply dark-themed
        but properly exposed.

        Returns:
            (is_dark: bool, diagnostics: dict)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        mean_brightness = np.mean(gray)
        p10 = np.percentile(gray, 10)
        p90 = np.percentile(gray, 90)
        dynamic_range = p90 - p10

        is_dark = (
            mean_brightness < AdaptiveRestorationRouter.BRIGHTNESS_MEAN_THRESHOLD
            and p10 < AdaptiveRestorationRouter.BRIGHTNESS_P10_THRESHOLD
        )

        diagnostics = {
            "mean_brightness": float(mean_brightness),
            "p10_intensity": float(p10),
            "p90_intensity": float(p90),
            "dynamic_range": float(dynamic_range),
            "is_dark": is_dark,
        }
        return is_dark, diagnostics

    @staticmethod
    def detect_motion_blur(image):
        """
        Detect whether an image suffers from motion blur using the
        Laplacian variance focus measure.

        The Laplacian operator responds strongly to edges and fine detail.
        In a sharp image, conv(I, Laplacian) has high variance because edges
        produce large responses while flat regions produce small ones.
        In a blurry image, edges are smeared out, so the Laplacian response
        is uniformly low -> low variance.

        Returns:
            (is_blurry: bool, diagnostics: dict)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(laplacian))
        lap_mean = float(np.mean(np.abs(laplacian)))

        is_blurry = lap_var < AdaptiveRestorationRouter.LAPLACIAN_VARIANCE_THRESHOLD

        diagnostics = {
            "laplacian_variance": lap_var,
            "laplacian_mean_abs": lap_mean,
            "is_blurry": is_blurry,
        }
        return is_blurry, diagnostics

    def analyze(self, image):
        """
        Run all degradation detectors on the input image.

        Returns:
            dict with keys 'low_light', 'motion_blur', each containing
            (detected: bool, diagnostics: dict)
        """
        is_dark, dark_diag = self.detect_low_light(image)
        is_blurry, blur_diag = self.detect_motion_blur(image)

        return {
            "low_light": {"detected": is_dark, "diagnostics": dark_diag},
            "motion_blur": {"detected": is_blurry, "diagnostics": blur_diag},
        }

    def restore(self, image):
        """
        Adaptively restore an image by applying only the needed corrections.

        Pipeline:
          1. Analyze image for degradation signatures
          2. If low-light detected -> apply Zero-DCE enhancement
          3. If motion blur detected -> apply Wiener deconvolution
          4. Return restored image + full diagnostic report

        This is the main entry point for the ADRR module.

        Returns:
            (restored_image, report: dict)
        """
        report = self.analyze(image)
        restored = image.copy()
        steps_applied = []

        # Step 1: Low-light enhancement (if detected)
        if report["low_light"]["detected"]:
            restored = enhance_lowlight(restored)
            steps_applied.append("zero_dce_enhancement")

        # Step 2: Motion deblurring (if detected)
        if report["motion_blur"]["detected"]:
            restored = enhance_blur(restored)
            steps_applied.append("wiener_deconvolution")

        # If nothing was detected, the image passes through unchanged
        if not steps_applied:
            steps_applied.append("none_required")

        report["steps_applied"] = steps_applied
        report["any_restoration_applied"] = len(steps_applied) > 0 and steps_applied[0] != "none_required"

        return restored, report


def adaptive_restore(image):
    """
    Convenience function for single-call adaptive restoration.

    Returns:
        (restored_image, report_dict)
    """
    router = AdaptiveRestorationRouter()
    return router.restore(image)
