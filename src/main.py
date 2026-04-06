"""
DrawingRouter — Main Pipeline Orchestrator

Orchestrates the full Robust Semantic-Aware Robotic Canvas Reconstruction
pipeline:

  1. Adaptive Degradation-Aware Restoration (ADRR)
  2. Learned Content Classification (Text vs Scene)
  3. Modality-Specific Perception (Volumetric or Linguistic)
  4. 2D Stroke Path Rendering (visual output of robot drawing)

Usage:
    python src/main.py --input <image_path> [--render] [--output <dir>]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from perception.restoration.adaptive_restoration import AdaptiveRestorationRouter
from perception.restoration.zero_dce import enhance_lowlight
from perception.restoration.deblur import enhance_blur
from perception.vision.depth_to_hatching import VolumetricPerception
from perception.text.ocr_hatching import LinguisticRestoration
from perception.content_classifier import ContentClassifier
from simulation.robot_sim_3d_env import RobotDrawingRenderer


class DrawingRouter:
    """
    Main pipeline controller that routes images through restoration,
    classification, perception, and rendering stages.
    """

    def __init__(self):
        print("\n--- Initializing Robust Semantic-Aware Canvas Bot ---")
        self.adrr = AdaptiveRestorationRouter()
        self.classifier = ContentClassifier(auto_train=True)
        self.vision_engine = VolumetricPerception()
        self.text_engine = LinguisticRestoration()
        self.renderer = RobotDrawingRenderer()
        print("--- Initialization Complete ---\n")

    def process(self, image_path, render_output=False, output_dir=None):
        """
        Process a single image through the full pipeline.

        Args:
            image_path: Path to the input image.
            render_output: If True, render stroke paths as a 2D image.
            output_dir: Directory to save rendered output.

        Returns:
            List of stroke paths (each path is a list of (x,y) tuples).
        """
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found.")
            return []

        print(f"[1] Reading Input: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Invalid image file.")
            return []

        # ── Stage 1: Adaptive Restoration ─────────────────────────────────
        print("[2] Adaptive Degradation-Aware Restoration Router (ADRR)...")
        cleaned_image, restoration_report = self.adrr.restore(img)

        steps = restoration_report["steps_applied"]
        print(f"    Degradation detected:")
        print(f"      Low-light: {restoration_report['low_light']['detected']} "
              f"(mean brightness: {restoration_report['low_light']['diagnostics']['mean_brightness']:.1f})")
        print(f"      Motion blur: {restoration_report['motion_blur']['detected']} "
              f"(Laplacian var: {restoration_report['motion_blur']['diagnostics']['laplacian_variance']:.1f})")
        print(f"    Steps applied: {steps}")

        # ── Stage 2: Content Classification ───────────────────────────────
        print("[3] Learned Content Classification Gate...")
        content_type, confidence, features = self.classifier.predict(cleaned_image)
        print(f"    Classification: {content_type} (confidence: {confidence:.3f})")

        # ── Stage 3: Modality-Specific Perception ─────────────────────────
        print(f"[4] Routing to {'Linguistic' if content_type == 'TEXT' else 'Volumetric'} Perception Engine...")
        if content_type == "TEXT":
            paths = self.text_engine.process_image(cleaned_image)
        else:
            paths = self.vision_engine.process_image(cleaned_image)

        print(f"    Generated {len(paths)} continuous vector stroke paths.")

        # ── Stage 4: Rendering ────────────────────────────────────────────
        if render_output:
            print("[5] Rendering stroke paths...")
            if output_dir is None:
                output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs'))
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            render_path = os.path.join(output_dir, f"render_{base_name}.png")

            comparison = self.renderer.render_comparison(cleaned_image, paths, render_path)
            print(f"    Rendered comparison saved to {render_path}")

            # Also save just the restored image
            restored_path = os.path.join(output_dir, f"restored_{base_name}.png")
            cv2.imwrite(restored_path, cleaned_image)
            print(f"    Restored image saved to {restored_path}")
        else:
            print("[5] Paths ready for robotic execution (use --render to visualize).")

        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robust Semantic-Aware Robotic Canvas Reconstruction Pipeline"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--render", action="store_true",
                        help="Render stroke paths as 2D image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for rendered images")
    args = parser.parse_args()

    router = DrawingRouter()
    paths = router.process(args.input, render_output=args.render, output_dir=args.output)

    if paths:
        print(f"\nPipeline complete. Total paths: {len(paths)}")
    else:
        print("\nPipeline produced no paths.")
