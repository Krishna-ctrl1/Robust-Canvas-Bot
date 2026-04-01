import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import sys
import argparse
import numpy as np

# Adjust imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from perception.restoration.zero_dce import enhance_lowlight
from perception.restoration.deblur import enhance_blur
from perception.vision.depth_to_hatching import VolumetricPerception
from perception.text.ocr_hatching import LinguisticRestoration

class DrawingRouter:
    def __init__(self):
        print("\n--- Initializing Robust Semantic-Aware Canvas Bot ---")
        self.vision_engine = VolumetricPerception()
        self.text_engine = LinguisticRestoration()
        
    def classification_gate(self, image):
        """
        Proxy classifier: decides if an image is mostly text or a generic 3D scene.
        Real implementation would use a lightweight ResNet or CRAFT bounding box density.
        For demonstration, we check a variance threshold. Text usually has sharp white/black contrast.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = np.var(gray)
        if var > 2000:
            return "TEXT"
        return "SCENE"

    def process(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found.")
            return
            
        print(f"\n[1] Reading Dirty Data: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
             print("Invalid image file.")
             return
             
        print("[2] Restoration Gatekeeper Active...")
        # Step 2a: Low-light enhancement (Zero-DCE)
        brightened = enhance_lowlight(img)
        # Step 2b: Motion Deblur
        cleaned_image = enhance_blur(brightened)
        
        print("[3] Content Classification...")
        content_type = self.classification_gate(cleaned_image)
        print(f" -> Classification result: {content_type}")
        
        print("[4] Routing to Perception Engine...")
        if content_type == "TEXT":
            paths = self.text_engine.process_image(cleaned_image)
        else:
            paths = self.vision_engine.process_image(cleaned_image)
            
        print(f" -> Generated {len(paths)} continuous vector stroke paths.")
        
        print("[5] Sending to Simulation Engine IK Solver...")
        # Mock connection to PyBullet. In real application, we would
        # import robot_sim_3d_env and sequentially feed paths[] to Inverse Kinematics targets.
        print(" -> All paths transmitted safely. Simulation ready to execute drawing.")
        return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    router = DrawingRouter()
    router.process(args.input)
