import cv2
import numpy as np
import os
import sys

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_degradation import make_dark, make_blurry

def create_synthetic_image():
    """Create a simple image with geometric shapes and text to test edge detection."""
    img = np.ones((400, 600), dtype=np.uint8) * 200  # Light gray background
    
    # Draw some shapes
    cv2.circle(img, (150, 200), 80, 0, -1)           # Black circle
    cv2.rectangle(img, (350, 100), (500, 300), 50, -1) # Dark gray rectangle
    
    # Put some text
    cv2.putText(img, "TEST STROKE", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    
    return img

def main():
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../outputs')), exist_ok=True)
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../outputs'))
    
    print("Generating synthetic image...")
    original = create_synthetic_image()
    cv2.imwrite(os.path.join(out_dir, '00_original.jpg'), original)
    
    print("Applying degradations...")
    dark_img = make_dark(original, factor=0.15)
    blurry_img = make_blurry(original, kernel_size=(25, 25))
    
    cv2.imwrite(os.path.join(out_dir, '01_dark.jpg'), dark_img)
    cv2.imwrite(os.path.join(out_dir, '02_blurry.jpg'), blurry_img)
    
    print("Running Canny Edge Detection...")
    # Baseline limits: 100, 200
    edges_orig = cv2.Canny(original, 100, 200)
    edges_dark = cv2.Canny(dark_img, 100, 200)
    edges_blur = cv2.Canny(blurry_img, 100, 200)
    
    cv2.imwrite(os.path.join(out_dir, '03_edges_original.jpg'), edges_orig)
    cv2.imwrite(os.path.join(out_dir, '04_edges_dark.jpg'), edges_dark)
    cv2.imwrite(os.path.join(out_dir, '05_edges_blurry.jpg'), edges_blur)
    
    # Calculate crude metrics: number of edge pixels detected
    orig_edge_count = np.sum(edges_orig > 0)
    dark_edge_count = np.sum(edges_dark > 0)
    blur_edge_count = np.sum(edges_blur > 0)
    
    print(f"Edges detected in Original: {orig_edge_count}")
    print(f"Edges detected in Dark:     {dark_edge_count} ({(dark_edge_count/orig_edge_count)*100:.2f}%)")
    print(f"Edges detected in Blurry:   {blur_edge_count} ({(blur_edge_count/orig_edge_count)*100:.2f}%)")
    print("\nBaseline test complete. Images saved to outputs/ directory.")
    
if __name__ == "__main__":
    main()
