import cv2
import torch
import numpy as np

class VolumetricPerception:
    def __init__(self):
        print("Loading MiDaS for Monocular Depth Estimation...")
        # In a real environment, we'd load MiDaS, but for stability in dl-env
        # without external huggingface-hub reliance we will use a small fallback variant
        # or load from torch hub cautiously.
        try:
            # Attempt to use MiDaS Small to keep it fast
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
        except Exception as e:
            print(f"Warning: Falling back to heuristic depth due to {e}")
            self.midas = None
            self.transform = None

    def get_depth_map(self, img_rgb):
        if self.midas is None:
            # Fallback: Use Grayscale as a very crude proxy for depth intensity
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            return gray.astype(np.float32)
            
        input_batch = self.transform(img_rgb)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth = prediction.cpu().numpy()
        return depth

    def get_surface_normals(self, depth_map):
        """
        Computes the X and Y gradient of the depth map to estimate 3D surface normals.
        """
        # Calculate gradients using Sobel
        gx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # The normal vector is essentially (-gx, -gy, 1) normalized
        length = np.sqrt(gx**2 + gy**2 + 1.0)
        nx = -gx / length
        ny = -gy / length
        nz = 1.0 / length
        
        normals = np.dstack((nx, ny, nz))
        return normals

    def generate_parametric_hatching(self, normals, mask=None):
        """
        Generates stroke lines (hatching) that follow the perpendicular surface curvature.
        Returns a list of 2D coordinates representing continuous vector paths for the robot.
        """
        h, w = normals.shape[:2]
        paths = []
        
        step_size = 15 # density of the hatching
        
        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                if mask is not None and mask[y, x] == 0:
                    continue
                # For simplicity, we create a short stroke tangent to the gradient
                nx = normals[y, x, 0]
                ny = normals[y, x, 1]
                
                # Tangent direction (perpendicular to normal gradient in 2D space)
                tx, ty = -ny, nx 
                
                stroke_length = 10
                pt1 = (int(x - tx * stroke_length), int(y - ty * stroke_length))
                pt2 = (int(x + tx * stroke_length), int(y + ty * stroke_length))
                
                paths.append([pt1, pt2])
                
        return paths

    def process_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        depth = self.get_depth_map(img_rgb)
        normals = self.get_surface_normals(depth)
        
        # Optional: threshold depth to create a mask for foreground objects
        depth_normalized = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, mask = cv2.threshold(depth_normalized, 100, 255, cv2.THRESH_BINARY)
        
        paths = self.generate_parametric_hatching(normals, mask)
        return paths

if __name__ == "__main__":
    print("Volumetric Perception Engine loaded.")
