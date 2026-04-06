"""
Robotic Drawing Simulation Environment

Provides two modes of operation:
  1. Interactive GUI mode — opens a PyBullet window with manual debug sliders
  2. Headless rendering mode — takes stroke paths, renders them as a 2D image

The simulation includes three safety protocols:
  - Geofencing: Virtual workspace boundaries
  - Singularity avoidance: Jacobian rank monitoring (det(J_t) threshold)
  - Collision prediction: Closest-point distance monitoring

For evaluation purposes, the RobotDrawingRenderer class renders stroke paths
directly as a 2D image without requiring PyBullet, enabling batch evaluation.
"""

import numpy as np
import cv2
import os

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


# ─── Geofencing Configuration ────────────────────────────────────────────────

WORKSPACE_BOUNDS = {
    'x_min': 0.2, 'x_max': 0.8,
    'y_min': -0.5, 'y_max': 0.5,
    'z_safe': 0.05,
}

SINGULARITY_DET_THRESHOLD = 1e-4  # Jacobian determinant threshold


class MotionHaltException(Exception):
    """Raised when a safety protocol triggers an emergency halt."""
    pass


# ─── Safety Protocols ────────────────────────────────────────────────────────

def check_geofence(pos):
    """
    Check whether a position is within the safe workspace boundaries.

    Args:
        pos: (x, y, z) end-effector position

    Returns:
        (is_safe: bool, violation: str or None)
    """
    if pos[0] < WORKSPACE_BOUNDS['x_min'] or pos[0] > WORKSPACE_BOUNDS['x_max']:
        return False, f"X-axis breach: {pos[0]:.3f} outside [{WORKSPACE_BOUNDS['x_min']}, {WORKSPACE_BOUNDS['x_max']}]"
    if pos[1] < WORKSPACE_BOUNDS['y_min'] or pos[1] > WORKSPACE_BOUNDS['y_max']:
        return False, f"Y-axis breach: {pos[1]:.3f} outside [{WORKSPACE_BOUNDS['y_min']}, {WORKSPACE_BOUNDS['y_max']}]"
    if pos[2] < WORKSPACE_BOUNDS['z_safe']:
        return False, f"Z-axis breach: {pos[2]:.3f} below {WORKSPACE_BOUNDS['z_safe']}"
    return True, None


def check_singularity(jacobian_t):
    """
    Check whether the robot's translational Jacobian is near-singular.

    Computes the determinant of J_t * J_t^T (which is a 3x3 matrix
    regardless of DOF count) and warns if it falls below threshold.

    Args:
        jacobian_t: Translational Jacobian matrix, shape (3, N_joints)

    Returns:
        (is_safe: bool, det_value: float)
    """
    J = np.array(jacobian_t)
    # J * J^T gives a 3x3 matrix; its determinant measures manipulability
    JJt = J @ J.T
    det_val = float(np.linalg.det(JJt))

    is_safe = abs(det_val) > SINGULARITY_DET_THRESHOLD
    return is_safe, det_val


# ─── 2D Stroke Path Renderer ─────────────────────────────────────────────────

class RobotDrawingRenderer:
    """
    Renders stroke paths as a 2D image, simulating what the robot would draw.

    This enables visual evaluation of drawing quality without requiring
    a full physics simulation. Paths are rendered as anti-aliased lines
    on a white canvas.
    """

    def __init__(self, canvas_height=400, canvas_width=600, line_thickness=1):
        self.canvas_h = canvas_height
        self.canvas_w = canvas_width
        self.line_thickness = line_thickness

    def render(self, paths, background_color=255):
        """
        Render a list of stroke paths onto a 2D canvas.

        Args:
            paths: List of paths, where each path is a list of (x, y) tuples.
            background_color: Background intensity (0-255).

        Returns:
            Canvas image (uint8, single channel) with the drawn strokes.
        """
        canvas = np.ones((self.canvas_h, self.canvas_w), dtype=np.uint8) * background_color

        strokes_drawn = 0
        strokes_clipped = 0

        for path in paths:
            if len(path) < 2:
                continue

            for i in range(len(path) - 1):
                x1, y1 = int(path[i][0]), int(path[i][1])
                x2, y2 = int(path[i + 1][0]), int(path[i + 1][1])

                # Clip to canvas bounds
                if (0 <= x1 < self.canvas_w and 0 <= y1 < self.canvas_h and
                        0 <= x2 < self.canvas_w and 0 <= y2 < self.canvas_h):
                    cv2.line(canvas, (x1, y1), (x2, y2), 0, self.line_thickness, cv2.LINE_AA)
                    strokes_drawn += 1
                else:
                    strokes_clipped += 1

        return canvas, strokes_drawn, strokes_clipped

    def render_comparison(self, original_image, paths, output_path=None):
        """
        Create a side-by-side comparison: original image | rendered drawing.

        Args:
            original_image: The original (or restored) input image.
            paths: Generated stroke paths.
            output_path: Optional path to save the comparison image.

        Returns:
            Comparison image (uint8, BGR).
        """
        h, w = original_image.shape[:2]
        self.canvas_h = h
        self.canvas_w = w

        drawing, n_drawn, n_clipped = self.render(paths)

        # Convert both to BGR for side-by-side
        if len(original_image.shape) == 2:
            orig_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            orig_bgr = original_image.copy()

        draw_bgr = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)

        # Add labels
        cv2.putText(orig_bgr, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(draw_bgr, f"Robot Drawing ({n_drawn} strokes)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        comparison = np.hstack([orig_bgr, draw_bgr])

        if output_path:
            cv2.imwrite(output_path, comparison)

        return comparison


# ─── Interactive Simulation (PyBullet) ────────────────────────────────────────

class RobotSimulator:
    """
    Full PyBullet simulation with 6-DOF KUKA IIWA arm and safety protocols.

    Use run_interactive() for GUI mode with manual debug sliders.
    Use execute_paths() to programmatically execute stroke paths.
    """

    def __init__(self, mode="DIRECT"):
        if not PYBULLET_AVAILABLE:
            raise RuntimeError("PyBullet not installed. Cannot create simulator.")

        self.mode = mode
        self.client = None
        self.robot_id = None
        self.table_id = None
        self.num_joints = None

    def setup(self):
        """Initialise the simulation environment."""
        if self.mode == "GUI":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load assets
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, -0.65],
                                    p.getQuaternionFromEuler([0, 0, 1.57]))
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0],
                                    p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True)

        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"  Simulator ready. Robot has {self.num_joints} joints.")

    def check_safety(self, target_angles):
        """
        Run all safety checks for the current robot configuration.

        Returns:
            (all_safe: bool, report: dict)
        """
        report = {"geofence": True, "singularity": True, "collision": True}

        # End-effector state
        ee_state = p.getLinkState(self.robot_id, 6)
        ee_pos = ee_state[0]

        # Geofencing
        geo_safe, geo_msg = check_geofence(ee_pos)
        if not geo_safe:
            report["geofence"] = False
            report["geofence_msg"] = geo_msg

        # Singularity check (actual Jacobian determinant computation)
        zero_vec = [0.0] * self.num_joints
        jac_t, jac_r = p.calculateJacobian(
            self.robot_id, 6, [0, 0, 0],
            list(target_angles), zero_vec, zero_vec
        )
        sing_safe, det_val = check_singularity(jac_t)
        report["singularity"] = sing_safe
        report["jacobian_det"] = det_val

        # Collision prediction
        overlap = p.getClosestPoints(self.robot_id, self.table_id, distance=0.01)
        if overlap:
            report["collision"] = False

        all_safe = report["geofence"] and report["singularity"] and report["collision"]
        return all_safe, report

    def disconnect(self):
        """Cleanly disconnect from the physics server."""
        if self.client is not None:
            p.disconnect()
            self.client = None

    def run_interactive(self):
        """Run interactive GUI mode with debug sliders."""
        import time

        if self.mode != "GUI":
            raise RuntimeError("Interactive mode requires GUI mode. Re-init with mode='GUI'.")

        self.setup()

        # Create debug sliders
        sliders = []
        for i in range(self.num_joints):
            sliders.append(p.addUserDebugParameter(f"Joint {i}", -3.14, 3.14, 0))

        print("Interactive simulation running. Move sliders to control robot.")
        try:
            while True:
                target_angles = []
                for i in range(self.num_joints):
                    angle = p.readUserDebugParameter(sliders[i])
                    target_angles.append(angle)
                    p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                            targetPosition=angle)

                p.stepSimulation()

                # Safety checks
                all_safe, report = self.check_safety(target_angles)
                if not all_safe:
                    if not report["geofence"]:
                        print(f"WARNING: Geofence breach! {report.get('geofence_msg', '')}")
                        raise MotionHaltException("Geofence breached.")
                    if not report["collision"]:
                        print("WARNING: Collision detected!")
                        raise MotionHaltException("Collision with table.")
                    if not report["singularity"]:
                        print(f"WARNING: Near singularity (det={report['jacobian_det']:.6f})!")

                time.sleep(1. / 240.)

        except KeyboardInterrupt:
            print("Simulation closed by user.")
        except MotionHaltException as e:
            print(f"Simulation halted: SAFETY PROTOCOL: {e}")
        finally:
            self.disconnect()


if __name__ == "__main__":
    # Demo: render some test paths
    renderer = RobotDrawingRenderer(400, 600)

    # Create sample hatching paths
    test_paths = []
    for y in range(50, 350, 15):
        for x in range(50, 550, 15):
            angle = (x + y) * 0.01
            dx = int(10 * np.cos(angle))
            dy = int(10 * np.sin(angle))
            test_paths.append([(x - dx, y - dy), (x + dx, y + dy)])

    canvas, drawn, clipped = renderer.render(test_paths)
    print(f"Rendered {drawn} strokes ({clipped} clipped)")

    out_dir = os.path.join(os.path.dirname(__file__), '../../outputs')
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, 'test_rendering.png'), canvas)
    print("Saved to outputs/test_rendering.png")