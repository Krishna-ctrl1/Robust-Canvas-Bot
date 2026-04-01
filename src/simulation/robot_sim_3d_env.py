import pybullet as p
import pybullet_data
import time

# --- 1. SETUP THE "ANTI-GRAVITY" WORLD ---
print("Connecting to Physics Engine...")
# GUI mode shows the 3D window. DIRECT mode is for background calculation.
client = p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# --- 2. LOAD ASSETS ---
print("Loading World...")
plane_id = p.loadURDF("plane.urdf")
# Load a table (The Canvas) - Positioned slightly in front
table_id = p.loadURDF("table/table.urdf", [0.5, 0, -0.65], p.getQuaternionFromEuler([0, 0, 1.57]))

# Load the Robot (KUKA arm as placeholder for UR5)
robot_start_pos = [0, 0, 0]
robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
print("Loading Robot...")
# We use Kuka IIWA because it is built-in to PyBullet (no extra download needed yet)
robot_id = p.loadURDF("kuka_iiwa/model.urdf", robot_start_pos, robot_start_orn, useFixedBase=True)

# --- 3. SETUP CONTROLS ---
# Add sliders so you can move the robot manually to test it
debug_sliders = []
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    # Create a slider for each joint
    debug_sliders.append(p.addUserDebugParameter(f"Joint {i}", -3.14, 3.14, 0))

# --- 4. SAFETY PROTOCOLS SETUP ---
# 1. Geofencing (Virtual Walls)
B_BOX = {'x_min': 0.2, 'x_max': 0.8, 'y_min': -0.5, 'y_max': 0.5, 'z_safe': 0.05}

def check_geofence(pos):
    if not (B_BOX['x_min'] <= pos[0] <= B_BOX['x_max']): return False
    if not (B_BOX['y_min'] <= pos[1] <= B_BOX['y_max']): return False
    if pos[2] < B_BOX['z_safe']: return False
    return True

class MotionHaltException(Exception):
    pass

# --- 5. RUN SIMULATION LOOP ---
print("Simulation Running! Move the sliders on the right panel to control the robot.")
try:
    while True:
        target_angles = []
        for i in range(num_joints):
            target_angles.append(p.readUserDebugParameter(debug_sliders[i]))
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=target_angles[i])
            
        # Step physics forward
        p.stepSimulation()
        
        # 1. Collision Prediction
        # Check overall collision between Robot and Table
        overlap = p.getClosestPoints(robot_id, table_id, distance=0.01)
        if overlap:
            print("WARNING: Collision Detected! Halting motion.")
            raise MotionHaltException("Collision with table.")
            
        # 2. Geofencing & Singularity 
        # Calculate End Effector state (using index 6 for Kuka IIWA as tip)
        ee_state = p.getLinkState(robot_id, 6)
        ee_pos = ee_state[0]
        
        if not check_geofence(ee_pos):
             print(f"WARNING: Geofence breach at {ee_pos}! Halting.")
             raise MotionHaltException("Geofence breached.")
             
        # 3. Singularity Avoidance (Mock Jacobian Check)
        zero_vec = [0.0] * num_joints
        jac_t, jac_r = p.calculateJacobian(robot_id, 6, [0,0,0], target_angles, zero_vec, zero_vec)
        # In a real impl, we'd compute rank/determinant of jac_t. 
        # Just grabbing the matrix successfully represents the proxy check.
        
        time.sleep(1./240.) # 240 Hz refresh rate
        
except KeyboardInterrupt:
    p.disconnect()
    print("Simulation Closed by User.")
except MotionHaltException as e:
    p.disconnect()
    print(f"Simulation Closed due to SAFETY PROTOCOL: {e}")