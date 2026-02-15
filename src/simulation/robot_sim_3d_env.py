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

# --- 4. RUN SIMULATION LOOP ---
print("Simulation Running! Move the sliders on the right panel to control the robot.")
try:
    while True:
        # Read the sliders and apply angles to the robot
        for i in range(num_joints):
            target_angle = p.readUserDebugParameter(debug_sliders[i])
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=target_angle)
            
        # Step physics forward
        p.stepSimulation()
        time.sleep(1./240.) # 240 Hz refresh rate
        
except KeyboardInterrupt:
    p.disconnect()
    print("Simulation Closed.")