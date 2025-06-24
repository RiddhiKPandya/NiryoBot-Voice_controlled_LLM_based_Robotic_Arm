from pyniryo import *

# ---- Connect to Robot ----
robot_ip = "10.10.10.10"
robot = NiryoRobot(robot_ip)

# Optional: Calibrate and home
# robot.calibrate_auto()
# robot.move_to_home_pose()

# ---- Define Target Pose ----
x = 0.179
y = 0.019
z = 0.165

roll = -2.961
pitch = 1.285
yaw = -2.979

# ---- Move to Target Pose ----
robot.move_pose(x, y, z, roll, pitch, yaw)

# ---- Disconnect ----
robot.close_connection()
