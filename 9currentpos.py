from pyniryo import NiryoRobot

def get_xyz_coordinates():
    """
    Get only the X, Y, Z coordinates of the Niryo Ned2 robotic arm.
    """
    robot_ip = "10.10.10.10"  # Replace with your robot's actual IP

    try:
        # Connect to the robot
        robot = NiryoRobot(robot_ip)
        print("Connected to robot")

        # Get pose object
        pose = robot.get_pose()

        # Extract pose attributes
        x = pose.x
        y = pose.y
        z = pose.z
        roll = pose.roll
        pitch = pose.pitch
        yaw = pose.yaw

        # Display results
        print("\n Cartesian Coordinates:")
        print(f"X: {x:.4f} m")
        print(f"Y: {y:.4f} m")
        print(f"Z: {z:.4f} m")

        print("\n🎯 Orientation:")
        print(f"Roll:  {roll:.3f} rad")
        print(f"Pitch: {pitch:.3f} rad")
        print(f"Yaw:   {yaw:.3f} rad")

        # Disconnect from robot
        robot.close_connection()
        print("🔌 Disconnected from robot")

        return x, y, z

    except Exception as e:
        print(f" Error connecting to robot: {e}")
        return None, None, None

if __name__ == "__main__":
    print("🛰️ Getting current X, Y, Z coordinates...\n")
    x, y, z = get_xyz_coordinates()

    if x is not None:
        print("\n Final Result:")
        print(f"X = {x:.3f} m")
        print(f"Y = {y:.3f} m")
        print(f"Z = {z:.3f} m")
    else:
        print(" Failed to get coordinates.")
