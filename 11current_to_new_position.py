#!/usr/bin/env python3

from pyniryo import *

def main():
    robot_ip = "10.10.10.10"  # Replace with your robot's actual IP address

    try:
        robot = NiryoRobot(robot_ip)
        print("Connected to robot.")

        # Get and display current pose
        current_pose = robot.get_pose()
        print("\nCurrent Position:")
        print(f"X = {current_pose.x:.3f} m")
        print(f"Y = {current_pose.y:.3f} m")
        print(f"Z = {current_pose.z:.3f} m")

        # Prompt user for new coordinates
        print("\nEnter new target coordinates:")
        x = float(input("New X (in meters): "))
        y = float(input("New Y (in meters): "))
        z = float(input("New Z (in meters): "))

        # Move to new position (keep orientation the same)
        target_pose = PoseObject(x, y, z, current_pose.roll, current_pose.pitch, current_pose.yaw)
        robot.move_pose(target_pose)
        print("\nRobot moved to new position successfully!")

        # Disconnect
        robot.close_connection()
        print("Disconnected from robot.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
