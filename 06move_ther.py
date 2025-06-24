#!/usr/bin/env python3

from pyniryo import *
import json

def load_target_coordinates():
    """Load target coordinates from converted_robot_coords.json"""
    try:
        with open('converted_robot_coords.json', 'r') as f:
            coords_data = json.load(f)
        
        # Extract x, y, z coordinates
        robot_coords = coords_data['robot_coordinates']
        x, y, z = robot_coords[0], robot_coords[1], robot_coords[2]
        
        print(f"Loaded target coordinates from JSON:")
        print(f"X = {x:.3f} m")
        print(f"Y = {y:.3f} m") 
        print(f"Z = {z:.3f} m")
        print(f"Source: {coords_data.get('color', 'Unknown')} object at pixel {coords_data.get('pixel_source', 'Unknown')}")
        
        return x, y, z
        
    except FileNotFoundError:
        print("Error: converted_robot_coords.json not found!")
        print("Please run the coordinate conversion script first.")
        return None, None, None
    except KeyError as e:
        print(f"Error: Missing key in JSON file: {e}")
        return None, None, None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in converted_robot_coords.json")
        return None, None, None
    except Exception as e:
        print(f"Error loading coordinates: {e}")
        return None, None, None

def main():
    robot_ip = "10.10.10.10"  # Replace with your robot's actual IP address

    # Load target coordinates from JSON file
    x, y, z = load_target_coordinates()
    
    if x is None or y is None or z is None:
        print("Failed to load coordinates. Exiting...")
        return

    try:
        robot = NiryoRobot(robot_ip)
        print("Connected to robot.")

        # Get and display current pose
        current_pose = robot.get_pose()
        print("\nCurrent Position:")
        print(f"X = {current_pose.x:.3f} m")
        print(f"Y = {current_pose.y:.3f} m")
        print(f"Z = {current_pose.z:.3f} m")

        # Display target coordinates
        print(f"\nTarget Position (from JSON):")
        print(f"X = {x:.3f} m")
        print(f"Y = {y:.3f} m")
        print(f"Z = {z:.3f} m")

        # Ask for confirmation before moving
        confirm = input("\nProceed with movement? (y/n): ").lower().strip()
        if confirm != 'y' and confirm != 'yes':
            print("Movement cancelled.")
            robot.close_connection()
            return

        # Move to new position (keep orientation the same)
        target_pose = PoseObject(x, y, z, current_pose.roll, current_pose.pitch, current_pose.yaw)
        
        print("\nMoving robot to target position...")
        robot.move_pose(target_pose)
        print("Robot moved to target position successfully!")

        # Verify final position
        final_pose = robot.get_pose()
        print(f"\nFinal Position:")
        print(f"X = {final_pose.x:.3f} m")
        print(f"Y = {final_pose.y:.3f} m")
        print(f"Z = {final_pose.z:.3f} m")

        # Disconnect
        robot.close_connection()
        print("Disconnected from robot.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()