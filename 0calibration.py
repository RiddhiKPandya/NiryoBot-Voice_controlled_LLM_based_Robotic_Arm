import cv2
import numpy as np
import json
import time

class NiryoCoordinateConverter:
    """
    Converts pixel coordinates to real-world robot coordinates
    """
    def __init__(self):
        # Camera intrinsic parameters (these need to be calibrated for your specific setup)
        # These are example values - you should calibrate your camera
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],      # fx, 0, cx
            [0.0, 600.0, 240.0],      # 0, fy, cy
            [0.0, 0.0, 1.0]           # 0, 0, 1
        ])
        
        self.dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])  # Distortion coefficients
        
        # Robot-to-camera transformation parameters
        # These define where the camera is relative to the robot base
        self.camera_height = 0.35  # Height of camera above the workspace (meters)
        self.camera_offset_x = 0.0  # Camera X offset from robot base (meters)
        self.camera_offset_y = 0.0  # Camera Y offset from robot base (meters)
        
        # Default workspace calibration points (you need to measure these)
        # Format: [pixel_x, pixel_y] -> [robot_x, robot_y, robot_z]
        self.calibration_points = {
            # Top-left corner
            (100, 100): (-0.15, 0.15, 0.02),
            # Top-right corner  
            (540, 100): (0.15, 0.15, 0.02),
            # Bottom-left corner
            (100, 380): (-0.15, -0.15, 0.02),
            # Bottom-right corner
            (540, 380): (0.15, -0.15, 0.02),
            # Center point (middle of workspace for better accuracy)
            (320, 240): (0.0, 0.0, 0.02)
        }
        
        # Calculate transformation matrix from calibration points
        self.setup_transformation()
    
    def setup_transformation(self):
        """
        Setup pixel-to-world coordinate transformation using calibration points
        """
        # Extract pixel and world coordinates
        pixel_coords = np.array(list(self.calibration_points.keys()), dtype=np.float32)
        world_coords = np.array([list(coord)[:2] for coord in self.calibration_points.values()], dtype=np.float32)
        
        # Calculate homography matrix for 2D transformation
        if len(pixel_coords) >= 4:
            self.homography_matrix, _ = cv2.findHomography(pixel_coords, world_coords)
        else:
            # Fallback: simple scaling transformation
            self.setup_simple_transformation()
    
    def setup_simple_transformation(self):
        """
        Simple transformation based on image dimensions and workspace size
        """
        # Assume workspace is 30cm x 30cm and image is 640x480
        self.pixels_to_meters_x = 0.30 / 640  # meters per pixel in X
        self.pixels_to_meters_y = 0.30 / 480  # meters per pixel in Y
        self.image_center_x = 320
        self.image_center_y = 240
        self.workspace_center_x = 0.0  # Robot X coordinate of workspace center
        self.workspace_center_y = 0.0  # Robot Y coordinate of workspace center
    
    def pixel_to_world(self, pixel_x, pixel_y, z_height=0.02):
        """
        Convert pixel coordinates to robot world coordinates
        """
        try:
            if hasattr(self, 'homography_matrix') and self.homography_matrix is not None:
                # Use homography transformation
                pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
                pixel_point = pixel_point.reshape(-1, 1, 2)
                world_point = cv2.perspectiveTransform(pixel_point, self.homography_matrix)
                world_x, world_y = world_point[0][0]
            else:
                # Use simple transformation
                world_x = self.workspace_center_x + (pixel_x - self.image_center_x) * self.pixels_to_meters_x
                world_y = self.workspace_center_y - (pixel_y - self.image_center_y) * self.pixels_to_meters_y
            
            return world_x, world_y, z_height
            
        except Exception as e:
            print(f"Error in coordinate conversion: {e}")
            # Fallback simple conversion
            world_x = (pixel_x - 320) * 0.0005  # Simple scaling
            world_y = -(pixel_y - 240) * 0.0005  # Invert Y axis
            return world_x, world_y, z_height
    
    def calibrate_workspace(self, robot, calibration_points_pixel):
        """
        Interactive calibration - move robot to corners and center of workspace
        """
        calibration_data = {}
        
        print("\nWORKSPACE CALIBRATION")
        print("="*50)
        print("Move the robot to each point of your workspace and press Enter")
        print("The 5 points are: 4 corners + 1 center point for better accuracy")
        
        point_names = ["Top-Left Corner", "Top-Right Corner", "Bottom-Left Corner", "Bottom-Right Corner", "Center Point"]
        
        for i, (point_name, pixel_coord) in enumerate(zip(point_names, calibration_points_pixel)):
            input(f"\nMove robot to {point_name} and press Enter...")
            
            try:
                # Get current robot position
                current_pose = robot.get_pose()
                robot_x, robot_y, robot_z = current_pose.x, current_pose.y, current_pose.z
                
                calibration_data[pixel_coord] = (robot_x, robot_y, robot_z)
                print(f"{point_name}: Pixel{pixel_coord} -> Robot({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f})")
                
            except Exception as e:
                print(f"Error getting robot position: {e}")
        
        # Update calibration points
        self.calibration_points = calibration_data
        self.setup_transformation()
        
        # Save calibration to file
        self.save_calibration(calibration_data)
        
        print("Calibration completed and saved!")
    
    def save_calibration(self, calibration_data):
        """Save calibration data to JSON file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            calib_data = {str(k): v for k, v in calibration_data.items()}
            
            with open("niryo_calibration.json", "w") as f:
                json.dump(calib_data, f, indent=2)
                
            print("Calibration saved to 'niryo_calibration.json'")
            
        except Exception as e:
            print(f"Error saving calibration: {e}")
    
    def load_calibration(self, filename="niryo_calibration.json"):
        """Load calibration data from JSON file"""
        try:
            with open(filename, "r") as f:
                calib_data = json.load(f)
            
            # Convert back to proper format
            self.calibration_points = {}
            for pixel_str, world_coords in calib_data.items():
                # Convert string key back to tuple
                pixel_coords = eval(pixel_str)  # Note: eval is used here, be careful in production
                self.calibration_points[pixel_coords] = tuple(world_coords)
            
            # Recalculate transformation
            self.setup_transformation()
            print(f"Calibration loaded from '{filename}'")
            
        except FileNotFoundError:
            print(f"Calibration file '{filename}' not found. Using default values.")
        except Exception as e:
            print(f"Error loading calibration: {e}")

def initialize_robot_safely(robot):
    """
    Safely initialize the robot by clearing any collision detection errors
    """
    try:
        print("Initializing robot safely...")
        
        # Clear any collision detection errors
        robot.clear_collision_detected()
        print(" Collision detection cleared")
        
        # Wait a moment for the robot to process the command
        time.sleep(1)
        
        # Try to calibrate the robot (this ensures joints are properly initialized)
        print("Calibrating robot...")
        robot.calibrate_auto()
        print(" Robot calibration completed")
        
        # Move to home position
        print(" Moving to home position...")
        robot.move_to_home_pose()
        print(" Robot moved to home position")
        
        return True
        
    except Exception as e:
        print(f" Error during robot initialization: {e}")
        print("Try manually moving the robot to clear any collision state")
        return False

def run_calibration():
    """
    Main calibration function with improved error handling
    """
    from pyniryo import NiryoRobot
    
    # Initialize coordinate converter
    coord_converter = NiryoCoordinateConverter()
    
    # Connect to robot
    robot_ip = "10.10.10.10"  # Replace with your robot's IP
    print(f"Connecting to Niryo Ned2 robot at {robot_ip}...")
    
    robot = None
    try:
        robot = NiryoRobot(robot_ip)
        print("✅ Successfully connected to Niryo Ned2!")
        
        # Safely initialize the robot
        if not initialize_robot_safely(robot):
            print(" Failed to initialize robot safely")
            print(" Manual steps to try:")
            print("   1. Physically move the robot arm to clear any collision state")
            print("   2. Press the robot's reset button if available")
            print("   3. Restart the robot and try again")
            return
        
    except Exception as e:
        print(f" Error connecting to robot: {e}")
        print(" Troubleshooting steps:")
        print("   1. Check if the robot IP address is correct")
        print("   2. Ensure the robot is powered on and connected to the network")
        print("   3. Try pinging the robot: ping 10.10.10.10")
        print("   4. Check if any other software is connected to the robot")
        return
    
    # Define pixel coordinates for calibration points
    # These correspond to where you expect the corners and center to appear in the camera image
    calibration_pixels = [
        (100, 100),   # Top-left corner
        (540, 100),   # Top-right corner  
        (100, 380),   # Bottom-left corner
        (540, 380),   # Bottom-right corner
        (320, 240)    # Center point
    ]
    
    print("\n CALIBRATION INSTRUCTIONS:")
    print("1. You will be asked to move the robot to 5 different positions")
    print("2. Use the robot's manual controls or teach mode to position it")
    print("3. Position the robot's end-effector (gripper) at each requested point")
    print("4. The center point helps improve calibration accuracy across the workspace")
    print("5. Press Enter after positioning the robot at each point")
    print("\n  SAFETY REMINDER:")
    print("   - Ensure the workspace is clear of obstacles")
    print("   - Keep emergency stop within reach")
    print("   - Move the robot slowly and carefully")
    
    try:
        # Start calibration
        coord_converter.calibrate_workspace(robot, calibration_pixels)
        
        # Test the calibration
        print("\n TESTING CALIBRATION:")
        test_pixel = (320, 240)  # Center of image
        world_coords = coord_converter.pixel_to_world(test_pixel[0], test_pixel[1])
        print(f"Test conversion - Pixel {test_pixel} -> World {world_coords}")
        
    except Exception as e:
        print(f"Error during calibration: {e}")
    
    finally:
        # Always close robot connection
        if robot:
            try:
                robot.close_connection()
                print("🔌 Robot connection closed")
            except:
                pass

if __name__ == "__main__":
    run_calibration()