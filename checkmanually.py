#!/usr/bin/env python3
"""
Niryo Robot Color Position Learning System - Fixed Version
1. Manually position robot at each color
2. Record positions for each color
3. Move automatically to selected colors

First install the library:
Try: pip install pyniryo2
Or: pip install pyniryo
"""

import json
import os

# Try importing pyniryo2 first, fallback to pyniryo
try:
    from pyniryo2 import *
    LIBRARY_VERSION = "pyniryo2"
    print("Using pyniryo2 library")
except ImportError:
    try:
        from pyniryo import *
        LIBRARY_VERSION = "pyniryo"
        print("Using pyniryo library")
    except ImportError:
        print("ERROR: Neither pyniryo2 nor pyniryo is installed!")
        print("Please install one of them:")
        print("  pip install pyniryo2")
        print("  or")
        print("  pip install pyniryo")
        exit(1)

class ColorPositionLearner:
    def __init__(self, robot_ip="169.254.200.200"):
        # Initialize Niryo robot connection
        self.robot_ip = robot_ip
        self.robot = None
        self.connect_robot()
        
        # File to store learned positions
        self.positions_file = "color_positions.json"
        
        # Available colors
        self.colors = ["GREEN", "PINK", "YELLOW", "LIME_GREEN"]
        
        # Dictionary to store learned positions
        self.learned_positions = self.load_positions()
    
    def connect_robot(self):
        """Connect to the Niryo robot"""
        try:
            print(f"Connecting to robot at {self.robot_ip}...")
            self.robot = NiryoRobot(self.robot_ip)
            print("✅ Connected to robot successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to robot: {e}")
            print("Please check:")
            print("1. Robot is powered on")
            print("2. Robot IP address is correct")
            print("3. You're connected to the same network")
            return False
        
    def load_positions(self):
        """Load previously learned positions from file"""
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, 'r') as f:
                    return json.load(f)
            except:
                print("Error loading positions file, starting fresh")
        return {}
    
    def save_positions(self):
        """Save learned positions to file"""
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(self.learned_positions, f, indent=2)
            print(f"Positions saved to {self.positions_file}")
        except Exception as e:
            print(f"Error saving positions: {e}")
    
    def get_current_position(self):
        """Get current robot position - handles different library versions"""
        if self.robot is None:
            print("❌ Robot not connected!")
            return None
            
        try:
            print("Getting robot position...")
            
            # Try different methods to get pose based on library version
            pose = None
            method_used = None
            
            # Method 1: get_pose() - common in newer versions
            if hasattr(self.robot, 'get_pose'):
                try:
                    pose = self.robot.get_pose()
                    method_used = "get_pose()"
                except Exception as e:
                    print(f"get_pose() failed: {e}")
            
            # Method 2: get_pose_as_list() - alternative method
            if pose is None and hasattr(self.robot, 'get_pose_as_list'):
                try:
                    pose = self.robot.get_pose_as_list()
                    method_used = "get_pose_as_list()"
                except Exception as e:
                    print(f"get_pose_as_list() failed: {e}")
            
            # Method 3: arm.get_pose() - nested arm object
            if pose is None and hasattr(self.robot, 'arm'):
                try:
                    pose = self.robot.arm.get_pose()
                    method_used = "arm.get_pose()"
                except Exception as e:
                    print(f"arm.get_pose() failed: {e}")
            
            # Method 4: get_joints() and forward kinematics (fallback)
            if pose is None and hasattr(self.robot, 'get_joints'):
                try:
                    joints = self.robot.get_joints()
                    if hasattr(self.robot, 'forward_kinematics'):
                        pose = self.robot.forward_kinematics(joints)
                        method_used = "forward_kinematics(get_joints())"
                    else:
                        print("❌ Cannot get pose: no suitable method found")
                        return None
                except Exception as e:
                    print(f"forward_kinematics failed: {e}")
            
            if pose is None:
                print("❌ All pose methods failed")
                # Print available methods for debugging
                print("Available robot methods:")
                methods = [method for method in dir(self.robot) if 'pose' in method.lower() or 'joint' in method.lower()]
                print(f"Pose/Joint methods: {methods}")
                return None
            
            print(f"✅ Got pose using: {method_used}")
            print(f"Raw pose data: {pose}")
            
            # Parse the pose data based on its structure
            position = self.parse_pose_data(pose)
            
            if position:
                print(f"Parsed position: {position}")
                return position
            else:
                print("❌ Failed to parse pose data")
                return None
                
        except Exception as e:
            print(f"❌ Error getting current position: {e}")
            print(f"Error type: {type(e)}")
            return None
    
    def parse_pose_data(self, pose):
        """Parse pose data from different possible formats"""
        try:
            # Format 1: Object with direct attributes (x, y, z, roll, pitch, yaw)
            if hasattr(pose, 'x') and hasattr(pose, 'y') and hasattr(pose, 'z'):
                return {
                    'x': float(pose.x),
                    'y': float(pose.y),
                    'z': float(pose.z),
                    'roll': float(getattr(pose, 'roll', 0)),
                    'pitch': float(getattr(pose, 'pitch', 0)),
                    'yaw': float(getattr(pose, 'yaw', 0))
                }
            
            # Format 2: Object with nested position and orientation
            elif hasattr(pose, 'position') and hasattr(pose, 'orientation'):
                return {
                    'x': float(pose.position.x),
                    'y': float(pose.position.y),
                    'z': float(pose.position.z),
                    'roll': float(pose.orientation.roll),
                    'pitch': float(pose.orientation.pitch),
                    'yaw': float(pose.orientation.yaw)
                }
            
            # Format 3: List or tuple [x, y, z, roll, pitch, yaw]
            elif isinstance(pose, (list, tuple)) and len(pose) >= 6:
                return {
                    'x': float(pose[0]),
                    'y': float(pose[1]),
                    'z': float(pose[2]),
                    'roll': float(pose[3]),
                    'pitch': float(pose[4]),
                    'yaw': float(pose[5])
                }
            
            # Format 4: Dictionary
            elif isinstance(pose, dict):
                required_keys = ['x', 'y', 'z']
                if all(key in pose for key in required_keys):
                    return {
                        'x': float(pose['x']),
                        'y': float(pose['y']),
                        'z': float(pose['z']),
                        'roll': float(pose.get('roll', 0)),
                        'pitch': float(pose.get('pitch', 0)),
                        'yaw': float(pose.get('yaw', 0))
                    }
            
            # If we can't parse it, show what we got for debugging
            print(f"Unknown pose format: {type(pose)}")
            print(f"Pose attributes: {dir(pose) if hasattr(pose, '__dict__') else 'No attributes'}")
            print(f"Pose content: {pose}")
            return None
            
        except Exception as e:
            print(f"Error parsing pose data: {e}")
            return None
    
    def move_to_position(self, position):
        """Move robot to specified position - handles different library versions"""
        try:
            print(f"Moving to position: {position}")
            
            # Method 1: Try PoseObject (common in pyniryo2)
            if 'PoseObject' in globals():
                try:
                    pose = PoseObject(
                        x=position['x'],
                        y=position['y'], 
                        z=position['z'],
                        roll=position['roll'],
                        pitch=position['pitch'],
                        yaw=position['yaw']
                    )
                    self.robot.move_pose(pose)
                    print("✅ Moved using PoseObject")
                    return True
                except Exception as e:
                    print(f"PoseObject method failed: {e}")
            
            # Method 2: Try move_pose with list
            try:
                pose_list = [
                    position['x'], position['y'], position['z'],
                    position['roll'], position['pitch'], position['yaw']
                ]
                self.robot.move_pose(pose_list)
                print("✅ Moved using pose list")
                return True
            except Exception as e:
                print(f"Pose list method failed: {e}")
            
            # Method 3: Try arm.move_pose with PoseObject
            if hasattr(self.robot, 'arm'):
                try:
                    # Try with PoseObject first
                    if 'PoseObject' in globals():
                        pose = PoseObject(
                            x=position['x'],
                            y=position['y'], 
                            z=position['z'],
                            roll=position['roll'],
                            pitch=position['pitch'],
                            yaw=position['yaw']
                        )
                        self.robot.arm.move_pose(pose)
                        print("✅ Moved using arm.move_pose with PoseObject")
                        return True
                except Exception as e:
                    print(f"arm.move_pose with PoseObject failed: {e}")
                
                try:
                    # Try with list
                    pose_list = [
                        position['x'], position['y'], position['z'],
                        position['roll'], position['pitch'], position['yaw']
                    ]
                    self.robot.arm.move_pose(pose_list)
                    print("✅ Moved using arm.move_pose with list")
                    return True
                except Exception as e:
                    print(f"arm.move_pose with list failed: {e}")
                
                try:
                    # Try with individual coordinates (original attempt)
                    self.robot.arm.move_pose(position['x'], position['y'], position['z'],
                                           position['roll'], position['pitch'], position['yaw'])
                    print("✅ Moved using arm.move_pose with coordinates")
                    return True
                except Exception as e:
                    print(f"arm.move_pose with coordinates failed: {e}")
            
            # Method 4: Try move_linear_pose (alternative method)
            if hasattr(self.robot, 'move_linear_pose'):
                try:
                    if 'PoseObject' in globals():
                        pose = PoseObject(
                            x=position['x'],
                            y=position['y'], 
                            z=position['z'],
                            roll=position['roll'],
                            pitch=position['pitch'],
                            yaw=position['yaw']
                        )
                        self.robot.move_linear_pose(pose)
                        print("✅ Moved using move_linear_pose")
                        return True
                except Exception as e:
                    print(f"move_linear_pose failed: {e}")
            
            # Method 5: Try arm.move_linear_pose
            if hasattr(self.robot, 'arm') and hasattr(self.robot.arm, 'move_linear_pose'):
                try:
                    if 'PoseObject' in globals():
                        pose = PoseObject(
                            x=position['x'],
                            y=position['y'], 
                            z=position['z'],
                            roll=position['roll'],
                            pitch=position['pitch'],
                            yaw=position['yaw']
                        )
                        self.robot.arm.move_linear_pose(pose)
                        print("✅ Moved using arm.move_linear_pose")
                        return True
                except Exception as e:
                    print(f"arm.move_linear_pose failed: {e}")
            
            # Method 6: Try joint-based movement via inverse kinematics
            if hasattr(self.robot, 'inverse_kinematics') and hasattr(self.robot, 'move_joints'):
                try:
                    pose_list = [
                        position['x'], position['y'], position['z'],
                        position['roll'], position['pitch'], position['yaw']
                    ]
                    joints = self.robot.inverse_kinematics(pose_list)
                    self.robot.move_joints(joints)
                    print("✅ Moved using inverse kinematics + move_joints")
                    return True
                except Exception as e:
                    print(f"Inverse kinematics method failed: {e}")
            
            # Method 7: Debug - show available move methods
            print("\n🔍 Available move methods:")
            move_methods = [method for method in dir(self.robot) if 'move' in method.lower()]
            print(f"Robot move methods: {move_methods}")
            
            if hasattr(self.robot, 'arm'):
                arm_move_methods = [method for method in dir(self.robot.arm) if 'move' in method.lower()]
                print(f"Arm move methods: {arm_move_methods}")
            
            print("❌ All move methods failed")
            return False
            
        except Exception as e:
            print(f"Error moving to position: {e}")
            return False
    
    def calibrate_robot(self):
        """Calibrate robot (run once at startup)"""
        print("Calibrating robot...")
        try:
            # Try different calibration methods
            if hasattr(self.robot, 'calibrate_auto'):
                self.robot.calibrate_auto()
            elif hasattr(self.robot, 'calibrate'):
                self.robot.calibrate()
            elif hasattr(self.robot, 'auto_calibrate'):
                self.robot.auto_calibrate()
            else:
                print("❌ No calibration method found")
                return False
            
            print("✅ Robot calibrated successfully")
            return True
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
            return False
    
    def test_connection(self):
        """Test robot connection and show available methods"""
        print("\n🔧 Testing robot connection...")
        
        if self.robot is None:
            print("❌ Robot not connected")
            return False
        
        print(f"✅ Robot object exists: {type(self.robot)}")
        print(f"📚 Using library: {LIBRARY_VERSION}")
        
        # Show available methods
        print("\n🔍 Available robot methods:")
        methods = [method for method in dir(self.robot) if not method.startswith('_')]
        pose_methods = [m for m in methods if 'pose' in m.lower()]
        joint_methods = [m for m in methods if 'joint' in m.lower()]
        move_methods = [m for m in methods if 'move' in m.lower()]
        
        print(f"Pose methods: {pose_methods}")
        print(f"Joint methods: {joint_methods}")  
        print(f"Move methods: {move_methods}")
        
        # Test basic connection
        try:
            # Try to get some basic info
            if hasattr(self.robot, 'get_learning_mode'):
                learning_mode = self.robot.get_learning_mode()
                print(f"✅ Learning mode status: {learning_mode}")
            
            if hasattr(self.robot, 'ping'):
                ping_result = self.robot.ping()
                print(f"✅ Ping result: {ping_result}")
            
            return True
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def learn_color_positions(self):
        """Interactive learning mode - teach robot where each color is"""
        print("\n" + "="*50)
        print("COLOR POSITION LEARNING MODE")
        print("="*50)
        print("Instructions:")
        print("1. Manually drag the robot arm to point at each color")
        print("2. Press ENTER when positioned correctly")
        print("3. Repeat for all colors")
        print("="*50)
        
        if self.robot is None:
            print("❌ Robot not connected! Please check connection.")
            return
        
        # Enable learning mode for manual positioning
        try:
            if hasattr(self.robot, 'set_learning_mode'):
                self.robot.set_learning_mode(True)
                print("✅ Learning mode enabled - you can now move the robot manually")
            elif hasattr(self.robot, 'enable_learning_mode'):
                self.robot.enable_learning_mode()
                print("✅ Learning mode enabled - you can now move the robot manually")
        except Exception as e:
            print(f"⚠️ Could not enable learning mode: {e}")
            print("Try moving the robot manually anyway...")
        
        successfully_learned = []
        
        for color in self.colors:
            print(f"\n📍 Learning position for: {color}")
            print(f"Manually move the robot arm to point at the {color} sticky note")
            
            # Wait for user confirmation
            input("Press ENTER when the robot is positioned correctly at the " + color + " sticky note...")
            
            # Record current position
            try:
                current_pos = self.get_current_position()
                if current_pos:
                    self.learned_positions[color] = current_pos
                    successfully_learned.append(color)
                    print(f"✅ Position recorded for {color}")
                    print(f"   Position: x={current_pos['x']:.3f}, y={current_pos['y']:.3f}, z={current_pos['z']:.3f}")
                else:
                    print(f"❌ Failed to record position for {color}")
                    print("   Could not get robot position - is the robot connected?")
            except Exception as e:
                print(f"❌ Error recording position for {color}: {e}")
        
        # Disable learning mode
        try:
            if hasattr(self.robot, 'set_learning_mode'):
                self.robot.set_learning_mode(False)
                print("✅ Learning mode disabled")
            elif hasattr(self.robot, 'disable_learning_mode'):
                self.robot.disable_learning_mode()
                print("✅ Learning mode disabled")
        except Exception as e:
            print(f"⚠️ Could not disable learning mode: {e}")
        
        # Save all learned positions
        if successfully_learned:
            self.save_positions()
            print(f"\n🎉 Positions learned and saved!")
            print(f"Successfully learned: {successfully_learned}")
        else:
            print(f"\n❌ No positions were successfully learned!")
            print("Please check robot connection and try again.")
        
        print(f"Total learned positions: {list(self.learned_positions.keys())}")
    
    def move_to_color(self, color):
        """Move robot to specified color position"""
        color = color.upper()
        
        if color not in self.learned_positions:
            print(f"❌ No learned position for color: {color}")
            print(f"Available colors: {list(self.learned_positions.keys())}")
            return False
        
        print(f"🤖 Moving to {color} position...")
        success = self.move_to_position(self.learned_positions[color])
        
        if success:
            print(f"✅ Successfully moved to {color}")
        else:
            print(f"❌ Failed to move to {color}")
        
        return success
    
    def interactive_mode(self):
        """Interactive mode for moving to colors"""
        print("\n" + "="*50)
        print("INTERACTIVE COLOR POINTING MODE")
        print("="*50)
        print("Available colors:", list(self.learned_positions.keys()))
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            color = input("\nEnter color to point to (or 'quit'): ").strip()
            
            if color.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if color.upper() in self.learned_positions:
                self.move_to_color(color)
            else:
                print(f"❌ Unknown color: {color}")
                print(f"Available colors: {list(self.learned_positions.keys())}")
    
    def show_learned_positions(self):
        """Display all learned positions"""
        print("\n" + "="*50)
        print("LEARNED POSITIONS")
        print("="*50)
        
        if not self.learned_positions:
            print("No positions learned yet. Run learning mode first.")
            return
        
        for color, pos in self.learned_positions.items():
            print(f"{color}:")
            print(f"  x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}")
            print(f"  roll={pos['roll']:.3f}, pitch={pos['pitch']:.3f}, yaw={pos['yaw']:.3f}")
            print()
    
    def main_menu(self):
        """Main menu for the application"""
        print("\n" + "="*60)
        print("🤖 NIRYO COLOR POSITION LEARNING SYSTEM")
        print("="*60)
        
        while True:
            print("\nChoose an option:")
            print("1. 📚 Learn color positions (teaching mode)")
            print("2. 🎯 Move to color (interactive mode)")
            print("3. 📋 Show learned positions")
            print("4. 🔄 Calibrate robot")
            print("5. 🔧 Test robot connection")
            print("6. 🚪 Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                self.learn_color_positions()
            elif choice == '2':
                if not self.learned_positions:
                    print("❌ No positions learned yet. Please run learning mode first (option 1)")
                else:
                    self.interactive_mode()
            elif choice == '3':
                self.show_learned_positions()
            elif choice == '4':
                self.calibrate_robot()
            elif choice == '5':
                self.test_connection()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")

def main():
    """Main function"""
    try:
        # Get robot IP from user
        robot_ip = input("Enter robot IP address (default: 169.254.200.200): ").strip()
        if not robot_ip:
            robot_ip = "169.254.200.200"
        
        # Create the color position learner
        learner = ColorPositionLearner(robot_ip)
        
        if learner.robot is None:
            print("❌ Cannot proceed without robot connection")
            return
        
        # Start the main menu
        learner.main_menu()
        
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        print("🔌 Disconnecting from robot...")

if __name__ == "__main__":
    main()