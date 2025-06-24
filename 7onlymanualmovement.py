import cv2
import numpy as np
from pyniryo import *
import time
import json
import os

# Connect to robot
robot_ip = "10.10.10.10"
robot = NiryoRobot(robot_ip)

# SAFE POSITIONS - These will be updated through calibration
SAFE_POSITIONS = {
    'green': [0.20, -0.05, 0.05, 0.0, 0.0, 0.0],    
    'orange': [0.20, 0.0, 0.05, 0.0, 0.0, 0.0],     
    'pink': [0.20, 0.05, 0.05, 0.0, 0.0, 0.0],      
    'yellow': [0.30, 0.0, 0.05, 0.0, 0.0, 0.0]      
}

# File to store calibrated positions
POSITIONS_FILE = "calibrated_positions.json"
# File where whisper saves the spoken color
SPOKEN_COLOR_FILE = "spoken_color.json"

def load_saved_positions():
    """Load previously saved positions"""
    global SAFE_POSITIONS
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                saved_positions = json.load(f)
                SAFE_POSITIONS.update(saved_positions)
                print(f" Loaded saved positions from {POSITIONS_FILE}")
                return True
        except Exception as e:
            print(f" Error loading positions: {e}")
    return False

def save_positions():
    """Save current positions to file"""
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(SAFE_POSITIONS, f, indent=2)
        print(f"Positions saved to {POSITIONS_FILE}")
    except Exception as e:
        print(f" Error saving positions: {e}")

def read_spoken_color():
    """Read the color from the JSON file created by whisper model"""
    try:
        if not os.path.exists(SPOKEN_COLOR_FILE):
            print(f" Spoken color file '{SPOKEN_COLOR_FILE}' not found")
            print("   Please run your whisper script first to generate the color command")
            return None
        
        with open(SPOKEN_COLOR_FILE, 'r') as f:
            data = json.load(f)
        
        spoken_color = data.get('color', '').lower()  # Convert to lowercase for consistency
        
        # Map common color variations to our supported colors
        color_mapping = {
            'green': 'green',
            'lime_green': 'green',  # Map lime_green to green if that's what whisper detects
            'orange': 'orange',
            'pink': 'pink',
            'yellow': 'yellow',
            'red': 'orange'  # Map red to orange if no red position exists
        }
        
        mapped_color = color_mapping.get(spoken_color)
        
        if mapped_color and mapped_color in SAFE_POSITIONS:
            print(f" Read spoken color: '{spoken_color}' -> mapped to '{mapped_color}'")
            return mapped_color
        else:
            print(f" Spoken color '{spoken_color}' is not supported")
            print(f"   Supported colors: {list(SAFE_POSITIONS.keys())}")
            return None
            
    except json.JSONDecodeError as e:
        print(f" Error reading JSON file: {e}")
        return None
    except Exception as e:
        print(f" Error reading spoken color: {e}")
        return None

def clear_spoken_color():
    """Clear the spoken color file after processing"""
    try:
        if os.path.exists(SPOKEN_COLOR_FILE):
            os.remove(SPOKEN_COLOR_FILE)
            print(" Cleared spoken color file")
    except Exception as e:
        print(f" Could not clear spoken color file: {e}")

def get_current_robot_position():
    """Get current robot position with error handling"""
    try:
        pose = robot.get_pose()
        
        # Handle different pose formats
        if hasattr(pose, 'x'):
            # Object format
            return [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]
        else:
            # List format
            return pose[:6] if len(pose) >= 6 else pose
            
    except Exception as e:
        print(f" Error getting robot position: {e}")
        return None

def show_current_position():
    """Display current robot position"""
    print("\n📍 CURRENT ROBOT POSITION:")
    pos = get_current_robot_position()
    if pos:
        print(f"   X: {pos[0]:.4f}")
        print(f"   Y: {pos[1]:.4f}")
        print(f"   Z: {pos[2]:.4f}")
        print(f"   Roll: {pos[3]:.4f}")
        print(f"   Pitch: {pos[4]:.4f}")
        print(f"   Yaw: {pos[5]:.4f}")
        return pos
    else:
        print("   Could not get position")
        return None

def test_robot_movement():
    """Test basic robot movement and positioning"""
    print("\n🔧 TESTING ROBOT MOVEMENT...")
    
    # Get starting position
    start_pos = get_current_robot_position()
    if not start_pos:
        print(" Cannot get starting position")
        return False
    
    print(f"Starting position: {[round(x, 3) for x in start_pos]}")
    
    # Try a small movement test
    test_pos = start_pos.copy()
    test_pos[2] += 0.02  # Move up 2cm
    
    print("Testing small upward movement (+2cm in Z)...")
    try:
        robot.move_pose(*test_pos)
        time.sleep(1)
        
        # Check if we actually moved
        new_pos = get_current_robot_position()
        if new_pos:
            z_diff = abs(new_pos[2] - test_pos[2])
            if z_diff < 0.005:  # Within 5mm
                print(" Movement test successful")
                
                # Move back to start
                robot.move_pose(*start_pos)
                time.sleep(1)
                return True
            else:
                print(f" Position mismatch. Expected Z: {test_pos[2]:.3f}, Got: {new_pos[2]:.3f}")
        
    except Exception as e:
        print(f" Movement test failed: {e}")
        return False
    
    return False

def move_to_color_with_verification(color):
    """Move robot to color position with verification"""
    
    if color not in SAFE_POSITIONS:
        print(f" No position defined for {color}")
        return False
    
    target_position = SAFE_POSITIONS[color]
    print(f"\n Moving to {color.upper()} position...")
    print(f"Target: {[round(x, 3) for x in target_position]}")
    
    try:
        # Clear any collision detection
        robot.clear_collision_detected()
        
        # Move to position
        robot.move_pose(*target_position)
        print(f" Move command sent for {color.upper()}")
        
        # Wait for movement to complete
        time.sleep(3)
        
        # Verify position
        actual_position = get_current_robot_position()
        if actual_position:
            print(f"Actual: {[round(x, 3) for x in actual_position]}")
            
            # Calculate position error
            errors = [abs(actual_position[i] - target_position[i]) for i in range(6)]
            max_error = max(errors[:3])  # Check X, Y, Z errors
            
            print(f"Position errors (X,Y,Z): {[round(e, 4) for e in errors[:3]]}")
            
            if max_error < 0.01:  # Within 1cm
                print(f" Successfully reached {color.upper()} position!")
                return True
            else:
                print(f" Position error too large: {max_error:.4f}m")
                return False
        else:
            print(" Could not verify final position")
            return False
        
    except Exception as e:
        print(f" Failed to move to {color}: {e}")
        return False

def calibrate_positions_enhanced():
    """Enhanced position calibration with verification"""
    
    print("\n" + "="*50)
    print(" ENHANCED POSITION CALIBRATION")
    print("="*50)
    print("Instructions:")
    print("1. Manually jog robot to each color position")
    print("2. Position should be directly above the color")
    print("3. Use safe height (Z = 0.05 or higher)")
    print("4. Press ENTER to save each position")
    print("="*50)
    
    # Enable learning mode for manual jogging
    try:
        robot.set_learning_mode(True)
        print(" Learning mode enabled - you can now jog the robot manually")
    except:
        print(" Could not enable learning mode - try jogging anyway")
    
    new_positions = {}
    colors = ['green', 'orange', 'pink', 'yellow']
    
    for color in colors:
        print(f"\n--- Setting up {color.upper()} position ---")
        print("Manually jog the robot above the color square...")
        
        while True:
            show_current_position()
            response = input(f"Press ENTER to save {color} position, or 'r' to retry positioning: ").strip().lower()
            
            if response == 'r':
                continue
            
            # Save position
            current_pos = get_current_robot_position()
            if current_pos:
                new_positions[color] = current_pos
                print(f" Saved {color} position: {[round(x, 3) for x in current_pos]}")
                
                # Safety check - warn if Z is too low
                if current_pos[2] < 0.02:
                    print(f" WARNING: Z position is very low ({current_pos[2]:.3f}m)")
                    print("   This might cause collision with the surface!")
                    confirm = input("   Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                break
            else:
                print("❌ Could not get current position, try again")
    
    # Disable learning mode
    try:
        robot.set_learning_mode(False)
        print("✅ Learning mode disabled")
    except:
        pass
    
    # Update global positions
    SAFE_POSITIONS.update(new_positions)
    
    # Save to file
    save_positions()
    
    print("\n" + "="*50)
    print("🎉 CALIBRATION COMPLETE!")
    print("="*50)
    print("New positions:")
    for color, pos in new_positions.items():
        print(f"    '{color}': {pos},")
    
    # Test the new positions
    test_new_positions = input("\nTest the new positions? (y/n): ").strip().lower()
    if test_new_positions == 'y':
        test_all_positions()
    
    return new_positions

def test_all_positions():
    """Test all saved positions"""
    print("\n🧪 TESTING ALL POSITIONS...")
    
    for color in SAFE_POSITIONS:
        print(f"\n--- Testing {color.upper()} ---")
        success = move_to_color_with_verification(color)
        if success:
            input("Press ENTER to continue to next color...")
        else:
            retry = input(f"❌ {color} failed. Retry? (y/n): ").strip().lower()
            if retry == 'y':
                move_to_color_with_verification(color)

def compare_positions():
    """Compare current positions with saved ones"""
    print("\n📊 POSITION COMPARISON:")
    print("="*60)
    
    for color, saved_pos in SAFE_POSITIONS.items():
        print(f"\n{color.upper()}:")
        print(f"  Saved:   {[round(x, 3) for x in saved_pos]}")
        
        # Move to position and check
        try:
            robot.move_pose(*saved_pos)
            time.sleep(2)
            actual_pos = get_current_robot_position()
            if actual_pos:
                print(f"  Actual:  {[round(x, 3) for x in actual_pos]}")
                errors = [abs(actual_pos[i] - saved_pos[i]) for i in range(3)]
                print(f"  Errors:  {[round(e, 4) for e in errors]} (X,Y,Z)")
            else:
                print("  Actual:  Could not get position")
        except Exception as e:
            print(f"  Error: {e}")

def detect_color_centers():
    """Detect centers of colored squares using OpenCV"""
    
    # Initialize camera (adjust camera index if needed)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return None
    
    print("📸 Taking photo in 3 seconds...")
    time.sleep(3)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to capture image")
        return None
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges (you may need to adjust these)
    color_ranges = {
        'green': ([40, 50, 50], [80, 255, 255]),      
        'orange': ([10, 100, 100], [25, 255, 255]),   
        'pink': ([140, 50, 50], [170, 255, 255]),     
        'yellow': ([20, 50, 50], [35, 255, 255])      
    }
    
    centers = {}
    
    # Create display image
    display_img = frame.copy()
    
    for color, (lower, upper) in color_ranges.items():
        # Create mask for this color
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (should be the colored square)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate center
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers[color] = (cx, cy)
                
                # Draw on display image
                cv2.circle(display_img, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(display_img, f"{color}: ({cx},{cy})", (cx-50, cy-20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                print(f"✅ {color.upper()}: center at pixel ({cx}, {cy})")
            else:
                print(f"❌ Could not find center for {color}")
        else:
            print(f"❌ No {color} detected")
    
    # Show the result
    cv2.imshow('Detected Colors', display_img)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()
    
    return centers

def voice_control_mode():
    """Voice control mode - reads color from whisper JSON file"""
    print("\n🎤 VOICE CONTROL MODE")
    print("="*40)
    print("Instructions:")
    print("1. Run your whisper script to record and process voice command")
    print("2. The script should save the color to 'spoken_color.json'")
    print("3. Press ENTER here to read the color and move the robot")
    print("="*40)
    
    while True:
        print(f"\nWaiting for voice command...")
        print(f"Make sure '{SPOKEN_COLOR_FILE}' exists after running your whisper script")
        
        choice = input("\nPress ENTER to check for voice command, or 'back' to return to main menu: ").strip().lower()
        
        if choice == 'back':
            break
        
        # Read the spoken color
        spoken_color = read_spoken_color()
        
        if spoken_color:
            print(f"\n🎯 Voice command received: Move to {spoken_color.upper()}")
            
            # Confirm before moving
            confirm = input("Execute this command? (y/n): ").strip().lower()
            if confirm == 'y':
                success = move_to_color_with_verification(spoken_color)
                if success:
                    print(f"✅ Successfully executed voice command!")
                    # Clear the file after successful execution
                    clear_spoken_color()
                else:
                    print(f"❌ Failed to execute voice command")
            else:
                print("Command cancelled")
                clear_spoken_color()
        else:
            print("❌ No valid voice command found")
            print("   Please run your whisper script first")

def main():
    """Enhanced main function with voice control option"""
    
    # Load any previously saved positions
    load_saved_positions()
    
    print("🎯 NIRYO COLOR TARGETING SYSTEM WITH VOICE CONTROL")
    print("="*55)
    
    while True:
        print("\nChoose an option:")
        print("1. 🎯 Move to color positions (manual input)")
        print("2. 🔧 Calibrate positions")
        print("3. 📸 Detect colors and move")
        print("4. 🎤 Voice control mode")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            # Move to specific color (manual input)
            while True:
                print("\nAvailable colors: green, orange, pink, yellow")
                color = input("Enter color to move to (or 'back' to return to main menu): ").strip().lower()
                
                if color == 'back':
                    break
                elif color in ['green', 'orange', 'pink', 'yellow']:
                    print(f"\n--- Moving to {color.upper()} ---")
                    move_to_color_with_verification(color)
                else:
                    print("❌ Invalid color. Please choose: green, orange, pink, or yellow")
        
        elif choice == "2":
            # Enhanced calibration
            calibrate_positions_enhanced()
        
        elif choice == "3":
            # Detect colors and move
            print("📸 Detecting color centers...")
            centers = detect_color_centers()
            
            if centers:
                print(f"\nDetected {len(centers)} colors")
                for color in centers:
                    print(f"\n--- Moving to detected {color.upper()} ---")
                    move_to_color_with_verification(color)
                    input("Press ENTER for next color...")
            else:
                print("❌ No colors detected")
        
        elif choice == "4":
            # Voice control mode
            voice_control_mode()
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        try:
            robot.close_connection()
            print("🔌 Robot connection closed")
        except:
            pass