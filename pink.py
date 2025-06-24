from pyniryo import *
import time
import math

# Connect to robot
robot_ip = "10.10.10.10"
robot = NiryoRobot(robot_ip)

# Your calibration data
calibration_data = {
    "top_left": {
        "pixel": [100, 100],
        "robot": [0.163, -0.081, 0.031]
    },
    "top_right": {
        "pixel": [540, 100],
        "robot": [0.156, 0.090, 0.032]
    },
    "bottom_left": {
        "pixel": [100, 540],
        "robot": [0.322, -0.072, 0.031]
    },
    "bottom_right": {
        "pixel": [540, 540],
        "robot": [0.314, 0.093, 0.031]
    },
    "center": {
        "pixel": [320, 240],
        "robot": [0.236, 0.011, 0.032]
    }
}


def calculate_robot_coordinates(px, py):
    """Calculate robot coordinates with boundary checking"""
    # Check if pixel is within calibrated bounds
    if px < 100 or px > 540 or py < 100 or py > 380:
        print(f"⚠️ WARNING: Pixel [{px}, {py}] is outside calibrated area!")
        print("Calibrated area: X=[100-540], Y=[100-380]")
        
        # Clamp to safe boundaries
        px = max(110, min(530, px))  # Leave 10px margin
        py = max(110, min(370, py))  # Leave 10px margin
        print(f"Clamped to safe pixel: [{px}, {py}]")
    
    # Calculate normalized coordinates
    u = (px - 100) / (540 - 100)
    v = (py - 100) / (380 - 100)
    
    # Bilinear interpolation
    tl = calibration_data["top_left"]["robot"]
    tr = calibration_data["top_right"]["robot"]
    bl = calibration_data["bottom_left"]["robot"]
    br = calibration_data["bottom_right"]["robot"]
    
    robot_coords = []
    for axis in range(3):
        top = tl[axis] * (1 - u) + tr[axis] * u
        bottom = bl[axis] * (1 - u) + br[axis] * u
        result = top * (1 - v) + bottom * v
        robot_coords.append(result)
    
    return robot_coords, px, py

def check_robot_limits(x, y, z):
    """Check if coordinates are within robot's physical limits"""
    # Basic reachability check for Niryo robot
    distance_from_base = math.sqrt(x**2 + y**2)
    
    # Niryo typical workspace limits (approximate)
    if distance_from_base > 0.35:  # 35cm from base
        print(f"⚠️ WARNING: Position may be outside robot reach!")
        print(f"Distance from base: {distance_from_base*100:.1f}cm")
        return False
    
    if z < 0.02 or z > 0.25:  # Z limits
        print(f"⚠️ WARNING: Z height {z:.3f} may be outside limits!")
        return False
        
    return True

def safe_move_to_pixel(target_px, target_py, safety_height_mm=20):
    """Safely move robot to pixel coordinates with comprehensive checking"""
    
    print(f"=== MOVING TO PIXEL [{target_px}, {target_py}] ===")
    
    try:
        # Clear collision detection
        robot.clear_collision_detected()
        
        # Get current position
        current_pose = robot.get_pose()
        print(f"Current position: {current_pose}")
        
        # Calculate target coordinates with boundary checking
        coords, safe_px, safe_py = calculate_robot_coordinates(target_px, target_py)
        TARGET_X, TARGET_Y, TARGET_Z = coords[0], coords[1], coords[2]
        
        # Add safety height
        SAFE_HEIGHT = TARGET_Z + (safety_height_mm / 1000.0)
        
        print(f"Calculated coordinates:")
        print(f"  Target: X={TARGET_X:.6f}, Y={TARGET_Y:.6f}, Z={TARGET_Z:.6f}")
        print(f"  With {safety_height_mm}mm clearance: Z={SAFE_HEIGHT:.6f}")
        
        # Check if coordinates are reachable
        if not check_robot_limits(TARGET_X, TARGET_Y, SAFE_HEIGHT):
            print("❌ Target position may not be reachable!")
            return False
        
        # Check distance from current position
        if hasattr(current_pose, 'x'):
            curr_x, curr_y, curr_z = current_pose.x, current_pose.y, current_pose.z
        else:
            curr_x, curr_y, curr_z = current_pose[0], current_pose[1], current_pose[2]
        
        distance = math.sqrt((TARGET_X - curr_x)**2 + (TARGET_Y - curr_y)**2 + (SAFE_HEIGHT - curr_z)**2)
        print(f"Distance to move: {distance*100:.1f}cm")
        
        # If movement is very large, break it into steps
        if distance > 0.15:  # More than 15cm
            print("Large movement detected - moving in steps...")
            
            # Move to safe intermediate position first
            mid_x = (curr_x + TARGET_X) / 2
            mid_y = (curr_y + TARGET_Y) / 2
            mid_z = max(curr_z, SAFE_HEIGHT) + 0.03  # Extra height for safety
            
            print(f"Intermediate position: X={mid_x:.6f}, Y={mid_y:.6f}, Z={mid_z:.6f}")
            robot.move_pose(mid_x, mid_y, mid_z, 0.0, 0.0, 0.0)
            time.sleep(4)
            
            intermediate_pose = robot.get_pose()
            print(f"Reached intermediate: {intermediate_pose}")
        
        # Move to final position
        print("Moving to final target position...")
        robot.move_pose(TARGET_X, TARGET_Y, SAFE_HEIGHT, 0.0, 0.0, 0.0)
        time.sleep(4)
        
        # Verify final position
        final_pose = robot.get_pose()
        print(f"Final position: {final_pose}")
        
        # Calculate actual movement achieved
        if hasattr(final_pose, 'x'):
            final_x, final_y = final_pose.x, final_pose.y
        else:
            final_x, final_y = final_pose[0], final_pose[1]
        
        actual_distance = math.sqrt((final_x - TARGET_X)**2 + (final_y - TARGET_Y)**2)
        
        if actual_distance < 0.01:  # Within 1cm
            print("✅ Successfully reached target!")
            print(f"Robot is now pointing at pixel [{safe_px}, {safe_py}]")
            return True
        else:
            print(f"⚠️ Position error: {actual_distance*100:.1f}cm from target")
            return False
            
    except Exception as e:
        print(f"❌ Movement failed: {e}")
        return False

# Test multiple positions
print("=== TESTING DIFFERENT PIXEL POSITIONS ===")

# Test positions around the working [295, 306]
test_positions = [
    [295, 306],  # Known working position
    [290, 300],  # Slightly different
    [300, 310],  # Slightly different
    [267, 145],  # Your problem position
    [320, 240],  # Center position (should work)
]

for px, py in test_positions:
    print(f"\n--- Testing pixel [{px}, {py}] ---")
    
    # Check if it's in safe bounds first
    if 110 <= px <= 530 and 110 <= py <= 370:
        print("✅ Pixel is within safe calibration bounds")
        
        # Calculate and check coordinates
        coords, _, _ = calculate_robot_coordinates(px, py)
        if check_robot_limits(coords[0], coords[1], coords[2] + 0.02):
            print("✅ Calculated position appears reachable")
        else:
            print("❌ Calculated position may be outside robot limits")
    else:
        print("❌ Pixel is outside calibration bounds")

print("\n=== MANUAL MOVEMENT TEST ===")
print("Choose a pixel position to test:")

# For interactive testing
target_pixel = [400, 450]  # Start with known working position
print(f"Testing movement to {target_pixel}")

try:
    success = safe_move_to_pixel(target_pixel[0], target_pixel[1])
    if success:
        print("Movement successful!")
    else:
        print("Movement had issues - check output above")
        
except Exception as e:
    print(f"Test failed: {e}")

finally:
    robot.close_connection()
    print("Robot connection closed.")

# Print safe movement zones
print("\n=== SAFE MOVEMENT RECOMMENDATIONS ===")
print("Based on your calibration:")
print("✅ SAFE PIXEL RANGE: X=[110-530], Y=[110-370]")
print("✅ CONFIRMED WORKING: [295, 306]")
print("✅ LIKELY SAFE ZONE: [280-330], [280-330] (around center)")
print("⚠️ EDGE ZONES: Near [100,100], [540,100], [100,380], [540,380]")
print("❌ OUTSIDE BOUNDS: < 100 or > 540 (X), < 100 or > 380 (Y)")