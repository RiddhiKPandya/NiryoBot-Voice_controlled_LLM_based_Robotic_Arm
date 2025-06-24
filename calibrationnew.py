from pyniryo import NiryoRobot
import json
import time
import numpy as np

class CalibrationDebugger:
    def __init__(self, robot_ip="10.10.10.10", safety_height_offset=0.005):
        self.robot_ip = robot_ip
        self.robot = None
        self.calibration_data = None
        self.safety_height_offset = safety_height_offset  # Height offset above table (in meters) - default 5mm
        
    def connect_robot(self):
        """Connect to the robot."""
        try:
            self.robot = NiryoRobot(self.robot_ip)
            print("✅ Connected to robot")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect_robot(self):
        """Disconnect from robot."""
        if self.robot:
            self.robot.close_connection()
            print("🔌 Disconnected from robot")
    
    def load_calibration(self, calibration_file=None):
        """Load calibration data."""
        if calibration_file:
            try:
                with open(calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                print(f"✅ Loaded calibration from {calibration_file}")
            except Exception as e:
                print(f"❌ Failed to load calibration: {e}")
                return False
        else:
            # Use your current calibration data
            self.calibration_data = {
                'top_left': {'pixel': [100, 100], 'robot': [0.158, -0.091, 0.031]},
                'top_right': {'pixel': [540, 100], 'robot': [0.145, 0.077, 0.031]},
                'bottom_left': {'pixel': [100, 380], 'robot': [0.317, -0.065, 0.031]},
                'bottom_right': {'pixel': [540, 380], 'robot': [0.295, 0.099, 0.031]},
                'center': {'pixel': [320, 240], 'robot': [0.234, 0.008, 0.032]}
            }
            print("✅ Using hardcoded calibration data")
            print(f"🛡️  Safety height offset: {self.safety_height_offset*1000:.1f}mm above calibrated Z positions")
        return True
    
    def pixel_to_robot_coordinates(self, pixel_x, pixel_y, apply_safety_offset=True):
        """Convert pixel to robot coordinates."""
        if not self.calibration_data:
            print("❌ No calibration data loaded")
            return None
            
        # Extract calibration points
        tl_pixel = self.calibration_data['top_left']['pixel']
        tr_pixel = self.calibration_data['top_right']['pixel']
        bl_pixel = self.calibration_data['bottom_left']['pixel']
        br_pixel = self.calibration_data['bottom_right']['pixel']
        
        tl_robot = self.calibration_data['top_left']['robot']
        tr_robot = self.calibration_data['top_right']['robot']
        bl_robot = self.calibration_data['bottom_left']['robot']
        br_robot = self.calibration_data['bottom_right']['robot']
        
        # Calculate bounds
        min_x = min(tl_pixel[0], bl_pixel[0])
        max_x = max(tr_pixel[0], br_pixel[0])
        min_y = min(tl_pixel[1], tr_pixel[1])
        max_y = max(bl_pixel[1], br_pixel[1])
        
        # Normalize coordinates
        width = max_x - min_x
        height = max_y - min_y
        
        u = (pixel_x - min_x) / width
        v = (pixel_y - min_y) / height
        
        print(f"   Normalized coordinates: u={u:.3f}, v={v:.3f}")
        
        # Bilinear interpolation
        robot_coords = []
        for axis in range(3):
            top_left_val = tl_robot[axis]
            top_right_val = tr_robot[axis]
            bottom_left_val = bl_robot[axis]
            bottom_right_val = br_robot[axis]
            
            top_interp = top_left_val * (1 - u) + top_right_val * u
            bottom_interp = bottom_left_val * (1 - u) + bottom_right_val * u
            final_val = top_interp * (1 - v) + bottom_interp * v
            
            robot_coords.append(round(final_val, 4))
        
        # Apply safety height offset to Z coordinate
        if apply_safety_offset:
            robot_coords[2] += self.safety_height_offset
            print(f"   Applied safety offset: Z = {robot_coords[2]:.4f}m (original + {self.safety_height_offset*1000:.1f}mm)")
        
        return robot_coords
    
    def move_to_safe_position(self, target_coords, description="target"):
        """Safely move to position with approach from above."""
        try:
            current_pose = self.robot.get_pose()
            
            # Step 1: Move to approach position (only 2cm above target for gentle approach)
            approach_z = target_coords[2] + 0.02  # Only 2cm above target
            approach_coords = [target_coords[0], target_coords[1], approach_z]
            
            print(f"   Step 1: Moving to approach position {approach_coords}")
            self.robot.move_pose(approach_coords[0], approach_coords[1], approach_coords[2],
                               current_pose.roll, current_pose.pitch, current_pose.yaw)
            time.sleep(1)
            
            # Step 2: Move down to target position
            print(f"   Step 2: Moving down to {description} position {target_coords}")
            self.robot.move_pose(target_coords[0], target_coords[1], target_coords[2],
                               current_pose.roll, current_pose.pitch, current_pose.yaw)
            time.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error during safe movement: {e}")
            return False
    
    def verify_calibration_point(self, point_name):
        """Move to a calibration point and verify accuracy."""
        if not self.robot or not self.calibration_data:
            print("❌ Robot not connected or calibration not loaded")
            return False
            
        if point_name not in self.calibration_data:
            print(f"❌ Point '{point_name}' not found in calibration data")
            return False
        
        # Get original coordinates and add safety offset
        original_coords = self.calibration_data[point_name]['robot']
        target_coords = [original_coords[0], original_coords[1], original_coords[2] + self.safety_height_offset]
        pixel_coords = self.calibration_data[point_name]['pixel']
        
        print(f"\n🎯 Verifying {point_name}")
        print(f"   Original robot coords: {original_coords}")
        print(f"   Safe target coords:    {target_coords}")
        print(f"   Corresponding pixels:  {pixel_coords}")
        
        # Use safe movement approach
        success = self.move_to_safe_position(target_coords, point_name)
        
        if success:
            # Get actual position
            actual_pose = self.robot.get_pose()
            actual_coords = [actual_pose.x, actual_pose.y, actual_pose.z]
            
            print(f"   Actual robot coords:   {[round(x, 4) for x in actual_coords]}")
            
            # Calculate error (compare with safe target, not original)
            error = [abs(target - actual) for target, actual in zip(target_coords, actual_coords)]
            total_error = sum(e**2 for e in error)**0.5
            
            print(f"   Position error: {[round(e, 4) for e in error]}")
            print(f"   Total error: {total_error:.4f}m")
            
            if total_error < 0.005:
                print("   ✅ Excellent accuracy!")
            elif total_error < 0.01:
                print("   ✅ Good accuracy!")
            else:
                print("   ⚠️  High positioning error!")
            
            return True
        
        return False
    
    def test_pixel_to_robot_conversion(self, test_pixels):
        """Test pixel to robot coordinate conversion."""
        print(f"\n🧪 TESTING PIXEL TO ROBOT CONVERSION")
        print("=" * 50)
        
        for i, pixel_coords in enumerate(test_pixels):
            print(f"\nTest {i+1}: Pixel {pixel_coords}")
            robot_coords = self.pixel_to_robot_coordinates(pixel_coords[0], pixel_coords[1])
            if robot_coords:
                print(f"   Converted to robot: {robot_coords}")
                
                # Check if coordinates are reasonable (accounting for safety offset)
                x, y, z = robot_coords
                if 0.1 <= x <= 0.4 and -0.15 <= y <= 0.15 and 0.03 <= z <= 0.08:
                    print("   ✅ Coordinates look reasonable")
                else:
                    print("   ⚠️  Coordinates might be outside workspace")
                    if z < 0.03:
                        print("   ⚠️  Z coordinate is very low - safety offset may be insufficient")
            else:
                print("   ❌ Conversion failed")
    
    def interactive_verification(self):
        """Interactive verification system."""
        print(f"\n🔍 INTERACTIVE CALIBRATION VERIFICATION")
        print("=" * 50)
        print(f"🛡️  Safety settings: {self.safety_height_offset*1000:.1f}mm height offset")
        
        while True:
            print("\nOptions:")
            print("1. Verify calibration point")
            print("2. Test pixel conversion (no movement)")
            print("3. Move to converted coordinates (safe)")
            print("4. Show current position")
            print("5. Test all calibration points")
            print("6. Adjust safety height offset")
            print("7. Emergency stop and move up")
            print("8. Exit")
            
            choice = input("\nSelect option (1-8): ")
            
            if choice == "1":
                print("\nAvailable points:", list(self.calibration_data.keys()))
                point = input("Enter point name: ")
                self.verify_calibration_point(point)
                
            elif choice == "2":
                try:
                    pixel_x = int(input("Enter pixel X: "))
                    pixel_y = int(input("Enter pixel Y: "))
                    robot_coords = self.pixel_to_robot_coordinates(pixel_x, pixel_y)
                    if robot_coords:
                        print(f"Safe coordinates: X={robot_coords[0]:.4f}, Y={robot_coords[1]:.4f}, Z={robot_coords[2]:.4f}")
                        
                        # Also show without safety offset for comparison
                        raw_coords = self.pixel_to_robot_coordinates(pixel_x, pixel_y, apply_safety_offset=False)
                        print(f"Raw coordinates:  X={raw_coords[0]:.4f}, Y={raw_coords[1]:.4f}, Z={raw_coords[2]:.4f}")
                except ValueError:
                    print("❌ Invalid input")
                    
            elif choice == "3":
                try:
                    pixel_x = int(input("Enter pixel X: "))
                    pixel_y = int(input("Enter pixel Y: "))
                    robot_coords = self.pixel_to_robot_coordinates(pixel_x, pixel_y)
                    
                    if robot_coords:
                        print(f"Target safe coordinates: {robot_coords}")
                        confirm = input("Move to this position safely? (y/N): ")
                        if confirm.lower() == 'y':
                            success = self.move_to_safe_position(robot_coords, "target")
                            if success:
                                actual_pose = self.robot.get_pose()
                                print(f"Moved to: X={actual_pose.x:.4f}, Y={actual_pose.y:.4f}, Z={actual_pose.z:.4f}")
                except ValueError:
                    print("❌ Invalid input")
                    
            elif choice == "4":
                pose = self.robot.get_pose()
                print(f"Current position: X={pose.x:.4f}, Y={pose.y:.4f}, Z={pose.z:.4f}")
                print(f"Orientation: Roll={pose.roll:.3f}, Pitch={pose.pitch:.3f}, Yaw={pose.yaw:.3f}")
                
            elif choice == "5":
                print("\n🧪 Testing all calibration points...")
                for point_name in ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'center']:
                    if point_name in self.calibration_data:
                        input(f"\nPress Enter to test {point_name}...")
                        self.verify_calibration_point(point_name)
                        
            elif choice == "6":
                try:
                    current_mm = self.safety_height_offset * 1000
                    new_offset_mm = float(input(f"Enter new safety height offset in mm (current: {current_mm:.1f}mm): "))
                    new_offset_m = new_offset_mm / 1000
                    if 0.5 <= new_offset_mm <= 20:
                        self.safety_height_offset = new_offset_m
                        print(f"✅ Safety height offset updated to {new_offset_mm:.1f}mm")
                    else:
                        print("❌ Safety offset must be between 0.5mm and 20mm")
                except ValueError:
                    print("❌ Invalid input")
                    
            elif choice == "7":
                print("🚨 Emergency: Moving to safe height...")
                current_pose = self.robot.get_pose()
                safe_z = max(0.08, current_pose.z + 0.02)  # Move up by 2cm or to 8cm, whichever is higher
                self.robot.move_pose(current_pose.x, current_pose.y, safe_z,
                                   current_pose.roll, current_pose.pitch, current_pose.yaw)
                print(f"✅ Moved to safe position: Z = {safe_z:.4f}m")
                        
            elif choice == "8":
                break
                
            else:
                print("❌ Invalid choice")
    
    def run_comprehensive_test(self):
        """Run comprehensive calibration test."""
        print(f"\n🔬 COMPREHENSIVE CALIBRATION TEST")
        print("=" * 50)
        print(f"🛡️  Using safety height offset: {self.safety_height_offset*1000:.1f}mm")
        
        # Test 1: Verify all calibration points
        print("\n1️⃣ CALIBRATION POINT VERIFICATION")
        print("-" * 30)
        
        calibration_errors = {}
        for point_name in ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'center']:
            if point_name in self.calibration_data:
                input(f"\nPress Enter to test {point_name}...")
                success = self.verify_calibration_point(point_name)
                if not success:
                    calibration_errors[point_name] = "Movement failed"
        
        # Test 2: Test pixel conversions
        print("\n2️⃣ PIXEL CONVERSION TEST")
        print("-" * 30)
        
        test_pixels = [
            [163, 356],  # Your red target
            [320, 240],  # Center
            [200, 200],  # Random test point 1
            [400, 300],  # Random test point 2
        ]
        
        self.test_pixel_to_robot_conversion(test_pixels)
        
        # Test 3: Check workspace boundaries
        print("\n3️⃣ WORKSPACE BOUNDARY ANALYSIS")
        print("-" * 30)
        
        robot_coords = [point['robot'] for point in self.calibration_data.values()]
        x_coords = [coord[0] for coord in robot_coords]
        y_coords = [coord[1] for coord in robot_coords]
        z_coords = [coord[2] for coord in robot_coords]
        
        print(f"Original Z range: {min(z_coords):.4f} to {max(z_coords):.4f}")
        print(f"Safe Z range:     {min(z_coords) + self.safety_height_offset:.4f} to {max(z_coords) + self.safety_height_offset:.4f} (+{self.safety_height_offset*1000:.1f}mm)")
        print(f"X range: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"Y range: {min(y_coords):.3f} to {max(y_coords):.3f}")
        
        # Test your red target
        red_target_coords = self.pixel_to_robot_coordinates(163, 356)
        if red_target_coords:
            print(f"\nRed target [163, 356] converts to: {red_target_coords}")
            x, y, z = red_target_coords
            
            x_in_range = min(x_coords) <= x <= max(x_coords)
            y_in_range = min(y_coords) <= y <= max(y_coords)
            z_safe = z >= min(z_coords) + self.safety_height_offset
            
            print(f"Within X range: {x_in_range}")
            print(f"Within Y range: {y_in_range}")
            print(f"Z is safe height: {z_safe}")
            
            if x_in_range and y_in_range and z_safe:
                print("✅ Red target is within safe calibrated workspace")
            else:
                print("⚠️  Red target might be outside safe workspace")
        
        print(f"\n📊 TEST SUMMARY")
        print("-" * 30)
        if calibration_errors:
            print("❌ Calibration errors found:")
            for point, error in calibration_errors.items():
                print(f"   {point}: {error}")
        else:
            print("✅ All calibration points verified successfully")
        
        print(f"🛡️  All movements performed with {self.safety_height_offset*1000:.1f}mm safety offset")

def main():
    print("🤖 NIRYO CALIBRATION DEBUGGER (SAFE MODE)")
    print("=" * 45)
    
    # Allow user to set safety offset
    try:
        safety_offset_mm = float(input("Enter safety height offset in mm (default 5mm): ") or "5")
        safety_offset = safety_offset_mm / 1000
        if safety_offset_mm < 0.5 or safety_offset_mm > 20:
            print("⚠️  Using default 5mm - offset should be between 0.5mm and 20mm")
            safety_offset = 0.005
    except ValueError:
        print("⚠️  Using default 5mm - invalid input")
        safety_offset = 0.005
    
    debugger = CalibrationDebugger(safety_height_offset=safety_offset)
    
    if not debugger.connect_robot():
        return
    
    if not debugger.load_calibration():
        debugger.disconnect_robot()
        return
    
    try:
        print("\nWhat would you like to do?")
        print("1. Interactive verification")
        print("2. Comprehensive test")
        print("3. Quick red target test (safe)")
        
        choice = input("\nSelect option (1-3): ")
        
        if choice == "1":
            debugger.interactive_verification()
        elif choice == "2":
            debugger.run_comprehensive_test()
        elif choice == "3":
            # Quick test for red target
            print(f"\n🎯 QUICK RED TARGET TEST (SAFE MODE)")
            print("=" * 35)
            
            red_coords = debugger.pixel_to_robot_coordinates(163, 356)
            print(f"Red target [163, 356] → {red_coords}")
            
            if red_coords:
                confirm = input("Move to red target position safely? (y/N): ")
                if confirm.lower() == 'y':
                    print("Using safe approach movement...")
                    success = debugger.move_to_safe_position(red_coords, "red target")
                    if success:
                        actual_pose = debugger.robot.get_pose()
                        print(f"Final safe position: X={actual_pose.x:.4f}, Y={actual_pose.y:.4f}, Z={actual_pose.z:.4f}")
        else:
            print("❌ Invalid choice")
            
    finally:
        # Move to safe position before disconnecting
        try:
            current_pose = debugger.robot.get_pose()
            safe_z = max(0.08, current_pose.z + 0.01)
            debugger.robot.move_pose(current_pose.x, current_pose.y, safe_z,
                                   current_pose.roll, current_pose.pitch, current_pose.yaw)
            print(f"🛡️  Moved to safe height: {safe_z:.4f}m before disconnecting")
            time.sleep(1)
        except:
            pass
        
        debugger.disconnect_robot()

if __name__ == "__main__":
    main()