import json
import numpy as np
import matplotlib.pyplot as plt

def pixel_to_robot_coordinates_improved(pixel_x, pixel_y, calibration_data):
    """
    Convert pixel coordinates to robot coordinates using proper bilinear interpolation
    with validation and error checking.
    """
    
    # Extract calibration points
    tl_pixel = np.array(calibration_data['top_left']['pixel'])
    tr_pixel = np.array(calibration_data['top_right']['pixel'])
    bl_pixel = np.array(calibration_data['bottom_left']['pixel'])
    br_pixel = np.array(calibration_data['bottom_right']['pixel'])
    
    tl_robot = np.array(calibration_data['top_left']['robot'])
    tr_robot = np.array(calibration_data['top_right']['robot'])
    bl_robot = np.array(calibration_data['bottom_left']['robot'])
    br_robot = np.array(calibration_data['bottom_right']['robot'])
    
    # Check if the pixel coordinates are within the calibrated region
    min_x = min(tl_pixel[0], bl_pixel[0])
    max_x = max(tr_pixel[0], br_pixel[0])
    min_y = min(tl_pixel[1], tr_pixel[1])
    max_y = max(bl_pixel[1], br_pixel[1])
    
    if not (min_x <= pixel_x <= max_x and min_y <= pixel_y <= max_y):
        print(f"Warning: Pixel coordinates ({pixel_x}, {pixel_y}) are outside calibrated region")
        print(f"Calibrated region: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")
    
    # Calculate normalized coordinates (0 to 1) within the calibrated area
    # Use proper normalization based on the actual pixel boundaries
    width = max_x - min_x
    height = max_y - min_y
    
    u = (pixel_x - min_x) / width  # normalized x (0 to 1)
    v = (pixel_y - min_y) / height  # normalized y (0 to 1)
    
    # Clamp values to [0, 1] range
    u = max(0, min(1, u))
    v = max(0, min(1, v))
    
    # Bilinear interpolation
    # For each axis (x, y, z), interpolate using the four corner points
    robot_coords = []
    
    for axis in range(3):  # x, y, z coordinates
        # Get the four corner values for this axis
        top_left_val = tl_robot[axis]
        top_right_val = tr_robot[axis]
        bottom_left_val = bl_robot[axis]
        bottom_right_val = br_robot[axis]
        
        # Bilinear interpolation formula
        # First interpolate along top edge
        top_interp = top_left_val * (1 - u) + top_right_val * u
        
        # Then interpolate along bottom edge
        bottom_interp = bottom_left_val * (1 - u) + bottom_right_val * u
        
        # Finally interpolate between top and bottom
        final_val = top_interp * (1 - v) + bottom_interp * v
        
        robot_coords.append(round(final_val, 4))
    
    return robot_coords

def validate_calibration(calibration_data):
    """
    Validate the calibration data for consistency and potential issues.
    """
    print(" CALIBRATION VALIDATION")
    print("=" * 50)
    
    # Check if calibration points form a proper rectangle
    tl = calibration_data['top_left']
    tr = calibration_data['top_right']
    bl = calibration_data['bottom_left']
    br = calibration_data['bottom_right']
    center = calibration_data.get('center', None)
    
    # Pixel coordinate checks
    print("Pixel Coordinate Analysis:")
    print(f"Top-Left:     {tl['pixel']}")
    print(f"Top-Right:    {tr['pixel']}")
    print(f"Bottom-Left:  {bl['pixel']}")
    print(f"Bottom-Right: {br['pixel']}")
    if center:
        print(f"Center:       {center['pixel']}")
    
    # Check if points form a proper rectangle in pixel space
    width_top = tr['pixel'][0] - tl['pixel'][0]
    width_bottom = br['pixel'][0] - bl['pixel'][0]
    height_left = bl['pixel'][1] - tl['pixel'][1]
    height_right = br['pixel'][1] - tr['pixel'][1]
    
    print(f"\nPixel Rectangle Check:")
    print(f"Top width:    {width_top}")
    print(f"Bottom width: {width_bottom}")
    print(f"Left height:  {height_left}")
    print(f"Right height: {height_right}")
    
    if abs(width_top - width_bottom) > 5:
        print("  Warning: Top and bottom widths differ significantly")
    if abs(height_left - height_right) > 5:
        print("  Warning: Left and right heights differ significantly")
    
    # Robot coordinate analysis
    print("\nRobot Coordinate Analysis:")
    for point_name, point_data in calibration_data.items():
        if point_name != 'center':
            print(f"{point_name:12}: {point_data['robot']}")
    if center:
        print(f"{'center':12}: {center['robot']}")
    
    # Check Z-coordinate consistency
    z_coords = [point['robot'][2] for point in calibration_data.values()]
    z_range = max(z_coords) - min(z_coords)
    print(f"\nZ-coordinate range: {z_range:.4f}m")
    if z_range > 0.005:  # More than 5mm difference
        print(" Warning: Z-coordinates vary significantly. Ensure consistent height.")
    
    # Test center point accuracy if available
    if center:
        calculated_center = pixel_to_robot_coordinates_improved(
            center['pixel'][0], center['pixel'][1], calibration_data
        )
        actual_center = center['robot']
        error = [abs(calc - actual) for calc, actual in zip(calculated_center, actual_center)]
        
        print(f"\nCenter Point Validation:")
        print(f"Calculated: {calculated_center}")
        print(f"Actual:     {actual_center}")
        print(f"Error:      {error}")
        print(f"Max error:  {max(error):.4f}m")
        
        if max(error) > 0.01:  # More than 1cm error
            print("  Warning: Large interpolation error detected!")

def visualize_calibration(calibration_data, target_pixel=None):
    """
    Create a visualization of the calibration points and target.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pixel space visualization
    ax1.set_title('Pixel Space Calibration')
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    ax1.invert_yaxis()  # Invert Y-axis to match image coordinates
    
    # Plot calibration points in pixel space
    points = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    colors = ['red', 'blue', 'green', 'orange']
    
    for point, color in zip(points, colors):
        px, py = calibration_data[point]['pixel']
        ax1.plot(px, py, 'o', color=color, markersize=10, label=point)
        ax1.annotate(f'{point}\n({px}, {py})', (px, py), 
                    xytext=(5, 5), textcoords='offset points')
    
    if 'center' in calibration_data:
        cx, cy = calibration_data['center']['pixel']
        ax1.plot(cx, cy, 's', color='purple', markersize=8, label='center')
        ax1.annotate(f'center\n({cx}, {cy})', (cx, cy),
                    xytext=(5, 5), textcoords='offset points')
    
    if target_pixel:
        ax1.plot(target_pixel[0], target_pixel[1], '*', color='red', 
                markersize=15, label='target')
        ax1.annotate(f'target\n({target_pixel[0]}, {target_pixel[1]})', 
                    target_pixel, xytext=(5, 5), textcoords='offset points')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Robot space visualization
    ax2.set_title('Robot Space Calibration')
    ax2.set_xlabel('Robot X (m)')
    ax2.set_ylabel('Robot Y (m)')
    
    for point, color in zip(points, colors):
        rx, ry = calibration_data[point]['robot'][:2]
        ax2.plot(rx, ry, 'o', color=color, markersize=10, label=point)
        ax2.annotate(f'{point}\n({rx:.3f}, {ry:.3f})', (rx, ry),
                    xytext=(5, 5), textcoords='offset points')
    
    if 'center' in calibration_data:
        crx, cry = calibration_data['center']['robot'][:2]
        ax2.plot(crx, cry, 's', color='purple', markersize=8, label='center')
        ax2.annotate(f'center\n({crx:.3f}, {cry:.3f})', (crx, cry),
                    xytext=(5, 5), textcoords='offset points')
    
    if target_pixel:
        target_robot = pixel_to_robot_coordinates_improved(
            target_pixel[0], target_pixel[1], calibration_data
        )
        ax2.plot(target_robot[0], target_robot[1], '*', color='red', 
                markersize=15, label='target')
        ax2.annotate(f'target\n({target_robot[0]:.3f}, {target_robot[1]:.3f})', 
                    target_robot[:2], xytext=(5, 5), textcoords='offset points')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('calibration_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# Updated calibration data based on your latest calibration
calibration_data = {
    'top_left': {
        'pixel': [100, 100],
        'robot': [0.158, -0.091, 0.031]
    },
    'top_right': {
        'pixel': [540, 100],
        'robot': [0.145, 0.077, 0.031]
    },
    'bottom_left': {
        'pixel': [100, 380],
        'robot': [0.317, -0.065, 0.031]
    },
    'bottom_right': {
        'pixel': [540, 380],
        'robot': [0.295, 0.099, 0.031]
    },
    'center': {
        'pixel': [320, 240],
        'robot': [0.234, 0.008, 0.032]
    }
}

# Validate the calibration
validate_calibration(calibration_data)

# Test with your RED target coordinates
target_pixel = [163, 356]
print(f"\n TARGET CONVERSION")
print("=" * 50)
print(f"Target pixel coordinates: {target_pixel}")

# Convert using improved method
robot_coordinates = pixel_to_robot_coordinates_improved(
    target_pixel[0], target_pixel[1], calibration_data
)

print(f"Converted robot coordinates: {robot_coordinates}")

# Create visualization
try:
    visualize_calibration(calibration_data, target_pixel)
    print("\n Calibration visualization saved as 'calibration_visualization.png'")
except Exception as e:
    print(f"Visualization failed: {e}")

# Save improved conversion results
result = {
    'calibration_info': {
        'description': 'Niryo robot workspace calibration data (improved)',
        'calibration_points': calibration_data,
        'validation_status': 'validated'
    },
    'target_object': {
        'color': 'RED',
        'pixel_coordinates': target_pixel,
        'robot_coordinates': robot_coordinates,
        'coordinates_format': '[x, y, z] in meters'
    },
    'conversion_method': 'Improved bilinear interpolation with validation',
    'notes': 'Includes boundary checking and error analysis'
}

# Save results
with open('improved_robot_coordinates.json', 'w') as f:
    json.dump(result, f, indent=4)

print(f"\n Results saved to 'improved_robot_coordinates.json'")

# Additional diagnostic information
print(f"\n DIAGNOSTIC INFORMATION")
print("=" * 50)

# Check if the target is within reasonable bounds
min_x_robot = min([p['robot'][0] for p in calibration_data.values()])
max_x_robot = max([p['robot'][0] for p in calibration_data.values()])
min_y_robot = min([p['robot'][1] for p in calibration_data.values()])
max_y_robot = max([p['robot'][1] for p in calibration_data.values()])

print(f"Robot workspace bounds:")
print(f"X: [{min_x_robot:.3f}, {max_x_robot:.3f}]")
print(f"Y: [{min_y_robot:.3f}, {max_y_robot:.3f}]")
print(f"Target robot coordinates: [{robot_coordinates[0]:.3f}, {robot_coordinates[1]:.3f}, {robot_coordinates[2]:.3f}]")

# Check if target is within bounds
within_bounds = (min_x_robot <= robot_coordinates[0] <= max_x_robot and 
                min_y_robot <= robot_coordinates[1] <= max_y_robot)
print(f"Target within bounds: {within_bounds}")

if not within_bounds:
    print("  Warning: Target coordinates are outside the calibrated workspace!")