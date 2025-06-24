import cv2
import numpy as np
from collections import defaultdict
import math
import time

class PolygonColorDetector:
    def __init__(self):
        # Very loose HSV color ranges for better detection
        self.color_ranges = {
            'lime_green': {  # Lime green/light green range
                'lower': np.array([35, 40, 40]),    # Lime green range
                'upper': np.array([65, 255, 255])
            },
            'green': {  # Darker green range
                'lower': np.array([65, 40, 40]),    # Darker green range
                'upper': np.array([85, 255, 255])
            },
            'yellow': {
                'lower': np.array([15, 50, 50]),    # Yellow range
                'upper': np.array([35, 255, 255])
            },
            'pink': {
                'lower': np.array([140, 50, 50]),   # Pink/magenta range
                'upper': np.array([180, 255, 255])
            },
            'red': {  # Additional red range for pink detection
                'lower': np.array([0, 50, 50]),
                'upper': np.array([15, 255, 255])
            }
        }
        
        # Morphological kernels
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = np.ones((7, 7), np.uint8)
        
        # Detection parameters
        self.min_contour_area = 1000
        self.max_contour_area = 50000
        self.min_polygon_sides = 3
        self.stability_threshold = 5  # Frames to consider detection stable
        
        # Tracking variables
        self.detected_polygons = []
        self.polygon_history = defaultdict(list)
        self.frame_count = 0
        
    def preprocess_image(self, frame):
        """
        Advanced preprocessing to enhance polygon detection
        """
        # Convert to different color spaces for better edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Multiple preprocessing approaches
        preprocessed_images = {}
        
        # 1. Standard edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_canny = cv2.Canny(blurred, 50, 150)
        preprocessed_images['canny'] = edges_canny
        
        # 2. Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        preprocessed_images['adaptive'] = adaptive_thresh
        
        # 3. Color-based edge detection using LAB
        l_channel = lab[:, :, 0]
        l_edges = cv2.Canny(l_channel, 50, 150)
        preprocessed_images['lab_edges'] = l_edges
        
        # 4. Saturation-based detection
        s_channel = hsv[:, :, 1]
        s_thresh = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY)[1]
        preprocessed_images['saturation'] = s_thresh
        
        # 5. Combined approach
        combined = cv2.bitwise_or(edges_canny, l_edges)
        combined = cv2.bitwise_or(combined, s_thresh)
        preprocessed_images['combined'] = combined
        
        return preprocessed_images, hsv
    
    def find_polygon_contours(self, preprocessed_images):
        """
        Find polygon contours using multiple approaches
        """
        all_contours = []
        
        for method_name, processed_img in preprocessed_images.items():
            # Apply morphological operations
            cleaned = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, self.kernel_medium)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self.kernel_small)
            
            # Find contours
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_contour_area < area < self.max_contour_area:
                    all_contours.append({
                        'contour': contour,
                        'area': area,
                        'method': method_name
                    })
        
        return all_contours
    
    def analyze_polygon_shape(self, contour):
        """
        Analyze contour to determine if it's a valid polygon
        """
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Approximate polygon
        epsilon = 0.02 * perimeter  # Adjustable approximation accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate various shape properties
        area = cv2.contourArea(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate solidity (convex hull ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = x + w//2, y + h//2
        
        return {
            'approx_polygon': approx,
            'vertices': len(approx),
            'center': (center_x, center_y),
            'bounding_rect': (x, y, w, h),
            'area': area,
            'perimeter': perimeter,
            'extent': extent,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'is_valid_polygon': (
                len(approx) >= self.min_polygon_sides and 
                extent > 0.5 and 
                solidity > 0.7 and
                0.3 < aspect_ratio < 3.0
            )
        }
    
    def classify_polygon_color(self, frame_hsv, polygon_info):
        """
        Classify polygon color using the center region
        """
        center_x, center_y = polygon_info['center']
        x, y, w, h = polygon_info['bounding_rect']
        
        # Create a mask for the polygon area
        mask = np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_info['approx_polygon']], 255)
        
        # Extract color information from the masked region
        masked_hsv = cv2.bitwise_and(frame_hsv, frame_hsv, mask=mask)
        
        # Calculate mean HSV values in the polygon region
        mean_hsv = cv2.mean(masked_hsv, mask=mask)[:3]  # Exclude alpha channel
        mean_hsv = np.array(mean_hsv, dtype=np.float32)  # Convert to numpy array
        
        # Also sample center region for more accurate color detection
        center_region_size = min(w, h) // 4
        center_x1 = max(0, center_x - center_region_size)
        center_y1 = max(0, center_y - center_region_size)
        center_x2 = min(frame_hsv.shape[1], center_x + center_region_size)
        center_y2 = min(frame_hsv.shape[0], center_y + center_region_size)
        
        center_region = frame_hsv[center_y1:center_y2, center_x1:center_x2]
        if center_region.size > 0:
            center_mean_hsv = np.mean(center_region.reshape(-1, 3), axis=0)
        else:
            center_mean_hsv = mean_hsv  # Fallback to polygon mean if center region is empty
        
        # Use weighted average of full polygon and center region
        final_hsv = (mean_hsv * 0.6 + center_mean_hsv * 0.4)
        
        # Classify color based on HSV values
        detected_colors = []
        confidence_scores = {}
        
        for color_name, color_range in self.color_ranges.items():
            lower = color_range['lower']
            upper = color_range['upper']
            
            # Check if HSV values fall within range
            if (lower[0] <= final_hsv[0] <= upper[0] and
                lower[1] <= final_hsv[1] <= upper[1] and
                lower[2] <= final_hsv[2] <= upper[2]):
                
                # Calculate confidence based on how close to center of range
                h_center = (lower[0] + upper[0]) / 2
                s_center = (lower[1] + upper[1]) / 2
                v_center = (lower[2] + upper[2]) / 2
                
                h_diff = abs(final_hsv[0] - h_center) / max(1, (upper[0] - lower[0]))
                s_diff = abs(final_hsv[1] - s_center) / max(1, (upper[1] - lower[1]))
                v_diff = abs(final_hsv[2] - v_center) / max(1, (upper[2] - lower[2]))
                
                confidence = 1.0 - (h_diff + s_diff + v_diff) / 3
                confidence_scores[color_name] = max(0.0, confidence)
                detected_colors.append(color_name)
        
        # Return the color with highest confidence, or 'unknown'
        if detected_colors:
            best_color = max(confidence_scores.keys(), key=lambda x: confidence_scores[x])
            return best_color, confidence_scores[best_color], final_hsv
        else:
            return 'unknown', 0.0, final_hsv
    
    def track_polygon_stability(self, polygon_info):
        """
        Track polygon detection stability over multiple frames
        """
        center = polygon_info['center']
        polygon_id = f"{center[0]//50}_{center[1]//50}"  # Grid-based ID
        
        self.polygon_history[polygon_id].append({
            'frame': self.frame_count,
            'center': center,
            'area': polygon_info['area'],
            'color': polygon_info.get('color', 'unknown')
        })
        
        # Keep only recent history
        self.polygon_history[polygon_id] = self.polygon_history[polygon_id][-10:]
        
        # Check if polygon is stable
        recent_detections = [p for p in self.polygon_history[polygon_id] 
                           if self.frame_count - p['frame'] < self.stability_threshold]
        
        return len(recent_detections) >= self.stability_threshold
    
    def process_frame(self, frame):
        """
        Main processing function
        """
        self.frame_count += 1
        results = []
        
        # Step 1: Preprocess image
        preprocessed_images, frame_hsv = self.preprocess_image(frame)
        
        # Step 2: Find polygon contours
        contour_candidates = self.find_polygon_contours(preprocessed_images)
        
        # Step 3: Analyze each contour
        valid_polygons = []
        for candidate in contour_candidates:
            polygon_info = self.analyze_polygon_shape(candidate['contour'])
            
            if polygon_info['is_valid_polygon']:
                polygon_info['detection_method'] = candidate['method']
                valid_polygons.append(polygon_info)
        
        # Step 4: Remove duplicate detections (same polygon detected by multiple methods)
        unique_polygons = self.remove_duplicate_polygons(valid_polygons)
        
        # Step 5: Classify colors for each unique polygon
        for polygon_info in unique_polygons:
            color, confidence, hsv_values = self.classify_polygon_color(frame_hsv, polygon_info)
            polygon_info['color'] = color
            polygon_info['color_confidence'] = confidence
            polygon_info['hsv_values'] = hsv_values
            
            # Step 6: Track stability
            is_stable = self.track_polygon_stability(polygon_info)
            polygon_info['is_stable'] = is_stable
            
            if is_stable:  # Only report stable detections
                results.append(polygon_info)
        
        return results, preprocessed_images
    
    def remove_duplicate_polygons(self, polygons):
        """
        Remove duplicate polygon detections based on center proximity
        """
        if not polygons:
            return []
        
        unique_polygons = []
        used_indices = set()
        
        for i, poly1 in enumerate(polygons):
            if i in used_indices:
                continue
                
            # Find all polygons close to this one
            similar_polygons = [poly1]
            used_indices.add(i)
            
            for j, poly2 in enumerate(polygons[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                # Calculate distance between centers
                dist = math.sqrt(
                    (poly1['center'][0] - poly2['center'][0])**2 + 
                    (poly1['center'][1] - poly2['center'][1])**2
                )
                
                # If centers are close, consider them the same polygon
                if dist < 50:  # Adjustable threshold
                    similar_polygons.append(poly2)
                    used_indices.add(j)
            
            # Keep the polygon with largest area among similar ones
            best_polygon = max(similar_polygons, key=lambda x: x['area'])
            unique_polygons.append(best_polygon)
        
        return unique_polygons
    
    def draw_results(self, frame, results, preprocessed_images=None):
        """
        Draw detection results on frame
        """
        output_frame = frame.copy()
        
        for i, polygon_info in enumerate(results):
            center = polygon_info['center']
            color = polygon_info['color']
            confidence = polygon_info['color_confidence']
            vertices = polygon_info['vertices']
            area = polygon_info['area']
            
            # Color mapping for drawing
            draw_colors = {
                'lime_green': (0, 255, 0),
                'green': (0, 200, 0),
                'yellow': (0, 255, 255),
                'pink': (255, 0, 255),
                'red': (0, 0, 255),
                'unknown': (128, 128, 128)
            }
            
            draw_color = draw_colors.get(color, (128, 128, 128))
            
            # Draw polygon outline
            cv2.drawContours(output_frame, [polygon_info['approx_polygon']], -1, draw_color, 3)
            
            # Draw center point
            cv2.circle(output_frame, center, 8, draw_color, -1)
            
            # Draw bounding rectangle
            x, y, w, h = polygon_info['bounding_rect']
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), draw_color, 2)
            
            # Add text information
            text_lines = [
                f"{color.upper()}",
                f"Conf: {confidence:.2f}",
                f"Vertices: {vertices}",
                f"Area: {int(area)}",
                f"Center: {center}"
            ]
            
            for j, line in enumerate(text_lines):
                cv2.putText(output_frame, line, (center[0] + 15, center[1] - 60 + j*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
        
        return output_frame

def analyze_detections(all_detections):
    """
    Analyze all detections to get stable final coordinates
    """
    if not all_detections:
        return {}
    
    # Group detections by color
    color_groups = {}
    for detection in all_detections:
        color = detection['color']
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(detection)
    
    # Calculate stable centers for each color
    final_targets = {}
    for color, detections in color_groups.items():
        if len(detections) >= 3:  # Need at least 3 detections for stability
            # Calculate weighted average (more recent detections have higher weight)
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            total_confidence = 0
            
            for i, det in enumerate(detections):
                weight = det['confidence'] * (1 + i * 0.1)  # Recency weight
                weighted_x += det['center'][0] * weight
                weighted_y += det['center'][1] * weight
                total_weight += weight
                total_confidence += det['confidence']
            
            if total_weight > 0:
                final_center = (
                    int(weighted_x / total_weight),
                    int(weighted_y / total_weight)
                )
                avg_confidence = total_confidence / len(detections)
                
                final_targets[color] = {
                    'center': final_center,
                    'confidence': avg_confidence,
                    'detections': len(detections)
                }
    
    return final_targets

def main():
    """
    Main function with optimized detection process:
    1. Detect for 5 seconds using Niryo camera
    2. List detected colors with coordinates
    3. Show available colors
    """
    from pyniryo import NiryoRobot
    
    # Initialize detector
    detector = PolygonColorDetector()
    
    # Connect to the Niryo Ned2 robot for camera access only
    robot_ip = "10.10.10.10"  # Change this to your robot's IP if different
    print(f"Connecting to Niryo Ned2 robot at {robot_ip} for camera access...")
    
    try:
        robot = NiryoRobot(robot_ip)
        print("Successfully connected to Niryo Ned2!")
    except Exception as e:
        print(f"Error connecting to robot: {e}")
        return
    
    # Detection Phase
    print("\n" + "="*60)
    print("🔍 COLOR POLYGON DETECTION")
    print("="*60)
    print("Starting color card detection...")
    print("The system will detect colored cards for 5 seconds!")
    
    start_time = time.time()
    detection_duration = 5.0
    frame_count = 0
    all_stable_detections = []
    
    try:
        while time.time() - start_time < detection_duration:
            try:
                # Get image from Niryo robot camera
                img_bytes = robot.get_img_compressed()
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Process frame
                results, preprocessed_images = detector.process_frame(frame)
                
                # Draw results
                output_frame = detector.draw_results(frame, results, preprocessed_images)
                
                # Add countdown timer to display
                remaining_time = detection_duration - (time.time() - start_time)
                cv2.putText(output_frame, f"Detection Time: {remaining_time:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Collect stable detections
                for result in results:
                    if result['is_stable'] and result['color_confidence'] > 0.5:
                        all_stable_detections.append({
                            'color': result['color'],
                            'center': result['center'],
                            'confidence': result['color_confidence'],
                            'frame': frame_count
                        })
                
                # Display the frame
                cv2.imshow('Niryo Ned2 - Color Polygon Detection', output_frame)
                
                # Allow early exit with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error in detection: {e}")
                continue
        
        cv2.destroyAllWindows()
        
        # Analysis and Results
        print("\n" + "="*60)
        print("📊 DETECTION RESULTS")
        print("="*60)
        
        # Process all detections to get final coordinates
        final_targets = analyze_detections(all_stable_detections)
        
        if not final_targets:
            print("No stable colored cards detected!")
            print("Please ensure colored cards are clearly visible and try again.")
            return
        
        print(f" DETECTED {len(final_targets)} STABLE COLORED CARDS:")
        print("\nDetected Colors with Coordinates and Centers:")
        print("-" * 50)
        
        for i, (color, data) in enumerate(final_targets.items(), 1):
            center = data['center']
            confidence = data['confidence']
            detections = data['detections']
            print(f"{i}. {color.upper():<12} | Center: {center} | Confidence: {confidence:.3f} | Detections: {detections}")
        
        print("\n" + "="*60)
        print(" AVAILABLE COLORS")
        print("="*60)
        
        available_colors = list(final_targets.keys())
        print("Colors successfully detected:")
        for i, color in enumerate(available_colors, 1):
            center = final_targets[color]['center']
            conf = final_targets[color]['confidence']
            print(f"  {i}. {color.upper()} - Center: {center} - Confidence: {conf:.3f}")
        
        print(f"\n🏁 Detection completed! Found {len(available_colors)} colored polygon(s).")
        
    except KeyboardInterrupt:
        print("\n Operation interrupted by user")
    
    except Exception as e:
        print(f"\n An error occurred: {e}")
    
    finally:
        cv2.destroyAllWindows()
        try:
            import json
            final_color_centers = {
                result['color'].lower(): result['center']
                for result in all_stable_detections
                if result['confidence'] > 0.5
            }
            with open("detected_centers.json", "w") as f:
                json.dump(final_color_centers, f)
            print("\n Saved detected centers to 'detected_centers.json'")
        except Exception as e:
            print(f"\n Could not save detected centers: {e}")
            
        robot.close_connection()
        print("\n Robot connection closed.")
        print("="*60)
        print(" DETECTION COMPLETED")
        print("="*60)

if __name__ == "__main__":
    main()
    