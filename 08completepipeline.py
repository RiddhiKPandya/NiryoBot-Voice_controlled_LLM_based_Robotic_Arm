import cv2
import numpy as np
from pyniryo import *
import time
import json
import os
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import spacy
import re

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

# Home position (adjust as needed)
HOME_POSITION = [0.25, 0.0, 0.15, 0.0, 0.0, 0.0]

# File to store calibrated positions
POSITIONS_FILE = "calibrated_positions.json"

# Load whisper model and spacy model globally
print("Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

colors = ["red", "green", "pink", "yellow", "lime_green"]  # Extended colors

# Enhanced intent detection patterns
MOVEMENT_INTENTS = {
    # Direct movement verbs
    'movement_verbs': [
        'go', 'move', 'point', 'reach', 'navigate', 'travel', 'head', 'proceed',
        'advance', 'approach', 'direct', 'aim', 'target', 'visit', 'touch'
    ],
    
    # Movement phrases
    'movement_phrases': [
        'go to', 'move to', 'point at', 'point to', 'reach for', 'head to',
        'navigate to', 'travel to', 'proceed to', 'advance to', 'approach the',
        'direct to', 'aim at', 'target the', 'visit the', 'touch the'
    ],
    
    # Object identifiers
    'object_identifiers': [
        'card', 'square', 'block', 'tile', 'piece', 'target', 'spot', 'area',
        'zone', 'position', 'location', 'place', 'object', 'item'
    ]
}

def enhanced_intent_detection(text):
    """
    Enhanced intent detection that works with various command formats
    Returns: (intent, color) or (None, None) if no valid command found
    """
    text_lower = text.lower().strip()
    
    # Method 1: Direct pattern matching with regex
    color_found = None
    intent_found = None
    
    # Find colors in the text
    for color in colors:
        color_variations = [color, color.replace('_', ' '), color.replace('_', '')]
        for variation in color_variations:
            if variation.lower() in text_lower:
                color_found = color
                break
        if color_found:
            break
    
    # Method 2: Check for movement intents using multiple approaches
    
    # Approach 2A: Check movement phrases first (more specific)
    for phrase in MOVEMENT_INTENTS['movement_phrases']:
        if phrase in text_lower:
            intent_found = 'move'
            break
    
    # Approach 2B: Check individual movement verbs
    if not intent_found:
        words = text_lower.split()
        for word in words:
            if word in MOVEMENT_INTENTS['movement_verbs']:
                intent_found = 'move'
                break
    
    # Method 3: Regex patterns for common structures
    if not intent_found or not color_found:
        # Pattern: "verb (to/at) color object"
        pattern = r'\b(go|move|point|reach|head|navigate|travel|proceed|advance|approach|direct|aim|target|visit|touch)\s*(to|at|for|the)?\s*(red|green|pink|yellow|orange|lime|lime_green)\s*(card|square|block|tile|piece|target|spot|area|zone|position|location|place|object|item)?'
        match = re.search(pattern, text_lower)
        if match:
            intent_found = 'move'
            matched_color = match.group(3)
            # Map color variations
            if matched_color == 'lime':
                color_found = 'lime_green'
            else:
                color_found = matched_color
    
    # Method 4: Fallback - if we found a color and any movement-related word
    if color_found and not intent_found:
        movement_keywords = MOVEMENT_INTENTS['movement_verbs'] + [phrase.split()[0] for phrase in MOVEMENT_INTENTS['movement_phrases']]
        for word in text_lower.split():
            if word in movement_keywords:
                intent_found = 'move'
                break
    
    # Method 5: Super fallback - if we have a color and the text seems like a command
    if color_found and not intent_found:
        # Check if text contains imperative indicators
        imperative_indicators = ['to', 'at', 'the', 'please', 'can', 'will', 'should']
        if any(indicator in text_lower for indicator in imperative_indicators):
            intent_found = 'move'
    
    print(f"🔍 Intent Analysis:")
    print(f"   Original text: '{text}'")
    print(f"   Detected intent: {intent_found}")
    print(f"   Detected color: {color_found}")
    
    return intent_found, color_found

def get_compound_noun(token):
    parts = [tok.text for tok in token.lefts if tok.dep_ == "compound"]
    parts.append(token.text)
    return " ".join(parts)

def get_objects(verb_token):
    objects = []
    for child in verb_token.children:
        if child.dep_ in ("dobj", "attr", "oprd", "pobj"):
            objects.append(get_compound_noun(child))
        elif child.dep_ == "prep":
            for subchild in child.children:
                if subchild.dep_ == "pobj":
                    objects.append(get_compound_noun(subchild))
    return objects

def extract_intents_and_objects(sentence):
    """Enhanced version that combines spaCy analysis with pattern matching"""
    
    # First, try the enhanced intent detection
    intent, color = enhanced_intent_detection(sentence)
    if intent and color:
        # Map colors to robot positions
        color_mapping = {
            'green': 'green',
            'lime_green': 'green',  
            'orange': 'orange',
            'red': 'orange',  
            'pink': 'pink',
            'yellow': 'yellow'
        }
        
        mapped_color = color_mapping.get(color.lower())
        if mapped_color:
            return [('move', [f"{color} card"])]
    
    # Fallback to original spaCy-based analysis
    doc = nlp(sentence)
    results = []
    seen_verbs = set()

    for token in doc:
        if token.pos_ == "VERB":
            if (token.lemma_, token.i) in seen_verbs:
                continue
            seen_verbs.add((token.lemma_, token.i))

            objs = get_objects(token)
            results.append((token.lemma_, objs))

            for conj in token.conjuncts:
                if conj.pos_ == "VERB" and (conj.lemma_, conj.i) not in seen_verbs:
                    seen_verbs.add((conj.lemma_, conj.i))
                    conj_objs = get_objects(conj)
                    results.append((conj.lemma_, conj_objs))

    # Original patch for 'point at <color> card' style
    if not results:
        for i, token in enumerate(doc):
            if token.text.lower() == "point":
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "at":
                        for subchild in child.children:
                            if subchild.dep_ == "pobj" and subchild.text.lower() == "card":
                                color = None
                                for grandchild in subchild.lefts:
                                    if grandchild.text.lower() in colors:
                                        color = grandchild.text.lower()
                                if color:
                                    results.append(("point", [f"{color} card"]))

    return results

def extract_color_from_objects(objects):
    for obj in objects:
        for color in colors:
            if color in obj.lower():
                return color.upper()  # Match with YOLO keys
    return None

def record_and_process_voice():
    """Record voice and extract color command with enhanced processing"""
    fs = 44100
    seconds = 5

    print("🎙️  Recording... (5 seconds)")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("mic_input.wav", fs, recording)
    print("✅ Recording complete. Processing...")

    try:
        result = whisper_model.transcribe("mic_input.wav", task="translate")
        
        print(f"\n📝 Translated Text: {result['text']}")
        
        # Try enhanced intent detection first
        intent, detected_color = enhanced_intent_detection(result['text'])
        
        if intent == 'move' and detected_color:
            # Map colors to robot positions
            color_mapping = {
                'green': 'green',
                'lime_green': 'green',  
                'orange': 'orange',
                'red': 'orange',  
                'pink': 'pink',
                'yellow': 'yellow'
            }
            
            mapped_color = color_mapping.get(detected_color.lower())
            if mapped_color and mapped_color in SAFE_POSITIONS:
                print(f"\n💡 Enhanced Detection - Color: {detected_color} → {mapped_color}")
                return mapped_color
        
        # Fallback to original spaCy analysis
        print("\n🧠 Extracted Intents and Objects:")
        output = extract_intents_and_objects(result["text"])
        
        for intent, objs in output:
            print(f" → Intent: {intent}, Objects: {objs}")

            # Accept both 'point' and 'move' intents, plus other movement verbs
            if intent in ['point', 'move', 'go', 'reach', 'navigate', 'head', 'travel', 'proceed', 'advance', 'approach', 'direct', 'aim', 'target', 'visit', 'touch']:
                color = extract_color_from_objects(objs)
                if color:
                    # Map colors to robot positions
                    color_mapping = {
                        'GREEN': 'green',
                        'LIME_GREEN': 'green',  
                        'ORANGE': 'orange',
                        'RED': 'orange',  
                        'PINK': 'pink',
                        'YELLOW': 'yellow'
                    }
                    
                    mapped_color = color_mapping.get(color.upper())
                    if mapped_color and mapped_color in SAFE_POSITIONS:
                        print(f"\n💡 spaCy Detection - Color: {color} → {mapped_color}")
                        return mapped_color
                    else:
                        print(f"⚠️ Color '{color}' not supported")
                else:
                    print("⚠️ No valid color found in spoken command.")
        
        print("⚠️ No movement command detected")
        return None
        
    except Exception as e:
        print(f"❌ Error processing voice: {e}")
        return None

def load_saved_positions():
    """Load previously saved positions"""
    global SAFE_POSITIONS
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                saved_positions = json.load(f)
                SAFE_POSITIONS.update(saved_positions)
                print(f"✅ Loaded saved positions from {POSITIONS_FILE}")
                return True
        except Exception as e:
            print(f"⚠️ Error loading positions: {e}")
    return False

def save_positions():
    """Save current positions to file"""
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(SAFE_POSITIONS, f, indent=2)
        print(f"✅ Positions saved to {POSITIONS_FILE}")
    except Exception as e:
        print(f"❌ Error saving positions: {e}")

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
        print(f"❌ Error getting robot position: {e}")
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
        print("   ❌ Could not get position")
        return None

def move_to_home():
    """Move robot to home position"""
    print("\n🏠 Moving to HOME position...")
    try:
        robot.move_pose(*HOME_POSITION)
        time.sleep(2)
        print("✅ Reached HOME position")
        return True
    except Exception as e:
        print(f"❌ Failed to move to home: {e}")
        return False

def move_to_color_with_verification(color):
    """Move robot to color position with verification"""
    
    if color not in SAFE_POSITIONS:
        print(f"❌ No position defined for {color}")
        return False
    
    target_position = SAFE_POSITIONS[color]
    print(f"\n🤖 Moving to {color.upper()} position...")
    print(f"Target: {[round(x, 3) for x in target_position]}")
    
    try:
        # Clear any collision detection
        robot.clear_collision_detected()
        
        # Move to position
        robot.move_pose(*target_position)
        print(f"✅ Move command sent for {color.upper()}")
        
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
                print(f"✅ Successfully reached {color.upper()} position!")
                return True
            else:
                print(f"⚠️ Position error too large: {max_error:.4f}m")
                return False
        else:
            print("❌ Could not verify final position")
            return False
        
    except Exception as e:
        print(f"❌ Failed to move to {color}: {e}")
        return False

def continuous_voice_control():
    """Continuous voice control mode"""
    print("\n🎤 CONTINUOUS VOICE CONTROL MODE")
    print("="*50)
    print("Instructions:")
    print("✅ Supported commands:")
    print("   • 'point at green card' / 'point to green card'")
    print("   • 'go to pink card' / 'move to pink card'")
    print("   • 'reach for yellow card' / 'head to yellow card'")
    print("   • 'navigate to orange card' / 'travel to orange card'")
    print("   • 'touch the green card' / 'visit the pink card'")
    print("   • And many more variations!")
    print("2. Robot will move to the color and return home")
    print("3. Press Ctrl+C to exit")
    print("="*50)
    
    # Move to home position first
    move_to_home()
    
    while True:
        try:
            print(f"\n{'='*30}")
            print("🎯 Ready for voice command!")
            print("Try commands like:")
            print("  • 'go to green card'")
            print("  • 'move to pink'") 
            print("  • 'reach yellow card'")
            print("  • 'head to orange'")
            print("Press Ctrl+C to exit")
            print("="*30)
            
            # Record and process voice
            detected_color = record_and_process_voice()
            
            if detected_color:
                print(f"\n🎯 Command: Move to {detected_color.upper()}")
                
                # Move to the detected color
                success = move_to_color_with_verification(detected_color)
                
                if success:
                    print(f"✅ Successfully moved to {detected_color.upper()}!")
                    
                    # Wait a moment at the position
                    time.sleep(1)
                    
                    print("✅ Ready for next command!")
                else:
                    print(f"❌ Failed to move to {detected_color}")
            else:
                print("❌ No valid color command detected. Try again.")
                print("   Examples:")
                print("   • 'go to green card'")
                print("   • 'move to pink'")
                print("   • 'reach yellow card'")
                print("   • 'point at orange card'")
            
            # Small delay before next recording
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n Exiting continuous voice control...")
            break
        except Exception as e:
            print(f" Error in voice control: {e}")
            continue

def calibrate_positions_enhanced():
    """Enhanced position calibration with verification"""
    
    print("\n" + "="*50)
    print("🎯 ENHANCED POSITION CALIBRATION")
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
        print("Learning mode enabled - you can now jog the robot manually")
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
                print("Could not get current position, try again")
    
    # Disable learning mode
    try:
        robot.set_learning_mode(False)
        print(" Learning mode disabled")
    except:
        pass
    
    # Update global positions
    SAFE_POSITIONS.update(new_positions)
    
    # Save to file
    save_positions()
    
    print("\n" + "="*50)
    print(" CALIBRATION COMPLETE!")
    print("="*50)
    print("New positions:")
    for color, pos in new_positions.items():
        print(f"    '{color}': {pos},")
    
    return new_positions

def test_all_positions():
    """Test all saved positions"""
    print("\n TESTING ALL POSITIONS...")
    
    for color in SAFE_POSITIONS:
        print(f"\n--- Testing {color.upper()} ---")
        success = move_to_color_with_verification(color)
        if success:
            input("Press ENTER to continue to next color...")
        else:
            retry = input(f" {color} failed. Retry? (y/n): ").strip().lower()
            if retry == 'y':
                move_to_color_with_verification(color)

def detect_color_centers():
    """Detect centers of colored squares using OpenCV"""
    
    # Initialize camera (adjust camera index if needed)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Cannot open camera")
        return None
    
    print("📸 Taking photo in 3 seconds...")
    time.sleep(3)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(" Failed to capture image")
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
                
                print(f" {color.upper()}: center at pixel ({cx}, {cy})")
            else:
                print(f" Could not find center for {color}")
        else:
            print(f" No {color} detected")
    
    # Show the result
    cv2.imshow('Detected Colors', display_img)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()
    
    return centers

def main():
    """Enhanced main function with continuous voice control"""
    
    # Load any previously saved positions
    load_saved_positions()
    
    print(" NIRYO COLOR TARGETING SYSTEM WITH ENHANCED VOICE CONTROL")
    print("="*70)
    
    while True:
        print("\nChoose an option:")
        print("1.  Move to color positions (manual input)")
        print("2.  Calibrate positions")
        print("3.  Detect colors and move")
        print("4.  Enhanced continuous voice control mode")
        print("5.  Move to home position")
        print("6.  Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
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
                    print("Invalid color. Please choose: green, orange, pink, or yellow")
        
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
                print(" No colors detected")
        
        elif choice == "4":
            # Enhanced continuous voice control mode
            continuous_voice_control()
        
        elif choice == "5":
            # Move to home position
            move_to_home()
        
        elif choice == "6":
            print(" Goodbye!")
            break
        
        else:
            print(" Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Stopped by user")
    except Exception as e:
        print(f"\n Error: {e}")
    finally:
        try:
            robot.close_connection()
            print("🔌 Robot connection closed")
        except:
            pass