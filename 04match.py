import json

# === Load Spoken Color ===
try:
    with open("spoken_color.json", "r") as f:
        spoken_data = json.load(f)
    spoken_color = spoken_data.get("color", "").upper().strip()  # normalize to UPPERCASE
    print(f" Spoken Color to point at: {spoken_color}")
except FileNotFoundError:
    print(" Error: 'spoken_color.json' not found.")
    exit()

# === Load YOLO-detected Centers ===
try:
    with open("detected_centers.json", "r") as f:
        centers = json.load(f)
    print(" Successfully loaded detected centers:")
    for color, coords in centers.items():
        print(f"  {color.lower()} - Center: {coords}")
except FileNotFoundError:
    print(" Error: 'detected_centers.json' not found.")
    exit()

# === Match and Point ===
normalized_centers = {k.upper(): v for k, v in centers.items()}
target_coords = normalized_centers.get(spoken_color)

if target_coords:
    print(f" Pointing to {spoken_color} at coordinates: {target_coords}")

    # === Save final matched color and coordinates ===
    target_data = {
        "color": spoken_color,
        "coordinates": target_coords
    }

    with open("final_target.json", "w") as f:
        json.dump(target_data, f)

    print(" Saved final target to 'final_target.json'")

else:
    print(f" Could not find coordinates for color: {spoken_color}")
    print(" Debug Tip: Check if color labels match exactly in both JSONs.")
