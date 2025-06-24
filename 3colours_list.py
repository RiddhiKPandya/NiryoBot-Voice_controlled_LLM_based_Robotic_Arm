import json

def get_detected_centers_from_yolo():
    try:
        with open("detected_centers.json", "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(" Error reading JSON file:", e)
        return {}

if __name__ == "__main__":
    centers = get_detected_centers_from_yolo()
    
    if centers:
        print(" Successfully loaded detected centers:")
        for color, center in centers.items():
            print(f"  {color.upper()} - Center: {center}")
    else:
        print(" No detected centers found.")
