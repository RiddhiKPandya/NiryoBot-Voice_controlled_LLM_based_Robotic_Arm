** Instructions **
Run all the codes in the same order after connecting to the Ned Niryo Robotic Arm.
Make sure to change the IP address 10.10.10.10 to your specific robot's IP.

** Code Overview **
00calibration.py → Manually calibrates the workspace by accessing the 4 corners and the center for accuracy.

01yolomodel.py → Detects the color using the HSV scale and identifies the object using the YOLOv8 model.
The detected output is saved in a JSON file called detected_centers.json.

02whisper&intent.py → Uses the Whisper model (base, small, or tiny as required) to accept voice input in 70+ languages.
It extracts the intent and object from the sentence, saving the result in spoken_color.json.

03colour_list.py → Displays the list of detected colors and their corresponding pixel coordinates (from the YOLO model) in a JSON format.

04match.py → Matches the color from detected_centers.json with the target color in spoken_color.json.
Once matched, it extracts the pixel coordinates of the required color.

05change_coordinate.py → Converts the pixel coordinates into x, y, z real-world coordinates so the robotic arm can move accordingly.

06move_there.py → Commands the robotic arm to move to the converted coordinates.

08completepipeline.py → Integrates all components and runs the full pipeline.
It takes continuous voice input and moves the robotic arm continuously based on detected objects and intents.
