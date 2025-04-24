import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model
model_path = "C:/Users/kalpe/OneDrive/Desktop/5g kapse/5g kapse/5G_project/best_mask.pt"
 # Using relative path since the file is in the same directory
model = YOLO(model_path)

# Check if CUDA (GPU) is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 = Default Camera

if not cap.isOpened():
    print("❌ Error: Could not open webcam!")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame!")
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(frame, conf=0.5)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the output
    cv2.imshow("Mask Detection - Press 'Q' to Quit", annotated_frame)

    # Exit if 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
