import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import cv2
import pandas as pd

# Load model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("data/sample_video.mp4")

# Data storage
data = []
frame_count = 0

# Check video
if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Run tracking
    results = model.track(frame, persist=True)

    if results is None or len(results) == 0:
        continue

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            try:
                # Safe ID handling
                person_id = int(box.id) if box.id is not None else -1

                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Center point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Zone logic
                zone = "Zone A" if cx < frame.shape[1] // 2 else "Zone B"

                # Store data
                data.append([person_id, frame_count, cx, cy, zone])

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {person_id} - {zone}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

            except Exception as e:
                print("Error processing box:", e)

    # Show video
    cv2.imshow("Tracking + Zones", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()

# Save data
if len(data) > 0:
    df = pd.DataFrame(data, columns=["PersonID", "Frame", "X", "Y", "Zone"])
    df.to_csv("customer_data.csv", index=False)
    print(f"✅ Data saved successfully! Total records: {len(data)}")
else:
    print("❌ No data collected. Check video or detection.")