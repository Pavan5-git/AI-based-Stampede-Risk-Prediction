from ultralytics import YOLO
import cv2
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt

# Create folders if not exist
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(r"C:\Users\pavan cherukupalli\Downloads\videoplayback.mp4")
log_data = []

print("Press 'q' to stop...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                person_count += 1

    # Risk Calculation
    if person_count < 5:
        risk = "LOW"
        color = (0, 255, 0)
    elif person_count < 10:
        risk = "MEDIUM"
        color = (0, 255, 255)
    else:
        risk = "HIGH"
        color = (0, 0, 255)

    cv2.putText(frame, f"People Count: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"Risk Level: {risk}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Crowd Risk Monitor", frame)

    log_data.append([datetime.datetime.now(), person_count, risk])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save logs
df = pd.DataFrame(log_data, columns=["Timestamp", "People_Count", "Risk_Level"])
df.to_csv("logs/crowd_log.csv", index=False)

print("Log saved to logs/crowd_log.csv")

# Generate Graph
plt.figure()
plt.plot(df["People_Count"])
plt.title("Crowd Density Over Time")
plt.xlabel("Frame Index")
plt.ylabel("People Count")
plt.savefig("outputs/crowd_graph.png")

print("Graph saved to outputs/crowd_graph.png")
