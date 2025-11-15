import cv2
import pandas as pd
from datetime import datetime

# Ask user to choose input source
print("Select input mode:")
print("1. Live Webcam Feed")
print("2. Pre-Recorded Video File")
choice = input("Enter 1 or 2: ")

# Set up video capture based on choice
if choice == "1":
    cap = cv2.VideoCapture(0)  
elif choice == "2":
    video_path = input("Enter video file path (e.g., baby_video.mp4): ")
    cap = cv2.VideoCapture(video_path)
else:
    print("Invalid choice! Exiting...")
    exit()

cv2.namedWindow("Smart Cradle Feed", cv2.WINDOW_NORMAL)

cv2.setWindowProperty("Smart Cradle Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("Smart Cradle Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

first_frame = None
status_list = [0, 0]
times = []
df = pd.DataFrame(columns=["Start", "End"])

while True:
    check, frame = cap.read()
    if not check:
        break  

    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours, _ = cv2.findContours(thresh_frame.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Smart Cradle Feed", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):  
        break

cap.release()
cv2.destroyAllWindows()

if len(times) % 2 != 0:
    times.append(datetime.now())

# Save motion timestamps into CSV
for i in range(0, len(times), 2):
    df.loc[len(df)] = {"Start": times[i], "End": times[i+1]}

df.to_csv("sleep_log.csv", index=False)

print("✅ Motion analysis complete. Sleep log saved as sleep_log.csv")
