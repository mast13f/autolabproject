"""
Live Camera View - Use this to adjust your camera position.
Press 'q' to quit.
"""

import cv2

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Live view running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera Live View - Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
