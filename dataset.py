import cv2
import os
import numpy as np
import mediapipe as mp
from collections import deque

# Setup
SEQUENCE_LENGTH = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
DATA_PATH = os.path.join(BASE_DIR, 'data')  # Data folder in the same directory as the script
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ['on', 'off']
SAVE_SAMPLES_PER_LABEL = 201

# Create data directories
for label in LABELS:
    os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ESP32-CAM MJPEG stream URL
#ESP32_CAM_URL = "http://192.168.8.104:81/stream"  # Replace with your ESP32-CAM's stream URL

# Video capture from ESP32-CAM
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open video stream from ESP32-CAM.")
    exit()

sequence = deque(maxlen=SEQUENCE_LENGTH)
label_index = 0
sample_count = 0

print("Press 'b' for back, 'n' for next label, 's' to save sequence, and 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from ESP32-CAM.")
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = frame.shape
    hand_landmarks = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            for lm in handLms.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])
        sequence.append(hand_landmarks)
    else:
        sequence.append([0] * 63)  # Empty frame fallback

    # Display current label
    label = LABELS[label_index]
    cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save sequence
    key = cv2.waitKey(1)
    if key == ord('s') and len(sequence) == SEQUENCE_LENGTH:
        seq_array = np.array(sequence)
        save_path = os.path.join(DATA_PATH, label, f"{label}_{sample_count}.npy")
        print(f"Saving to: {save_path}")
        try:
            np.save(save_path, seq_array)
            print(f"[✔] Saved {save_path}")
            sample_count += 1
        except Exception as e:
            print(f"Error saving file: {e}")

    elif key == ord('q'):
        break

    elif key == ord('b'):  # 'b' key for back
        label_index = (label_index - 1) % len(LABELS)
        sample_count = len(os.listdir(os.path.join(DATA_PATH, LABELS[label_index])))
        print(f"Switched to label: {LABELS[label_index]}, sample count: {sample_count}")

    elif key == ord('n'):  # 'n' key for next
        label_index = (label_index + 1) % len(LABELS)
        sample_count = len(os.listdir(os.path.join(DATA_PATH, LABELS[label_index])))
        print(f"Switched to label: {LABELS[label_index]}, sample count: {sample_count}")

    cv2.imshow("ASL Data Collection", frame)

cap.release()
cv2.destroyAllWindows()
