import cv2
import os
import numpy as np
import mediapipe as mp

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
DATA_PATH = os.path.join(BASE_DIR, 'staticData')  # Data folder in the same directory as the script
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ['on', 'off']
SAVE_SAMPLES_PER_LABEL = 201

# Create data directories
for label in LABELS:
    os.makedirs(os.path.join(DATA_PATH, label), exist_ok=True)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,  # Static mode for single frames
                       max_num_hands=1,
                       min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Video capture from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

label_index = 0
sample_count = 0

print("Press 'b' for back, 'n' for next label, 's' to save frame, and 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from webcam.")
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_landmarks = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            for lm in handLms.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])

    # Display current label
    label = LABELS[label_index]
    cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save frame or landmarks
    key = cv2.waitKey(1)
    if key == ord('s'):
        if len(hand_landmarks) == 63:  # Ensure landmarks are detected
            save_path = os.path.join(DATA_PATH, label, f"{label}_{sample_count}.npy")
            print(f"Saving to: {save_path}")
            try:
                np.save(save_path, np.array(hand_landmarks))
                print(f"[✔] Saved {save_path}")
                sample_count += 1
            except Exception as e:
                print(f"Error saving file: {e}")
        else:
            print("No hand landmarks detected. Try again.")

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