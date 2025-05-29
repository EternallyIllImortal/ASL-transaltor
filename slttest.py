import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import json

# Load trained LSTM model
model = load_model("C:/Users/Edson Longares/OneDrive/Documents/EmbedProject/LSTM_training/asl_lstm_model_optimizedv3.keras")

# Load labels dynamically from label_map.json
with open("C:/Users/Edson Longares/OneDrive/Documents/EmbedProject/label_map.json", "r") as f:
    label_map = json.load(f)
LABELS = [label_map[str(i)] for i in range(len(label_map))]

print(f"Loaded labels: {LABELS}")
print(f"Model output layer shape: {model.output_shape}")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Pad the input to match the expected shape (1, 30, 63)
        input_data = np.expand_dims(np.tile(landmarks, (30, 1)), axis=0)

        # Predict the gesture
        prediction = model.predict(input_data)[0]
        print(f"Model output shape: {prediction.shape}")
        print(f"Prediction: {prediction}")

        if len(LABELS) != prediction.shape[0]:
            raise ValueError(f"Mismatch between LABELS length ({len(LABELS)}) and model output size ({prediction.shape[0]}).")

        label = LABELS[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display prediction
        cv2.putText(image, f'{label} ({confidence:.2f})', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Translator", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
