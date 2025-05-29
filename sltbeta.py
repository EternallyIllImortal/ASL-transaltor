import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import serial  # For serial communication
import time

# Load trained LSTM model
model = load_model("C:/Users/Edson Longares/OneDrive/Documents/EmbedProject/LSTM_training/asl_lstm_model_optimized.keras")

# Load labels dynamically from label_map.json
with open("C:/Users/Edson Longares/OneDrive/Documents/EmbedProject/label_map.json", "r") as f:
    label_map = json.load(f)
LABELS = [label_map[str(i)] for i in range(len(label_map))]

print(f"Loaded labels: {LABELS}")
print(f"Model output layer shape: {model.output_shape}")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Sequence buffer
sequence = []
SEQUENCE_LENGTH = 30

# ESP32-CAM MJPEG stream URL
ESP32_CAM_URL = "http://192.168.8.104:81/stream"  # Replace with your ESP32-CAM's stream URL

# Start video capture from ESP32-CAM
cap = cv2.VideoCapture(ESP32_CAM_URL)

if not cap.isOpened():
    print("Error: Unable to open video stream from ESP32-CAM.")
    exit()

# Initialize serial communication with ESP32
SERIAL_PORT = "COM8"  # Replace with your ESP32's COM port
BAUD_RATE = 115200
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for the serial connection to initialize
    print(f"Serial connection established on {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"Error: Unable to open serial port {SERIAL_PORT}. {e}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from ESP32-CAM.")
        break

    image = cv2.flip(frame, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        sequence.append(landmarks)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Predict when sequence is full
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data)[0]
            print(f"Model output shape: {prediction.shape}")
            print(f"Prediction: {prediction}")

            if len(LABELS) != prediction.shape[0]:
                raise ValueError(f"Mismatch between LABELS length ({len(LABELS)}) and model output size ({prediction.shape[0]}).")

            label = LABELS[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Send the predicted label to ESP32 via serial
            try:
                ser.write((label + "\n").encode('utf-8'))  # Send label followed by a newline
                print(f"Sent to ESP32: {label}")
            except serial.SerialException as e:
                print(f"Error: Unable to send data to ESP32. {e}")

            # Display prediction
            cv2.putText(image, f'{label} ({confidence:.2f})', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            sequence = []

    cv2.imshow("ASL Translator", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close serial connection
ser.close()
print("Serial connection closed.")