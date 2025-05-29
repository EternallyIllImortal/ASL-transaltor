import cv2
import serial
import time
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import logging
from threading import Thread, Lock
from queue import Queue

# -- Configuration --
MJPEG_URL   = 0
SERIAL_PORT = 'COM9'
BAUD_RATE   = 115200
MODEL_PATH  = 'C:/Users/Edson Longares/OneDrive/Documents/EmbedProject/LSTM_training/asl_lstm_model_optimized.keras'
LABEL_MAP   = 'c:/Users/Edson Longares/OneDrive/Documents/EmbedProject/label_map.json'
T           = 30
# ----------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Shared resources
frame_queue = Queue(maxsize=1)  # Queue for frames
result_queue = Queue(maxsize=1)  # Queue for gesture results
buffer = deque(maxlen=T)  # Buffer for feature vectors
buffer_lock = Lock()  # Lock for buffer access

# Setup functions
def setup_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # ESP32-WROVER reset delay
        return ser
    except serial.SerialException as e:
        logging.error(f"Serial setup failed: {e}")
        raise

def setup_camera(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise IOError(f"Cannot open stream at {url}")
    return cap

def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands, mp_hands

def load_label_map(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Label map file not found: {path}")
        raise

# Thread 1: Capture frames
def capture_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Stream read error.")
            break
        if not frame_queue.full():
            frame_queue.put(frame)

# Thread 2: Process frames and perform inference
def process_frames(hands, model, label_map):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Construct feature vector(s)
            if results.multi_hand_landmarks:
                feature = np.array([
                    [[lm.x, lm.y, lm.z] for lm in hl.landmark]
                    for hl in results.multi_hand_landmarks
                ])
                feature = feature[:1]  # Limit to 2 hands

                # Validate feature shape
                if feature.shape == (1, 21, 3):  # Ensure consistent shape
                    with buffer_lock:
                        buffer.append(feature.flatten())
                else:
                    logging.warning(f"Inconsistent feature shape: {feature.shape}")
            else:
                logging.warning("No hand landmarks detected.")

            # Sequence inference when buffer full
            gesture_text = "No Hands"
            with buffer_lock:
                if len(buffer) == T:
                    try:
                        X = np.array(buffer)[None, ...]  # Add batch dimension
                        probs = model.predict(X, verbose=0)
                        idx = np.argmax(probs)
                        gesture_text = label_map.get(str(idx), "UNK")
                    except Exception as e:
                        logging.error(f"Error during inference: {e}")

            # Send result to result queue
            if not result_queue.full():
                result_queue.put((frame, gesture_text))
        time.sleep(0.1)  # Add this line to limit processing rate (~30 FPS)

# Thread 3: Send results via serial and display
def send_results(ser):
    while True:
        if not result_queue.empty():
            frame, gesture_text = result_queue.get()

            # Send result via serial
            ser.write((gesture_text + '\n').encode('utf-8'))

            # Display locally (optional)
            cv2.putText(frame, gesture_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Sign Language Translator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Main function
if __name__ == "__main__":
    ser = None
    cap = None
    hands = None
    try:
        logging.info("Initializing components...")
        ser = setup_serial(SERIAL_PORT, BAUD_RATE)
        cap = setup_camera(MJPEG_URL)
        hands, mp_hands = init_mediapipe()
        model = tf.keras.models.load_model(MODEL_PATH)
        label_map = load_label_map(LABEL_MAP)
        logging.info("Initialization complete.")

        # Start threads
        capture_thread = Thread(target=capture_frames, args=(cap,))
        process_thread = Thread(target=process_frames, args=(hands, model, label_map))
        result_thread = Thread(target=send_results, args=(ser,))

        capture_thread.start()
        process_thread.start()
        result_thread.start()

        capture_thread.join()
        process_thread.join()
        result_thread.join()

    except Exception as e:
        logging.critical(f"Error occurred: {e}")
    finally:
        logging.info("Releasing resources...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()
        if hands is not None:
            hands.close()
        logging.info("Resources released.")