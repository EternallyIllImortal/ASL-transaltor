import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import logging

# -- Configuration --
WEBCAM_INDEX = 0  # Index of the built-in webcam
MODEL_PATH = 'C:/Users/Edson Longares/OneDrive/Documents/EmbedProject/LSTM_training/asl_lstm_model_optimized.keras'
LABEL_MAP = 'c:/Users/Edson Longares/OneDrive/Documents/EmbedProject/label_map.json'
FRAME_SKIP = 30  # Process every 30th frame
# ----------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load TensorFlow model
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP, 'r') as f:
        label_map = json.load(f)
    return model, label_map

# Initialize MediaPipe Hands
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands, mp_hands

# Main function
def main():
    logging.info("Initializing components...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        logging.critical("Failed to open webcam. Exiting...")
        return

    model, label_map = load_model_and_labels()
    hands, mp_hands = init_mediapipe()
    logging.info("Initialization complete.")

    frame_count = 0
    buffer = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture frame. Exiting...")
                break

            frame_count += 1

            # Process every 30th frame
            if frame_count % FRAME_SKIP == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    # Extract hand landmarks
                    feature = np.array([
                        [[lm.x, lm.y, lm.z] for lm in hl.landmark]
                        for hl in results.multi_hand_landmarks
                    ])
                    feature = feature[:1]  # Limit to 1 hand
                    buffer.append(feature.flatten())

                    # Perform inference if buffer is full
                    if len(buffer) == FRAME_SKIP:
                        X = np.array(buffer)[None, ...]  # Add batch dimension
                        probs = model.predict(X, verbose=0)
                        idx = np.argmax(probs)
                        gesture_text = label_map.get(str(idx), "UNK")
                        buffer = []  # Clear buffer after inference
                    else:
                        gesture_text = "Buffering..."
                else:
                    gesture_text = "No Hands"

                # Display results
                cv2.putText(frame, gesture_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow("Sign Language Translator", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.critical(f"Error occurred: {e}")
    finally:
        logging.info("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        logging.info("Resources released.")

if __name__ == "__main__":
    main()