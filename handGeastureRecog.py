import cv2
import serial
import time
import mediapipe as mp
import threading
from queue import Queue

# Configuration
MJPEG_URL = 'http://192.168.1.100/stream'
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

def initialize_serial(port, baud_rate):
    """Initialize the serial connection."""
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Allow time for the connection to establish
        return ser
    except serial.SerialException as e:
        raise Exception(f"Failed to initialize serial connection: {e}")

def initialize_video_stream(url):
    """Initialize the video stream."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise Exception(f"Cannot open stream {url}")
    return cap

def initialize_mediapipe_hands():
    """Initialize MediaPipe Hands."""
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def video_processing_thread(cap, hands, gesture_queue, stop_event):
    """Thread for video processing."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting video thread...")
            break

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Default gesture text
        gesture_text = "No hands"
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hl, mp.solutions.hands.HAND_CONNECTIONS
                )
                # Placeholder for gesture logic
                gesture_text = "Gesture"

        # Add gesture text to the queue
        gesture_queue.put(gesture_text)

        # Display the gesture text on the frame
        cv2.putText(frame, gesture_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ESP32-CAM Stream", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

def serial_communication_thread(ser, gesture_queue, stop_event):
    """Thread for serial communication."""
    while not stop_event.is_set():
        if not gesture_queue.empty():
            gesture_text = gesture_queue.get()
            ser.write((gesture_text + '\n').encode('utf-8'))

    ser.close()
    print("Serial port closed.")

def main():
    ser = initialize_serial(SERIAL_PORT, BAUD_RATE)
    cap = initialize_video_stream(MJPEG_URL)
    hands = initialize_mediapipe_hands()

    # Queue for sharing gesture data between threads
    gesture_queue = Queue()

    # Event to signal threads to stop
    stop_event = threading.Event()

    # Create and start threads
    video_thread = threading.Thread(target=video_processing_thread, args=(cap, hands, gesture_queue, stop_event))
    serial_thread = threading.Thread(target=serial_communication_thread, args=(ser, gesture_queue, stop_event))

    video_thread.start()
    serial_thread.start()

    # Wait for threads to finish
    video_thread.join()
    serial_thread.join()
    print("Exiting program.")

if __name__ == "__main__":
    main()