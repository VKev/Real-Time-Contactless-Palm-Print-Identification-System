import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands with desired parameters
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # For video input
    max_num_hands=1,               # Detect one hand
    model_complexity=1,            # Full model complexity
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def calculate_palm_center(hand_landmarks, image_shape):
    h, w, _ = image_shape
    # Landmarks corresponding to the wrist and base of fingers
    palm_landmarks = [
        mp_hands.HandLandmark.WRIST,               # 0
        mp_hands.HandLandmark.THUMB_CMC,           # 1
        mp_hands.HandLandmark.INDEX_FINGER_MCP,    # 5
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,   # 9
        mp_hands.HandLandmark.RING_FINGER_MCP,     # 13
        mp_hands.HandLandmark.PINKY_MCP            # 17
    ]
    # Calculate the average x and y coordinates of the palm landmarks
    cx = int(np.mean([hand_landmarks.landmark[l].x * w for l in palm_landmarks]))
    cy = int(np.mean([hand_landmarks.landmark[l].y * h for l in palm_landmarks]))
    return cx, cy

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            handedness = results.multi_handedness[idx].classification[0].label
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            wrist_coords = (int(wrist.x * w), int(wrist.y * h))
            cv2.putText(frame, f'{handedness} Hand', wrist_coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cx, cy = calculate_palm_center(hand_landmarks, frame.shape)

            roi_size = 100
            half_size = roi_size // 2

            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w, cx + half_size)
            y2 = min(h, cy + half_size)

            palm_roi = frame[y1:y2, x1:x2]
            if palm_roi.size > 0:
                palm_roi_resized = cv2.resize(palm_roi, (roi_size, roi_size))

                display_size = 300
                palm_roi_big = cv2.resize(palm_roi_resized, (display_size, display_size))

                cv2.imshow('Palm ROI', palm_roi_big)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
