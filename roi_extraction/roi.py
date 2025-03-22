import cv2
import mediapipe as mp
import numpy as np
from collections import namedtuple

# Define a simple Landmark structure with x, y, and z attributes.
Landmark = namedtuple("Landmark", ["x", "y", "z"])

# Define a container class that mimics the MediaPipe hand_landmarks format.
class HandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

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
        mp_hands.HandLandmark.WRIST,          # 0
        mp_hands.HandLandmark.THUMB_CMC,        # 1
        mp_hands.HandLandmark.INDEX_FINGER_MCP, # 5
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,# 9
        mp_hands.HandLandmark.RING_FINGER_MCP,  # 13
        mp_hands.HandLandmark.PINKY_MCP         # 17
    ]
    # Calculate the average x and y coordinates of the palm landmarks in pixel values
    cx = int(np.mean([hand_landmarks.landmark[l].x * w for l in palm_landmarks]))
    cy = int(np.mean([hand_landmarks.landmark[l].y * h for l in palm_landmarks]))
    # Calculate the average depth (z) from the palm landmarks (remains normalized)
    cz = np.mean([hand_landmarks.landmark[l].z for l in palm_landmarks])
    return cx, cy, cz

def calculate_hand_rotation(hand_landmarks, image_shape):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    h, w, _ = image_shape
    
    # Convert the coordinates from normalized (0-1) to pixel values
    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
    index_finger_x, index_finger_y = int(index_finger.x * w), int(index_finger.y * h)
    
    # Calculate the angle between the wrist and the index finger using atan2
    angle = np.arctan2(index_finger_y - wrist_y, index_finger_x - wrist_x)
    
    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle)
    
    return angle_deg

def get_rotated_landmarks(hand_landmarks, rotation_matrix, image_shape):
    h, w, _ = image_shape
    rotated_landmarks = []
    for lm in hand_landmarks.landmark:
        # Convert normalized coordinates to pixel values.
        x = int(lm.x * w)
        y = int(lm.y * h)
        # Create a homogeneous coordinate [x, y, 1] for transformation.
        point = np.array([x, y, 1])
        rotated_point = rotation_matrix.dot(point)
        # Convert rotated pixel coordinates back to normalized values.
        new_x = rotated_point[0] / w
        new_y = rotated_point[1] / h
        # Preserve the original depth (z) since 2D rotation doesn't affect depth.
        rotated_landmarks.append(Landmark(new_x, new_y, lm.z))
    return HandLandmarks(rotated_landmarks)

def rotate_image(image, angle):
    # Get the image center
    center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # Get the rotation matrix (using angle+115 as in your original code)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle + 115, 1.0)
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    return rotated_image, rotation_matrix

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip for a mirror-like effect
    frame = cv2.flip(frame, 1)
    
    # Process with MediaPipe Hands
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks on the original frame
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            handedness = results.multi_handedness[idx].classification[0].label
            h, w, _ = frame.shape
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_coords = (int(wrist.x * w), int(wrist.y * h))
            cv2.putText(frame, f'{handedness} Hand', wrist_coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Calculate the hand rotation angle
            hand_rotation = calculate_hand_rotation(hand_landmarks, frame.shape)
            print(f"Hand rotation angle: {hand_rotation:.2f} degrees")

            # Rotate the frame and get the rotation matrix
            rotated_frame, rotation_matrix = rotate_image(frame, hand_rotation)

            # Get rotated landmarks using the same rotation matrix
            rotated_landmarks = get_rotated_landmarks(hand_landmarks, rotation_matrix, frame.shape)
            
            # (Optional) Draw the rotated landmarks on the rotated frame (convert normalized back to pixels)
            h_r, w_r, _ = rotated_frame.shape
            # for lm in rotated_landmarks.landmark:
            #     x = int(lm.x * w_r)
            #     y = int(lm.y * h_r)
            #     cv2.circle(rotated_frame, (x, y), 6, (0, 255, 0), -1)
            
            # Calculate the palm center (and depth) from rotated landmarks
            cx, cy, cz = calculate_palm_center(rotated_landmarks, rotated_frame.shape)
            
            # Draw a circle at the palm center
            cv2.circle(rotated_frame, (cx, cy), 8, (255, 0, 0), -1)
            # Print the depth (z value) near the palm center on the rotated frame
            cv2.putText(rotated_frame, f'Depth: {cz:.2f}', (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Optionally, extract a region of interest (ROI) around the palm center
            roi_size = 100
            half_size = roi_size // 2
            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(w_r, cx + half_size)
            y2 = min(h_r, cy + half_size)
            palm_roi = rotated_frame[y1:y2, x1:x2]
            if palm_roi.size > 0:
                palm_roi_resized = cv2.resize(palm_roi, (roi_size, roi_size))
                display_size = 300
                palm_roi_big = cv2.resize(palm_roi_resized, (display_size, display_size))
                cv2.imshow('Palm ROI', palm_roi_big)

            # Show the rotated frame with landmarks and depth info
            cv2.imshow('Hand Tracking (Rotated)', rotated_frame)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
