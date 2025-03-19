import mediapipe as mp
import cv2
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

image_path = '../../Dataset/Palm-Print/IITD Palmprint V1/Right Hand/001_1.JPG'
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Error: Unable to load image at {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

output_path = './test/output.jpg'

success = cv2.imwrite(output_path, image)
if success:
    print(f"Processed image saved successfully at {output_path}")
else:
    print("Error: Failed to save the processed image.")
