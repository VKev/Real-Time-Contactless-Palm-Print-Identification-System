import cv2
import mediapipe as mp
import os
try:
    from util import extract_palm_roi
except ImportError:
    from .util import extract_palm_roi

class HandLandmarkVisualizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def visualize_landmarks(self, image_path, output_dir='visualize'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        roi = extract_palm_roi(image.copy())
        if roi is not None:
            roi_filename = os.path.join(output_dir, f"roi_{os.path.basename(image_path)}")
            cv2.imwrite(roi_filename, roi)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=8, circle_radius=8),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=8, circle_radius=8)
                )

        output_filename = os.path.join(output_dir, f"landmarks_{os.path.basename(image_path)}")
        cv2.imwrite(output_filename, image)
        return output_filename, roi_filename if roi is not None else None

    def __del__(self):
        self.hands.close()

def visualize_hand_landmarks(image_path, output_dir='visualize'):

    visualizer = HandLandmarkVisualizer()
    return visualizer.visualize_landmarks(image_path, output_dir)

if __name__ == "__main__":
    # Example usage with hardcoded input path
    image_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\RealisticSet\Roi\augmentations\aug_nobg_1_1_6.png"  # Replace with your image path
    try:
        landmark_path, roi_path = visualize_hand_landmarks(image_path)
        print(f"Landmark visualization saved to: {landmark_path}")
        if roi_path:
            print(f"ROI extraction saved to: {roi_path}")
        else:
            print("No ROI could be extracted")
    except Exception as e:
        print(f"Error processing image: {str(e)}") 