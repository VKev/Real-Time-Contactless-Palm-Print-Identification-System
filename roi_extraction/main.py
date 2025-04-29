if __name__ == "__main__":
    import cv2
    import mediapipe as mp
    import numpy as np
    from collections import namedtuple

    Landmark = namedtuple("Landmark", ["x", "y", "z"])

    class HandLandmarks:
        def __init__(self, landmarks):
            self.landmark = landmarks

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    def calculate_palm_center(landmarks, image_shape):
        h, w, _ = image_shape
        indices = [mp_hands.HandLandmark.WRIST,
                mp_hands.HandLandmark.THUMB_CMC,
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_MCP,
                mp_hands.HandLandmark.PINKY_MCP]

        xs = [landmarks.landmark[i].x * w for i in indices]
        ys = [landmarks.landmark[i].y * h for i in indices]
        zs = [landmarks.landmark[i].z for i in indices]

        return int(np.mean(xs)), int(np.mean(ys)), np.mean(zs)

    def calculate_hand_rotation(landmarks, image_shape):
        h, w, _ = image_shape
        wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
        index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

        x1, y1 = wrist.x * w, wrist.y * h
        x2, y2 = index_mcp.x * w, index_mcp.y * h

        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        return np.degrees(angle_rad)

    def get_rotated_landmarks(landmarks, rot_matrix, image_shape):
        h, w, _ = image_shape
        rotated = []
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            point = np.array([x, y, 1])
            rx, ry = rot_matrix.dot(point)[:2]
            rotated.append(Landmark(rx / w, ry / h, lm.z))
        return HandLandmarks(rotated)

    def rotate_image(image, angle, offset_angle=120):
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle + offset_angle, 1.0)
        rotated_img = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
        return rotated_img, rot_matrix

    def extract_palm_roi(image, cx, cy, size=224):
        """
        Extracts a square ROI centered at (cx, cy), with side length = size.
        """
        half = size // 2
        h, w = image.shape[:2]
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        return image[y1:y2, x1:x2]

    def draw_depth_info(image, cx, cy, cz):
        cv2.circle(image, (cx, cy), 8, (255, 0, 0), -1)
        cv2.putText(image, f'Depth: {cz:.2f}', (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    def main():
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        # Make the palm ROI window resizable
        cv2.namedWindow('Palm ROI', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for idx, landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label
                    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    h, w, _ = frame.shape

                    # Draw the label on the original frame
                    cv2.putText(frame, f'{label} Hand',
                                (int(wrist.x * w), int(wrist.y * h)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # 1) DRAW THE LINE BETWEEN INDEX_FINGER_MCP AND PINKY_MCP
                    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                    x1, y1 = int(index_mcp.x * w), int(index_mcp.y * h)
                    x2, y2 = int(pinky_mcp.x * w), int(pinky_mcp.y * h)

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                    # 2) MEASURE THAT LINE'S LENGTH IN PIXELS
                    line_length = int(np.hypot(x2 - x1, y2 - y1))

                    angle = calculate_hand_rotation(landmarks, frame.shape)
                    rotated_frame, rot_matrix = rotate_image(frame, angle)
                    rotated_landmarks = get_rotated_landmarks(landmarks, rot_matrix, frame.shape)

                    # 3) GET THE CENTER OF THE PALM IN THE ROTATED FRAME
                    cx, cy, cz = calculate_palm_center(rotated_landmarks, rotated_frame.shape)
                    draw_depth_info(rotated_frame, cx, cy, cz)

                    # 4) DYNAMIC ROI SIZE BASED ON THE LINE LENGTH
                    #    You can tweak the formula below depending on how large you want the ROI to get.
                    #    Here, we use a minimum of 224 and a maximum of 600 as an example.
                    min_roi = 100
                    max_roi = 600
                    roi_size = int(np.clip(line_length * 1.1, min_roi, max_roi))

                    palm_roi = extract_palm_roi(rotated_frame, cx, cy, roi_size)
                    if palm_roi.size > 0:
                        # Resize the palm ROI to the roi_size for display
                        palm_roi_resized = cv2.resize(palm_roi, (roi_size, roi_size))

                        # Show and resize the "Palm ROI" window
                        cv2.imshow('Palm ROI', palm_roi_resized)
                        cv2.resizeWindow('Palm ROI', roi_size, roi_size)

                    cv2.imshow('Hand Tracking (Rotated)', rotated_frame)

            cv2.imshow('Hand Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    main()
