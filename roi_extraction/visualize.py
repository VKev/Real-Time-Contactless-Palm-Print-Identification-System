import cv2
import mediapipe as mp
import os
import sys
import numpy as np
import torch

# Add the ROOT directory to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from depth_estimation.depth_anything_v2.dpt import DepthAnythingV2
from depth_estimation.utils import apply_depth_mask, image2tensor, interpolate
from utils.preprocess import preprocess

try:
    from util import extract_palm_roi, _PALM_IDS, _mp, _calculate_baseline, _calculate_hand_rotation, _calculate_palm_center, _rotate_image
except ImportError:
    from .util import extract_palm_roi, _PALM_IDS, _mp, _calculate_baseline, _calculate_hand_rotation, _calculate_palm_center, _rotate_image

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class HandLandmarkVisualizer:
    def __init__(self):
        # Initialize mediapipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # Initialize depth model for background removal
        encoder = 'vits'
        self.depth_model = DepthAnythingV2(**model_configs[encoder])
        state_dict = torch.load(os.path.join(ROOT, 'depth_estimation', 'checkpoints', f'depth_anything_v2_{encoder}.pth'), map_location="cpu")
        self.depth_model.load_state_dict(state_dict)
        self.depth_model = self.depth_model.to(DEVICE).eval()

    def remove_background(self, img, threshold=0.27):
        # img: BGR np.ndarray
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor, (h, w) = image2tensor(rgb_img, input_size=518)
        depth = self.depth_model.forward(tensor).detach()
        depth = interpolate(depth, h, w)
        print(depth.min(), depth.max())
        depth = (depth - 0) / (10 - 0) * 255.0
        depth = depth.astype(np.uint8)
        masked_pil = apply_depth_mask(rgb_img, depth, threshold=threshold)
        masked_np = np.array(masked_pil)
        # Convert back to BGR for downstream processing
        return cv2.cvtColor(masked_np, cv2.COLOR_RGB2BGR), depth

    def visualize_roi_preprocessing(self, roi, output_dir='visualize'):
        """
        Visualize the ROI preprocessing steps:
        1. Original ROI
        2. Grayscale version
        3. CLAHE enhanced version
        4. Final preprocessed version
        
        Args:
            roi: Input ROI image
            output_dir: Directory to save output images
        
        Returns:
            Tuple of paths to the visualization images
        """
        if roi is None:
            return None, None, None, None
        
        # Create a copy of the ROI for visualization
        vis_roi = roi.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Get preprocessed version
        preprocessed, _ = preprocess(roi)
        # Convert preprocessed back to uint8 for visualization
        preprocessed_vis = (preprocessed[0, 0] * 255).astype(np.uint8)
        
        # Save all versions
        roi_filename = os.path.join(output_dir, "roi_original.jpg")
        gray_filename = os.path.join(output_dir, "roi_gray.jpg")
        enhanced_filename = os.path.join(output_dir, "roi_enhanced.jpg")
        preprocessed_filename = os.path.join(output_dir, "roi_preprocessed.jpg")
        
        cv2.imwrite(roi_filename, vis_roi)
        cv2.imwrite(gray_filename, gray)
        cv2.imwrite(enhanced_filename, enhanced)
        cv2.imwrite(preprocessed_filename, preprocessed_vis)
        
        return roi_filename, gray_filename, enhanced_filename, preprocessed_filename

    def visualize_landmarks(self, image_path, output_dir='visualize'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Extract ROI
        roi = extract_palm_roi(image.copy())
        roi_filename = None
        if roi is not None:
            roi_filename = os.path.join(output_dir, f"roi_{os.path.basename(image_path)}")
            cv2.imwrite(roi_filename, roi)
            
            # Add ROI preprocessing visualization
            roi_orig, roi_gray, roi_enhanced, roi_preprocessed = self.visualize_roi_preprocessing(roi, output_dir)
            print(f"ROI preprocessing visualizations saved:")
            print(f"Original ROI: {roi_orig}")
            print(f"Grayscale ROI: {roi_gray}")
            print(f"Enhanced ROI: {roi_enhanced}")
            print(f"Preprocessed ROI: {roi_preprocessed}")

        # Background removal
        no_bg_image, depth_map = self.remove_background(image.copy())
        no_bg_filename = os.path.join(output_dir, f"nobg_{os.path.basename(image_path)}")
        cv2.imwrite(no_bg_filename, no_bg_image)
        
        # Save depth map for visualization
        depth_filename = os.path.join(output_dir, f"depth_{os.path.basename(image_path)}")
        cv2.imwrite(depth_filename, depth_map)
        
        # Extract ROI from background-removed image
        roi_nobg = extract_palm_roi(no_bg_image.copy())
        roi_nobg_filename = None
        if roi_nobg is not None:
            roi_nobg_filename = os.path.join(output_dir, f"roi_nobg_{os.path.basename(image_path)}")
            cv2.imwrite(roi_nobg_filename, roi_nobg)

        # Enhanced visualization with palm center, ROI region, and landmarks
        # Only create visualization with y_shift=10 (default in util.py)
        enhanced_visualization = self.visualize_palm_features(image.copy())
        enhanced_filename = os.path.join(output_dir, f"enhanced_{os.path.basename(image_path)}")
        cv2.imwrite(enhanced_filename, enhanced_visualization)

        # Hand landmarks visualization
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
        
        return output_filename, roi_filename, no_bg_filename, depth_filename, roi_nobg_filename, enhanced_filename

    def visualize_palm_features(self, image, y_shift=50):
        """
        Visualize the palm center, ROI region, baseline, and landmarks used for palm center calculation
        
        Args:
            image: Input image
            y_shift: Number of pixels to shift the palm center down
        """
        # Flip image for hand detection
        flipped_image = cv2.flip(image.copy(), 1)
        h, w = flipped_image.shape[:2]
        
        # Process image to get hand landmarks
        image_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
        results = _mp.Hands().process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return image
            
        # Get the first hand landmarks
        lms = results.multi_hand_landmarks[0]
        label = results.multi_handedness[0].classification[0].label
        
        # Calculate baseline and angle
        baseline = _calculate_baseline(lms, w, h)
        angle = _calculate_hand_rotation(lms, w, h)
        
        # Calculate offset based on hand label
        offset = 120
        if label == 'Left':
            offset = 120 - 70
            
        # Create rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle + offset, 1.0)
        
        # Rotate landmarks
        rot_lms = []
        for lm in lms.landmark:
            x, y = lm.x * w, lm.y * h
            px, py = M.dot(np.array([x, y, 1.0]))[:2]
            rot_lms.append(type('obj', (object,), {'x': px / w, 'y': py / h, 'z': lm.z}))
        
        # Calculate palm center with y_shift
        palm_cx, palm_cy = _calculate_palm_center(rot_lms, w, h, y_shift)
        
        # Calculate original palm center (for comparison)
        orig_cx, orig_cy = _calculate_palm_center(rot_lms, w, h, 0)
        
        # Calculate ROI size and corners with shifted palm center
        roi_sz = int(np.clip(baseline * 1.1, 100, 600))
        half = roi_sz // 2
        x1, y1 = max(0, palm_cx - half), max(0, palm_cy - half)
        x2, y2 = min(w, palm_cx + half), min(h, palm_cy + half)
        
        # Create a copy of the flipped image for visualization
        vis_image = flipped_image.copy()
        
        # Draw baseline (line between index finger MCP and pinky MCP)
        idx = lms.landmark[_mp.HandLandmark.INDEX_FINGER_MCP]
        pky = lms.landmark[_mp.HandLandmark.PINKY_MCP]
        idx_x, idx_y = int(idx.x * w), int(idx.y * h)
        pky_x, pky_y = int(pky.x * w), int(pky.y * h)
        cv2.line(vis_image, (idx_x, idx_y), (pky_x, pky_y), (0, 255, 255), 2)  # Yellow line for baseline
        
        # Draw original landmarks
        for id_point in _PALM_IDS:
            lm = lms.landmark[id_point]
            cx, cy = int(lm.x * w), int(lm.y * h)
            # Draw landmark points used for palm center in red
            cv2.circle(vis_image, (cx, cy), 8, (0, 0, 255), -1)
        
        # Add labels for specific landmarks
        landmark_labels = {
            _mp.HandLandmark.WRIST: "WRIST",
            _mp.HandLandmark.THUMB_CMC: "THUMB CMC",
            _mp.HandLandmark.INDEX_FINGER_MCP: "INDEX FINGER MCP",
            _mp.HandLandmark.MIDDLE_FINGER_MCP: "MIDDLE FINGER MCP",
            _mp.HandLandmark.RING_FINGER_MCP: "RING FINGER MCP",
            _mp.HandLandmark.PINKY_MCP: "PINKY MCP"
        }
        
        # We'll only add the labels to the flipped image later
        
        # If y_shift is used, show both the original and shifted centers
        if y_shift > 0:
            # Draw original palm center (smaller circle)
            cv2.circle(vis_image, (orig_cx, orig_cy), 8, (255, 255, 0), -1)  # Yellow for original
            # Draw arrow from original to shifted center
            cv2.arrowedLine(vis_image, (orig_cx, orig_cy), (palm_cx, palm_cy), (255, 255, 0), 2)
        
        # Draw rotated palm center (with shift if applied)
        cv2.circle(vis_image, (palm_cx, palm_cy), 12, (255, 0, 255), -1)  # Magenta for palm center
        
        # Draw ROI region with the shifted center
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle for ROI
        
        # Flip back for consistent visualization
        vis_image = cv2.flip(vis_image, 1)
        
        # Add text labels on the flipped image
        flipped_w = vis_image.shape[1]
        
        # New coordinates after flipping
        flipped_palm_cx = flipped_w - palm_cx
        flipped_palm_cy = palm_cy
        flipped_x1 = flipped_w - x2
        flipped_y1 = y1
        flipped_idx_x = flipped_w - idx_x
        flipped_idx_y = idx_y
        flipped_pky_x = flipped_w - pky_x
        flipped_pky_y = pky_y
        flipped_orig_cx = flipped_w - orig_cx
        flipped_orig_cy = orig_cy
        
        # Add flipped labels for the specific landmarks
        for lm_id, label_text in landmark_labels.items():
            lm = lms.landmark[lm_id]
            cx, cy = int(lm.x * w), int(lm.y * h)
            flipped_cx = flipped_w - cx
            flipped_cy = cy
            # Draw circle with different color for specific landmarks
            cv2.circle(vis_image, (flipped_cx, flipped_cy), 10, (0, 165, 255), -1)
            # Add text label slightly offset from the landmark (move higher up and to the left)
            cv2.putText(vis_image, label_text, (flipped_cx - 40, flipped_cy - 25), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        cv2.putText(vis_image, "PALM CENTER", (flipped_palm_cx - 80, flipped_palm_cy + 30), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(vis_image, "BASELINE", (min(flipped_idx_x, flipped_pky_x) + abs(flipped_idx_x - flipped_pky_x)//2 - 50, 
                                         min(flipped_idx_y, flipped_pky_y) - 10), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis_image, "ROI", (flipped_x1 + 10, flipped_y1 - 10), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add y_shift text if applicable
        if y_shift > 0:
            cv2.putText(vis_image, "DELTA Y", (flipped_orig_cx - 90, flipped_orig_cy - 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return vis_image

    def __del__(self):
        self.hands.close()

def visualize_hand_landmarks(image_path, output_dir='visualize'):
    visualizer = HandLandmarkVisualizer()
    return visualizer.visualize_landmarks(image_path, output_dir)

def visualize_rotation_correction(image_path, output_dir='visualize'):
    """
    Demonstrates the image rotation process:
    1. Rotates an input image by 40 degrees
    2. Uses _rotate_image from util to rotate it back
    3. Saves both images for comparison
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images
    
    Returns:
        Tuple of paths to the rotated and corrected images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # Get image dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotate the image by 40 degrees to simulate a tilted hand
    rotation_angle = -40
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    # Save the rotated image
    rotated_filename = os.path.join(output_dir, f"rotated_{os.path.basename(image_path)}")
    cv2.imwrite(rotated_filename, rotated_image)
    
    # Create a copy for visualizing landmarks and angle on rotated image
    rotated_vis = rotated_image.copy()
    
    # Process the rotated image to detect hand landmarks
    rotated_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
    hands = _mp.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    results = hands.process(rotated_rgb)
    
    # Variables to store angle information
    calculated_angle = None
    
    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0]
        
        # Draw all landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            rotated_vis,
            lms,
            _mp.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
        
        # Highlight the landmarks used for angle calculation (wrist and index finger MCP)
        wrist = lms.landmark[_mp.HandLandmark.WRIST]
        index_mcp = lms.landmark[_mp.HandLandmark.INDEX_FINGER_MCP]
        
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        index_x, index_y = int(index_mcp.x * w), int(index_mcp.y * h)
        
        # Draw larger circles at these key landmarks
        cv2.circle(rotated_vis, (wrist_x, wrist_y), 10, (0, 128, 255), -1)  # Orange for wrist
        cv2.circle(rotated_vis, (index_x, index_y), 10, (255, 255, 0), -1)  # Yellow for index MCP (same as line)
        
        # Draw a line between these landmarks
        cv2.line(rotated_vis, (wrist_x, wrist_y), (index_x, index_y), (255, 255, 0), 2)
        
        # Calculate the angle using the same method as in util._calculate_hand_rotation
        calculated_angle = _calculate_hand_rotation(lms, w, h)
        
        # Draw horizontal reference line from wrist
        horizontal_end_x = wrist_x + max(w - wrist_x, 500)  # Extend to edge of image or at least 500px
        cv2.line(rotated_vis, (wrist_x, wrist_y), (horizontal_end_x, wrist_y), (0, 255, 255), 2)  # Yellow horizontal line
        
        # Label the horizontal line
        cv2.putText(rotated_vis, "HORIZONTAL LINE", (wrist_x + 120, wrist_y - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw arc to visualize the angle
        radius = 120  # Increased radius of the arc in pixels
        start_angle = 0  # Horizontal line is 0 degrees
        end_angle = calculated_angle  # Angle to the index finger
        
        # Convert angle to correct format for cv2.ellipse (which uses 0 degrees as pointing right)
        # If angle is negative, adjust it for proper visualization
        if end_angle < 0:
            end_angle = 360 + end_angle
        
        # Draw the arc - changed to red
        cv2.ellipse(rotated_vis, (wrist_x, wrist_y), (radius, radius), 
                  0, start_angle, end_angle, (0, 0, 255), 2)
        
        # Add small text with the angle value - now says "Before" and is positioned below
        cv2.putText(rotated_vis, "BEFORE", 
                  (wrist_x + int(radius/2) + 10, wrist_y + radius + 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(rotated_vis, "WRIST", (wrist_x-20, wrist_y + 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)  # Orange for wrist
        cv2.putText(rotated_vis, "INDEX MCP", (index_x + 10, index_y - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Yellow for index
    
    # Save the visualized rotated image
    rotated_vis_filename = os.path.join(output_dir, f"rotated_vis_{os.path.basename(image_path)}")
    cv2.imwrite(rotated_vis_filename, rotated_vis)
    
    # Use _rotate_image from util to rotate it back
    corrected_image, _ = _rotate_image(rotated_image, -rotation_angle, offset=0)
    
    # Create a copy for visualizing result after correction
    corrected_vis = corrected_image.copy()
    
    # Process the corrected image to detect hand landmarks
    corrected_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
    results_corrected = hands.process(corrected_rgb)
    
    if results_corrected.multi_hand_landmarks:
        lms = results_corrected.multi_hand_landmarks[0]
        
        # Draw all landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            corrected_vis,
            lms,
            _mp.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
        
        # Highlight the landmarks used for angle calculation
        wrist = lms.landmark[_mp.HandLandmark.WRIST]
        index_mcp = lms.landmark[_mp.HandLandmark.INDEX_FINGER_MCP]
        
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        index_x, index_y = int(index_mcp.x * w), int(index_mcp.y * h)
        
        # Draw larger circles at these key landmarks
        cv2.circle(corrected_vis, (wrist_x, wrist_y), 10, (0, 128, 255), -1)  # Orange for wrist
        cv2.circle(corrected_vis, (index_x, index_y), 10, (255, 255, 0), -1)  # Yellow for index MCP
        
        # Draw a line between these landmarks
        cv2.line(corrected_vis, (wrist_x, wrist_y), (index_x, index_y), (255, 255, 0), 2)
        
        # Calculate the angle after correction
        corrected_angle = _calculate_hand_rotation(lms, w, h)
        
        # Draw horizontal reference line from wrist
        horizontal_end_x = wrist_x + max(w - wrist_x, 500)  # Extend to edge of image or at least 500px
        cv2.line(corrected_vis, (wrist_x, wrist_y), (horizontal_end_x, wrist_y), (0, 255, 255), 2)  # Yellow horizontal line
        
        # Label the horizontal line
        cv2.putText(corrected_vis, "HORIZONTAL LINE", (wrist_x + 120, wrist_y - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw arc to visualize the angle
        radius = 120  # Increased radius of the arc in pixels
        start_angle = 0  # Horizontal line is 0 degrees
        end_angle = corrected_angle  # Angle to the index finger
        
        # Convert angle to correct format for cv2.ellipse
        if end_angle < 0:
            end_angle = 360 + end_angle
        
        # Draw the arc
        cv2.ellipse(corrected_vis, (wrist_x, wrist_y), (radius, radius), 
                  0, start_angle, end_angle, (0, 255, 0), 2)
        
        # Draw the before angle with a smaller arc in different color
        if calculated_angle is not None:
            before_end_angle = calculated_angle
            if before_end_angle < 0:
                before_end_angle = 360 + before_end_angle
                
            cv2.ellipse(corrected_vis, (wrist_x, wrist_y), (radius - 60, radius - 60), 
                      0, start_angle, before_end_angle, (0, 0, 255), 2)
                      
            # Calculate end point of the Before arc to connect to wrist
            before_radius = radius - 60  # Same as the radius used for the before arc
            end_angle_rad = np.radians(before_end_angle)
            end_x = int(wrist_x + before_radius * np.cos(end_angle_rad))
            end_y = int(wrist_y + before_radius * np.sin(end_angle_rad))
            
            # Draw line from wrist to end of before arc
            cv2.line(corrected_vis, (wrist_x, wrist_y), (end_x, end_y), (0, 0, 255), 2)
        
        # Add small text labels for the arcs
        cv2.putText(corrected_vis, "BEFORE", 
                  (wrist_x - 150, wrist_y - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                      
        cv2.putText(corrected_vis, "AFTER", 
                  (wrist_x - 60, wrist_y + radius + 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add labels
        cv2.putText(corrected_vis, "WRIST", (wrist_x - 20, wrist_y +40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)  # Orange for wrist
        cv2.putText(corrected_vis, "INDEX MCP", (index_x + 10, index_y - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Yellow for index
    
    # Close the hand detector
    hands.close()
    
    # Save the visualized corrected image
    corrected_vis_filename = os.path.join(output_dir, f"corrected_vis_{os.path.basename(image_path)}")
    cv2.imwrite(corrected_vis_filename, corrected_vis)
    
    # Save the raw corrected image
    corrected_filename = os.path.join(output_dir, f"corrected_{os.path.basename(image_path)}")
    cv2.imwrite(corrected_filename, corrected_image)
    
    return rotated_vis_filename, corrected_vis_filename

if __name__ == "__main__":
    # Example usage with hardcoded input path
    image_path = r"C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\RealisticSet\Raw\video_frames_14\frame_1.jpg"  # Replace with your image path 002_F_R_9.JPG 002_S_L_27.JPG
    try:
        # Regular palm visualization
        landmark_path, roi_path, no_bg_path, depth_path, roi_nobg_path, enhanced_path = visualize_hand_landmarks(image_path)
        print(f"Landmark visualization saved to: {landmark_path}")
        if roi_path:
            print(f"ROI extraction saved to: {roi_path}")
        else:
            print("No ROI could be extracted")
        print(f"Background removed image saved to: {no_bg_path}")
        print(f"Depth map saved to: {depth_path}")
        if roi_nobg_path:
            print(f"ROI from background-removed image saved to: {roi_nobg_path}")
        else:
            print("No ROI could be extracted from background-removed image")
        print(f"Enhanced visualization saved to: {enhanced_path}")
        
        # Rotation visualization
        rotated_vis_path, corrected_vis_path = visualize_rotation_correction(image_path)
        print("\nRotation visualization results:")
        print(f"Rotated image with landmarks saved to: {rotated_vis_path}")
        print(f"Corrected image with landmarks saved to: {corrected_vis_path}")
        
        
    except Exception as e:
        print(f"Error processing image: {str(e)}") 