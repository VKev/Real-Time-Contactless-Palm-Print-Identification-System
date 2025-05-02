import cv2
import numpy as np
import mediapipe as mp
from collections import namedtuple

_LM = namedtuple("Landmark", ["x", "y", "z"])
_mp = mp.solutions.hands
_hands = _mp.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

_PALM_IDS = [
    _mp.HandLandmark.WRIST,
    _mp.HandLandmark.THUMB_CMC,
    _mp.HandLandmark.INDEX_FINGER_MCP,
    _mp.HandLandmark.MIDDLE_FINGER_MCP,
    _mp.HandLandmark.RING_FINGER_MCP,
    _mp.HandLandmark.PINKY_MCP,
]

def _calculate_baseline(lms, w, h):
    idx = lms.landmark[_mp.HandLandmark.INDEX_FINGER_MCP]
    pky = lms.landmark[_mp.HandLandmark.PINKY_MCP]
    x1, y1 = idx.x * w, idx.y * h
    x2, y2 = pky.x * w, pky.y * h
    return np.hypot(x2 - x1, y2 - y1)

def _calculate_hand_rotation(lms, w, h):
    wrist = lms.landmark[_mp.HandLandmark.WRIST]
    idx   = lms.landmark[_mp.HandLandmark.INDEX_FINGER_MCP]
    x1, y1 = wrist.x * w, wrist.y * h
    x2, y2 = idx.x   * w, idx.y   * h
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def _rotate_image(img, angle_deg, offset=120):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg + offset, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated, M

def _rotate_landmarks(lms, M, w, h):
    out = []
    for lm in lms.landmark:
        x, y = lm.x * w, lm.y * h  # Remove int() to preserve precision
        px, py = M.dot(np.array([x, y, 1.0]))[:2]
        out.append(_LM(px / w, py / h, lm.z))
    return out

def _calculate_palm_center(rot_lms, w, h):
    xs = [rot_lms[i].x * w for i in _PALM_IDS]
    ys = [rot_lms[i].y * h for i in _PALM_IDS]
    return int(np.mean(xs)), int(np.mean(ys))

def extract_palm_roi(frame_bgr, min_size=100, max_size=600, scale=1.1):
    frame_bgr = cv2.flip(frame_bgr, 1)
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None

    lms = res.multi_hand_landmarks[0]
    label = res.multi_handedness[0].classification[0].label
    baseline = _calculate_baseline(lms, w, h)
    angle    = _calculate_hand_rotation(lms, w, h)
    offset = 120
    if label == 'Left':
        offset = 120 - 70
    rot_img, M    = _rotate_image(frame_bgr, angle, offset)
    rot_lms       = _rotate_landmarks(lms, M, w, h)
    cx, cy        = _calculate_palm_center(rot_lms, w, h)

    roi_sz = int(np.clip(baseline * scale, min_size, max_size))
    half   = roi_sz // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    roi    = rot_img[y1:y2, x1:x2]

    if roi.size == 0:
        return None
    roi_resized = cv2.resize(roi, (roi_sz, roi_sz))
    
    return roi_resized