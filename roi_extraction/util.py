import os
import time
import urllib.request
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np

WRIST_IDX = 0
THUMB_CMC_IDX = 1
INDEX_MCP_IDX = 5
MIDDLE_MCP_IDX = 9
RING_MCP_IDX = 13
PINKY_MCP_IDX = 17

PALM_KEYPOINT_IDS = [
    WRIST_IDX,
    THUMB_CMC_IDX,
    INDEX_MCP_IDX,
    MIDDLE_MCP_IDX,
    RING_MCP_IDX,
    PINKY_MCP_IDX,
]

_backend_lock = Lock()
_hands_backend = None
_using_tasks_backend = False
_video_t0 = None
_last_video_ts_ms = 0

_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_TASK_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
_TASK_MODEL_ENV_VARS = (
    "MP_HAND_LANDMARKER_MODEL",
    "MP_HAND_LANDMARKER_MODEL_PATH",
    "MEDIAPIPE_HAND_LANDMARKER_MODEL",
)


def _read_env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


_ALLOW_LEGACY_SOLUTIONS = _read_env_flag("MP_ALLOW_LEGACY_SOLUTIONS", False)
_REFRESH_LATEST_MODEL = _read_env_flag("MP_REFRESH_LATEST_MODEL", True)
_REQUIRE_FULL_MODEL = _read_env_flag("MP_REQUIRE_FULL_MODEL", True)
_ROTATION_BIAS_DEG = float(os.getenv("ROI_ROTATION_BIAS_DEG", "70"))


def _resolve_task_model_path() -> str:
    for env_name in _TASK_MODEL_ENV_VARS:
        configured = os.getenv(env_name)
        if not configured:
            continue
        path = Path(configured).expanduser().resolve()
        if path.exists():
            if _REQUIRE_FULL_MODEL and "lite" in path.name.lower():
                raise RuntimeError(
                    "Configured MediaPipe model appears to be a lite variant. "
                    "For best accuracy use the full hand_landmarker.task."
                )
            return str(path)
        raise FileNotFoundError(
            f"Model path set in {env_name} was not found: {path}"
        )

    _TASK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    should_refresh = _REFRESH_LATEST_MODEL or (not _TASK_MODEL_PATH.exists())
    if should_refresh:
        tmp_path = _TASK_MODEL_PATH.with_suffix(".task.tmp")
        try:
            urllib.request.urlretrieve(_TASK_MODEL_URL, tmp_path)
            os.replace(tmp_path, _TASK_MODEL_PATH)
        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if not _TASK_MODEL_PATH.exists():
                raise RuntimeError(
                    "Could not download MediaPipe hand landmarker model. "
                    f"URL: {_TASK_MODEL_URL}. "
                    f"Set one of {_TASK_MODEL_ENV_VARS} to a local .task model path."
                ) from exc

    return str(_TASK_MODEL_PATH)


def _init_hands_backend():
    global _hands_backend, _using_tasks_backend, _video_t0, _last_video_ts_ms

    if _hands_backend is not None:
        return

    with _backend_lock:
        if _hands_backend is not None:
            return

        tasks_error = None
        try:
            # Preferred path: Tasks API + latest full model bundle.
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = _resolve_task_model_path()
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            _hands_backend = vision.HandLandmarker.create_from_options(options)
            _using_tasks_backend = True
            _video_t0 = time.monotonic()
            _last_video_ts_ms = 0
            return
        except Exception as exc:
            tasks_error = exc

        if not _ALLOW_LEGACY_SOLUTIONS:
            raise RuntimeError(
                "Failed to initialize MediaPipe Tasks HandLandmarker with the latest "
                "full model. Upgrade mediapipe or provide a valid .task model path. "
                "Set MP_ALLOW_LEGACY_SOLUTIONS=1 only if you explicitly want old API fallback."
            ) from tasks_error

        # Optional compatibility fallback for old mediapipe installs.
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            _hands_backend = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            _using_tasks_backend = False
            return

        raise RuntimeError(
            "MediaPipe hand backend initialization failed for both Tasks API and legacy solutions API."
        ) from tasks_error


def _extract_handedness_label(handedness_result) -> str:
    if not handedness_result:
        return "Right"

    first = handedness_result[0]
    category = None

    if isinstance(first, list) and first:
        category = first[0]
    elif hasattr(first, "categories") and first.categories:
        category = first.categories[0]
    elif hasattr(first, "classification") and first.classification:
        category = first.classification[0]

    if category is None:
        return "Right"

    return (
        getattr(category, "category_name", None)
        or getattr(category, "display_name", None)
        or getattr(category, "label", None)
        or "Right"
    )


def _detect_hand_landmarks(rgb_frame):
    global _last_video_ts_ms
    _init_hands_backend()

    if not _using_tasks_backend:
        results = _hands_backend.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None, None
        landmarks = results.multi_hand_landmarks[0].landmark
        label = _extract_handedness_label(results.multi_handedness)
        return landmarks, label

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    ts_ms = int((time.monotonic() - _video_t0) * 1000) if _video_t0 is not None else 0
    if ts_ms <= _last_video_ts_ms:
        ts_ms = _last_video_ts_ms + 1
    _last_video_ts_ms = ts_ms

    try:
        results = _hands_backend.detect_for_video(mp_image, ts_ms)
    except Exception:
        # Fallback if backend does not expose video mode as expected.
        results = _hands_backend.detect(mp_image)

    if not results.hand_landmarks:
        return None, None
    landmarks = results.hand_landmarks[0]
    label = _extract_handedness_label(results.handedness)
    return landmarks, label


def _calculate_baseline(landmarks, width, height):
    idx = landmarks[INDEX_MCP_IDX]
    pky = landmarks[PINKY_MCP_IDX]
    x1, y1 = _landmark_xy(idx, width, height)
    x2, y2 = _landmark_xy(pky, width, height)
    return np.hypot(x2 - x1, y2 - y1)


def _landmark_xy(landmark, width, height):
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        return landmark.x * width, landmark.y * height
    return landmark[0] * width, landmark[1] * height


def _calculate_hand_rotation(landmarks, width, height):
    """
    Match the old main-branch logic: measure the direction from wrist to
    index_mcp, then use a handedness-dependent offset during frame rotation.
    """
    wrist = landmarks[WRIST_IDX]
    index_mcp = landmarks[INDEX_MCP_IDX]

    x1, y1 = _landmark_xy(wrist, width, height)
    x2, y2 = _landmark_xy(index_mcp, width, height)

    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def _get_rotation_offset(handedness_label: str) -> float:
    offset = -10.0
    if str(handedness_label).strip().lower() == "left":
        offset = 50.0
    return offset + _ROTATION_BIAS_DEG


def _rotate_image(img, angle_deg, offset_deg=0.0, center_xy=None):
    height, width = img.shape[:2]
    if center_xy is None:
        cx, cy = width / 2.0, height / 2.0
    else:
        cx, cy = center_xy
    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg + offset_deg, 1.0)
    rotated = cv2.warpAffine(
        img,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return rotated, matrix


def _rotate_landmarks(landmarks, matrix, width, height):
    rotated = []
    for lm in landmarks:
        x, y = _landmark_xy(lm, width, height)
        z = lm.z if hasattr(lm, "z") else lm[2]
        px, py = matrix.dot(np.array([x, y, 1.0]))[:2]
        rotated.append((px / width, py / height, z))
    return rotated


def _calculate_palm_center(rotated_landmarks, width, height, y_shift):
    xs = [_landmark_xy(rotated_landmarks[i], width, height)[0] for i in PALM_KEYPOINT_IDS]
    ys = [_landmark_xy(rotated_landmarks[i], width, height)[1] for i in PALM_KEYPOINT_IDS]
    return int(np.mean(xs)), int(np.mean(ys) + y_shift)


def _localize_roi(rotated_img, landmarks, y_shift, min_size, max_size, scale):
    height, width = rotated_img.shape[:2]
    baseline = _calculate_baseline(landmarks, width, height)
    cx, cy = _calculate_palm_center(landmarks, width, height, y_shift)

    roi_size = int(np.clip(baseline * scale, min_size, max_size))
    half = roi_size // 2

    pad = half + 8
    padded = cv2.copyMakeBorder(
        rotated_img,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_REFLECT,
    )

    cx_p, cy_p = cx + pad, cy + pad
    x1, y1 = int(cx_p - half), int(cy_p - half)
    x2, y2 = int(cx_p + half), int(cy_p + half)
    roi = padded[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)


def extract_palm_roi(frame_bgr, min_size=120, max_size=700, scale=1, y_shift=40):
    """
    Extract palm ROI from a BGR frame using MediaPipe hand landmarks.
    Returns a square BGR ROI image or None if no hand is detected.
    """
    mirrored = cv2.flip(frame_bgr, 1)
    height, width = mirrored.shape[:2]
    rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)

    landmarks, handedness_label = _detect_hand_landmarks(rgb)
    if landmarks is None:
        return None

    angle_deg = _calculate_hand_rotation(landmarks, width, height)
    offset_deg = _get_rotation_offset(handedness_label)
    rotated_img, rotation_matrix = _rotate_image(
        mirrored,
        angle_deg,
        offset_deg=offset_deg,
    )
    rotated_landmarks = _rotate_landmarks(landmarks, rotation_matrix, width, height)
    roi = _localize_roi(
        rotated_img,
        rotated_landmarks,
        y_shift,
        min_size,
        max_size,
        scale,
    )
    if roi is None:
        return None

    return cv2.flip(roi, 1)
