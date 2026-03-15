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
THUMB_IP_IDX = 3
THUMB_TIP_IDX = 4
INDEX_PIP_IDX = 6
INDEX_TIP_IDX = 8
MIDDLE_PIP_IDX = 10
MIDDLE_TIP_IDX = 12
RING_PIP_IDX = 14
RING_TIP_IDX = 16
PINKY_PIP_IDX = 18
PINKY_TIP_IDX = 20

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
_gesture_backend = None
_using_gesture_tasks_backend = False
_gesture_video_t0 = None
_last_gesture_video_ts_ms = 0
_gesture_backend_warned = False
_roi_smoother_lock = Lock()
_roi_smoother = None

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
_GESTURE_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
)
_GESTURE_TASK_MODEL_PATH = Path(__file__).resolve().parent / "models" / "gesture_recognizer.task"
_GESTURE_TASK_MODEL_ENV_VARS = (
    "MP_GESTURE_RECOGNIZER_MODEL",
    "MP_GESTURE_RECOGNIZER_MODEL_PATH",
    "MEDIAPIPE_GESTURE_RECOGNIZER_MODEL",
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
_ROI_SMOOTHING_ENABLED = _read_env_flag("ROI_SMOOTHING_ENABLED", True)
_ROI_SMOOTH_MIN_CUTOFF = float(os.getenv("ROI_SMOOTH_MIN_CUTOFF", "1.2"))
_ROI_SMOOTH_BETA = float(os.getenv("ROI_SMOOTH_BETA", "0.08"))
_ROI_SMOOTH_D_CUTOFF = float(os.getenv("ROI_SMOOTH_D_CUTOFF", "1.0"))
_ROI_SMOOTH_RESET_SEC = float(os.getenv("ROI_SMOOTH_RESET_SEC", "0.35"))
_ROI_REQUIRE_PALM_FACING = _read_env_flag("ROI_REQUIRE_PALM_FACING", True)
_ROI_PALM_FACING_X_TOL = float(os.getenv("ROI_PALM_FACING_X_TOL", "0.0"))
_ROI_REQUIRE_OPEN_PALM = _read_env_flag("ROI_REQUIRE_OPEN_PALM", True)
_ROI_OPEN_PALM_MIN_EXTENDED = int(os.getenv("ROI_OPEN_PALM_MIN_EXTENDED", "3"))
_ROI_OPEN_PALM_MIN_SPREAD = float(os.getenv("ROI_OPEN_PALM_MIN_SPREAD", "0.22"))
_ROI_OPEN_PALM_DIST_RATIO = float(os.getenv("ROI_OPEN_PALM_DIST_RATIO", "1.08"))
_ROI_USE_GESTURE_RECOGNIZER = _read_env_flag("ROI_USE_GESTURE_RECOGNIZER", True)


class _LowPassFilter:
    def __init__(self):
        self.initialized = False
        self.value = 0.0

    def reset(self):
        self.initialized = False
        self.value = 0.0

    def filter(self, sample, alpha):
        if not self.initialized:
            self.value = float(sample)
            self.initialized = True
            return self.value

        self.value = (alpha * float(sample)) + ((1.0 - alpha) * self.value)
        return self.value


class _OneEuroFilter:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = _LowPassFilter()
        self.dx_filter = _LowPassFilter()
        self.last_ts = None

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_ts = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + (tau / dt))

    def filter(self, sample, ts):
        sample = float(sample)
        if self.last_ts is None:
            self.last_ts = ts
            return self.x_filter.filter(sample, 1.0)

        dt = max(ts - self.last_ts, 1e-3)
        self.last_ts = ts

        prev = self.x_filter.value if self.x_filter.initialized else sample
        deriv = (sample - prev) / dt
        deriv_hat = self.dx_filter.filter(deriv, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + (self.beta * abs(deriv_hat))
        return self.x_filter.filter(sample, self._alpha(cutoff, dt))


def _unwrap_angle_deg(angle_deg, reference_deg):
    if reference_deg is None:
        return float(angle_deg)
    delta = (float(angle_deg) - reference_deg + 180.0) % 360.0 - 180.0
    return reference_deg + delta


class _RoiSmoother:
    def __init__(self):
        self.angle_filter = _OneEuroFilter(
            _ROI_SMOOTH_MIN_CUTOFF,
            _ROI_SMOOTH_BETA,
            _ROI_SMOOTH_D_CUTOFF,
        )
        self.cx_filter = _OneEuroFilter(
            _ROI_SMOOTH_MIN_CUTOFF,
            _ROI_SMOOTH_BETA,
            _ROI_SMOOTH_D_CUTOFF,
        )
        self.cy_filter = _OneEuroFilter(
            _ROI_SMOOTH_MIN_CUTOFF,
            _ROI_SMOOTH_BETA,
            _ROI_SMOOTH_D_CUTOFF,
        )
        self.size_filter = _OneEuroFilter(
            _ROI_SMOOTH_MIN_CUTOFF,
            _ROI_SMOOTH_BETA,
            _ROI_SMOOTH_D_CUTOFF,
        )
        self.last_seen_ts = None
        self.last_angle = None
        self.frame_shape = None
        self.handedness_label = None

    def reset(self):
        self.angle_filter.reset()
        self.cx_filter.reset()
        self.cy_filter.reset()
        self.size_filter.reset()
        self.last_seen_ts = None
        self.last_angle = None
        self.frame_shape = None
        self.handedness_label = None

    def _should_reset(self, ts, frame_shape, handedness_label):
        if self.last_seen_ts is None:
            return False
        if ts - self.last_seen_ts > _ROI_SMOOTH_RESET_SEC:
            return True
        if self.frame_shape != frame_shape:
            return True
        return self.handedness_label != handedness_label

    def _touch(self, ts, frame_shape, handedness_label):
        if self._should_reset(ts, frame_shape, handedness_label):
            self.reset()
        self.last_seen_ts = ts
        self.frame_shape = frame_shape
        self.handedness_label = handedness_label

    def smooth_rotation(self, rotation_deg, ts, frame_shape, handedness_label):
        self._touch(ts, frame_shape, handedness_label)
        unwrapped = _unwrap_angle_deg(rotation_deg, self.last_angle)
        smoothed = self.angle_filter.filter(unwrapped, ts)
        self.last_angle = smoothed
        return smoothed

    def smooth_roi_box(self, cx, cy, roi_size, ts, frame_shape, handedness_label):
        self._touch(ts, frame_shape, handedness_label)
        smoothed_cx = self.cx_filter.filter(cx, ts)
        smoothed_cy = self.cy_filter.filter(cy, ts)
        smoothed_size = self.size_filter.filter(roi_size, ts)
        return smoothed_cx, smoothed_cy, smoothed_size


def _get_roi_smoother():
    global _roi_smoother
    if _roi_smoother is None:
        _roi_smoother = _RoiSmoother()
    return _roi_smoother


def _reset_roi_smoother():
    global _roi_smoother
    with _roi_smoother_lock:
        if _roi_smoother is not None:
            _roi_smoother.reset()


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


def _resolve_gesture_task_model_path() -> str:
    for env_name in _GESTURE_TASK_MODEL_ENV_VARS:
        configured = os.getenv(env_name)
        if not configured:
            continue
        path = Path(configured).expanduser().resolve()
        if path.exists():
            return str(path)
        raise FileNotFoundError(
            f"Gesture model path set in {env_name} was not found: {path}"
        )

    _GESTURE_TASK_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    should_refresh = _REFRESH_LATEST_MODEL or (not _GESTURE_TASK_MODEL_PATH.exists())
    if should_refresh:
        tmp_path = _GESTURE_TASK_MODEL_PATH.with_suffix(".task.tmp")
        try:
            urllib.request.urlretrieve(_GESTURE_TASK_MODEL_URL, tmp_path)
            os.replace(tmp_path, _GESTURE_TASK_MODEL_PATH)
        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if not _GESTURE_TASK_MODEL_PATH.exists():
                raise RuntimeError(
                    "Could not download MediaPipe gesture recognizer model. "
                    f"URL: {_GESTURE_TASK_MODEL_URL}. "
                    f"Set one of {_GESTURE_TASK_MODEL_ENV_VARS} to a local .task model path."
                ) from exc

    return str(_GESTURE_TASK_MODEL_PATH)


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
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
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


def _init_gesture_backend():
    global _gesture_backend, _using_gesture_tasks_backend
    global _gesture_video_t0, _last_gesture_video_ts_ms, _gesture_backend_warned

    if not _ROI_USE_GESTURE_RECOGNIZER:
        return
    if _gesture_backend is not None:
        return

    with _backend_lock:
        if _gesture_backend is not None:
            return

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = _resolve_gesture_task_model_path()
            options = vision.GestureRecognizerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
            )
            _gesture_backend = vision.GestureRecognizer.create_from_options(options)
            _using_gesture_tasks_backend = True
            _gesture_video_t0 = time.monotonic()
            _last_gesture_video_ts_ms = 0
        except Exception as exc:
            _gesture_backend = None
            _using_gesture_tasks_backend = False
            if not _gesture_backend_warned:
                print(
                    "[WARN] Gesture recognizer backend unavailable. "
                    f"Falling back to landmark open-palm heuristic. Detail: {exc}"
                )
                _gesture_backend_warned = True


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


def _is_open_palm_gesture(rgb_frame):
    global _last_gesture_video_ts_ms

    if not _ROI_USE_GESTURE_RECOGNIZER:
        return None

    _init_gesture_backend()
    if _gesture_backend is None or not _using_gesture_tasks_backend:
        return None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    ts_ms = (
        int((time.monotonic() - _gesture_video_t0) * 1000)
        if _gesture_video_t0 is not None
        else 0
    )
    if ts_ms <= _last_gesture_video_ts_ms:
        ts_ms = _last_gesture_video_ts_ms + 1
    _last_gesture_video_ts_ms = ts_ms

    try:
        results = _gesture_backend.recognize_for_video(mp_image, ts_ms)
    except Exception:
        try:
            results = _gesture_backend.recognize(mp_image)
        except Exception:
            return None

    if not getattr(results, "gestures", None):
        return False
    if not results.gestures or not results.gestures[0]:
        return False

    top = results.gestures[0][0]
    name = (
        getattr(top, "category_name", None)
        or getattr(top, "display_name", None)
        or getattr(top, "label", None)
        or ""
    )
    return str(name).strip().lower() == "open_palm"


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


def _is_palm_facing_camera(landmarks, handedness_label) -> bool:
    """
    Heuristic in mirrored selfie space:
    - Right hand palm-facing: index MCP is right of pinky MCP.
    - Left hand palm-facing:  index MCP is left of pinky MCP.
    """
    label = str(handedness_label).strip().lower()
    if label not in {"left", "right"}:
        return True

    idx_x, _ = _landmark_xy(landmarks[INDEX_MCP_IDX], 1.0, 1.0)
    pky_x, _ = _landmark_xy(landmarks[PINKY_MCP_IDX], 1.0, 1.0)
    tol = _ROI_PALM_FACING_X_TOL

    if label == "right":
        return (idx_x - pky_x) > tol
    return (pky_x - idx_x) > tol


def _dist_sq(landmarks, a_idx: int, b_idx: int) -> float:
    ax, ay = _landmark_xy(landmarks[a_idx], 1.0, 1.0)
    bx, by = _landmark_xy(landmarks[b_idx], 1.0, 1.0)
    dx = ax - bx
    dy = ay - by
    return float(dx * dx + dy * dy)


def _is_open_palm_pose(landmarks) -> bool:
    wrist_idx = WRIST_IDX
    extended = 0

    finger_triplets = (
        (INDEX_TIP_IDX, INDEX_PIP_IDX, INDEX_MCP_IDX),
        (MIDDLE_TIP_IDX, MIDDLE_PIP_IDX, MIDDLE_MCP_IDX),
        (RING_TIP_IDX, RING_PIP_IDX, RING_MCP_IDX),
        (PINKY_TIP_IDX, PINKY_PIP_IDX, PINKY_MCP_IDX),
    )
    ratio = max(1.01, _ROI_OPEN_PALM_DIST_RATIO)
    for tip_idx, pip_idx, _mcp_idx in finger_triplets:
        tip_d = _dist_sq(landmarks, tip_idx, wrist_idx)
        pip_d = _dist_sq(landmarks, pip_idx, wrist_idx)
        if tip_d > (pip_d * ratio):
            extended += 1

    spread = np.sqrt(_dist_sq(landmarks, INDEX_TIP_IDX, PINKY_TIP_IDX))
    min_extended = max(1, _ROI_OPEN_PALM_MIN_EXTENDED)
    return extended >= min_extended and spread >= _ROI_OPEN_PALM_MIN_SPREAD


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
    return float(np.mean(xs)), float(np.mean(ys) + y_shift)


def _localize_roi(
    rotated_img,
    landmarks,
    y_shift,
    min_size,
    max_size,
    scale,
    handedness_label=None,
    use_smoothing=True,
):
    height, width = rotated_img.shape[:2]
    baseline = _calculate_baseline(landmarks, width, height)
    cx, cy = _calculate_palm_center(landmarks, width, height, y_shift)

    roi_size = float(np.clip(baseline * scale, min_size, max_size))
    if use_smoothing and _ROI_SMOOTHING_ENABLED:
        now_ts = time.monotonic()
        with _roi_smoother_lock:
            smoother = _get_roi_smoother()
            cx, cy, roi_size = smoother.smooth_roi_box(
                cx,
                cy,
                roi_size,
                now_ts,
                (height, width),
                handedness_label,
            )

    roi_size = int(np.clip(round(roi_size), min_size, max_size))
    cx = int(round(cx))
    cy = int(round(cy))
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
        if _ROI_SMOOTHING_ENABLED:
            _reset_roi_smoother()
        return None
    if _ROI_REQUIRE_PALM_FACING and not _is_palm_facing_camera(landmarks, handedness_label):
        if _ROI_SMOOTHING_ENABLED:
            _reset_roi_smoother()
        return None
    if _ROI_REQUIRE_OPEN_PALM:
        open_palm_ok = _is_open_palm_gesture(rgb)
        if open_palm_ok is None:
            open_palm_ok = _is_open_palm_pose(landmarks)
        if not open_palm_ok:
            if _ROI_SMOOTHING_ENABLED:
                _reset_roi_smoother()
            return None

    angle_deg = _calculate_hand_rotation(landmarks, width, height)
    applied_rotation_deg = angle_deg + _get_rotation_offset(handedness_label)
    if _ROI_SMOOTHING_ENABLED:
        now_ts = time.monotonic()
        with _roi_smoother_lock:
            smoother = _get_roi_smoother()
            applied_rotation_deg = smoother.smooth_rotation(
                applied_rotation_deg,
                now_ts,
                (height, width),
                handedness_label,
            )
    rotated_img, rotation_matrix = _rotate_image(
        mirrored,
        applied_rotation_deg,
    )
    rotated_landmarks = _rotate_landmarks(landmarks, rotation_matrix, width, height)
    roi = _localize_roi(
        rotated_img,
        rotated_landmarks,
        y_shift,
        min_size,
        max_size,
        scale,
        handedness_label=handedness_label,
        use_smoothing=_ROI_SMOOTHING_ENABLED,
    )
    if roi is None:
        return None

    return cv2.flip(roi, 1)
