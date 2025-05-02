from pathlib import Path
import cv2, mediapipe as mp
from mediapipe.tasks.python import vision

class OpenPalmDetector:
    """Detects an ‘Open_Palm’ gesture with MediaPipe Tasks."""

    def __init__(self, model_path: str, threshold: float = 0.3):
        task_file = Path(model_path).resolve()
        if not task_file.is_file():
            raise FileNotFoundError(task_file)

        # --- safest: load as bytes until you’re sure you’re on >=0.10.14 ---
        with task_file.open('rb') as f:
            model_buf = f.read()

        base_opts = mp.tasks.BaseOptions(model_asset_buffer=model_buf)
        options   = vision.GestureRecognizerOptions(base_options=base_opts)
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._thr = threshold

    def is_open_palm(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._recognizer.recognize(mp.Image(mp.ImageFormat.SRGB, rgb))
        if not result.gestures:
            return False
        top = result.gestures[0][0]
        return top.category_name == "Open_Palm" and top.score >= self._thr

    def close(self):
        self._recognizer.close()
