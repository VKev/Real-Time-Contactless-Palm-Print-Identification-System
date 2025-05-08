import cv2
import numpy as np



def preprocess(
    roi: np.ndarray,
    target_size: int = 224,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int,int] = (4, 4),
    mean: float = 0.5,
    std: float = 0.5
) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    resized = cv2.resize(enhanced,
                         (target_size, target_size),
                         interpolation=cv2.INTER_AREA)

    img = resized.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.repeat(img[np.newaxis, ...], 3, axis=0)
    batch = img[np.newaxis, ...]

    return batch, resized
