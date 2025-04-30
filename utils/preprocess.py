import numpy as np
import cv2

def preprocess(roi: np.ndarray,
               target_size: int = 224,
               clahe_clip: float = 2.0,
               clahe_grid: tuple = (4, 4)
              ) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)
    
    preprocessed = cv2.resize(gray, (target_size, target_size),
                              interpolation=cv2.INTER_AREA)
    
    return preprocessed