import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from typing import Union
import numpy as np
from PIL import Image
import time
try:
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
except ImportError:
    from depth_estimation.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    

def apply_depth_mask(image: np.ndarray, depth_map: np.ndarray, threshold: float = 0.5) -> Image.Image:
    if depth_map.max() <= 1.0:
        depth_map = (depth_map * 255).astype(np.uint8)
    else:
        depth_map = depth_map.astype(np.uint8)

    threshold_value = int(threshold * 255)
    mask = depth_map >= threshold_value

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    return Image.fromarray(masked_image)


def apply_depth_mask_np(image: np.ndarray, depth_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if depth_map.dtype == np.uint8:
        thr = int(threshold * 255)
    else:
        thr = threshold

    out = np.empty_like(image)

    if image.ndim == depth_map.ndim + 1:
        np.multiply(image, depth_map[..., None] > thr, out=out)
    else:
        np.multiply(image, depth_map > thr, out=out)

    return out


_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INV255 = 1.0 / 255.0

def resize(raw_image, input_size=518):   
    h, w = raw_image.shape[:2]

    resized = cv2.resize(
        raw_image,
        (input_size, input_size),
        interpolation=cv2.INTER_CUBIC
    )

    img = resized[..., ::-1].astype(np.float32) * _INV255
    img = (img - _MEAN) / _STD

    img = img.transpose(2, 0, 1)
    image = torch.from_numpy(img).unsqueeze(0)

    return image, (h, w)


def image2tensor(raw_image, input_size=518):        
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    h, w = raw_image.shape[:2]
    
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    image = image.to(DEVICE)
    
    return image, (h, w)

def interpolate(depth, h, w):
    depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    return depth.cpu().numpy()


def interpolate_np(
    depth: Union[np.ndarray, dict],
    h: int,
    w: int
) -> np.ndarray:
    arr = depth[0]

    resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

    normalized = cv2.normalize(
        src=resized,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )
    return normalized
    
    
