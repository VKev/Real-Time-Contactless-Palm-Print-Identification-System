import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from typing import Union
import numpy as np
from PIL import Image
try:
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
except ImportError:
    from depth_estimation.depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    

def apply_depth_mask(image: np.ndarray, depth_map: np.ndarray, threshold: float = 0.5) -> Image.Image:
    """
    Applies a depth-based mask to the input image. Pixels with depth values below the threshold are blacked out.

    Parameters:
        image (np.ndarray): The original RGB image as a NumPy array.
        depth_map (np.ndarray): The corresponding depth map as a NumPy array with values in [0, 255].
        threshold (float): Threshold value between 0 and 1 to determine which pixels to keep.

    Returns:
        PIL.Image.Image: The masked image.
    """
    # Ensure depth_map is in the range [0, 255]
    if depth_map.max() <= 1.0:
        depth_map = (depth_map * 255).astype(np.uint8)
    else:
        depth_map = depth_map.astype(np.uint8)

    # Create a binary mask where depth >= threshold
    threshold_value = int(threshold * 255)
    mask = depth_map >= threshold_value

    # Apply the mask to the image
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    # Convert the result to a PIL Image
    return Image.fromarray(masked_image)


def apply_depth_mask_np(image: np.ndarray, depth_map: np.ndarray, threshold: float = 0.5) -> Image.Image:
    """
    Applies a depth-based mask to the input image. Pixels with depth values below the threshold are blacked out.

    Parameters:
        image (np.ndarray): The original RGB image as a NumPy array.
        depth_map (np.ndarray): The corresponding depth map as a NumPy array with values in [0, 255].
        threshold (float): Threshold value between 0 and 1 to determine which pixels to keep.

    Returns:
        PIL.Image.Image: The masked image.
    """
    # Ensure depth_map is in the range [0, 255]
    if depth_map.max() <= 1.0:
        depth_map = (depth_map * 255).astype(np.uint8)
    else:
        depth_map = depth_map.astype(np.uint8)

    threshold_value = int(threshold * 255)
    mask = depth_map >= threshold_value

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    return masked_image

def resize(raw_image, input_size=518):        
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
    if isinstance(depth, dict):
        depth = next(iter(depth.values()))

    arr = depth

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

    elif arr.ndim == 3:
        batch_resized = []
        for i in range(arr.shape[0]):
            batch_resized.append(
                cv2.resize(arr[i], (w, h), interpolation=cv2.INTER_LINEAR)
            )
        return np.stack(batch_resized, axis=0)

    else:
        raise ValueError(f"Unsupported depth array shape: {depth.shape}")