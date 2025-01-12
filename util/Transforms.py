import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np

# Convert PIL Image to numpy array if needed
def to_numpy(image):
    if not isinstance(image, np.ndarray):
        return np.array(image)
    return image

# Normal transformation
def transform(image):
    image = to_numpy(image)
    transform_pipeline = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    return transform_pipeline(image=image)['image']

# Augmentation transformation
def augmentation(image):
    image = to_numpy(image)
    augmentation_pipeline = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.RandomResizedCrop(height=224, width=224, scale=(0.9, 1.0), p=0.5),
        ], p=0.7),
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=0.5),
        ], p=0.3),

        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    return augmentation_pipeline(image=image)['image']
