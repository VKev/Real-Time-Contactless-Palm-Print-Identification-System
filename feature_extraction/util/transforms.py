import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os


def to_numpy(image):
    if not isinstance(image, np.ndarray):
        return np.array(image)
    return image

_MEAN = [0.5, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]

_BASE_PIPELINE = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]
)

_AUG_IMAGE_PIPELINE = A.Compose(
    [
        A.Resize(224, 224),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.03, 0.03),
            rotate=(-12, 12),
            shear=(-5, 5),
            p=0.6,
        ),
        A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.95, 1.05), p=0.25),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ],
            p=0.15,
        ),
        A.OneOf(
            [
                A.ImageCompression(quality_range=(85, 100), p=1.0),
                A.ISONoise(p=1.0),
            ],
            p=0.2,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(4, 14),
            hole_width_range=(4, 14),
            fill=0,
            p=0.1,
        ),
    ]
)

_AUG_PIPELINE = A.Compose(
    [
        *_AUG_IMAGE_PIPELINE.transforms,
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ]
)

def transform(image):
    image = to_numpy(image)
    return _BASE_PIPELINE(image=image)["image"]

def augmentation(image):
    image = to_numpy(image)
    return _AUG_PIPELINE(image=image)["image"]


def augment_and_save_images(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):  # Check for image files
                input_file_path = os.path.join(root, file)

                image = Image.open(input_file_path).convert("RGB")
                augmented_image = _AUG_IMAGE_PIPELINE(image=np.array(image))["image"]
                augmented_image = Image.fromarray(augmented_image)
                output_file_path = os.path.join(output_path, file)
                augmented_image.save(output_file_path)
                print(f"Augmented image saved to: {output_file_path}")


if __name__ == "__main__":
    input_path = "../../../Dataset/Palm-Print/TrainAndTest/train"
    output_path = "../../../Dataset/Palm-Print/AugmentationTest"
    augment_and_save_images(input_path, output_path)
