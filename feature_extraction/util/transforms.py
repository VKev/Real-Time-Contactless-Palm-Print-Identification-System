import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import os

# Convert PIL Image to numpy array if needed
def to_numpy(image):
    if not isinstance(image, np.ndarray):
        return np.array(image)
    return image


# Normal transformation
def transform(image):
    image = to_numpy(image)
    transform_pipeline = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ]
    )
    return transform_pipeline(image=image)["image"]


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
            A.Affine(translate_percent=(0.0625,0.1), scale=(0.9, 1.1), rotate=25, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), p=0.5)
        ], p=0.7),
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.5),
            A.RandomShadow(num_shadows_limit=(1, 3), p=0.5),
        ], p=0.5),
        
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.5), p=0.5),  
        ], p=0.2),
        
        A.OneOf([
            A.GridDistortion(p=0.3),  # Random grid distortion
            A.OpticalDistortion(distort_limit=0.1, p=0.3),  # Optical distortions
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5),  # RGB shifting
        ], p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),  # Contrast limited adaptive histogram equalization
        
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    return augmentation_pipeline(image=image)['image']


def augment_and_save_images(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):  # Check for image files
                input_file_path = os.path.join(root, file)

                image = Image.open(input_file_path).convert("RGB")
                augmented_image = augmentation(image)
                augmented_image = Image.fromarray(augmented_image)
                output_file_path = os.path.join(output_path, file)
                augmented_image.save(output_file_path)
                print(f"Augmented image saved to: {output_file_path}")


if __name__ == "__main__":
    input_path = "../../../Dataset/Palm-Print/TrainAndTest/train"
    output_path = "../../../Dataset/Palm-Print/AugmentationTest"
    augment_and_save_images(input_path, output_path)
