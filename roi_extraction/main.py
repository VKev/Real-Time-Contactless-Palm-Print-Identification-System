import os
import sys
import argparse
import cv2
import numpy as np
import shutil
try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from util import extract_palm_roi
except ImportError:
    from roi_extraction.util import extract_palm_roi
from utils.preprocess import preprocess


def augment_image(
    img: np.ndarray,
    max_rot: float = 30.0,
    blur_range: tuple[int, int] = (11, 21),
    hue_delta: int = 10,
    sat_range: tuple[float, float] = (0.8, 1.1),
    val_range: tuple[float, float] = (0.8, 1.1),
    brightness_delta: int = 30,
    contrast_range: tuple[float, float] = (0.8, 1.1),
    translation_frac: tuple[float, float] = (0.1, 0.1),
    channel_shift_range: tuple[int, int] = (-20, 20),
    gamma_range: tuple[float, float] = (0.8, 1.1)
) -> np.ndarray:
    h, w = img.shape[:2]

    # Rotation
    angle = np.random.uniform(-max_rot, max_rot)
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Translation
    tx = int(np.random.uniform(-translation_frac[0], translation_frac[0]) * w)
    ty = int(np.random.uniform(-translation_frac[1], translation_frac[1]) * h)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_trans, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Blur
    k = np.random.choice(range(blur_range[0], blur_range[1] + 1, 2))
    img = cv2.GaussianBlur(img, (k, k), 0)

    # HSV jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    dh = np.random.uniform(-hue_delta, hue_delta)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    ds = np.random.uniform(sat_range[0], sat_range[1])
    hsv[..., 1] = np.clip(hsv[..., 1] * ds, 0, 255)
    dv = np.random.uniform(val_range[0], val_range[1])
    hsv[..., 2] = np.clip(hsv[..., 2] * dv, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Brightness shift
    delta_b = np.random.randint(-brightness_delta, brightness_delta + 1)
    img = cv2.convertScaleAbs(img, alpha=1.0, beta=delta_b)

    # Contrast scaling
    alpha_c = np.random.uniform(contrast_range[0], contrast_range[1])
    img = cv2.convertScaleAbs(img, alpha=alpha_c, beta=0)

    # Gamma correction
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    invGamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** invGamma * 255
    table = table.astype(np.uint8)
    img = cv2.LUT(img, table)

    # Channel shifts
    chans = cv2.split(img)
    shifted = []
    for c in chans:
        delta = np.random.randint(channel_shift_range[0], channel_shift_range[1] + 1)
        c = np.clip(c.astype(int) + delta, 0, 255).astype(np.uint8)
        shifted.append(c)
    img = cv2.merge(shifted)

    return img


def process_and_save(img: np.ndarray, out_path: str) -> bool:
    """
    Extract palm ROI, preprocess it, and save 224×224 grayscale crop.
    """
    roi = extract_palm_roi(img)
    if roi is None:
        return False
    _, resized_gray = preprocess(roi)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return cv2.imwrite(out_path, resized_gray)


def process_folder(class_folder: str, class_id: int, num_aug: int):
    IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = sorted(f for f in os.listdir(class_folder)
                   if os.path.splitext(f)[1].lower() in IMG_EXT)

    for idx, fname in enumerate(files, start=1):
        path = os.path.join(class_folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Cannot read {path}; skipping.")
            continue

        # Save original ROI
        roi_out = os.path.join(class_folder, f"{idx}_0_{class_id}.png")
        if process_and_save(img, roi_out):
            print(f"✅ ROI saved: {roi_out}")
        else:
            print(f"ℹ️ No hand in original: {path}")

        # Generate augmentations
        for v in range(1, num_aug + 1):
            aug = augment_image(img)
            # Save raw augmented image
            raw_out = os.path.join(class_folder, f"{idx}_{v}_{class_id}_raw.png")
            cv2.imwrite(raw_out, aug)
            print(f"✅ Raw augmentation saved: {raw_out}")

            # Then save ROI of augmented
            roi_aug_out = os.path.join(class_folder, f"{idx}_{v}_{class_id}.png")
            if process_and_save(aug, roi_aug_out):
                print(f"✅ ROI saved: {roi_aug_out}")
            else:
                print(f"ℹ️ No hand in augmentation #{v}: {path}")


def organize_outputs(root_dir: str):
    """
    Move all ROI images (without '_raw') to the root folder,
    and move all raw augmentations ('_raw.png') into a subfolder 'raw_augmentation'.
    """
    raw_folder = os.path.join(root_dir, 'raw_augmentation')
    os.makedirs(raw_folder, exist_ok=True)

    # Traverse class subfolders
    for entry in os.listdir(root_dir):
        class_path = os.path.join(root_dir, entry)
        if not os.path.isdir(class_path) or entry == 'raw_augmentation':
            continue
        for fname in os.listdir(class_path):
            src = os.path.join(class_path, fname)
            if not fname.endswith('.png'):
                continue
            if fname.endswith('_raw.png'):
                dst = os.path.join(raw_folder, fname)
            else:
                dst = os.path.join(root_dir, fname)
            shutil.move(src, dst)
    print(f"Organized outputs: ROI images → {root_dir}, raw → {raw_folder}")


def main(root_dir: str, num_aug: int):
    # entries = sorted(os.listdir(root_dir))
    # class_folders = [e for e in entries if os.path.isdir(os.path.join(root_dir, e))]
    # class_id_map = {name: i for i, name in enumerate(class_folders, start=1)}

    # for cls_name, cls_id in class_id_map.items():
    #     folder = os.path.join(root_dir, cls_name)
    #     print(f"\n→ Processing class '{cls_name}' (ID {cls_id})")
    #     process_folder(folder, cls_id, num_aug)

    # After generation, organize outputs
    organize_outputs(root_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract & preprocess ROI from originals + augmentations."
    )
    parser.add_argument(
        "--root_dir",
        default=r"C:\\Vkev\\Repos\\Mamba-Environment\\Dataset\\Palm-Print\\RealisticSet\\Roi",
        help="Root folder with class subfolders."
    )
    parser.add_argument(
        "--num_aug",
        type=int,
        default=9,
        help="Number of augmentations per image."
    )
    args = parser.parse_args()
    main(args.root_dir, args.num_aug)
