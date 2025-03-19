import os
import re

def load_images(base_folder):
    image_paths = []
    labels = []

    for root, _, files in os.walk(base_folder):
        files = [
            file for file in files if file.lower().endswith((".bmp", ".jpg", ".jpeg"))
        ]
        files.sort()

        for file in files:
            img_path = os.path.join(root, file)
            image_paths.append(img_path)

            label_match = re.search(r"_(\d+)\.(bmp|jpg|jpeg)$", file, re.IGNORECASE)
            if label_match:
                label = int(
                    label_match.group(1)
                )  
                labels.append(label)

    return image_paths, labels


def get_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths