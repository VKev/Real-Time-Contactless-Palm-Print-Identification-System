import random
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import sys
# def apply_cutmix(image1, image2, beta=1.0):

#     np_img1 = np.array(image1)
#     np_img2 = np.array(image2)
    
#     lam = np.random.beta(beta, beta)
#     H, W, C = np_img1.shape
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)
    
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)
    
#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)
    
#     np_img1[bby1:bby2, bbx1:bbx2, :] = np_img2[bby1:bby2, bbx1:bbx2, :]
    
#     return Image.fromarray(np_img1)

def apply_mixing(image1, image2, beta=1.0):
    """
    Blends two images by mixing their pixel values using a weight sampled from a Beta distribution.
    
    Parameters:
        image1 (PIL.Image): The first image.
        image2 (PIL.Image): The second image.
        beta (float): Beta distribution parameter; default is 1.0.
        
    Returns:
        PIL.Image: The resulting mixed image.
    """
    # Convert images to numpy arrays (float32 for computation)
    np_img1 = np.array(image1).astype(np.float32)
    np_img2 = np.array(image2).astype(np.float32)
    
    # Sample lambda from the Beta distribution
    lam = np.random.beta(beta, beta)
    
    # Create the mixed image: weighted average of image1 and image2
    mixed_img = lam * np_img1 + (1 - lam) * np_img2
    
    # Ensure the pixel values are valid and of type uint8
    mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(mixed_img)

class TripletDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        n_negatives,
        num_classes_for_negative,
        transform=None,
        augmentation=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.n_negatives = (
            n_negatives 
        )
        self.transform = transform
        self.augmentation = augmentation

        # Create a dictionary mapping labels to their indices
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        print(f"Number of labels: {len(self.label_to_indices)}")
        # Calculate the maximum possible different labels for negatives
        self.max_possible_neg_classes = (
            len(self.label_to_indices) - 1
        )  # -1 for anchor's class

        # Adjust num_classes_for_negative if it exceeds maximum possible
        self.num_classes_for_negative = min(
            num_classes_for_negative, self.max_possible_neg_classes
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_image = Image.open(self.image_paths[idx]).convert("RGB")
        anchor_label = self.labels[idx]

        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
        positive_idx = random.choice(positive_indices)
        positive_image = Image.open(self.image_paths[positive_idx]).convert("RGB")

        available_labels = [
            label for label in self.label_to_indices.keys() if label != anchor_label
        ]

        selected_labels = random.sample(available_labels, self.num_classes_for_negative)

        samples_per_label = [
            self.n_negatives // self.num_classes_for_negative
        ] * self.num_classes_for_negative
        remaining = self.n_negatives % self.num_classes_for_negative
        for i in range(remaining):
            samples_per_label[i] += 1

        negative_images = []
        for label, num_samples in zip(selected_labels, samples_per_label):
            label_indices = self.label_to_indices[label]

            num_samples = min(num_samples, len(label_indices))
            selected_indices = random.sample(label_indices, num_samples)

            for neg_idx in selected_indices:
                # print(self.image_paths[neg_idx])
                negative_image = Image.open(self.image_paths[neg_idx]).convert("RGB")

                if self.augmentation and random.random() < 0.7:
                    negative_image = self.augmentation(negative_image)
                else:
                    negative_image = self.transform(negative_image)
                negative_images.append(negative_image)

        if self.augmentation:
            if random.random() < 0.5:
                anchor_image = self.augmentation(anchor_image)
            else:
                anchor_image = self.transform(anchor_image)

            if random.random() < 0.7:
                positive_image = self.augmentation(positive_image)
            else:
                positive_image = self.transform(positive_image)
        else:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)

        return anchor_image, positive_image, negative_images


def triplet_collate_fn(batch):
    anchors = []
    positives = []
    negatives = []

    for anchor, positive, negative_list in batch:
        anchors.append(anchor)
        positives.append(positive)
        negatives.extend(negative_list)

    # Stack images
    anchors = torch.stack(anchors)  # Batch of anchor images
    positives = torch.stack(positives)  # Batch of positive images
    negatives = torch.stack(negatives)  # Batch of all negative images
    all_images = torch.cat([anchors, positives, negatives], dim=0)
    return all_images, len(anchors), len(negatives) // len(anchors)

if __name__ == "__main__":
    
    image1_path = r'C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\AugmentationTest\00001_s1_0.bmp'
    image2_path = r'C:\Vkev\Repos\Mamba-Environment\Dataset\Palm-Print\AugmentationTest\00002_s2_0.bmp'
    
    # Open the images and convert them to RGB.
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    
    result_image = apply_mixing(image1, image2, beta=1)
    
    result_image.show()