import torchvision.transforms as transforms
import random
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

augmentation = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(25),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=random.uniform(0.1, 0.2),
                    contrast=random.uniform(0.1, 0.3),
                    saturation=random.uniform(0.1, 0.3),
                    hue=random.uniform(0.1, 0.3),
                )
            ],
            p=0.5,
        ),
        transforms.RandomApply(
            [transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0))], p=0.5
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3
        ),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))], p=0.7
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)