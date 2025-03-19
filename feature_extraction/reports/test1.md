**Description**

Implement vision transformer and spatial transformer network

**Performance**

Top 1 test accuracy: 0.7755

Top 1 train accuracy: 0.9998

---
**Model structure**
```
MyModel(
  (stn): STN(
    (localization): Sequential(
      (0): Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(32, 80, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False)
      (4): BatchNorm2d(80, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): Conv2d(80, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (7): BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
      (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (9): ReLU()
    )
    (fc_loc): Sequential(
      (0): Linear(in_features=10816, out_features=128, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=128, out_features=6, bias=True)
    )
  )
  (stem): StemBlock(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (patchEmbed): PatchEmbed(
    (proj): Conv2d(64, 256, kernel_size=(8, 8), stride=(8, 8))
    (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  )
  (posEmbed): Summer(
    (penc): PositionalEncoding1D()
  )
  (selfAttn): ScaledDotProductAttention(
    (fc_q): Linear(in_features=256, out_features=1024, bias=True)
    (fc_k): Linear(in_features=256, out_features=1024, bias=True)
    (fc_v): Linear(in_features=256, out_features=1024, bias=True)
    (fc_o): Linear(in_features=1024, out_features=256, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (fc): Linear(in_features=12544, out_features=128, bias=True)
)
```
---
**Online argumentation**

```
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
```