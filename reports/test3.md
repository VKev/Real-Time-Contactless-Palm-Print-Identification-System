**Detail**

Change data augmentation (Albumentations)

**Performance**

Top 1 test accuracy: 0.8287

Top 1 train accuracy: 0.9985

---

**Weight initialization**

Conv2D: init.kaiming_normal_(m.weight, nonlinearity='relu')

Linear: init.kaiming_normal_(m.weight, nonlinearity='relu'), init.constant_(m.bias, 0)

BatchNorm2d: init.constant_(m.weight, 1), init.constant_(m.bias, 0)

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

```