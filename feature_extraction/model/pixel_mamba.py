import torch
import torch.nn as nn
from mamba_ssm import Mamba2
try:
    from utils import print_total_params
except ImportError:
    from model.utils import print_total_params
    

class PixelMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = Mamba2(
            d_model=128,    
            d_state=256,
            d_conv=4,
            expand=4,
            headdim = 32
        )
        self.fc0 = nn.Linear(3, 128)

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat_1 = x.permute(0, 2, 3, 1).reshape(B, -1, C)

        x_flipped = x.flip(dims=[2, 3])
        x_flat_2 = x_flipped.permute(0, 2, 3, 1).reshape(B, -1, C)

        x_cat = torch.cat([x_flat_1, x_flat_2], dim=1)
        x = self.fc0(x_cat)
        out = self.mamba(x)
        return out


if __name__ == "__main__":
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).cuda()

    model = PixelMamba().cuda()
    print("Model summary:")
    print(model)
    print_total_params(model)
    y = model(x)
    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)  

