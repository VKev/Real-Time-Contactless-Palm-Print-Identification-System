import torch
import torch.nn as nn
from mamba_ssm import Mamba2

try:
    from utils import print_total_params
except ImportError:
    from model.utils import print_total_params

class PixelMamba(nn.Module):
    def __init__(self, height=224, width=224):
        super().__init__()
        self.H = height
        self.W = width

        self.mamba = Mamba2(
            d_model=128,   # Input dim to Mamba
            d_state=256,
            d_conv=4,
            expand=4,
            headdim=32
        )

        self.fc0 = nn.Linear(3, 128)

        self.avgPool1d = nn.Sequential(
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc1 = nn.Linear(self.H * self.W, 128)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.H and W == self.W, "Height/Width mismatch!"

        x_flat_1 = x.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, H*W, 3)
        x_flipped = x.flip(dims=[2, 3])  # flips H and W
        x_flat_2 = x_flipped.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, H*W, 3)

        x_cat = torch.cat([x_flat_1, x_flat_2], dim=1)

        x_embed = self.fc0(x_cat)

        out = self.mamba(x_embed)

        out_half = out[:, H*W:, :] #get last half of the sequence

        out_half = self.avgPool1d(out_half) 
        out_half = out_half.squeeze(2)

        out_vec = self.fc1(out_half)

        return out_vec


if __name__ == "__main__":
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).cuda()
    model = PixelMamba(height=224, width=224).cuda()
    print_total_params(model)

    y = model(x)
    print("Input shape: ", x.shape)   # (2, 3, 224, 224)
    print("Output shape:", y.shape)   # (2, 128)
