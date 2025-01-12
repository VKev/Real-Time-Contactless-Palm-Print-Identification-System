import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .LinearDeformableConv import *

class BranchCNN(nn.Module):
    def __init__(self, in_channels=3):
        super(BranchCNN, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, 49, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(49),
            nn.ReLU(inplace=True),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(49, 49, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(49),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.GELU()
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(49, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Additional conv layer
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.channel_reducer = nn.Conv1d(
            in_channels=128,
            out_channels=49,
            kernel_size=1
        )
        
    def forward(self, x):
        x = self.convblock1(x)
        local_output_1 = self.convblock2(x)
        x = self.convblock3(x)

        x = x.reshape(x.size(0), x.size(1), -1)
        local_output_final = self.channel_reducer(x)
        local_output_1 = local_output_1.reshape(local_output_1.size(0), local_output_1.size(1), -1)
        return local_output_1, local_output_final

if __name__ == "__main__":
    # Test the model
    model = BranchCNN()
    x = torch.randn(32, 3, 224, 224)
    
    
    print(f"Input shape: {x.shape}")
    local_attn, x = model(x)
    print(f"local_attn: {local_attn.shape}")
    print(f"x: {x.shape}")