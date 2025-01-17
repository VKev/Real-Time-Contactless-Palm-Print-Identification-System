import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .LinearDeformableConv import *

class BranchCNN(nn.Module):
    def __init__(self, in_channels=3):
        super(BranchCNN, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(128, 205, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(205),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((16, 16)),
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(205, 256, kernel_size=3, stride=1, padding=1),  # Standard convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 205, kernel_size=3, stride=1, padding=1),  # Pointwise convolution
            nn.BatchNorm2d(205),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16))  # Global pooling
        )

        # self.channel_reducer = nn.Conv1d(
        #     in_channels=128,
        #     out_channels=49,
        #     kernel_size=1
        # )
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use Kaiming Normal initialization for Conv2d layers
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm weights to 1 and biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.convblock1(x)
        local_output_1 = self.convblock2(x)
        x = self.convblock3(local_output_1)

        x = x.reshape(x.size(0), x.size(1), -1)
        local_output_1 = local_output_1.reshape(local_output_1.size(0), local_output_1.size(1), -1)
        return local_output_1, x

if __name__ == "__main__":
    # Test the model
    model = BranchCNN()
    x = torch.randn(32, 3, 224, 224)
    
    
    print(f"Input shape: {x.shape}")
    local_attn, x = model(x)
    print(f"local_attn: {local_attn.shape}")
    print(f"x: {x.shape}")