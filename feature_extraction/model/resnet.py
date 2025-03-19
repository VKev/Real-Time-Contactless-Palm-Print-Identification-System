import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from .LinearDeformableConv import *

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class BranchResNet(nn.Module):
    def __init__(self, in_channels=3):
        super(BranchResNet, self).__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.resblock1 = ResidualBlock(128, 205, stride=2)
        self.resblock2 = ResidualBlock(205, 256, stride=1)
        self.resblock3 = ResidualBlock(256, 205, stride=1)
        
        self.adptAvgPool2d = nn.AdaptiveAvgPool2d((16, 16))
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.resblock1(x)
        local_output_1 = self.adptAvgPool2d(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.adptAvgPool2d(x)
        
        x = x.reshape(x.size(0), x.size(1), -1)
        local_output_1 = local_output_1.reshape(local_output_1.size(0), local_output_1.size(1), -1)
        return local_output_1, x

if __name__ == "__main__":
    # Test the model
    model = BranchResNet()
    x = torch.randn(32, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    local_attn, x = model(x)
    print(f"local_attn: {local_attn.shape}")
    print(f"x: {x.shape}")