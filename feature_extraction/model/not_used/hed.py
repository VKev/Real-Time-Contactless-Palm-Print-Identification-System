import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class HED(nn.Module):
    """
    Module inspired by Holistically-Nested Edge Detection for principal line extraction.
    """
    def __init__(self, in_channels, out_channels):
        super(HED, self).__init__()
        
        # Ensure out_channels is divisible by 4 for side outputs to work
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")
        
        # Multi-scale feature extraction
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Side outputs at different scales
        self.side1 = nn.Conv2d(64, out_channels // 4, kernel_size=1)
        self.side2 = nn.Conv2d(64, out_channels // 4, kernel_size=1)
        self.side3 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1)
        
        # Fusion layer
        self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # First level
        x1 = F.relu(self.bn1(self.conv1(x)))
        side1 = self.side1(x1)
        
        # Second level
        x2 = F.relu(self.bn2(self.conv2(x1)))
        side2 = self.side2(x2)
        
        # Third level
        x3 = F.relu(self.bn3(self.conv3(x2)))
        side3 = self.side3(x3)
        
        # Ensure all side outputs have the same size
        if side1.size()[2:] != side3.size()[2:]:
            side1 = F.interpolate(side1, size=side3.size()[2:], mode='bilinear', align_corners=True)
            side2 = F.interpolate(side2, size=side3.size()[2:], mode='bilinear', align_corners=True)
        
        # Concatenate side outputs
        fused = torch.cat([side1, side2, side3], dim=1)
        out = self.fuse(fused)
        
        return out

if __name__ == "__main__":
    in_channels = 3  # Example: RGB image
    out_channels = 4  # Ensure that out_channels is divisible by 4
    hed_model = HED(in_channels, out_channels)

    batch_size = 1
    input_tensor = torch.randn(batch_size, in_channels, 256, 256)

    output_tensor = hed_model(input_tensor)

    print("Output tensor shape:", output_tensor.shape)
