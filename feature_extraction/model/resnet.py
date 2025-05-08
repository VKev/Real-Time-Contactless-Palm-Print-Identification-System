import torch
import torch.nn as nn
import torch.nn.init as init
from fightingcv_attention.attention.CBAM import CBAMBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.cbam = CBAMBlock(out_channels, 16, 3) if use_cbam else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights using Kaiming Normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        residual = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class BranchResNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 block_channels=[205, 256, 205], 
                 strides=[2, 1, 1],
                 pooling_size=(16, 16),
                 use_cbam=True,
                 local_out_channels=None,
                 pooled_out_channels=None):
        super(BranchResNet, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, block_channels[0], kernel_size=3, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(block_channels[0]),
            nn.ReLU(inplace=True),
        )
        
        self.resblocks = nn.ModuleList()
        current_channels = block_channels[0]
        for out_channels, stride in zip(block_channels, strides):
            self.resblocks.append(ResidualBlock(current_channels, out_channels, stride=stride, use_cbam=use_cbam))
            current_channels = out_channels
        
        self.adptAvgPool2d = nn.AdaptiveAvgPool2d(pooling_size)
        
        if local_out_channels is not None:
            self.local_proj = nn.Sequential(
                nn.Conv2d(block_channels[0], local_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(local_out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.local_proj = None
        
        if pooled_out_channels is not None:
            self.pooled_proj = nn.Sequential(
                nn.Conv2d(block_channels[-1], pooled_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(pooled_out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.pooled_proj = None
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize weights using Kaiming Normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.resblocks[0](x)
        local_feature = self.adptAvgPool2d(x)
        
        if self.local_proj is not None:
            local_feature = self.local_proj(local_feature)
        
        x = self.resblocks[1](x)
        x_raw = self.resblocks[2](x)
        pooled_feature = self.adptAvgPool2d(x_raw)
        
        if self.pooled_proj is not None:
            pooled_feature = self.pooled_proj(pooled_feature)
        
        local_feature = local_feature.reshape(local_feature.size(0), local_feature.size(1), -1)
        pooled_feature = pooled_feature.reshape(pooled_feature.size(0), pooled_feature.size(1), -1)
        
        return local_feature, pooled_feature, x_raw

if __name__ == "__main__":
    model = BranchResNet(
        block_channels=[128, 256, 512],
        strides=[1, 1, 1],
        local_out_channels=205, 
        pooled_out_channels=205)
    x = torch.randn(32, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    local_attn, x_final, x_raw = model(x)
    print(f"local_attn (after projection and pooling): {local_attn.shape}")
    print(f"x_final (after projection and pooling): {x_final.shape}")
    print(f"x_raw (raw last block output): {x_raw.shape}")
