import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .Utils import *

class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        self.initialize_weights()   


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)  
        x = self.bn1(x)    
        x = self.relu(x)   
        x = self.maxpool(x)
        return x


if __name__ == "__main__":
    stem = StemBlock()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = stem(x)
    print_total_params(stem)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
