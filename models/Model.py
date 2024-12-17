from .PatchEmbedding import PatchEmbed
from .SpatialTransformer import STN
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.stn = STN()
        self.stem = StemBlock()
        self.patchEmbed = PatchEmbed(
            img_size=56,
            patch_size=8,
            in_chans=64,
            embed_dim=256,
            norm_layer= lambda dim: nn.LayerNorm(dim)
        )
        self.posEmbed = Summer(PositionalEncoding1D(256))
        self.selfAttn = ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=4)
        self.fc = nn.Linear(12544,128)

    def forward(self, x):
        x = self.stn(x)
        x = self.stem(x)
        x = self.patchEmbed(x)
        x = self.posEmbed(x)
        x = self.selfAttn(x,x,x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




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