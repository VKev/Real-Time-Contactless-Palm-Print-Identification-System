from .PatchEmbedding import PatchEmbed
from .SpatialTransformer import STN
from .Stem import StemBlock
from .Local import BranchCNN
from .LinearDeformableConv import LDConv
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # self.stn = STN()
        self.stem = StemBlock()
        self.localbranch = BranchCNN()
        self.patchEmbed =  PatchEmbed(
            img_size=56,
            in_chans=64,
            patch_sizes=[16, 8],
            strides=[8, 4],
            embed_dim=256,
            norm_layer=lambda dim: nn.LayerNorm(dim),
        )
        self.fc_0 = nn.Linear(512, 256)
        self.posEmbed = Summer(PositionalEncoding1D(512))
        self.selfAttn = ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=4)
        self.fc_1 = nn.Linear(512, 128)
        self.fc_2 = nn.Linear(128*205, 128)
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 32, 3, 224, 224
        local_output_1, final_local = self.localbranch(x)
        # x = self.stn(x)
        x = self.stem(x)
        x = self.patchEmbed(x)  # 32, 49, 256
        x = torch.cat((x, local_output_1), dim=-1)
        x = self.fc_0(x)
        x = self.posEmbed(x)
        x = self.selfAttn(x, x, x)
        x = torch.cat((x, final_local), dim=-1)
        x = self.fc_1(x)
        # x = x.mean(dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc_2(x)

        return x
