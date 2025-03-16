from .PatchEmbedding import PatchEmbed
from .Stem import StemBlock
# from .LinearDeformableConv import LDConv
from .darts import DARTSMultiHeadAttention
from .ResNet import *
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MyModel(nn.Module):
    def __init__(self, candidate_heads=[2, 4, 8]):
        super(MyModel, self).__init__()
        # self.stn = STN()
        self.stem = StemBlock()
        self.localbranch = BranchResNet(kernel_size=[3, 5, 7])
        self.patchEmbed = PatchEmbed(
            img_size=56,
            in_chans=64,
            patch_sizes=[16, 8],
            strides=[8, 4],
            embed_dim=256,
            norm_layer=lambda dim: nn.LayerNorm(dim),
        )
        self.fc_0 = nn.Linear(512, 256)
        self.posEmbed = Summer(PositionalEncoding1D(512))
        # Use the DARTS-style multi-head attention module.
        self.selfAttn = DARTSMultiHeadAttention(d_model=256, d_k=256, d_v=256, candidate_heads=candidate_heads)
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
        # x is assumed to be (B, 3, 224, 224)
        local_output_1, final_local = self.localbranch(x)
        # x = self.stn(x)
        x = self.stem(x)
        x = self.patchEmbed(x)  # e.g., (B, 49, 256)
        x = torch.cat((x, local_output_1), dim=-1)
        x = self.fc_0(x)
        x = self.posEmbed(x)
        # Apply DARTS-style multi-head self-attention.
        x = self.selfAttn(x, x, x)
        x = torch.cat((x, final_local), dim=-1)
        x = self.fc_1(x)
        # Flatten and apply final fully-connected layer.
        x = x.view(x.size(0), -1)
        x = self.fc_2(x)
        return x

if __name__ == "__main__":
    # For testing purposes, create a dummy input tensor.
    img = torch.randn(1, 3, 224, 224)
    model = MyModel()
    out = model(img)
    print("Output shape before discretization:", out.shape)
    
    # After training, call discretize() to fix the architecture.
    model.selfAttn.discretize()
    out_discrete = model(img)
    print("Output shape after discretization:", out_discrete.shape)
