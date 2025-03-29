from .patch_embedding import PatchEmbed
from .stem import StemBlock
from .resnet import *
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionBlock(nn.Module):
    """
    A helper module to fuse two features:
    - Concatenates the main input with a skip connection.
    - Applies a linear projection.
    - Applies LayerNorm, GELU activation, and dropout.
    """
    def __init__(self, input_dim, skip_dim, output_dim, dropout_rate=0.1):
        super(FusionBlock, self).__init__()
        self.fc = nn.Linear(input_dim + skip_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=-1)
        x = self.fc(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x

class ResidualAttentionBlock(nn.Module):
    """
    A helper module for a self-attention block with:
    - Self-attention computation.
    - A residual connection.
    - Layer normalization.
    """
    def __init__(self, attention_module):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = attention_module
        # Using the d_model from the attention module for normalization
        self.norm = nn.LayerNorm(attention_module.d_model)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.stem = StemBlock()
        self.localbranch1 = BranchResNet(
            in_channels=3,
            block_channels=[128, 256, 512],
            strides=[1, 1, 1],
            local_out_channels=205, 
            pooled_out_channels=205
        )
        self.localbranch2 = BranchResNet(
            in_channels=512,
            block_channels=[512, 512, 512],
            strides=[1, 1, 1],
            local_out_channels=205, 
            pooled_out_channels=205
        )
        self.patchEmbed = PatchEmbed(
            img_size=56,
            in_chans=64,
            patch_sizes=[16, 8],
            strides=[8, 4],
            embed_dim=256,
            norm_layer=lambda dim: nn.LayerNorm(dim),
        )
        
        # Fusion blocks replacing the repeated pattern:
        # 1. Fuse patch embedding output with early_features_1.
        # 2. Fuse attention output with late_features_1.
        # 3. Fuse attention output with early_features_2.
        self.fusion0 = FusionBlock(input_dim=256, skip_dim=256, output_dim=256, dropout_rate=0.1)
        self.fusion1 = FusionBlock(input_dim=256, skip_dim=256, output_dim=256, dropout_rate=0.1)
        self.fusion2 = FusionBlock(input_dim=256, skip_dim=256, output_dim=256, dropout_rate=0.1)
        self.fusion3 = FusionBlock(input_dim=256, skip_dim=256, output_dim=128, dropout_rate=0.1)
        
        # Positional embedding
        self.posEmbed = Summer(PositionalEncoding1D(256))
        
        # Attention blocks using the residual attention module.
        self.attn1 = ResidualAttentionBlock(
            ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=4)
        )
        self.attn2 = ResidualAttentionBlock(
            ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=4)
        )
        self.attn3 = ResidualAttentionBlock(
            ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=4)
        )
        
        # Final classifier layer.
        self.fc_3 = nn.Linear(128 * 205, 128)
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        early_features_1, late_features_1, raw_1 = self.localbranch1(x)
        early_features_2, late_features_2, raw_2 = self.localbranch2(raw_1)
        
        x = self.stem(x)
        x = self.patchEmbed(x)
        
        x = self.fusion0(x, early_features_1)
        x = self.posEmbed(x)
        
        x = self.attn1(x)
        x = self.fusion1(x, late_features_1)
        
        x = self.attn2(x)
        x = self.fusion2(x, early_features_2)
        
        x = self.attn3(x)
        x = self.fusion3(x, late_features_2)
        
        x = x.view(x.size(0), -1)
        x = self.fc_3(x)
        return x
