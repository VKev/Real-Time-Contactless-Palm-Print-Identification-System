try:
    from model.utils import print_total_params
    from model.patch_embedding import PatchEmbed
    from model.stem import StemBlock
    from model.resnet import *
except ImportError:
    try:
        from patch_embedding import PatchEmbed
        from stem import StemBlock
        from resnet import *
        from utils import print_total_params
    except ImportError:
        from feature_extraction.model.utils import print_total_params
        from feature_extraction.model.patch_embedding import PatchEmbed
        from feature_extraction.model.stem import StemBlock
        from feature_extraction.model.resnet import *
    
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

class FusionBlock(nn.Module):
    def __init__(self, input_dim, skip_dim, output_dim, dropout_rate=0.1):
        super(FusionBlock, self).__init__()
        self.fc = nn.Linear(input_dim + skip_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.silu = nn.SiLU()

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=-1)
        x = self.fc(x)
        x = self.norm(x)
        x = self.silu(x)
        x = self.dropout(x)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, attention_module, drop_path=0.1):
        super(ResidualAttentionBlock, self).__init__()
        self.attn = attention_module
        self.norm = nn.LayerNorm(attention_module.d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x +  self.drop_path(self.attn(x, x, x))
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
            local_out_channels=269, 
            pooled_out_channels=269
        )
        self.localbranch2 = BranchResNet(
            in_channels=512,
            block_channels=[512, 512, 512],
            strides=[1, 1, 1],
            local_out_channels=269, 
            pooled_out_channels=269
        )
        self.patchEmbed = PatchEmbed(
            img_size=56,
            in_chans=64,
            patch_sizes=[16, 12, 8],
            strides=[8, 6 ,4],
            embed_dim=256,
            norm_layer=lambda dim: nn.LayerNorm(dim),
        )
        
        self.fusion0 = FusionBlock(input_dim=256, skip_dim=256, output_dim=256, dropout_rate=0.2)
        self.fusion1 = FusionBlock(input_dim=256, skip_dim=256, output_dim=256, dropout_rate=0.2)
        self.fusion2 = FusionBlock(input_dim=256, skip_dim=256, output_dim=256, dropout_rate=0.1)
        self.fusion3 = FusionBlock(input_dim=256, skip_dim=256, output_dim=256, dropout_rate=0.1)
        
        self.posEmbed = Summer(PositionalEncoding1D(256))
        
        self.attn1 = ResidualAttentionBlock(
            ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=4, dropout=0.2),
            drop_path = 0
        )
        self.attn2 = ResidualAttentionBlock(
            ScaledDotProductAttention(d_model=256, d_k=256, d_v=256, h=4, dropout=0.2),
            drop_path = 0
        )
        self.attn3 = ResidualAttentionBlock(
            ScaledDotProductAttention(d_model=256, d_k=256, d_v=128, h=4, dropout=0.2),
            drop_path = 0
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 269, 128),
        )
        
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if hasattr(m, 'initialize_weights') and callable(m.initialize_weights):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        
        return x


if __name__ == "__main__":
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).cuda()

    model = MyModel().cuda()
    print("Model summary:")
    print(model)
    print_total_params(model)
    y = model(x)
    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)  

