import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from utils import print_total_params
except ImportError:
    try:
        from model.utils import print_total_params
    except ImportError:
        from feature_extraction.model.utils import print_total_params
        
import torch.nn.init as init


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_sizes=[16, 8],
        strides=[8, 4],
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.flatten = flatten
        self.embed_dim = embed_dim

        self.embed_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_chans,
                        embed_dim,
                        kernel_size=patch_size,
                        stride=stride,
                        bias=bias,
                    ),
                )
                for patch_size, stride in zip(patch_sizes, strides)
            ]
        )

        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.num_patches = sum(
            ((img_size - patch_size) // stride + 1) ** 2
            for patch_size, stride in zip(patch_sizes, strides)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        embeddings = []
        for embed_layer in self.embed_layers:
            out = embed_layer(x)
            if self.flatten:
                out = out.flatten(2).transpose(1, 2)  # BCHW -> BNC
            embeddings.append(out)

        x = torch.cat(embeddings, dim=1)
        x = self.norm_layer(x)
        return x


if __name__ == "__main__":

    patch_embed = PatchEmbed(
        img_size=56,
        in_chans=72,
        patch_sizes=[16, 8],
        strides=[8, 4],
        embed_dim=256,
        norm_layer=lambda dim: nn.LayerNorm(dim),
    )

    img = torch.randn(1, 72, 56, 56)

    embeddings = patch_embed(img)

    print_total_params(patch_embed)
    print(f"Input shape: {img.shape}")
    print(f"Output shape: {embeddings.shape}")
