import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .darts import MixedConv2d
import copy

class MixedResidualBlock(nn.Module):
    """
    A residual block that can handle multiple kernel sizes via MixedConv2d.
    If 'kernel_size' is a list, each MixedConv2d becomes a parallel set of convs
    (one per kernel size), and we do a weighted sum via the alpha parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(MixedResidualBlock, self).__init__()
        
        self.conv1 = MixedConv2d(
            in_channels, 
            out_channels, 
            kernel_sizes=kernel_size, 
            stride=stride, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = MixedConv2d(
            out_channels, 
            out_channels, 
            kernel_sizes=kernel_size, 
            stride=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path if dimension changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Typically a 1x1 conv with stride for dimension matching
            self.shortcut = nn.Sequential(
                MixedConv2d(
                    in_channels, 
                    out_channels, 
                    kernel_sizes=1, 
                    stride=stride, 
                    bias=False
                ),
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
    """
    Example ResNet-like branch that can use multiple kernel sizes with learnable
    'alphas' for searching. Pass 'kernel_size' as an int or a list of ints.
    """
    def __init__(self, in_channels=3, kernel_size=3):
        super(BranchResNet, self).__init__()
        
        # First conv block
        self.convblock1 = nn.Sequential(
            MixedConv2d(
                in_channels, 
                128, 
                kernel_sizes=kernel_size, 
                stride=2, 
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.resblock1 = MixedResidualBlock(128, 205, kernel_size=kernel_size, stride=2)
        self.resblock2 = MixedResidualBlock(205, 256, kernel_size=kernel_size, stride=1)
        self.resblock3 = MixedResidualBlock(256, 205, kernel_size=kernel_size, stride=1)
        
        self.adptAvgPool2d = nn.AdaptiveAvgPool2d((16, 16))

        self.initialize_weights()
        
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

def discretize_branch_resnet(model: nn.Module) -> nn.Module:

    if isinstance(model, MixedConv2d):
        with torch.no_grad():
            alpha_index = model.alphas.argmax().item()

        chosen_kernel_size = model.kernel_sizes[alpha_index]
        padding = chosen_kernel_size // 2

        in_ch = model.convs[0].in_channels
        out_ch = model.convs[0].out_channels
        stride = model.convs[0].stride
        bias = (model.convs[0].bias is not None)

        # Create the discrete Conv2d
        new_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=chosen_kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        # Copy weights from chosen sub-conv
        with torch.no_grad():
            chosen_conv = model.convs[alpha_index]
            new_conv.weight.copy_(chosen_conv.weight)
            if bias:
                new_conv.bias.copy_(chosen_conv.bias)

        return new_conv  # Return the new discrete conv in place of MixedConv2d

    # Otherwise, recurse on children
    for name, child in list(model.named_children()):
        # discretize child
        discrete_submodule = discretize_branch_resnet(child)
        # set the child to the (possibly replaced) submodule
        setattr(model, name, discrete_submodule)

    return model

if __name__ == "__main__":
    model = BranchResNet(in_channels=3, kernel_size=[3, 5, 7]).to('cuda')
    x_single = torch.randn(8, 3, 224, 224).to('cuda')
    local_attn, x_out = model(x_single)
    print("local_attn:", local_attn.shape)
    print("x_out     ", x_out.shape)

    convblock_mixed = model.convblock1[0]
    print("\nLearnable alphas in first MixedConv2d:", convblock_mixed.alphas)
    print("Softmaxed alphas:", F.softmax(convblock_mixed.alphas, dim=0))
    
    discretize_branch_resnet(model )
    print(model)
