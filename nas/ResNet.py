import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .darts import MixedConv2d

class MixedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.conv1 = MixedConv2d(in_channels, out_channels, kernel_sizes=kernel_size, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = MixedConv2d(out_channels, out_channels, kernel_sizes=kernel_size, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                MixedConv2d(in_channels, out_channels, kernel_sizes=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return self.relu(out)

class BranchResNet(nn.Module):
    def __init__(self, in_channels=3, kernel_size=[3,5,7]):
        super(BranchResNet, self).__init__()
        
        self.convblock1 = nn.Sequential(
            MixedConv2d(in_channels, 128, kernel_sizes=kernel_size, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
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

    def discretize(self):
        """
        Discretize the architecture by recursively replacing all MixedConv2d modules
        with standard Conv2d modules using the kernel size with highest alpha.
        """
        def _discretize_module(module):
            for name, child in module.named_children():
                if isinstance(child, MixedConv2d):
                    with torch.no_grad():
                        alpha_index = child.alphas.argmax().item()
                        chosen_kernel_size = child.kernel_sizes[alpha_index]
                        padding = chosen_kernel_size // 2
                        
                        in_ch = child.convs[0].in_channels
                        out_ch = child.convs[0].out_channels
                        stride = child.convs[0].stride
                        bias = child.convs[0].bias is not None

                        new_conv = nn.Conv2d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=chosen_kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias
                        )

                        # Copy the weights
                        chosen_conv = child.convs[alpha_index]
                        new_conv.weight.data.copy_(chosen_conv.weight.data)
                        if bias:
                            new_conv.bias.data.copy_(chosen_conv.bias.data)

                        setattr(self, name, new_conv)
                else:
                    child = child.discretize() if hasattr(child, "discretize") else self.discretize_recursive(child)

        def discretize_module(module):
            for name, child in module.named_children():
                if isinstance(child, MixedConv2d):
                    with torch.no_grad():
                        alpha_index = child.alphas.argmax().item()
                    chosen_kernel_size = child.kernel_sizes[alpha_index]
                    padding = chosen_kernel_size // 2
                    
                    in_ch = child.convs[0].in_channels
                    out_ch = child.convs[0].out_channels
                    stride = child.convs[0].stride
                    bias = child.convs[0].bias is not None

                    new_conv = nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=chosen_kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias
                    )

                    # Copy the weights
                    chosen_conv = child.convs[alpha_index]
                    new_conv.weight.data.copy_(chosen_conv.weight.data)
                    if bias:
                        new_conv.bias.data.copy_(chosen_conv.bias.data)

                    setattr(module, name, new_conv)
                else:
                    discretize_module(child)

        discretize_module(self)

# Example usage
if __name__ == "__main__":
    model = BranchResNet(in_channels=3, kernel_size=[3,5,7]).to('cuda')
    x_single = torch.randn(1, 3, 224, 224).to('cuda')
    local_attn, x_out = model(x_single)
    print("Before discretization:", local_attn.shape, x_out.shape)

    # Discretize using the new method:
    model.discretize()
    print("After discretization:")
    print(model)
