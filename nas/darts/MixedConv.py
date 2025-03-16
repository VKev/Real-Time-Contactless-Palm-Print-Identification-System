# dartconv.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedConv2d(nn.Module):
    """
    A convolution layer that supports multiple kernel sizes in parallel.
    Instead of averaging their outputs, we learn a set of 'alpha' (weights)
    for each kernel size and do a weighted sum (DARTS-style).
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1, bias=False):
        super(MixedConv2d, self).__init__()

        # If the user passes a single int, wrap it in a list
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]

        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.convs.append(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=k, 
                    stride=stride, 
                    padding=padding, 
                    bias=bias
                )
            )

        # Alphas (weights) for each possible kernel size
        self.alphas = nn.Parameter(torch.zeros(len(kernel_sizes)))  

    def forward(self, x):
        # If single kernel size, no weighting needed
        if len(self.convs) == 1:
            return self.convs[0](x)

        # Multiple kernel sizes
        # Softmax to ensure each alpha >= 0 and sum of alphas = 1
        weights = F.softmax(self.alphas, dim=0)  # shape: (num_kernels,)

        # Weighted sum of each conv output
        out = 0
        for conv, w in zip(self.convs, weights):
            out = out + w * conv(x)
        return out


