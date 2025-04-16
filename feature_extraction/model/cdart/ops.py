import torch
import torch.nn as nn


activation_ops = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'conv1d': lambda: nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding=(0, 1)),
}


class ActivationModule(nn.Module):
    def __init__(self, act_type='relu', inplace=True):
        super().__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=inplace)
        elif act_type in ('silu', 'swish'):
            self.activation = nn.SiLU(inplace=inplace)
        elif act_type == 'tanh':
            self.activation = nn.Tanh()
        elif act_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

def forward(self, x):
        return self.activation(x)

class MixedOp(nn.Module):
    def __init__(self, primitives, is_slim=False):
        super().__init__()
        self._ops = nn.ModuleList()
        self.is_slim = is_slim

        for name, op_class in primitives.items():
            self._ops.append(op_class())

    def forward(self, x, weights):
        assert len(weights) == len(self._ops), (
            f"Mismatch: {len(weights)} weights provided, "
            f"but {len(self._ops)} operations are available."
        )
        
        if self.is_slim:
            if torch.sum(weights) == 0:
                return torch.zeros_like(x)
            else:
                index = torch.argmax(weights).item()
                return self._ops[index](x)
        else:
            return sum(w * op(x) for w, op in zip(weights, self._ops))


if __name__ == "__main__":
    x = torch.linspace(-2, 2, steps=5).reshape(1, 1, 1, 5)
    print(f"Input Tensor:\n{x}\n")

    weights = torch.tensor([0.2, 0.3, 0.4, 0.7, 0.6])
    print(f"Weights:\n{weights}\n")

    mixed_op = MixedOp(activation_ops,is_slim=False)
    output = mixed_op(x, weights)
    print(f"Output in standard mode:\n{output}\n")

    mixed_op_slim = MixedOp(activation_ops,is_slim=True)
    output_slim = mixed_op_slim(x, weights)
    sigmoid = nn.Sigmoid()
    expected_output = sigmoid(x)
    print(f"Output in slim mode:\n{output_slim}\n")
    difference = output_slim - expected_output
    print(f"Difference between slim output and expected sigmoid output:\n{difference}")