import torch
import torch.nn as nn
import torch.nn.functional as F
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.tutorials.cnn_utils import evaluate, load_mnist, train
from torch._tensor import Tensor

torch.manual_seed(42)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
train_loader, valid_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)

ax_client = AxClient()
ax_client.create_experiment(
    name="tune_cnn_on_mnist",
    parameters=[
        {
            "name": "lr",
            "type": "range",
            "bounds": [1e-6, 0.4],
            "value_type": "float",
            "log_scale": True,
        },
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        {
            "name": "kernel_size",
            "type": "choice",
            "values": [3, 5, 7],
            "value_type": "int",
        },
        {
            "name": "num_conv_layers",
            "type": "choice",
            "values": [1, 2, 3],
            "value_type": "int",
        },
        {
            "name": "hidden_size",
            "type": "choice",
            "values": [32, 64, 128, 256],
            "value_type": "int",
        },
    ],
    objectives={"accuracy": ObjectiveProperties(minimize=False)},
)


class CNN(nn.Module):
    def __init__(
        self, kernel_size: int, num_conv_layers: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_conv_layers):
            in_channels = 1 if i == 0 else 20
            self.convs.append(
                nn.Conv2d(in_channels, 20, kernel_size=kernel_size, stride=1)
            )
        conv_out_size = 28 - num_conv_layers * (kernel_size - 1)
        pooled_size = (conv_out_size - 3) // 3 + 1
        self.fc1 = nn.Linear(pooled_size * pooled_size * 20, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x))
        x = F.max_pool2d(x, 3, 3)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train_evaluate(parameterization):
    net = CNN(
        kernel_size=int(parameterization["kernel_size"]),
        num_conv_layers=int(parameterization["num_conv_layers"]),
        hidden_size=int(parameterization["hidden_size"]),
    )
    net = train(
        net=net,
        train_loader=train_loader,
        parameters=parameterization,
        dtype=dtype,
        device=device,
    )
    return evaluate(net=net, data_loader=valid_loader, dtype=dtype, device=device)


ax_client.attach_trial(
    parameters={
        "lr": 0.000026,
        "momentum": 0.58,
        "kernel_size": 5,
        "num_conv_layers": 2,
        "hidden_size": 64,
    }
)

baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
ax_client.complete_trial(trial_index=0, raw_data=train_evaluate(baseline_parameters))

for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=train_evaluate(parameters)
    )

print(ax_client.get_trials_data_frame())
