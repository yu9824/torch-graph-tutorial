from typing import Optional, Union
from collections.abc import Callable

import torch
import torch.nn
import torch.nn.modules.loss
import torch.optim

import torch_geometric.nn
import torch_geometric.data


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 32)
        self.conv2 = torch_geometric.nn.GCNConv(32, 64)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(64, out_channels)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        edge_attr: Optional[torch.Tensor] = data.edge_attr
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x_conv1 = self.conv1(x, edge_index, edge_attr)
        x_conv1 = self.relu(x_conv1)
        x_conv2 = self.conv2(x_conv1, edge_index, edge_attr)
        x_conv2 = self.relu(x_conv2)
        x_linear = self.linear(x_conv2)
        return x_linear


def fit(
    model: GCN,
    train_loader: torch_geometric.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[
        torch.nn.modules.loss._Loss,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ],
    device: torch.device,
):
    model.train()
    running_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    return train_loss
