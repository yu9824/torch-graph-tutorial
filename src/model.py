from typing import Optional, Union

import torch
import torch.nn
import torch.nn.functional
import torch.nn.modules.loss
import torch.optim

import torch_geometric.nn
import torch_geometric.data
import torch_geometric.data.batch


# for type hint
DataBatch = Union[torch_geometric.data.Batch, torch_geometric.data.Data]

# class GCN(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.conv1 = torch_geometric.nn.GCNConv(in_channels, 32)
#         self.conv2 = torch_geometric.nn.GCNConv(32, 64)
#         self.relu = torch.nn.ReLU()
#         self.linear = torch.nn.Linear(64 * , out_channels)

#     def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
#         x: torch.Tensor = data.x
#         edge_index: torch.Tensor = data.edge_index
#         edge_attr: Optional[torch.Tensor] = data.edge_attr
#         # x: Node feature matrix of shape [num_nodes, in_channels]
#         # edge_index: Graph connectivity matrix of shape [2, num_edges]
#         x_conv1 = self.conv1(x, edge_index, edge_attr)
#         x_conv1 = self.relu(x_conv1)
#         x_conv2 = self.conv2(x_conv1, edge_index, edge_attr)
#         x_conv2 = self.relu(x_conv2)
#         x_linear = self.linear(torch.flatten(x_conv2))
#         return x_linear


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, embedding_size: int = 64):
        """Graph Convolutional Network

        Reference
        - https://seunghan96.github.io/gnn/PyG_review1/

        Parameters
        ----------
        in_channels : int
            _description_
        embedding_size : int, optional
            _description_, by default 64
        """
        # Init parent
        super(GCN, self).__init__()

        # GCN layers ( for Message Passing )
        self.initial_conv = torch_geometric.nn.GCNConv(
            in_channels, embedding_size
        )
        self.conv1 = torch_geometric.nn.GCNConv(embedding_size, embedding_size)
        self.conv2 = torch_geometric.nn.GCNConv(embedding_size, embedding_size)
        self.conv3 = torch_geometric.nn.GCNConv(embedding_size, embedding_size)

        # Output layer ( for scalar output ... REGRESSION )
        self.out = torch.nn.Linear(embedding_size * 2, 1)

    def forward(self, data: DataBatch) -> float:
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        batch_index: torch.Tensor = data.batch
        edge_attr: Optional[torch.Tensor] = data.edge_attr

        hidden = torch.nn.functional.tanh(
            self.initial_conv(
                x=x, edge_index=edge_index, edge_weight=edge_attr
            )
        )
        hidden = torch.nn.functional.tanh(
            self.conv1(x=hidden, edge_index=edge_index, edge_weight=edge_attr)
        )
        hidden = torch.nn.functional.tanh(
            self.conv2(x=hidden, edge_index=edge_index, edge_weight=edge_attr)
        )
        hidden = torch.nn.functional.tanh(
            self.conv3(x=hidden, edge_index=edge_index, edge_weight=edge_attr)
        )

        # Global Pooling (stack different aggregations)
        # (reason) multiple nodes in one graph....
        # how to make 1 representation for graph??
        # use POOLING!
        # ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        hidden = torch.cat(
            [
                torch_geometric.nn.global_max_pool(hidden, batch_index),
                torch_geometric.nn.global_mean_pool(hidden, batch_index),
            ],
            dim=1,
        )

        out = self.out(hidden).flatten()
        return out
        # return out, hidden
