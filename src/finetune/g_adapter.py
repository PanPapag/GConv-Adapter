import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj

class GAdapter(nn.Module):
    """
    GAdapter module as described in the paper https://arxiv.org/pdf/2305.10329

    Args:
        hidden_size (int): Size of the hidden layer.
        bottleneck_size (int): Size of the bottleneck layer.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int):
        super(GAdapter, self).__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the GAdapter module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, no_nodes, hidden_size].
            edge_index (torch.Tensor): Edge index matrix of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix with shape [num_edges, num_edge_features].

        Returns:
            torch.Tensor: Output tensor of the same shape as the input tensor.
        """
        s = to_dense_adj(edge_index, max_num_nodes=x.squeeze(0).shape[0])[0]
        x = self.pre_ln(x)
        x = self.act_fn(self.up(self.down(torch.matmul(s, x)))) + x
        x = self.post_ln(x)
        return x
