import torch
import torch.nn as nn


class Adapter(nn.Module):
    """
    Adapter module as described in the paper https://arxiv.org/pdf/1902.00751

    Args:
        hidden_size (int): Size of the hidden layer.
        bottleneck_size (int): Size of the bottleneck layer.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int):
        super(Adapter, self).__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.act_fn = nn.ReLU()

        # Initialize weights to be near identity
        self.down.weight.data.normal_(mean=0.0, std=1e-2)
        self.down.bias.data.zero_()
        self.up.weight.data.normal_(mean=0.0, std=1e-2)
        self.up.bias.data.zero_()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Adapter module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, no_nodes, hidden_size].
            edge_index (torch.Tensor): Edge index matrix of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix with shape [num_edges, num_edge_features].

        Returns:
            Tensor: Output tensor of the same shape as the input tensor.
        """
        x = self.up(self.act_fn(self.down(x))) + x
        return x
