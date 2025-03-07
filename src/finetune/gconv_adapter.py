import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GConvAdapter(nn.Module):
    """
    Proposed GConv-Adapter for ablation studies.

    Args:
        hidden_size (int): Size of the hidden layer.
        bottleneck_size (int): Size of the bottleneck layer.
        conv_type (str): Type of graph convolution ('gcn', 'sage', 'gat').
        non_linearity (str): Type of non-linearity ('relu', 'silu', 'none').
        normalization (str): Type of normalization ('batch_norm', 'layer_norm', 'none').
        learnable_scalar (bool): Whether to use a learnable scalar to multiply the output.
        skip_connection (bool): Whether to include a skip connection in the forward pass.
        normalize (bool): Whether to add self-loops and compute symmetric normalization coefficients on-the-fly.

    Raises:
        ValueError: If an invalid conv_type, non_linearity, or normalization is provided.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int, 
                 conv_type: str = 'gcn', non_linearity: str = 'relu', 
                 normalization: str = 'none', learnable_scalar: bool = False, 
                 skip_connection: bool = True, normalize: bool = True):
        super(GConvAdapter, self).__init__()

        # Select the graph convolution type
        if conv_type == 'gcn':
            ConvLayer = GCNConv
        elif conv_type == 'sage':
            ConvLayer = SAGEConv
        elif conv_type == 'gat':
            ConvLayer = GATConv
        else:
            raise ValueError("Invalid conv_type. Supported types: 'gcn', 'sage', 'gat'.")

        # Initialize graph convolution layers with the normalize flag
        self.conv_down = ConvLayer(in_channels=hidden_size, out_channels=bottleneck_size, normalize=normalize)
        self.conv_up = ConvLayer(in_channels=bottleneck_size, out_channels=hidden_size, normalize=normalize)

        # Select the non-linearity
        if non_linearity == 'relu':
            self.act_fn = nn.ReLU()
        elif non_linearity == 'silu':
            self.act_fn = nn.SiLU()
        elif non_linearity == 'none':
            self.act_fn = nn.Identity()
        else:
            raise ValueError("Invalid non_linearity. Supported types: 'relu', 'silu', 'none'.")

        # Select the normalization technique
        if normalization == 'batch_norm':
            self.normalization = nn.BatchNorm1d(hidden_size)
        elif normalization == 'layer_norm':
            self.normalization = nn.LayerNorm(hidden_size)
        elif normalization == 'none':
            self.normalization = None
        else:
            raise ValueError("Invalid normalization. Supported types: 'batch_norm', 'layer_norm', 'none'.")

        # Learnable scalar for output
        if learnable_scalar:
            self.scalar = nn.Parameter(torch.ones(1))
        else:
            self.scalar = None

        # Skip connection flag
        self.skip_connection = skip_connection

        # Initialize weights to be near identity
        torch.nn.init.normal_(self.conv_down.lin.weight, mean=0.0, std=1e-5)
        if self.conv_down.bias is not None:
            torch.nn.init.zeros_(self.conv_down.bias)
        torch.nn.init.normal_(self.conv_up.lin.weight, mean=0.0, std=1e-5)
        if self.conv_up.bias is not None:
            torch.nn.init.zeros_(self.conv_up.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the GConv-Adapter module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, no_nodes, hidden_size].
            edge_index (torch.Tensor): Edge index matrix of shape [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix with shape [num_edges, num_edge_features].

        Returns:
            torch.Tensor: Output tensor of the same shape as the input tensor.
        """
        out = self.conv_up(self.act_fn(self.conv_down(x, edge_index)), edge_index)

        if self.skip_connection:
            out += x

        # Apply normalization if set
        if isinstance(self.normalization, nn.BatchNorm1d):
            if out.size(0) > 1:  # Apply batch normalization only if batch size is greater than 1
                out = self.normalization(out)
        elif isinstance(self.normalization, nn.LayerNorm):
            out = self.normalization(out)
        
        # Multiply by learnable scalar if set
        if self.scalar is not None:
            out = out * self.scalar
        
        return out

