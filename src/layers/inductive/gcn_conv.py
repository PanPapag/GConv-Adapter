
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

NUM_BOND_TYPE = 6 
NUM_BOND_DIRECTION = 3 

class MolecularGCNConv(MessagePassing):
    """
    Graph Convolutional Network (GCN) layer with edge feature integration.
    Designed for molecular property prediction in chemistry.

    Args:
        emb_dim (int): Dimensionality of embeddings.
        aggr (str, optional): Aggregation method ('add', 'mean', 'max'). Default is 'add'.
    """
    def __init__(self, emb_dim, aggr="add"):
        super(MolecularGCNConv, self).__init__(aggr=aggr)
        self.aggr = aggr

        self.emb_dim = emb_dim
        
        # Linear transformation layer
        self.linear = nn.Linear(emb_dim, emb_dim)
        
        # Embeddings for edge attributes
        self.edge_embedding_type = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding_direction = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        
        # Initialize edge embeddings
        torch.nn.init.xavier_uniform_(self.edge_embedding_type.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding_direction.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        """
        Compute normalization coefficients for edges.

        Args:
            edge_index (torch.Tensor): Edge indices.
            num_nodes (int): Number of nodes.
            dtype (torch.dtype): Data type.

        Returns:
            torch.Tensor: Normalization coefficients.
        """
        device = edge_index[0].device
        edge_weight = torch.ones((edge_index[0].size(1),), dtype=dtype, device=device)
        row, col = edge_index[0]
        degree = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=device)
        degree.scatter_add_(0, row, edge_weight)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

        return degree_inv_sqrt[row] * edge_weight * degree_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GCN layer.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
            torch.Tensor: Updated node features.
        """
        # Add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        # Add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 # Bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding_type(edge_attr[:,0]) + self.edge_embedding_direction(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        """
        Message computation for GCN layer.

        Args:
            x_j (torch.Tensor): Node features of neighboring nodes.
            edge_attr (torch.Tensor): Edge attributes.
            norm (torch.Tensor): Normalization coefficients.

        Returns:
            torch.Tensor: Computed messages.
        """
        return norm.view(-1, 1) * (x_j + edge_attr)