import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

NUM_BOND_TYPE = 6 
NUM_BOND_DIRECTION = 3 

class MolecularGINConv(MessagePassing):
    """
    Molecular Graph Isomorphism Network (GIN) layer with edge feature integration.
    Designed for molecular property prediction in chemistry.

    Args:
        emb_dim (int): Dimensionality of embeddings for nodes and edges.
        aggr (str, optional): Aggregation method ('add', 'mean', 'max'). Default is 'add'.
    """
    def __init__(self, emb_dim, aggr="add"):
        super(MolecularGINConv, self).__init__(aggr=aggr)
        
        self.aggr = aggr

        # Multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        
        # Embeddings for edge attributes
        self.edge_embedding_type = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding_direction = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        
        # Initialize edge embeddings
        torch.nn.init.xavier_uniform_(self.edge_embedding_type.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding_direction.weight.data)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the Molecular GIN layer.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
            torch.Tensor: Updated node features.
        """
        # Add self-loops to the edge index
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # Add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4  # Bond type for self-loop edge
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        edge_embeddings = self.edge_embedding_type(edge_attr[:, 0]) + self.edge_embedding_direction(edge_attr[:, 1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
    
    def message(self, x_j, edge_attr):
        """
        Message computation for Molecular GIN layer.

        Args:
            x_j (torch.Tensor): Node features of neighboring nodes.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
            torch.Tensor: Computed messages.
        """
        return x_j + edge_attr

    def update(self, aggr_out):
        """
        Update function for Molecular GIN layer.

        Args:
            aggr_out (torch.Tensor): Aggregated node features.

        Returns:
            torch.Tensor: Updated node features after applying MLP.
        """
        return self.mlp(aggr_out)
