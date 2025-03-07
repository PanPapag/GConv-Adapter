import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

NUM_BOND_TYPE = 6 
NUM_BOND_DIRECTION = 3 

class MolecularGraphSAGEConv(MessagePassing):
    """
    Molecular GraphSAGE layer with edge feature integration.
    Designed for molecular property prediction in chemistry.

    Args:
        emb_dim (int): Dimensionality of embeddings.
        aggr (str, optional): Aggregation method ('mean', 'add', 'max'). Default is 'mean'.
    """
    def __init__(self, emb_dim, aggr = "mean"):
        super(MolecularGraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding_type = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding_direction = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding_type.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding_direction.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the Molecular GraphSAGE layer.

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

        edge_embeddings = self.edge_embedding_type(edge_attr[:, 0]) + self.edge_embedding_direction(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        """
        Message computation for Molecular GraphSAGE layer.

        Args:
            x_j (torch.Tensor): Node features of neighboring nodes.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
        """            
        return x_j + edge_attr

    def update(self, aggr_out):
        """
        Update function for Molecular GraphSAGE layer.

        Args:
            aggr_out (torch.Tensor): Aggregated node features.

        Returns:
            torch.Tensor: Normalized node features.
        """
        return F.normalize(aggr_out, p = 2, dim = -1)