import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F

NUM_BOND_TYPE = 6 
NUM_BOND_DIRECTION = 3 

class MolecularGATConv(MessagePassing):
    """
    Molecular Graph Attention Network (GAT) layer with edge feature integration.
    Designed for molecular property prediction in chemistry.

    Args:
        emb_dim (int): Dimensionality of embeddings.
        heads (int, optional): Number of attention heads. Default is 2.
        negative_slope (float, optional): Negative slope for LeakyReLU. Default is 0.2.
        aggr (str, optional): Aggregation method ('add', 'mean', 'max'). Default is 'add'.
    """
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(MolecularGATConv, self).__init__(aggr=aggr)

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding_type = torch.nn.Embedding(NUM_BOND_TYPE, heads * emb_dim)
        self.edge_embedding_direction = torch.nn.Embedding(NUM_BOND_DIRECTION, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding_type.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding_direction.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters to their initial values.
        """
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the Molecular GAT layer.

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
        self_loop_attr[:, 0] = 4 # Bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding_type(edge_attr[:, 0]) + self.edge_embedding_direction(edge_attr[:, 1])

        x = self.weight_linear(x)
        
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        """
        Message computation for Molecular GAT layer.

        Args:
            edge_index (torch.Tensor): Edge indices.
            x_i (torch.Tensor): Source node features.
            x_j (torch.Tensor): Target node features.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
            torch.Tensor: Computed messages.
        """
        edge_attr = edge_attr
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1).view(-1, self.heads, 2 * self.emb_dim) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return (x_j.view(-1, self.heads, self.emb_dim) * alpha.unsqueeze(-1)).view(-1, self.heads * self.emb_dim)


    def update(self, aggr_out):
        """
        Update function for Molecular GAT layer.

        Args:
            aggr_out (torch.Tensor): Aggregated node features.

        Returns:
            torch.Tensor: Updated node features after applying the bias term.
        """
        aggr_out = aggr_out.view(-1, self.heads, self.emb_dim).mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out