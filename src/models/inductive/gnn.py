import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from hydra.utils import instantiate

from src.layers.inductive.gin_conv import MolecularGINConv
from src.layers.inductive.gcn_conv import MolecularGCNConv
from src.layers.inductive.gat_conv import MolecularGATConv
from src.layers.inductive.sage_conv import MolecularGraphSAGEConv

NUM_ATOM_TYPE = 120 
NUM_CHIRALITY_TAG = 3


class MolecularGNN(torch.nn.Module):
    """
    General Graph Neural Network (GNN) model for molecular property prediction.

    Args:
        num_layers (int): Number of GNN layers.
        emb_dim (int): Dimensionality of embeddings.
        jk_mode (str): Jumping Knowledge mode: 'last', 'concat', 'max', or 'sum'.
        dropout_rate (float): Dropout rate.
        gnn_type (str): Type of GNN: 'gin', 'gcn', 'graphsage', 'gat'.

    Output:
        torch.Tensor: Node representations.
    """
    def __init__(self, num_layers, emb_dim, jk_mode="last", dropout_rate=0, gnn_type="gin"):
        super(MolecularGNN, self).__init__()
        
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.jk_mode = jk_mode
        self.dropout_rate = dropout_rate

        # List of GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gin":
                self.gnn_layers.append(MolecularGINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnn_layers.append(MolecularGCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnn_layers.append(MolecularGATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnn_layers.append(MolecularGraphSAGEConv(emb_dim))
            else:
                raise ValueError("Invalid GNN type specified")

        # List of batch normalization layers
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(self.num_layers)])

        # Initialize adapters as None
        self.pre_adapters = nn.ModuleDict()
        self.post_adapters = nn.ModuleDict()
        self.adapter_type = None

    def add_adapter(self, adapter_class, position, type):
        """
        Add an adapter module to the model at the specified position.

        Args:
            adapter_class (Union[nn.Module, str]): The adapter class to be added. If a string is provided, it should be the full
                                                path to the module.
            position (str): Position to insert the adapter. Valid options are:
                            'pre' - Insert adapter before each GNN layer.
                            'post' - Insert adapter after each GNN layer.

            type (str, optional): The type of insertion of the adapters ('sequential' or 'parallel'). Defaults to 'sequential'.
        """
        self.adapter_type = type 
        if position == 'pre':
            for layer in range(self.num_layers):
                self.pre_adapters[str(layer)] = instantiate(adapter_class, hidden_size=self.emb_dim)
        elif position == 'post':
            for layer in range(self.num_layers):
                self.post_adapters[str(layer)] = instantiate(adapter_class, hidden_size=self.emb_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GNN model.

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, num_node_features].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix with shape [num_edges, num_edge_features].

        Returns:
            torch.Tensor: Node representations.
        """
        hidden_states = [x]
        for layer in range(self.num_layers):
            layer_str = str(layer)  # Ensure consistent string conversion

            # Pre-adapter processing
            pre_adapter_output = None
            if layer_str in self.pre_adapters:
                pre_adapter_output = self.pre_adapters[layer_str](hidden_states[layer], edge_index, edge_attr)
                if self.adapter_type == "sequential":
                    hidden_states[layer] = pre_adapter_output

            # Main GNN layer processing
            h = self.gnn_layers[layer](hidden_states[layer], edge_index, edge_attr)
            if h.size(0) > 1:  # Apply BatchNorm only if batch size > 1
                h = self.batch_norm_layers[layer](h)

            # Post-adapter processing
            post_adapter_output = None
            if layer_str in self.post_adapters:
                post_adapter_output = self.post_adapters[layer_str](h, edge_index, edge_attr)
                if self.adapter_type == "sequential":
                    h = post_adapter_output

            # Handle parallel mode explicitly and consistently
            if self.adapter_type == "parallel":
                if pre_adapter_output is not None:
                    h = h + pre_adapter_output
                if post_adapter_output is not None:
                    h = h + post_adapter_output

            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout_rate, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout_rate, training=self.training)
            hidden_states.append(h)

        # Different implementations of Jumping Knowledge (JK)
        if self.jk_mode == "concat":
            node_representation = torch.cat(hidden_states, dim=1)
        elif self.jk_mode == "last":
            node_representation = hidden_states[-1]
        elif self.jk_mode == "max":
            hidden_states = [h.unsqueeze(0) for h in hidden_states]
            node_representation = torch.max(torch.cat(hidden_states, dim=0), dim=0)[0]
        elif self.jk_mode == "sum":
            hidden_states = [h.unsqueeze(0) for h in hidden_states]
            node_representation = torch.sum(torch.cat(hidden_states, dim=0), dim=0)[0]
        else:
            raise ValueError("Invalid JK mode specified")

        return node_representation
    

class MolecularGraphPredictionModel(torch.nn.Module):
    """
    General Graph Prediction Model for molecule function prediction, capable of extending various GNN types

    Args:
        num_layers (int): The number of GNN layers.
        emb_dim (int): Dimensionality of embeddings.
        num_tasks (int): Number of tasks in a multi-task learning scenario.
        dropout_rate (float): Dropout rate.
        jk_mode (str): Jumping Knowledge mode: 'last', 'concat', 'max', or 'sum'.
        pooling_method (str): Graph pooling method: 'sum', 'mean', 'max', 'attention', 'set2set'.
        gnn_type (str): Type of GNN: 'gin', 'gcn', 'graphsage', 'gat'.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    def __init__(self, num_layers, emb_dim, num_tasks, jk_mode="last", dropout=0, pooling_method="mean", gnn_type="gin", device="cpu"):
        super(MolecularGraphPredictionModel, self).__init__()
        
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.jk_mode = jk_mode
        self.dropout = dropout
        self.device = device

        # Node feature embeddings
        self.node_embedding_type = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.node_embedding_chirality = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        torch.nn.init.xavier_uniform_(self.node_embedding_type.weight.data)
        torch.nn.init.xavier_uniform_(self.node_embedding_chirality.weight.data)

        # Initialize the GNN with the specified parameters
        self.gnn = MolecularGNN(num_layers, emb_dim, jk_mode, dropout, gnn_type=gnn_type)

        # Define the graph pooling method
        pooling_methods = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool
        }

        # Select the appropriate pooling method
        if pooling_method in pooling_methods:
            self.pool = pooling_methods[pooling_method]
        elif pooling_method == "attention":
            # Global attention pooling
            gate_nn_input_dim = (num_layers + 1) * emb_dim if jk_mode == "concat" else emb_dim
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(gate_nn_input_dim, 1))
        elif pooling_method.startswith("set2set"):
            # Set2Set pooling
            set2set_iterations = int(pooling_method[-1])
            s2s_input_dim = (num_layers + 1) * emb_dim if jk_mode == "concat" else emb_dim
            self.pool = Set2Set(s2s_input_dim, set2set_iterations)
        else:
            raise ValueError("Invalid graph pooling type.")
        
        # Determine the multiplier for the pooling output
        self.pool_output_multiplier = 2 if pooling_method.startswith("set2set") else 1
        
        # Define the final linear layer 
        pred_input_dim = self.pool_output_multiplier * (num_layers + 1) * emb_dim if jk_mode == "concat" else self.pool_output_multiplier * emb_dim
        self.graph_pred_linear = torch.nn.Linear(pred_input_dim, self.num_tasks)
    
    def add_adapter(self, adapter_class, positions, type='sequential'):
        """
        Add an adapter module to the model.

        Args:
            adapter_class (nn.Module): The adapter class to be added. 
            positions (List[str]): List of positions to insert the adapter. Valid options are:
                                    'pre' - Insert adapter before each GNN layer.
                                    'post' - Insert adapter after each GNN layer.
            type (str, optional): The type of insertion of the adapters ('sequential' or 'parallel'). Defaults to 'sequential'.
        """
        for position in positions:
            self.gnn.add_adapter(adapter_class, position, type)
        
        # Freeze all parameters apart from Adapter Layers, Gating Parameters, BatchNorm/LayerNorm and the Final Prediction Layer 
        for name, module in self.named_modules():
            for _, param in module.named_parameters(recurse=False):
                if 'adapter' in name:
                    param.requires_grad = True
                elif 'graph_pred_linear' in name:
                    param.requires_grad = True
                elif isinstance(module, nn.LayerNorm):
                    param.requires_grad = True
                elif (isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d)):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def print_all_learnable_scalars(self):
        """
        Print the learnable scalar for each adapter in the model, if present.
        """
        for position, adapters in [('pre', self.gnn.pre_adapters)]:
            for layer_id, adapter in adapters.items():
                if hasattr(adapter, 'get_learnable_scalar'):
                    print(f"{position.capitalize()}-Adapter at Layer {int(layer_id) + 1}: {adapter.get_learnable_scalar()}")
        print()
        for position, adapters in [('post', self.gnn.post_adapters)]:
            for layer_id, adapter in adapters.items():
                if hasattr(adapter, 'get_learnable_scalar'):
                    print(f"{position.capitalize()}-Adapter at Layer {int(layer_id) + 1}: {adapter.get_learnable_scalar()}")
                
           
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, num_node_features].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix with shape [num_edges, num_edge_features].
            batch (torch.Tensor): Batch vector, which assigns each node to a specific example in the batch with shape [num_nodes].

        Returns:
            torch.Tensor: The output predictions of the model.
        """
        # Compute initial node embeddings
        x = self.node_embedding_type(x[:, 0]) + self.node_embedding_chirality(x[:, 1])

        # Get node representations from the GNN
        node_representations = self.gnn(x, edge_index, edge_attr)

        # Apply the pooling method to get graph-level representation
        pooled_representation = self.pool(node_representations, batch)

        # Use only the pooled representation
        graph_representation = pooled_representation

        # Pass the graph representation through the final linear layer
        return self.graph_pred_linear(graph_representation)