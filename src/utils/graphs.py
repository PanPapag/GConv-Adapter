import json
import logging
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from typing import Dict, Union


def print_graph_statistics(stats: Dict[str, Union[int, float]], logger: logging.Logger = None):
    """
    Print the graph statistics from the dictionary. If a logger is provided,
    use the logger to print the statistics; otherwise, print to the console.

    Args:
        stats (dict): Dictionary containing graph statistics.
        logger (logging.Logger, optional): Logger to use for printing the statistics.
    """
    output = [
        f"Number of nodes: {stats['num_nodes']}",
        f"Number of edges: {stats['num_edges']}",
        f"Number of Features: {stats['num_features']}",
        f"Number of Classes: {stats['num_classes']}",
        "",
        f"Average Degree: {stats['average_degree']}",
        f"Average Neighbor Degree: {stats['average_neighbor_degree']}",
        f"Average Degree Connectivity: {stats['average_degree_connectivity']}",
        f"Average Clustering Coefficient: {stats['average_clustering_coefficient']}",
        f"Average Path Length: {stats['average_path_length']}",
        f"Density: {stats['density']}",
        f"Transitivity: {stats['transitivity']}",
        f"Diameter: {stats['diameter']}",
        "",
        f"Degree Centrality: {stats['degree_centrality']}",
        f"Eigenvector Centrality: {stats['eigenvector_centrality']}",
        f"Betweenness Centrality for Nodes: {stats['node_betweenness_centrality']}",
        f"Betweenness Centrality for Edges: {stats['edge_betweenness_centrality']}",
        f"Closeness Centrality: {stats['closeness_centrality']}",
        "",
        f"Number of Connected Components: {stats['num_connected_components']}",
        f"Size of Smallest Connected Component: {stats['size_smallest_cc']}",
        f"Size of Largest Connected Component: {stats['size_largest_cc']}",
        f"Average Size of Connected Components: {stats['average_cc_size']}",
        "",
    ]
    if logger:
        for line in output:
            logger.info(line)
    else:
        for line in output:
            print(line)


def save_graph_statistics_to_json(stats: Dict[str, Union[int, float]], filename: str):
    """
    Save the graph statistics to a JSON file.

    Args:
        stats (dict): Dictionary of graph statistics.
        filename (str): Path to the JSON file.
    """
    with open(filename, "w") as f:
        json.dump(stats, f, indent=4)


def update_data_with_perturbed_graph(
    original_data: Data, perturbed_graph: nx.Graph
) -> Data:
    """
    Update the original Data object with the perturbed graph.

    Args:
        original_data (torch_geometric.data.Data): The original Data object.
        perturbed_graph (nx.Graph): The perturbed NetworkX graph.

    Returns:
        Data: The updated Data object with the perturbed graph structure.
    """
    perturbed_data = original_data.clone()
    perturbed_data.edge_index = from_networkx(perturbed_graph).edge_index
    perturbed_data.edge_attr = from_networkx(perturbed_graph).edge_attr

    num_new_nodes = perturbed_graph.number_of_nodes() - original_data.x.size(0)

    if num_new_nodes > 0:
        new_x = torch.zeros((num_new_nodes, original_data.x.size(1)))
        new_y = -1 * torch.ones(num_new_nodes, dtype=original_data.y.dtype)
        new_train_mask = torch.zeros(num_new_nodes, dtype=torch.bool)
        new_val_mask = torch.zeros(num_new_nodes, dtype=torch.bool)
        new_test_mask = torch.zeros(num_new_nodes, dtype=torch.bool)

        perturbed_data.x = torch.cat([original_data.x, new_x], dim=0)
        perturbed_data.y = torch.cat([original_data.y, new_y], dim=0)
        perturbed_data.train_mask = torch.cat(
            [original_data.train_mask, new_train_mask], dim=0
        )
        perturbed_data.val_mask = torch.cat(
            [original_data.val_mask, new_val_mask], dim=0
        )
        perturbed_data.test_mask = torch.cat(
            [original_data.test_mask, new_test_mask], dim=0
        )

    return perturbed_data
