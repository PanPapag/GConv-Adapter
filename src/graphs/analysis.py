import networkx as nx
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Union


def calculate_graph_statistics(G: nx.Graph, data: Data) -> Dict[str, Union[int, float]]:
    """
    Calculate various statistics of a single graph.

    Args:
        G (nx.Graph): The graph.
        data (torch_geometric.data.Data): The PyTorch Geometric data object.

    Returns:
        dict: Dictionary containing various graph statistics.
    """
    stats = {}
    # Basic Graph Statistics
    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    stats["num_features"] = data.x.shape[1]
    stats["num_classes"] = len(np.unique(data.y.numpy()))

    # Structural Properties
    stats["average_degree"] = np.mean([degree for _, degree in G.degree()])
    stats["average_neighbor_degree"] = np.mean(
        list(nx.average_neighbor_degree(G).values())
    )
    stats["average_degree_connectivity"] = np.mean(
        list(nx.average_degree_connectivity(G).values())
    )
    stats["density"] = nx.density(G)
    stats["average_clustering_coefficient"] = nx.average_clustering(G)
    stats["diameter"] = (
        nx.diameter(G) if nx.is_connected(G) else "Graph is not connected"
    )
    stats["average_path_length"] = (
        nx.average_shortest_path_length(G)
        if nx.is_connected(G)
        else "Graph is not connected"
    )
    stats["transitivity"] = nx.transitivity(G)

    # Centrality Measures
    stats["degree_centrality"] = np.mean(list(nx.degree_centrality(G).values()))
    stats["eigenvector_centrality"] = np.mean(
        list(nx.eigenvector_centrality(G).values())
    )
    stats["node_betweenness_centrality"] = np.mean(
        list(nx.betweenness_centrality(G).values())
    )
    stats["edge_betweenness_centrality"] = np.mean(
        list(nx.edge_betweenness_centrality(G).values())
    )
    stats["closeness_centrality"] = np.mean(list(nx.closeness_centrality(G).values()))

    # Connectivity measures
    stats["num_connected_components"] = nx.number_connected_components(G)
    connected_components_sizes = [len(c) for c in nx.connected_components(G)]
    stats["size_smallest_cc"] = min(connected_components_sizes)
    stats["size_largest_cc"] = max(connected_components_sizes)
    stats["average_cc_size"] = np.mean(connected_components_sizes)

    return stats
