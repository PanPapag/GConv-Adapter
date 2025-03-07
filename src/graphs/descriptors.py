"""
Code taken and augmented from: https://github.com/BorgwardtLab/ggme/blob/main/src/metrics/descriptor_functions.py
"""

import numpy as np
import networkx as nx
from typing import Tuple


def degree_distribution(G: nx.Graph, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the degree distribution of a graph.

    Args:
        G (networkx.Graph): The input graph.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The degree distribution histogram.
    """
    hist = nx.degree_histogram(G)

    if density:
        hist = np.divide(hist, np.sum(hist))

    return np.asarray(hist)


def clustering_coefficient(G: nx.Graph, n_bins: int = 200, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the clustering coefficient histogram of a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The clustering coefficient histogram.
    """
    coefficient_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        coefficient_list, bins=n_bins, range=(0.0, 1.0), density=density
    )

    return hist


def normalised_laplacian_spectrum(
    G: nx.Graph, n_bins: int = 200, bin_range: Tuple[int, int] = (0, 2), density: bool = False, **kwargs
) -> np.ndarray:
    """
    Calculate the normalized Laplacian spectrum of a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        bin_range (tuple, optional): The lower and upper range of the bins. Default is (0, 2).
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The normalized Laplacian spectrum histogram.
    """
    spectrum = nx.normalized_laplacian_spectrum(G)
    hist, _ = np.histogram(spectrum, bins=n_bins, density=density, range=bin_range)

    return hist


def neighbor_degree_distribution(G: nx.Graph, n_bins: int = 200, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the average neighbor degree distribution of a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The average neighbor degree distribution histogram.
    """
    neighbor_degree_list = list(nx.average_neighbor_degree(G).values())
    hist, _ = np.histogram(neighbor_degree_list, bins=n_bins, density=density)

    return hist


def degree_connectivity_distribution(G: nx.Graph, n_bins: int = 200, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the degree connectivity distribution of a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The degree connectivity distribution histogram.
    """
    degree_connectivity_list = list(nx.average_degree_connectivity(G).values())
    hist, _ = np.histogram(degree_connectivity_list, bins=n_bins, density=density)

    return hist


def degree_centrality_distribution(G: nx.Graph, n_bins: int = 200, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the degree centrality histogram of a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The degree centrality histogram.
    """
    centrality_list = list(nx.degree_centrality(G).values())
    hist, _ = np.histogram(centrality_list, bins=n_bins, density=density)

    return hist


def eigenvector_centrality_distribution(G: nx.Graph, n_bins: int = 200, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the eigenvector centrality histogram of a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The eigenvector centrality histogram.
    """
    centrality_list = list(nx.eigenvector_centrality_numpy(G).values())
    hist, _ = np.histogram(centrality_list, bins=n_bins, density=density)

    return hist


def closeness_centrality_distribution(G: nx.Graph, n_bins: int = 200, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the closeness centrality histogram of a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The closeness centrality histogram.
    """
    centrality_list = list(nx.closeness_centrality(G).values())
    hist, _ = np.histogram(centrality_list, bins=n_bins, density=density)

    return hist


def path_length_distribution(G: nx.Graph, n_bins: int = 200, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the path length distribution of a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The path length distribution histogram.
    """
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    path_lengths = []
    for _, target_lengths in lengths.items():
        path_lengths.extend(target_lengths.values())

    max_length = max(path_lengths)
    n_bins = max_length + 1

    hist, _ = np.histogram(
        path_lengths, bins=n_bins, density=density, range=(0, max_length + 1)
    )

    return hist


def connected_components_size_distribution(G: nx.Graph, n_bins: int = 200, density: bool = False, **kwargs) -> np.ndarray:
    """
    Calculate the size distribution of connected components in a graph.

    Args:
        G (networkx.Graph): The input graph.
        n_bins (int, optional): Number of bins for the histogram.
        density (bool, optional): If True, the histogram is normalized to form a probability density.
        **kwargs: Additional arguments for customization (not used).

    Returns:
        np.ndarray: The size distribution histogram of connected components.
    """
    component_sizes = [len(c) for c in nx.connected_components(G)]
    hist, _ = np.histogram(component_sizes, bins=n_bins, density=density)

    return hist
