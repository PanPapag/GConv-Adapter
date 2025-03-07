"""
Code taken and slightly modified: https://github.com/BorgwardtLab/ggme/blob/main/src/perturbations.py
"""

from typing import Union
import networkx as nx
import numpy as np


class GraphPerturbation:
    def __init__(self, random_state: Union[int, np.random.RandomState]):
        """
        Initialize the GraphPerturbation class.

        Args:
            random_state (Union[int, np.random.RandomState]): Seed or RandomState for reproducibility.
        """
        self.random_state = self._init_random_state(random_state)

    def remove_edges(self, graph: nx.Graph, probability: float) -> nx.Graph:
        """
        Randomly remove edges from the graph.

        Args:
            graph (nx.Graph): The input graph.
            probability (float): Probability of removing each edge.

        Returns:
            nx.Graph: The graph with edges removed.
        """
        self._validate_probability(probability)
        graph = graph.copy()
        edges_to_remove = self.random_state.binomial(
            1, probability, size=graph.number_of_edges()
        )
        edge_indices_to_remove = np.where(edges_to_remove == 1)[0]
        edges = list(graph.edges())

        for edge_index in edge_indices_to_remove:
            edge = edges[edge_index]
            graph.remove_edge(*edge)

        return graph

    def add_edges(self, graph: nx.Graph, probability: float) -> nx.Graph:
        """
        Randomly add edges to the graph.

        Args:
            graph (nx.Graph): The input graph.
            probability (float): Probability of adding each edge.

        Returns:
            nx.Graph: The graph with edges added.
        """
        self._validate_probability(probability)
        graph = graph.copy()
        nodes = list(graph.nodes())
        for i, node1 in enumerate(nodes):
            nodes_to_connect = self.random_state.binomial(
                1, probability, size=len(nodes)
            )
            nodes_to_connect[i] = 0  # Never introduce self connections
            node_indices_to_connect = np.where(nodes_to_connect == 1)[0]
            for j in node_indices_to_connect:
                node2 = nodes[j]
                graph.add_edge(node1, node2)

        return graph

    def rewire_edges(self, graph: nx.Graph, probability: float) -> nx.Graph:
        """
        Randomly rewire edges in the graph.

        Args:
            graph (nx.Graph): The input graph.
            probability (float): Probability of rewiring each edge.

        Returns:
            nx.Graph: The graph with edges rewired.
        """
        self._validate_probability(probability)
        graph = graph.copy()
        edges_to_rewire = self.random_state.binomial(
            1, probability, size=graph.number_of_edges()
        )
        edge_indices_to_rewire = np.where(edges_to_rewire == 1)[0]
        edges = list(graph.edges())
        nodes = list(graph.nodes())

        for edge_index in edge_indices_to_rewire:
            edge = edges[edge_index]
            graph.remove_edge(*edge)

            # Randomly pick one of the nodes which should be detached
            if self.random_state.random() > 0.5:
                keep_node, detach_node = edge
            else:
                detach_node, keep_node = edge

            # Pick a random node besides detach node and keep node to attach to
            possible_nodes = [n for n in nodes if n not in [keep_node, detach_node]]
            attach_node = self.random_state.choice(possible_nodes)
            graph.add_edge(keep_node, attach_node)
        return graph

    @staticmethod
    def _validate_probability(p: float):
        if not 0 <= p <= 1:
            raise ValueError("Probability must be between 0 and 1")

    @staticmethod
    def _init_random_state(
        random_state: Union[int, np.random.RandomState]
    ) -> np.random.RandomState:
        if isinstance(random_state, int):
            return np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            return random_state
        else:
            raise ValueError("random_state must be an int or np.random.RandomState")
