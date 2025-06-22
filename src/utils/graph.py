"""Custom graph object."""

import random
import numpy as np
from collections import defaultdict, deque

class Graph:
    """
    Represents an undirected graph, loaded from an edge list file.
    Nodes are expected to be integers or convertible to integers.
    Provides methods for edge management and feature extraction.
    """
    def __init__(self):
        """Construct a new graph instance."""
        self._adjacency_list = defaultdict(set)
        self._vertices = set()
        self._edge_index = [[], []]
        self._num_edges = 0
        self._zero_indexed = False
        self._core_numbers = None
        self._distances = {}

    def initialize_synthetic(
        self, 
        edges: list[tuple[int]], 
        zero_indexed: bool = True
    ):
        """
        Initialize the graph with synthetic edge data.
        
        Args:
            edges: List of tuples (u, v) representing edges
            zero_indexed: Boolean indicating if nodes are 0-indexed or 1-indexed
        """
        # Clear existing data
        self._adjacency_list.clear()
        self._vertices.clear()
        self._edge_index = [[], []]
        self._num_edges = 0
        self._core_numbers = None
        self._zero_indexed = zero_indexed
        
        # Add all edges
        for u, v in edges:
            if u != v:  # Skip self-loops
                # Adjust indices if not zero-indexed
                if not zero_indexed:
                    u_adj, v_adj = u, v
                else:
                    u_adj, v_adj = u, v
                
                # Add edge if not already present
                if v_adj not in self._adjacency_list[u_adj]:
                    self._adjacency_list[u_adj].add(v_adj)
                    self._adjacency_list[v_adj].add(u_adj)
                    self._vertices.update([u_adj, v_adj])
                    self._num_edges += 1
                    self._edge_index[0].append(u_adj)
                    self._edge_index[1].append(v_adj)
        
        print(f"Synthetic graph initialized: {self.num_vertices} vertices, {self.num_edges} edges, Density: {self.density:.3f}")

    @property
    def vertices(self):
        return self._vertices

    @property
    def num_vertices(self):
        return len(self._vertices)

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def density(self):
        n = self.num_vertices
        if n < 2:
            return 0.0
        max_edges = n * (n - 1) / 2
        return self.num_edges / max_edges if max_edges > 0 else 0.0

    @property
    def edge_index(self):
        # Considering node_features are built on sorted vertices, need to map the edge indices the same.
        vertices_sorted = sorted(self._vertices)
        id_2_idx = {v:i for i,v in enumerate(vertices_sorted)}
        return [
            [id_2_idx[u] for u in self._edge_index[0]],
            [id_2_idx[v] for v in self._edge_index[1]],
        ]

    def load_from_edgelist_file(self, filepath: str):
        """
        Loads graph data from an edge list file and extracts features. Supports two formats:
        1. Direct format: "node1 node2" per line (e.g., brock200-2.txt)
        2. DIMACS format: "e node1 node2" per line (e.g., p_hat300-1.txt)
        
        Skips malformed lines and self-loops.
        Builds adjacency list and edge_index (0-based).

        args:
            filepath: location at which the edge list is located
        """
        self._adjacency_list.clear()
        self._vertices.clear()
        self._edge_index = [[], []]
        self._num_edges = 0
        self._core_numbers = None
        self._cluster_coeffs = {}
        line_num = 0

        checked_indexing = False
        try:
            with open(filepath, 'r') as infile:
                for line in infile:
                    line_num += 1
                    parts = line.split()
                    
                    # Skip empty lines
                    if not parts:
                        continue
                    
                    # Handle different formats
                    if len(parts) >= 2:
                        try:
                            if parts[0] == 'e' and len(parts) >= 3:
                                u, v = int(parts[1]), int(parts[2])
                            elif parts[0] != 'e':
                                u, v = int(parts[0]), int(parts[1])
                            else:
                                continue
                            
                            if not checked_indexing:
                                if u == 0 or v == 0:
                                    self._zero_indexed = True
                                checked_indexing = True
                            
                            if u != v:
                                self.add_edge(u, v)
                                
                        except ValueError:
                            print(f"Warning (Line {line_num}): non-integer nodes skipped: '{line.strip()}'")
                        except Exception as e:
                            print(f"Warning (Line {line_num}): error '{e}' for line: '{line.strip()}' and file {filepath}")
                            
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            raise
        except Exception as e:
            print(f"Error loading graph: {e}")
            raise

        self._core_numbers = self.compute_core_numbers()
        for u in sorted(self._vertices):
            self._cluster_coeffs[u] = self.clustering_coefficient(u)
        
        print(f"Graph loaded: {self.num_vertices} vertices, {self.num_edges} edges, Density: {self.density:.3f}")

    def add_edge(
        self, 
        u: int, 
        v: int
    ):
        """Adds an undirected edge between u and v.
        
        args:
            u: one of the nodes participating in the edge
            v: one of the nodes participating in the edge
        """
        if self._zero_indexed:
            u0, v0 = u, v
        else:
            u0, v0 = u - 1, v - 1

        if v0 not in self._adjacency_list[u0]:
            self._adjacency_list[u0].add(v0)
            self._adjacency_list[v0].add(u0)
            self._vertices.update([u0, v0])
            self._num_edges += 1
            self._edge_index[0].append(u0)
            self._edge_index[1].append(v0)
            self._core_numbers = None

    def get_neighbors(self, u: int) -> set[int]:
        """
        Returns the neighborhood of a node
        
        args:
            u: the node for which the N(u) is extracted
        """
        return self._adjacency_list.get(u, set())

    def has_edge(self, u: int, v: int) -> bool:
        """
        Checkes whether an edge is present between two nodes.
        
        args:
            u: one of the nodes potentially participating in the edge
            v: one of the nodes potentially participating in the edge
        
        returns:
            bool: whether edge (u,v) is present
        """
        return v in self._adjacency_list.get(u, set())

    def get_induced_subgraph_edges(
        self, 
        subset: set[int]
    ) -> int:
        """
        Compute the number of edges in the induced subgraph.
        
        args:
            subset: the vertices of the induced subgraph
        
        returns:
            count: the number of edges in the induced subgraph
        """
        count = 0
        subset_list = list(subset)
        for i in range(len(subset_list)):
            for j in range(i + 1, len(subset_list)):
                u = subset_list[i]
                v = subset_list[j]
                if v in self._adjacency_list.get(u, set()):
                    count += 1
        return count

    def clustering_coefficient(self, u: int) -> float:
        """
        Computes the clustering coefficient: (# of links between neighbors) / (d(u) * (d(u) - 1) / 2).

        args:
            u: node for which to compute C(u)
        
        returns:
            float: value of C(u) according to Definition 4 in the thesis
        """
        neighbors = self._adjacency_list.get(u, set())
        k = len(neighbors)
        if k < 2:
            return 0.0
        links = 0
        # count each edge once
        for v in neighbors:
            for w in neighbors:
                if v < w and w in self._adjacency_list.get(v, set()):
                    links += 1
        return (2 * links) / (k * (k - 1))

    def compute_core_numbers(self) -> dict[int, int]:
        """
        Computes the core number k-core for each vertex using the standard algorithm.
        
        The core number of a vertex is the maximum k such that the vertex exists in a k-core,
        where a k-core is a maximal subgraph in which each vertex has at least k neighbors.

        returns:
            core_numbers: dictionary mapping each vertex in the graph to its core number
        """
        if self._core_numbers is not None:
            return self._core_numbers
        core_numbers = {}
        degrees = {}
        
        for v in self._vertices:
            degrees[v] = len(self._adjacency_list[v])
            core_numbers[v] = degrees[v]
        
        vertices_by_degree = sorted(self._vertices, key=lambda v: degrees[v])
        processed = set()
        
        for v in vertices_by_degree:
            if v in processed:
                continue

            current_core = min(degrees[v], core_numbers[v])
            core_numbers[v] = current_core
            processed.add(v)
            for neighbor in self._adjacency_list[v]:
                if neighbor not in processed:
                    degrees[neighbor] = max(0, degrees[neighbor] - 1)
                    core_numbers[neighbor] = min(core_numbers[neighbor], max(current_core, degrees[neighbor]))
        
        self._core_numbers = core_numbers
        return core_numbers

    def get_degree_into_subset(
        self, 
        node: int, 
        subset: set[int]
    ) -> int:
        """
        Computes the degree of a node into a given subset of vertices.
        
        args:
            node: The node for which to compute the degree into subset
            subset: set of vertices defining the subset
        
        returns:
            degree_into_subset: the degree of the node wrt the subset
        """
        if node not in self._vertices:
            return 0
        
        subset_set = set(subset) if not isinstance(subset, set) else subset
        neighbors = self._adjacency_list.get(node, set())
        
        degree_into_subset = len(neighbors.intersection(subset_set))
        return degree_into_subset

    def _bfs_multi_source(self, subset: set[int]) -> dict[int, int]:
        """
        Computes the shortest paths for all nodes into the subset, using
        a multi-source BFS (breadth-first search)

        args:
            subset: the subset to which the distances are related
        
        returns:
            distances: dictionary mapping a node to its shortest path to any node in subset
        """
        distances = {}
        queue = deque()
        for u in subset:
            if u in self._vertices and u not in distances:
                distances[u] = 0
                queue.append(u)
        while queue:
            u = queue.popleft()
            for v in self._adjacency_list.get(u, set()):
                if v not in distances:
                    distances[v] = distances[u] + 1
                    queue.append(v)
        return distances

    def get_node_features(self, subset: set[int], k: int, gamma: float) -> list[list[float]]:
        """
        Extract the node feature matrix at a certain state in the algortithms.

        args:
            subset: iterable of nodes defining the target set for shortest-path features
            k: parameter for the quasi-clique size
            gamma: parameter for the quasi-clique density
        
        returns:
            features: the node feature matrix X
        """
        distances = self._bfs_multi_source(subset)
        subset_set = set(subset) if not isinstance(subset, set) else subset
        
        features = []
        for u in sorted(self._vertices):
            degree = len(self._adjacency_list[u])
            clustering = self._cluster_coeffs.get(u, 0)
            dist = distances.get(u, self.num_vertices)
            core_number = self._core_numbers.get(u, 0)
            degree_into_subset = self.get_degree_into_subset(u, subset_set)
            
            # Combine all features into one row
            features.append([
                degree,
                degree_into_subset,
                core_number,
                clustering,  
                dist,        
                self.density,
                k,          
                gamma,
            ])
        
        return features
