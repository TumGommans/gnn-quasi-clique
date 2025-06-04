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

    def initialize_synthetic(self, edges, zero_indexed=True):
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
        """Considering node_features are built on sorted vertices, need to map the edge indices the same."""
        vertices_sorted = sorted(self._vertices)
        id_2_idx = {v:i for i,v in enumerate(vertices_sorted)}
        return [
            [id_2_idx[u] for u in self._edge_index[0]],
            [id_2_idx[v] for v in self._edge_index[1]],
        ]

    def load_from_edgelist_file(self, filepath):
        """
        Loads graph data from an edge list file. Supports two formats:
        1. Direct format: "node1 node2" per line (e.g., brock200-2.txt)
        2. DIMACS format: "e node1 node2" per line (e.g., p_hat300-1.txt)
        
        Skips malformed lines and self-loops.
        Builds adjacency list and edge_index (0-based).
        """
        self._adjacency_list.clear()
        self._vertices.clear()
        self._edge_index = [[], []]
        self._num_edges = 0
        self._core_numbers = None  # Reset cache
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
            
        print(f"Graph loaded: {self.num_vertices} vertices, {self.num_edges} edges, Density: {self.density:.3f}")

    load_from_csv = load_from_edgelist_file

    def add_edge(self, u, v):
        """Adds an undirected edge between u and v."""
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

    def get_neighbors(self, u):
        return self._adjacency_list.get(u, set())

    def has_edge(self, u, v):
        return v in self._adjacency_list.get(u, set())

    def get_induced_subgraph_edges(self, subset):
        count = 0
        subset_list = list(subset)
        for i in range(len(subset_list)):
            for j in range(i + 1, len(subset_list)):
                u = subset_list[i]
                v = subset_list[j]
                if v in self._adjacency_list.get(u, set()):
                    count += 1
        return count

    def clustering_coefficient(self, u):
        """
        Computes the clustering coefficient for node u:
        (# of links between neighbors) / (k * (k - 1) / 2)
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

    def compute_core_numbers(self):
        """
        Computes the core number (k-core) for each vertex using the standard algorithm.
        Returns a dictionary mapping vertex -> core number.
        
        The core number of a vertex is the maximum k such that the vertex exists in a k-core,
        where a k-core is a maximal subgraph in which each vertex has at least k neighbors.
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

    def get_degree_into_subset(self, node, subset):
        """
        Computes the degree of a node into a given subset of vertices.
        
        Args:
            node: The node for which to compute the degree into subset
            subset: Set/list of vertices defining the subset
        """
        if node not in self._vertices:
            return 0
        
        subset_set = set(subset) if not isinstance(subset, set) else subset
        neighbors = self._adjacency_list.get(node, set())
        
        degree_into_subset = len(neighbors.intersection(subset_set))
        return degree_into_subset

    def _bfs_multi_source(self, subset):
        """
        Performs a multi-source BFS from all nodes in subset.
        Returns a dict node -> shortest distance to any subset node.
        Unreachable nodes are omitted (or can be treated as infinite).
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

    def get_node_features(self, subset, k, gamma):
        """
        Returns a 2D list where each row contains augmented features for each node:
        [degree, degree_into_subset, clustering_coeff, core_number, dist_to_subset, density, k, gamma]
        
        Args:
            subset: iterable of nodes defining the target set for shortest-path features
            k: parameter for the quasi-clique size
            gamma: parameter for the quasi-clique density
        """
        core_numbers = self.compute_core_numbers()
        distances = self._bfs_multi_source(subset)
        
        subset_set = set(subset) if not isinstance(subset, set) else subset
        
        features = []
        for u in sorted(self._vertices):
            degree = len(self._adjacency_list[u])
            clustering = self.clustering_coefficient(u)
            dist = distances.get(u, self.num_vertices)
            core_number = core_numbers.get(u, 0)
            degree_into_subset = self.get_degree_into_subset(u, subset_set)
            
            # Combine all features
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

def generate_synthetic_graph():
    """
    Generates a synthetic Graph instance with varying properties.
    Each call produces graphs with different numbers of nodes and sparseness levels.
    
    Returns:
        Graph: A synthetic graph instance with random structure
    """
    num_nodes = random.randint(300, 1000)
    graph_type = random.choice(['sparse', 'medium', 'scale_free'])
    
    if graph_type == 'sparse':
        edge_prob = random.uniform(0.01, 0.05)
        edges = _generate_erdos_renyi_edges(num_nodes, edge_prob)
    
    elif graph_type == 'medium':
        edge_prob = random.uniform(0.1, 0.3)
        edges = _generate_erdos_renyi_edges(num_nodes, edge_prob)
    
    else:
        edges = _generate_scale_free_graph(num_nodes)
    
    zero_indexed = random.choice([True, False])
    
    # Create and initialize the graph
    graph = Graph()
    graph.initialize_synthetic(edges, zero_indexed)
    
    return graph

def _generate_erdos_renyi_edges(num_nodes, edge_prob):
    """Generate edges for an Erdos Renyi random graph."""
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                edges.append((i, j))
    return edges

def _generate_scale_free_graph(num_nodes):
    """Generate a scale-free network using preferential attachment."""
    if num_nodes < 3:
        return [(0, 1)] if num_nodes == 2 else []
    
    edges = []
    degrees = defaultdict(int)
    
    initial_nodes = min(3, num_nodes)
    for i in range(initial_nodes):
        for j in range(i + 1, initial_nodes):
            edges.append((i, j))
            degrees[i] += 1
            degrees[j] += 1
    
    for new_node in range(initial_nodes, num_nodes):
        m = random.randint(1, min(3, new_node))
    
        existing_nodes = list(range(new_node))
        if not existing_nodes:
            continue
            
        total_degree = sum(degrees[node] for node in existing_nodes)
        if total_degree == 0:
            targets = random.sample(existing_nodes, min(m, len(existing_nodes)))
        else:
            probs = [degrees[node] / total_degree for node in existing_nodes]
            targets = np.random.choice(existing_nodes, size=min(m, len(existing_nodes)), 
                                     replace=False, p=probs)
        
        for target in targets:
            edges.append((new_node, target))
            degrees[new_node] += 1
            degrees[target] += 1
    
    return edges