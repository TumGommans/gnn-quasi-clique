# graph.py
import random
from collections import defaultdict

class Graph:
    """
    Represents an undirected graph, loaded from an edge list file.
    Handles files where each line represents an edge with two node numbers
    separated by whitespace (e.g., space or tab).
    Nodes are expected to be integers or convertible to integers.
    """
    def __init__(self):
        self._adjacency_list = defaultdict(set)
        self._vertices = set()
        self._num_edges = 0

    @property
    def vertices(self):
        """Returns the set of vertices in the graph."""
        return self._vertices

    @property
    def num_vertices(self):
        """Returns the number of vertices in the graph."""
        return len(self._vertices)

    @property
    def num_edges(self):
        """Returns the number of edges in the graph."""
        return self._num_edges

    @property
    def density(self):
        """Calculates the density of the graph."""
        n = self.num_vertices
        if n < 2:
            return 0.0
        max_possible_edges = n * (n - 1) / 2
        return self.num_edges / max_possible_edges if max_possible_edges > 0 else 0.0

    # Updated loading function:
    def load_from_edgelist_file(self, filepath):
        """
        Loads graph data from an edge list file (e.g., .txt).
        Expects each line to contain two node IDs separated by whitespace.
        Skips lines that do not conform to this format or contain non-integer IDs.

        Args:
            filepath (str): Path to the edge list file.
        """
        self._adjacency_list = defaultdict(set)
        self._vertices = set()
        self._num_edges = 0
        line_num = 0
        try:
            with open(filepath, 'r') as infile:
                for line in infile:
                    line_num += 1
                    parts = line.split() # Split by any whitespace
                    if len(parts) >= 2:
                        try:
                            # Attempt to convert the first two parts to integers
                            u, v = int(parts[0]), int(parts[1])
                            if u != v: # Avoid self-loops
                                self.add_edge(u, v)
                        except ValueError:
                            print(f"Warning (Line {line_num}): Skipping line due to non-integer nodes: '{line.strip()}'")
                        except Exception as e:
                            print(f"Warning (Line {line_num}): Skipping line due to error: '{line.strip()}' - {e}")
                    # Silently ignore lines with less than 2 parts after splitting
                    # (Could add a warning here if needed)

        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            raise
        except Exception as e:
            print(f"Error loading graph from {filepath}: {e}")
            raise

        print(f"Graph loaded: {self.num_vertices} vertices, {self.num_edges} edges, Density: {self.density:.4f}")


    # --- Alias for backward compatibility with main.py if needed ---
    # If you prefer main.py to keep calling load_from_csv, you can add this alias:
    def load_from_csv(self, filepath):
        """Alias for load_from_edgelist_file."""
        self.load_from_edgelist_file(filepath)
    # Or, alternatively, just update main.py to call load_from_edgelist_file directly.
    # Let's assume you update main.py for clarity.

    def add_edge(self, u, v):
        """Adds an edge between vertex u and vertex v."""
        # Check prevents double counting edges if file lists both (u,v) and (v,u)
        if v not in self._adjacency_list.get(u, set()):
            self._adjacency_list[u].add(v)
            self._adjacency_list[v].add(u)
            self._vertices.add(u)
            self._vertices.add(v)
            self._num_edges += 1

    def get_neighbors(self, u):
        """Returns the set of neighbors for vertex u."""
        return self._adjacency_list.get(u, set())

    def has_edge(self, u, v):
        """Checks if an edge exists between vertex u and vertex v."""
        # Check both directions just in case, though add_edge ensures symmetry
        return u in self._adjacency_list.get(v, set()) or v in self._adjacency_list.get(u, set())


    def get_induced_subgraph_edges(self, subset):
        """
        Calculates the number of edges within the subgraph induced by a subset of vertices.
        """
        count = 0
        # Convert to list for indexed iteration to avoid checking pairs twice
        subset_list = list(subset)
        for i in range(len(subset_list)):
            u = subset_list[i]
            # Check only neighbors of u that are ALSO in the subset AND appear later in the list
            for j in range(i + 1, len(subset_list)):
                v = subset_list[j]
                # Use the adjacency list check which is efficient
                if v in self._adjacency_list.get(u, set()):
                    count += 1
        return count