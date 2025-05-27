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
        Loads graph data from an edge list file (whitespace-separated integers per line).
        Skips malformed lines and self-loops.
        Builds adjacency list and edge_index (0-based).
        """
        self._adjacency_list.clear()
        self._vertices.clear()
        self._edge_index = [[], []]
        self._num_edges = 0
        line_num = 0

        checked_indexing = False
        try:
            with open(filepath, 'r') as infile:
                for line in infile:
                    line_num += 1
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            u, v = int(parts[0]), int(parts[1])
                            if not checked_indexing:
                                if u == 0 or v == 0:
                                    self._zero_indexed = True
                                    checked_indexing = True
                            # if u != v:
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

    def get_node_features(self, subset, g_freq):
        """
        Returns a 2D list where each row is [degree, clustering_coeff, g_v, dist_to_subset, in_subset_flag]
        - subset: iterable of nodes defining the target set for shortest-path features.
        - g_freq: dict mapping node -> g_v value; defaults to internal _g_freq.
        Rows are ordered by sorted node id.
        """
        distances = self._bfs_multi_source(subset)
        features = []
        for u in sorted(self._vertices):
            degree = len(self._adjacency_list[u])
            clustering = self.clustering_coefficient(u)
            g_v = g_freq.get(u, 0)
            ### Some graphs can be not-connected, therefore having large distance
            dist = distances.get(u, self.num_vertices)
            in_subset = 1 if u in subset else 0
            features.append([degree, clustering, g_v, dist, in_subset])
        return features
