"""Custom path relinking class."""

from src.utils.graph import Graph

class PathRelinker:
    """
    Implements path relinking between two k-size solutions for the k-gamma-quasi-clique problem.
    Tie-breaking by g_v - g_u from TSQC's long-term frequency memory.
    Best intermediate solution is the one with lowest total frequency sum.
    """
    def __init__(self, graph: Graph, gamma: float, g_freq: dict):
        """
        Initializes the PathRelinker.

        Args:
            graph (Graph): The graph object.
            gamma (float): The quasi-clique density threshold (0<gamma<=1).
            g_freq (dict): Long-term frequency memory mapping vertices to counts.
        """
        if not isinstance(graph, Graph):
            raise TypeError("graph must be an instance of Graph")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be between 0 (exclusive) and 1 (inclusive)")
        if not isinstance(g_freq, dict):
            raise TypeError("g_freq must be a dictionary mapping vertex to frequency")

        self._graph = graph
        self._gamma = gamma
        self._g_freq = g_freq

    def relink(self, S_init: set, S_target: set):
        """
        Performs path relinking from S_init toward S_target.

        Both S_init and S_target must be subsets of vertices of the same cardinality k.

        Returns:
            path (list of sets): Sequence of intermediate solutions.
            best_solution (set): The intermediate solution with lowest cumulative frequency.
        """
        if not isinstance(S_init, set) or not isinstance(S_target, set):
            raise TypeError("S_init and S_target must be sets of vertex IDs")
        if len(S_init) != len(S_target):
            raise ValueError("S_init and S_target must have the same cardinality")

        current = set(S_init)
        path = [set(current)]

        # Track best solution by frequency sum
        best_solution = set(current)
        best_freq_sum = sum(self._g_freq.get(v, 0) for v in current)

        # Determine elements to swap
        to_remove = list(current - S_target)
        to_add = list(S_target - current)

        # Iterate until aligned or no moves
        while to_remove and to_add:
            best_pair = None
            best_score = float('-inf')
            best_gv_gu = float('-inf')

            # Evaluate all possible swaps
            for u in to_remove:
                for v in to_add:
                    # Compute induced edges after swap
                    new_set = (current - {u}) | {v}
                    score = self._graph.get_induced_subgraph_edges(new_set)

                    if score > best_score:
                        best_score = score
                        best_pair = (u, v)
                        best_gv_gu = self._g_freq.get(v, 0) - self._g_freq.get(u, 0)
                    elif score == best_score:
                        gv_gu = self._g_freq.get(v, 0) - self._g_freq.get(u, 0)
                        if gv_gu > best_gv_gu:
                            best_pair = (u, v)
                            best_gv_gu = gv_gu

            if best_pair is None:
                break

            u_sel, v_sel = best_pair
            # Apply swap
            current.remove(u_sel)
            current.add(v_sel)
            to_remove.remove(u_sel)
            to_add.remove(v_sel)
            path.append(set(current))

            # Update best by frequency sum
            freq_sum = sum(self._g_freq.get(v, 0) for v in current)
            if freq_sum < best_freq_sum:
                best_freq_sum = freq_sum
                best_solution = set(current)

            if current == S_target:
                break

        return path, best_solution
