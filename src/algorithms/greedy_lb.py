"""Greedy algorithm for a lower bound."""

import heapq
from src.utils.graph import Graph

class GreedyLB:
    """Object to obtain greedy lowerbounds."""
    def __init__(self, graph: Graph, gamma: float):
        """Initializes an instance.
        
        args:
            graph: the graph instance
            gamma: the density threshold
        """
        if not (0 < gamma <= 1):
            raise ValueError("Gamma must be in (0,1].")
        self.graph = graph
        self.gamma = gamma

    def run(self):
        """Runs the greedy heuristic.
        
        Implements a peeling strategy, by removing low-degree
        vertices until the density threshold is satisfied.
        """
        nbrs = {v: set(self.graph.get_neighbors(v)) for v in self.graph.vertices}
        deg = {v: len(nbrs[v]) for v in nbrs}
        remaining = set(self.graph.vertices)
        E = sum(deg.values()) // 2

        heap = [(deg[v], v) for v in remaining]
        heapq.heapify(heap)

        best_size = 0

        while remaining:
            if len(remaining) > 1:
                density = (2 * E) / (len(remaining) * (len(remaining) - 1))
                if density >= self.gamma and len(remaining) > best_size:
                    best_size     = len(remaining)

            d, v = heapq.heappop(heap)
            if v not in remaining or d != deg[v]:
                continue

            remaining.remove(v)
            for u in nbrs[v]:
                if u in remaining:
                    E      -= 1
                    deg[u] -= 1
                    heapq.heappush(heap, (deg[u], u))

        return best_size
