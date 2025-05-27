"""Greedy algorithm for a lower bound."""

import heapq
from src.utils.graph import Graph

class GreedyLB:
    def __init__(self, graph: Graph, gamma: float):
        if not (0 < gamma <= 1):
            raise ValueError("Gamma must be in (0,1].")
        self.graph = graph
        self.gamma = gamma

    def run(self):
        # Build adj-list and degrees
        nbrs      = {v: set(self.graph.get_neighbors(v)) for v in self.graph.vertices}
        deg       = {v: len(nbrs[v]) for v in nbrs}
        remaining = set(self.graph.vertices)
        E         = sum(deg.values()) // 2

        # Min-heap of (degree, vertex)
        heap = [(deg[v], v) for v in remaining]
        heapq.heapify(heap)

        best_size     = 0

        # Peel
        while remaining:
            if len(remaining) > 1:
                density = (2 * E) / (len(remaining) * (len(remaining) - 1))
                if density >= self.gamma and len(remaining) > best_size:
                    best_size     = len(remaining)

            # Extract current min-degree vertex
            d, v = heapq.heappop(heap)
            if v not in remaining or d != deg[v]:
                continue  # skip stale entry

            # Remove v
            remaining.remove(v)
            # Update E and neighbor-degrees
            for u in nbrs[v]:
                if u in remaining:
                    E      -= 1
                    deg[u] -= 1
                    heapq.heappush(heap, (deg[u], u))

        return best_size
