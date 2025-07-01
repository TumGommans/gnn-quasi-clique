"""The TSQC algorithm."""

import random
import math
import time
import copy

from collections import defaultdict

from src.utils.graph import Graph 

from src.algorithms.tsq import TSQ

class TSQC:
    """
    Implements the TSQC algorithm structure.

    Coordinates the search for k-gamma-quasi-cliques for increasing k
    by utilizing the TSQ procedure (inner loop) and handling restarts.
    """
    def __init__(
        self, 
        graph: Graph, 
        gamma: float, 
        max_iterations_It: int = 10**6, 
        search_depth_L: int = 1000, 
        rng: random.Random = None,
        best_known: bool = False, 
        time_limit: int = 600
    ):
        """
        Initializes the TSQC process.

        Args:
            graph:             The graph object
            gamma:             The quasi-clique density threshold
            max_iterations_It: Overall maximum iterations for the entire process
            search_depth_L:    Search depth L for TSQ restarts
            rng:               random number generator
            best_known:        whether the target k given to the solve method is the best known
            time_limit:        speaks for itself
        """
        if not isinstance(graph, Graph):
            raise TypeError("graph must be an instance of the Graph class.")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be between 0 (exclusive) and 1 (inclusive).")

        self._graph = graph
        self._gamma = gamma
        self._It_max = max_iterations_It
        self._L = search_depth_L
        self._num_all_vertices = graph.num_vertices
        self._best_known = best_known
        self._rng = rng
        self._time_limit = time_limit

        # Long term frequency memory
        self._g_freq = defaultdict(int)

        self.best_quasi_clique_found = set()
        self.best_quasi_clique_size = 0

        self.intensification_count = 0
        self.tie_breaker_count = 0

    def solve(
        self, 
        initial_k: int = 2
    ) -> tuple[set[int], float] :
        """
        Approximates the maximum gamma-quasi-clique by finding k-gamma-quasi-cliques
        for increasing values of k.

        Args:
            initial_k: The starting size k to search for.
        
        returns:
            best_quasi_clique_found: the best solution found by DeepTSQC
            elapsed_overall: the runtime until the largest quasi-clique was found
        """
        self._initial_k = initial_k
        k = max(1, initial_k)
        current_best_clique = set()
        elapsed_overall = float('inf')
        start_time_overall = time.time()
        iteration_count_It = 0

        print(f"Starting TSQC search with gamma = {self._gamma}, initial_k = {k}, It_max = {self._It_max}, L = {self._L}")

        while iteration_count_It < self._It_max:
            print(f"\n--- Searching for k = {k} (Global Iteration: {iteration_count_It}) ---")
            if k > self._num_all_vertices:
                 print(f"Stopping: k ({k}) exceeds number of vertices ({self._num_all_vertices}).")
                 break

            found_clique_for_k = None
            is_first_attempt_for_k = True
            start_time_k = time.time()

            if is_first_attempt_for_k:
                 current_S = self._generate_initial_solution(k)
                 self.initial_S = copy.deepcopy(current_S)
                 is_first_attempt_for_k = False
            else:
                 current_S = self._generate_restart_solution(k)

            if not current_S:
                print(f"  Stopping k={k} search: Could not generate initial/restart solution.")
                break
            
            restart_counter = 1
            while iteration_count_It < self._It_max:
                print(f"  Calling TSQ for k={k} (Starts at Global It: {iteration_count_It})")
                if not current_S:
                     print("  Error: current_S is None before calling TSQ.")
                     found_clique_for_k = None
                     break

                tsq = TSQ(
                    graph=self._graph,
                    gamma=self._gamma,
                    k=k,
                    L=self._L,
                    initial_S=current_S,
                    current_It=iteration_count_It,
                    update_freq_method=self._update_frequency_memory,
                    max_total_It=self._It_max,
                    rng=self._rng,
                )
                S_star, it_consumed_tsq = tsq.run()
                iteration_count_It += it_consumed_tsq
                self.intensification_count += tsq.intensification_count
                self.tie_breaker_count += tsq.tie_breaker_count

                if self._is_legal_quasi_clique(S_star, k, self._gamma):
                    print(f"  SUCCESS: Legal {k}-quasi-clique found by TSQ!")
                    found_clique_for_k = S_star
                    break
                else:
                    print("  TSQ finished without legal clique. Generating restart solution...")
                    current_S = self._generate_restart_solution(k)
                    if not current_S:
                        print(f"  Stopping k={k} search: Could not generate restart solution.")
                        found_clique_for_k = None
                        break
                    if time.time() - start_time_overall > self._time_limit:
                        print(f"  Timeout reached for finding k={k}. Stopping search for this k.")
                        found_clique_for_k = None
                        break
                    restart_counter += 1
                    
            if found_clique_for_k:
                end_time_k = time.time()
                elapsed = end_time_k - start_time_k
                print(f"  Time taken for k={k}: {elapsed:.2f} seconds")
                current_best_clique = found_clique_for_k
                self.best_quasi_clique_found = current_best_clique
                self.best_quasi_clique_size = k
                if self._best_known:
                    elapsed_overall = elapsed
                    print(f"The {k}-quasi-clique was the best known for TSQC. Stopping search.")
                    break
                else:
                    elapsed_overall = end_time_k - start_time_overall
                    k += 1
            else:
                print(f"Could not find a {k}-quasi-clique within iteration limits. Stopping search.")
                break

        print(f"\n--- TSQC Search Complete ---")
        print(f"Maximum {self._gamma}-quasi-clique size found: {self.best_quasi_clique_size}")
        print(f"Total execution time: {elapsed_overall:.2f} seconds")
        print(f"Total global iterations consumed: {iteration_count_It}")

        return self.best_quasi_clique_found, elapsed_overall


# ------------------------------------------------------------------------------------------------------------
# Solution Generation & Frequency Memory
# ------------------------------------------------------------------------------------------------------------

    def _generate_initial_solution(self, k: int) -> set[int]:
        """Generates initial solution S.
        
        args:
            k: the target quasi clique size
        
        returns:
            S: the set of vertices in the initial random solution
        """
        if k > self._num_all_vertices or k <= 0: return None
        available_vertices = list(self._graph.vertices)
        if not available_vertices: return None

        S = set()
        first_vertex = self._rng.choice(available_vertices) if self._rng else random.choice(available_vertices)
        S.add(first_vertex)
        available_vertices.remove(first_vertex)

        while len(S) < k and available_vertices:
            max_neighbors_in_S = -1; candidates = []
            for v_out in available_vertices:
                neighbors_in_S = sum(1 for v_in in S if self._graph.has_edge(v_out, v_in))
                if neighbors_in_S > max_neighbors_in_S:
                    max_neighbors_in_S = neighbors_in_S; candidates = [v_out]
                elif neighbors_in_S == max_neighbors_in_S:
                    candidates.append(v_out)

            if candidates: 
                chosen_vertex = self._rng.choice(candidates) if self._rng else random.choice(candidates)
            elif available_vertices: 
                chosen_vertex = self._rng.choice(available_vertices) if self._rng else random.choice(available_vertices)
            else: break

            S.add(chosen_vertex)
            available_vertices.remove(chosen_vertex)

        return S

    def _generate_restart_solution(self, k: int) -> set[int]:
        """
        Generates restart solution. Designed for potential override.
        
        args:
            k: the target quasi clique size
        
        returns:
            set[int]: the restart solution
        """
        return self._default_restart_strategy(k)

    def _default_restart_strategy(self, k: int) -> set[int]:
        """
        Default restart strategy using g_freq from Djeddi et al. (2019).
        
        According to Section 3.5 of the paper:
        1. First select the vertex with the lowest g_v 
        2. Then, iteratively select the next vertex that maximizes the degree 
        wrt the partial restart solution, breaking ties with g_v and secondly at random
        
        args:
            k: the target quasi clique size
        
        returns:
            S: the restart solution
        """
        if k > self._num_all_vertices or k <= 0: 
            return None
        
        available_vertices = list(self._graph.vertices)
        if not available_vertices: 
            return None

        S = set()
        
        min_freq = min(self._g_freq.get(v, 0) for v in available_vertices)
        candidates_min_freq = [v for v in available_vertices if self._g_freq.get(v, 0) == min_freq]
        first_vertex = self._rng.choice(candidates_min_freq) if self._rng else random.choice(candidates_min_freq)
        S.add(first_vertex)
        
        while len(S) < k:
            remaining_vertices = [v for v in available_vertices if v not in S]
            if not remaining_vertices:
                break
                
            max_degree_to_S = -1
            best_candidates = []
            
            for v in remaining_vertices:
                degree_to_S = sum(1 for u in S if self._graph.has_edge(v, u))
                if degree_to_S > max_degree_to_S:
                    max_degree_to_S = degree_to_S
                    best_candidates = [v]
                elif degree_to_S == max_degree_to_S:
                    best_candidates.append(v)
            
            if len(best_candidates) == 1:
                chosen_vertex = best_candidates[0]
            else:
                min_freq_among_best = min(self._g_freq.get(v, 0) for v in best_candidates)
                final_candidates = [v for v in best_candidates if self._g_freq.get(v, 0) == min_freq_among_best]
        
                chosen_vertex = self._rng.choice(final_candidates) if self._rng else random.choice(final_candidates)
            
            S.add(chosen_vertex)
        
        if len(S) != k:
            print(f"Warning: Restart strategy generated {len(S)} vertices, expected {k}.")
            remaining = [v for v in available_vertices if v not in S]
            while len(S) < k and remaining:
                S.add(remaining.pop(self._rng.randrange(len(remaining)) if self._rng else random.randrange(len(remaining))))

        return S

    def _update_frequency_memory(self, u_out, v_in, k):
        """
        Updates the long-term frequency memory g_freq and handles reset.
        
        args:
            u_out: the vertex removed in the swap
            v_in: the vertex added in the swap
            k: the target quasi clique size
        """
        self._g_freq[u_out] = self._g_freq.get(u_out, 0) + 1
        self._g_freq[v_in] = self._g_freq.get(v_in, 0) + 1

        if any(freq > k for freq in self._g_freq.values()):
            self._g_freq = defaultdict(int)

# ------------------------------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------------------------------

    def _is_legal_quasi_clique(
        self, 
        subset: set[int], 
        k: int, 
        gamma: float
    ) -> bool:
        """
        Checks feasibility of given solution based on edge count.
        
        args:
            subset: the current solution to check
            k: the target quasi clique size
            gamma: the density threshold
        
        returns:
            bool: whether the current solution is feasible
        """
        if not subset or len(subset) != k:
            return False
        num_edges = self._graph.get_induced_subgraph_edges(subset)
        target_edges = math.ceil(gamma * k * (k - 1) / 2.0) if k >= 2 else 0
        return num_edges >= target_edges
