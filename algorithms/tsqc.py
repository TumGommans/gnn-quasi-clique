"""The TSQC algorithm."""

import random
import math
import time
from collections import defaultdict
from utils.graph import Graph 
from algorithms.tsq import TSQ

class TSQC:
    """
    Implements the main TSQC algorithm structure (Algorithm 1 from the paper),
    coordinating the search for k-gamma-quasi-cliques for increasing k
    by utilizing the TSQ procedure (inner loop) and handling restarts.
    """
    def __init__(self, graph, gamma, max_iterations_It=10**8, search_depth_L=1000):
        """
        Initializes the TSQC process.

        Args:
            graph (Graph): The graph object.
            gamma (float): The quasi-clique density threshold.
            max_iterations_It (int): Overall maximum iterations for the entire process.
            search_depth_L (int): Search depth L for TSQ restarts.
        """
        if not isinstance(graph, Graph):
            raise TypeError("graph must be an instance of the Graph class.")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be between 0 (exclusive) and 1 (inclusive).")

        self._graph = graph
        self._gamma = gamma
        self._It_max = max_iterations_It
        self._L = search_depth_L # L parameter for TSQ restarts
        self._num_all_vertices = graph.num_vertices

        # Long term frequency memory
        self._g_freq = defaultdict(int)

        # Store the best quasi-clique found overall
        self.best_quasi_clique_found = set()
        self.best_quasi_clique_size = 0

    # --- Public Solve Method ---
    def solve(self, initial_k=1):
        """
        Approximates the maximum gamma-quasi-clique by finding k-gamma-quasi-cliques
        for increasing values of k (Algorithm 1).

        Args:
            initial_k (int): The starting size k to search for.

        Returns:
            set: The best gamma-quasi-clique found.
        """
        k = max(1, initial_k) # Ensure k starts at least at 1
        current_best_clique = set()
        start_time_overall = time.time()
        iteration_count_It = 0 # Global iteration counter

        print(f"Starting TSQC search with gamma = {self._gamma}, initial_k = {k}, It_max = {self._It_max}, L = {self._L}")

        while iteration_count_It < self._It_max:
            print(f"\n--- Searching for k = {k} (Global Iteration: {iteration_count_It}) ---")
            if k > self._num_all_vertices:
                 print(f"Stopping: k ({k}) exceeds number of vertices ({self._num_all_vertices}).")
                 break

            # --- Find k-Quasi-Clique ---
            # This part corresponds to Algorithm 1's main loop body
            found_clique_for_k = None
            is_first_attempt_for_k = True
            start_time_k = time.time()

            # Generate initial/restart solution for this k
            if is_first_attempt_for_k:
                 current_S = self._generate_initial_solution(k)
                 is_first_attempt_for_k = False
            else: # Generate restart solution if previous TSQ failed
                 current_S = self._generate_restart_solution(k)

            if not current_S: # Handle cases where solution generation fails
                print(f"  Stopping k={k} search: Could not generate initial/restart solution.")
                # If we can't even start for this k, we likely can't find larger ones either.
                break

            # Loop specifically for finding a k-clique (includes restarts)
            while iteration_count_It < self._It_max:
                print(f"  Calling TSQ for k={k} (Starts at Global It: {iteration_count_It})")
                if not current_S: # Safety check
                     print("  Error: current_S is None before calling TSQ.")
                     found_clique_for_k = None
                     break

                # Instantiate and run the inner TSQ loop
                tsq_instance = TSQ(graph=self._graph,
                                   gamma=self._gamma,
                                   k=k,
                                   L=self._L,
                                   initial_S=current_S,
                                   current_It=iteration_count_It,
                                   update_freq_method=self._update_frequency_memory, # Pass method reference
                                   max_total_It=self._It_max)

                best_S_star_tsq, it_consumed_tsq = tsq_instance.run()
                iteration_count_It += it_consumed_tsq # Update global counter

                # Check if TSQ found a legal clique
                if self._is_legal_quasi_clique(best_S_star_tsq, k, self._gamma):
                    print(f"  SUCCESS: Legal {k}-quasi-clique found by TSQ!")
                    found_clique_for_k = best_S_star_tsq
                    break # Exit the inner while loop (for this k)
                else:
                    # TSQ failed, need to restart
                    print("  TSQ finished without legal clique. Generating restart solution...")
                    current_S = self._generate_restart_solution(k)
                    if not current_S:
                         print(f"  Stopping k={k} search: Could not generate restart solution.")
                         found_clique_for_k = None # Mark as not found
                         break # Exit the inner while loop (for this k)

                    # Optional: Add timeout for finding a specific k
                    if time.time() - start_time_k > 3600: # e.g., 1 hour limit per k
                         print(f"  Timeout reached for finding k={k}. Stopping search for this k.")
                         found_clique_for_k = None
                         break

            # --- Process result for k ---
            if found_clique_for_k:
                end_time_k = time.time()
                print(f"  Time taken for k={k}: {end_time_k - start_time_k:.2f} seconds")
                current_best_clique = found_clique_for_k
                self.best_quasi_clique_found = current_best_clique
                self.best_quasi_clique_size = k
                k += 1 # Increment k and search for a larger one
            else:
                print(f"Could not find a {k}-quasi-clique within iteration limits. Stopping search.")
                break # Stop if we can't find a quasi-clique for the current k

        end_time_overall = time.time()
        print(f"\n--- TSQC Search Complete ---")
        print(f"Maximum {self._gamma}-quasi-clique size found: {self.best_quasi_clique_size}")
        # print(f"Best quasi-clique found: {sorted(list(self.best_quasi_clique_found))}") # Optional detail
        print(f"Total execution time: {end_time_overall - start_time_overall:.2f} seconds")
        print(f"Total global iterations consumed: {iteration_count_It}")

        return self.best_quasi_clique_found


    # --- Solution Generation & Frequency Memory ---

    def _generate_initial_solution(self, k):
        """ Generates initial solution S (greedy random heuristic). """
        if k > self._num_all_vertices or k <= 0: return None
        available_vertices = list(self._graph.vertices)
        if not available_vertices: return None

        S = set()
        first_vertex = random.choice(available_vertices)
        S.add(first_vertex)
        available_vertices.remove(first_vertex)

        while len(S) < k and available_vertices:
            best_vertex = -1; max_neighbors_in_S = -1; candidates = []
            for v_out in available_vertices:
                neighbors_in_S = sum(1 for v_in in S if self._graph.has_edge(v_out, v_in))
                if neighbors_in_S > max_neighbors_in_S:
                    max_neighbors_in_S = neighbors_in_S; candidates = [v_out]
                elif neighbors_in_S == max_neighbors_in_S:
                    candidates.append(v_out)

            if candidates: chosen_vertex = random.choice(candidates)
            elif available_vertices: chosen_vertex = random.choice(available_vertices)
            else: break

            S.add(chosen_vertex)
            available_vertices.remove(chosen_vertex)

        return S if len(S) == k else None

    def _generate_restart_solution(self, k):
        """ Generates restart solution. Designed for potential override. """
        return self._default_restart_strategy(k)

    def _default_restart_strategy(self, k):
        """ Paper's default restart strategy using g_freq. """
        if k > self._num_all_vertices or k <= 0: return None
        available_vertices = list(self._graph.vertices)
        if not available_vertices: return None

        # Calculate degrees needed for tie-breaking
        all_degrees = {v: len(self._graph.get_neighbors(v)) for v in available_vertices}
        sorted_vertices = sorted(available_vertices, key=lambda v: (self._g_freq.get(v, 0), -all_degrees.get(v, 0)))
        S = set(sorted_vertices[:k])

        # Ensure correct size if possible
        idx = k
        while len(S) < k and idx < len(sorted_vertices):
             S.add(sorted_vertices[idx])
             idx += 1

        if len(S) != k:
             print(f"Warning: Restart strategy generated {len(S)} vertices, expected {k}.")
             # Fallback might be needed if graph is too small or disconnected badly
             remaining = [v for v in available_vertices if v not in S]
             while len(S) < k and remaining:
                 S.add(remaining.pop(random.randrange(len(remaining))))

        return S if len(S) == k else None

    def _update_frequency_memory(self, u_out, v_in, k):
        """ Updates the long-term frequency memory g_freq and handles reset. """
        self._g_freq[u_out] = self._g_freq.get(u_out, 0) + 1
        self._g_freq[v_in] = self._g_freq.get(v_in, 0) + 1

        # Check for reset condition (if *any* freq > k)
        if any(freq > k for freq in self._g_freq.values()):
            self._g_freq = defaultdict(int) # Reset all frequencies


    # --- Helper Methods ---

    def _is_legal_quasi_clique(self, subset, k, gamma):
        """ Checks legality based on edge count. """
        if not subset or len(subset) != k:
            return False
        # Need to recalculate edges here as TSQ's f_S is internal to it
        num_edges = self._graph.get_induced_subgraph_edges(subset)
        target_edges = math.ceil(gamma * k * (k - 1) / 2.0) if k >= 2 else 0
        return num_edges >= target_edges