"""The TSQC algorithm."""

import random
import math
import time
import copy

from collections import defaultdict

from src.utils.graph import Graph 
from src.utils.state_action_pair import (
    GraphStructure,
    GraphFeatures,
    State,
    Action,
    StateActionPair
)

from src.algorithms.tsq import TSQ

class TSQC:
    """
    Implements the core TSQC algorithm structure.

    Coordinates the search for k-gamma-quasi-cliques for increasing k
    by utilizing the TSQ procedure (inner loop) and handling restarts.
    """
    def __init__(
        self, 
        graph, 
        gamma, 
        max_iterations_It=10**6, 
        search_depth_L=1000, 
        rng=None,
        best_known=False, 
        time_limit=600,
        collect_train_data=False,
        train_data_filepath='',
        search_depth_grid=[]
    ):
        """
        Initializes the TSQC process.

        Args:
            graph (Graph):           The graph object.
            gamma (float):           The quasi-clique density threshold.
            max_iterations_It (int): Overall maximum iterations for the entire process.
            search_depth_L (int):    Search depth L for TSQ restarts.
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
        self._collect_train_data = collect_train_data
        self._search_depth_grid = sorted(search_depth_grid)

        # Long term frequency memory
        self._g_freq = defaultdict(int)

        self.best_quasi_clique_found = set()
        self.best_quasi_clique_size = 0

        self.intensification_count = 0
        self.tie_breaker_count = 0

        self._initial_k = 1

        # Training data output if applicable
        self._train_data_fh = None
        if self._collect_train_data:
            self._train_data_fh = open(train_data_filepath, 'a', buffering=1)

    def solve(self, initial_k=1):
        """
        Approximates the maximum gamma-quasi-clique by finding k-gamma-quasi-cliques
        for increasing values of k.

        Args:
            initial_k (int): The starting size k to search for.

        Returns:
            set: The best gamma-quasi-clique found.
        """
        self._initial_k = initial_k
        k = max(1, initial_k)
        current_best_clique = set()
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
                
                if self._collect_train_data:
                    best_S_star_tsq = None
                    best_f_S_star = 0
                    best_local_g_freq = None
                    iterations_to_add = None
                    best_L = None
                    best_L_index = None
                    for i, L in enumerate(self._search_depth_grid):
                        self._local_g_freq = copy.deepcopy(self._g_freq)
                        tsq = TSQ(
                            graph=self._graph,
                            gamma=self._gamma,
                            k=k,
                            L=L,
                            initial_S=current_S,
                            current_It=iteration_count_It,
                            update_freq_method=self._local_update_frequency_memory,
                            max_total_It=self._It_max,
                            rng=self._rng
                        )
                        S_star, it_consumed_tsq = tsq.run()
                        if self._is_legal_quasi_clique(S_star, k, self._gamma):
                            best_S_star_tsq = S_star
                            best_local_g_freq = copy.deepcopy(self._local_g_freq)
                            iterations_to_add = it_consumed_tsq
                            best_L = L
                            best_L_index = i
                            break
                        else:
                            f_S_star = tsq._evaluate_solution(S_star)
                            if f_S_star > best_f_S_star:
                                best_f_S_star = f_S_star
                                best_S_star_tsq = S_star
                                best_local_g_freq = copy.deepcopy(self._local_g_freq)
                                iterations_to_add = it_consumed_tsq
                                best_L = L
                                best_L_index = i
                    
                    train_data_pair = self._store_state_action_pair(
                        current_S, k, restart_counter, best_L, best_L_index
                    )

                    if self._train_data_fh:
                        self._train_data_fh.write(train_data_pair + "\n")
                        # self._train_data_fh.flush()
    
                    self._g_freq = copy.deepcopy(best_local_g_freq)
                    iteration_count_It += iterations_to_add
                else:
                    best_S_star_tsq = current_S
                    tsq = TSQ(
                            graph=self._graph,
                            gamma=self._gamma,
                            k=k,
                            L=self._L,
                            initial_S=current_S,
                            current_It=iteration_count_It,
                            update_freq_method=self._update_frequency_memory,
                            max_total_It=self._It_max,
                            rng=self._rng
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
                    if time.time() - start_time_k > self._time_limit:
                        print(f"  Timeout reached for finding k={k}. Stopping search for this k.")
                        found_clique_for_k = None
                        break
                    restart_counter += 1
                    
            if found_clique_for_k:
                end_time_k = time.time()
                print(f"  Time taken for k={k}: {end_time_k - start_time_k:.2f} seconds")
                current_best_clique = found_clique_for_k
                self.best_quasi_clique_found = current_best_clique
                self.best_quasi_clique_size = k
                if self._best_known:
                    print(f"The {k}-quasi-clique was the best known for TSQC. Stopping search.")
                    break
                else:
                    k += 1
            else:
                print(f"Could not find a {k}-quasi-clique within iteration limits. Stopping search.")
                break

        # close file handle
        if self._train_data_fh:
            self._train_data_fh.close()
    
        end_time_overall = time.time()
        runtime = end_time_overall - start_time_overall
        print(f"\n--- TSQC Search Complete ---")
        print(f"Maximum {self._gamma}-quasi-clique size found: {self.best_quasi_clique_size}")
        print(f"Total execution time: {runtime:.2f} seconds")
        print(f"Total global iterations consumed: {iteration_count_It}")

        return self.best_quasi_clique_found, runtime


# ------------------------------------------------------------------------------------------------------------
# Solution Generation & Frequency Memory
# ------------------------------------------------------------------------------------------------------------

    def _generate_initial_solution(self, k):
        """Generates initial solution S."""
        if k > self._num_all_vertices or k <= 0: return None
        available_vertices = list(self._graph.vertices)
        if not available_vertices: return None

        S = set()
        first_vertex = self._rng.choice(available_vertices) if self._rng else random.choice(available_vertices)
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

            if candidates: 
                chosen_vertex = self._rng.choice(candidates) if self._rng else random.choice(candidates)
            elif available_vertices: 
                chosen_vertex = self._rng.choice(available_vertices) if self._rng else random.choice(available_vertices)
            else: break

            S.add(chosen_vertex)
            available_vertices.remove(chosen_vertex)

        return S if len(S) == k else None

    def _generate_restart_solution(self, k):
        """Generates restart solution. Designed for potential override."""
        return self._default_restart_strategy(k)

    def _default_restart_strategy(self, k):
        """Paper's default restart strategy using g_freq."""
        if k > self._num_all_vertices or k <= 0: return None
        available_vertices = list(self._graph.vertices)
        if not available_vertices: return None

        all_degrees = {v: len(self._graph.get_neighbors(v)) for v in available_vertices}
        sorted_vertices = sorted(available_vertices, key=lambda v: (self._g_freq.get(v, 0), -all_degrees.get(v, 0)))
        S = set(sorted_vertices[:k])

        idx = k
        while len(S) < k and idx < len(sorted_vertices):
             S.add(sorted_vertices[idx])
             idx += 1

        if len(S) != k:
             print(f"Warning: Restart strategy generated {len(S)} vertices, expected {k}.")
             remaining = [v for v in available_vertices if v not in S]
             while len(S) < k and remaining:
                 S.add(remaining.pop(self._rng.randrange(len(remaining)) if self._rng else random.randrange(len(remaining))))

        return S if len(S) == k else None

    def _update_frequency_memory(self, u_out, v_in, k):
        """Updates the long-term frequency memory g_freq and handles reset."""
        self._g_freq[u_out] = self._g_freq.get(u_out, 0) + 1
        self._g_freq[v_in] = self._g_freq.get(v_in, 0) + 1

        if any(freq > k for freq in self._g_freq.values()):
            self._g_freq = defaultdict(int)

    def _local_update_frequency_memory(self, u_out, v_in, k):
        """Updates the long-term frequency memory g_freq and handles reset.
        
        Performs local updates for the current TSQ iteration."""
        self._local_g_freq[u_out] = self._local_g_freq.get(u_out, 0) + 1
        self._local_g_freq[v_in] = self._local_g_freq.get(v_in, 0) + 1
        if any(freq > k for freq in self._local_g_freq.values()):
            for key in self._local_g_freq:
                self._local_g_freq[key] = 0

# ------------------------------------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------------------------------------

    def _is_legal_quasi_clique(self, subset, k, gamma):
        """Checks legality of given solution based on edge count."""
        if not subset or len(subset) != k:
            return False
        num_edges = self._graph.get_induced_subgraph_edges(subset)
        target_edges = math.ceil(gamma * k * (k - 1) / 2.0) if k >= 2 else 0
        return num_edges >= target_edges
    
    def _store_state_action_pair(self, subset, k, restart_counter, L, L_index):
        """Store an optimal state-action pair for training the GNN policy."""
        return StateActionPair(
            graph_structure=GraphStructure(
                num_vertices=self._graph.num_vertices,
                edge_index=self._graph.edge_index
            ),
            state=State(
                node_features=self._graph.get_node_features(subset, self._g_freq),
                graph_features=GraphFeatures(
                    gamma=self._gamma,
                    initial_target_k=self._initial_k,
                    current_target_k=k,
                    graph_density=self._graph.density,
                    restart_count=restart_counter
                )
            ),
            action=Action(
                optimal_L_index=L_index,
                optimal_L_value=L
            )
        ).to_json()