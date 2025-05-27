"""

"""
import time
import torch

from torch_geometric.data import Data

from src.utils.graph import Graph
from src.gnn.model import SearchDepthGNN

from src.algorithms.tsq import TSQ
from src.algorithms.tsqc import TSQC

INDEX_TO_L = {0: 500, 1: 1000, 2: 5000}

class DeepTSQC(TSQC):
    """

    """
    def __init__(
        self,
        graph: Graph,
        gamma: float,
        gnn: SearchDepthGNN,
        max_iterations_It: int = 10**8,
        search_depth_L: int = 1000,
        rng=None,
        best_known: bool = False,
        time_limit: int = 3600,
    ):
        super().__init__(graph, gamma, max_iterations_It, search_depth_L, rng, best_known, time_limit)
        self._gnn = gnn

    def solve(self, initial_k: int = 1):
        k = max(1, initial_k)
        current_best_clique = set()
        start_time_overall = time.time()
        iteration_count_It = 0
        best_runtime_found = 0.0

        k_is_incremented = False

        print(f"Starting DeepTSQC search with gamma = {self._gamma}, initial_k = {k}, "
              f"It_max = {self._It_max}")

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
                current_S = self._default_restart_strategy(k)

            if not current_S:
                print(f"  Stopping k={k} search: Could not generate initial solution.")
                break
            
            restart_counter = 1
            while iteration_count_It < self._It_max:
                print(f"  Calling TSQ for k={k} (Starts at Global It: {iteration_count_It})")

                L = self._predict_L(current_S, k, restart_counter)

                tsq = TSQ(
                    graph=self._graph,
                    gamma=self._gamma,
                    k=k,
                    L=L,
                    initial_S=current_S,
                    current_It=iteration_count_It,
                    update_freq_method=self._update_frequency_memory,
                    max_total_It=self._It_max,
                    rng=self._rng,
                )

                best_S_star_tsq, it_consumed_tsq = tsq.run()
                iteration_count_It += it_consumed_tsq

                if self._is_legal_quasi_clique(best_S_star_tsq, k, self._gamma):
                    end_time_k = time.time()
                    elapsed = end_time_k - start_time_overall
                    print(f"  DeepTSQ: SUCCESS: Legal {k}-quasi-clique found by TSQ! "
                          f"(Time to find: {elapsed:.2f}s)")
                    found_clique_for_k = best_S_star_tsq.copy()
                    current_best_clique = found_clique_for_k
                    self.best_quasi_clique_found = current_best_clique
                    self.best_quasi_clique_size = k
                    best_runtime_found = elapsed
                    break
                else:
                    print("  TSQ finished without legal clique. Generating restart...")
                    current_S = self._generate_restart_solution(k)
                    if not current_S:
                        print(f"  DeepTSQ: Could not generate restart for k={k}. Stopping search.")
                        found_clique_for_k = None
                        break
                    if time.time() - start_time_k > self._time_limit:
                        print(f"  Timeout reached for finding k={k}. Stopping search.")
                        found_clique_for_k = None
                        break
                    restart_counter += 1

            if found_clique_for_k:
                if not self._best_known:
                    k += 1
                    k_is_incremented = True
                else:
                    break
            else:
                if not k_is_incremented:
                    termination_time = time.time()
                    best_runtime_found = termination_time - start_time_overall
                break

        print(f"\n--- DeepTSQC Complete ---")
        print(f"Max {self._gamma}-quasi-clique size: {self.best_quasi_clique_size}")
        print(f"Time to best found clique: {best_runtime_found:.2f} seconds")
        print(f"Global iterations consumed: {iteration_count_It}")

        return self.best_quasi_clique_found, best_runtime_found

    def _predict_L(self, current_S, k, restart_counter):
        state = self._get_state(current_S, k, restart_counter)
        data = self._create_data_from_state(state)
        with torch.no_grad():
            output = self._gnn(data)
            predicted_class = output.argmax(dim=-1).item()
            return INDEX_TO_L[predicted_class]

    def _get_state(self, current_S, k, restart_counter):
        return {
            "graph_structure": {
                "num_vertices": self._graph.num_vertices, 
                "edge_index": self._graph.edge_index
            }, 
            "state": {
                "node_features": self._graph.get_node_features(current_S, self._g_freq), 
                "graph_features": {
                    "gamma": self._gamma, 
                    "initial_target_k": self._initial_k,
                    "current_target_k": k, 
                    "graph_density": self._graph.density,
                    "restart_count": restart_counter
                }
            }
        }

    @staticmethod
    def _create_data_from_state(state):
        x = torch.tensor(state['state']['node_features'], dtype=torch.float)
        edge_index = torch.tensor(state['graph_structure']['edge_index'], dtype=torch.long)
        gf = state['state']['graph_features']
        u = torch.tensor([
            gf['gamma'],
            gf['initial_target_k'],
            gf['current_target_k'],
            gf['graph_density'],
            gf['restart_count']
        ], dtype=torch.float).unsqueeze(0)
        
        return Data(x=x, edge_index=edge_index, u=u)
