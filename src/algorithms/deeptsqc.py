"""Script implementing the DeepTSQC algorithm."""

import time
import torch
import random

from torch_geometric.data import Data

from src.utils.graph import Graph
from src.gnn.model import SearchDepthGNN

from src.algorithms.tsq import TSQ
from src.algorithms.tsqc import TSQC

INDEX_TO_L = {0: 500, 1: 1000, 2: 5000}

class DeepTSQC(TSQC):
    """Object to run the DeepTSQC algorithm."""
    def __init__(
        self,
        graph: Graph,
        gamma: float,
        gnn: SearchDepthGNN,
        max_iterations_It: int = 10**8,
        rng: random.Random = None,
        best_known: bool = False,
        time_limit: int = 3600,
    ):
        """Initializes an instance.
        
        args:
            graph: the graph instance
            gamma: the density threshold
            gnn: the trained search depth predictor
            max_iterations_It: the max iterations allowed throughout the search
            rng: random number generator
            best_known: whether the target k given to the solve method is the best known
            time_limit: speaks for itself
        """
        super().__init__(
            graph=graph, 
            gamma=gamma, 
            max_iterations_It=max_iterations_It,
            rng=rng, 
            best_known=best_known, 
            time_limit=time_limit
        )
        self._gnn = gnn

    def solve(self, initial_k: int = 2) -> tuple[set[int], float]:
        """Solves the MQCP using DeepTSQC.

        The key difference with TSQC.solve() is the prediction of L
        
        args:
            initial_k: the target clique size to begin with
        
        returns:
            best_quasi_clique_found: the best solution found by DeepTSQC
            elapsed_overall: the runtime until the largest quasi-clique was found
        """
        k = max(1, initial_k)
        current_best_clique = set()
        start_time_overall = time.time()
        iteration_count_It = 0
        elapsed_overall = float('inf')

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
                L = self._predict_L(current_S, k)
                is_first_attempt_for_k = False
            else:
                current_S = self._default_restart_strategy(k)

            if not current_S:
                print(f"  Stopping k={k} search: Could not generate initial solution.")
                break
            
            restart_counter = 1
            while iteration_count_It < self._It_max:
                print(f"  Calling TSQ for k={k} (Starts at Global It: {iteration_count_It})")

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

                S_star, it_consumed_tsq = tsq.run()
                iteration_count_It += it_consumed_tsq

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

        print(f"\n--- DeepTSQC Complete ---")
        print(f"Maximum {self._gamma}-quasi-clique size found: {self.best_quasi_clique_size}")
        print(f"Total execution time: {elapsed_overall:.2f} seconds")
        print(f"Global iterations consumed: {iteration_count_It}")

        return self.best_quasi_clique_found, elapsed_overall

    def _predict_L(
            self, 
            current_S: set[int], 
            k: int
    ) -> int:
        """Predict the optimal search depth for the algorithm.
        
        args:
            current_S: the current initial solution that the alg begins with
            k: the target quasi clique size
        
        returns:
            int: the predicted value for L
        """
        state = self._get_state(current_S, k)
        data = self._create_data_from_state(state)
        with torch.no_grad():
            output = self._gnn(data)
            predicted_class = output.argmax(dim=-1).item()
            return INDEX_TO_L[predicted_class]

    def _get_state(
        self, 
        current_S: set[int], 
        k: int
    ) -> dict:
        """Helper method, to get the state prior to predicting L.
        
        args:
            current_S: the current initial solution that the alg begins with
            k: the target quasi clique size
        
        returns:
            dict: state dict needed for running a forward pass on the GNN
        """
        return {
            "graph_structure": {
                "num_vertices": self._graph.num_vertices, 
                "edge_index": self._graph.edge_index
            }, 
            "state": {
                "node_features": self._graph.get_node_features(current_S, self._gamma, k), 
            }
        }

    @staticmethod
    def _create_data_from_state(
        state: dict
    ) -> Data:
        """Helper method to convert the state dictionary to a Data() object.

        This allows for a forward pass through the trained GNN.
        
        args:
            state: dictionary containing the input data
            
        returns:
            Data: object for running a forward pass on the GNN"""
        x = torch.tensor(state['state']['node_features'], dtype=torch.float)
        edge_index = torch.tensor(state['graph_structure']['edge_index'], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
