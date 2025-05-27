"""
TSQC-PRR: TSQC with Path-Relinking Restarts
Extends TSQC by maintaining an elite pool of solutions and using path relinking
for intelligent restarts.
"""
import time

from src.utils.graph import Graph
from src.algorithms.tsq import TSQ
from src.algorithms.tsqc import TSQC
from src.utils.path_relinker import PathRelinker


class PRR_TSQC(TSQC):
    """
    TSQC with Path-Relinking Restarts (PRR).

    Maintains an elite pool of up to `elite_size` best solutions (by highest density),
    and when TSQ stagnates, selects the one with greatest density to relink towards.
    """
    def __init__(
        self,
        graph: Graph,
        gamma: float,
        max_iterations_It: int = 10**8,
        search_depth_L: int = 1000,
        rng=None,
        best_known: bool = False,
        time_limit: int = 3600,
        elite_size: int = 5,
    ):
        super().__init__(graph, gamma, max_iterations_It, search_depth_L, rng, best_known, time_limit)
        self._elite_pool = []  # list of sets
        self._elite_size = elite_size
        self._relinker = PathRelinker(self._graph, self._gamma, self._g_freq)

    def solve(self, initial_k: int = 1):
        k = max(1, initial_k)
        current_best_clique = set()
        start_time_overall = time.time()
        iteration_count_It = 0
        best_runtime_found = 0.0

        k_is_incremented = False

        print(f"Starting TSQC-PRR search with gamma = {self._gamma}, initial_k = {k}, "
              f"It_max = {self._It_max}, L = {self._L}, elite_size = {self._elite_size}")

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

            while iteration_count_It < self._It_max:
                print(f"  Calling TSQ for k={k} (Starts at Global It: {iteration_count_It})")
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

                best_S_star_tsq, it_consumed_tsq = tsq.run()
                iteration_count_It += it_consumed_tsq

                if self._is_legal_quasi_clique(best_S_star_tsq, k, self._gamma):
                    end_time_k = time.time()
                    elapsed = end_time_k - start_time_overall
                    print(f"  PRR: SUCCESS: Legal {k}-quasi-clique found by TSQ! "
                          f"(Time to find: {elapsed:.2f}s)")
                    found_clique_for_k = best_S_star_tsq.copy()
                    current_best_clique = found_clique_for_k
                    self.best_quasi_clique_found = current_best_clique
                    self.best_quasi_clique_size = k
                    best_runtime_found = elapsed
                    self._update_elite_pool(found_clique_for_k)
                    break
                else:
                    print("  TSQ finished without legal clique. Generating PRR-based restart...")
                    current_S = self._prr_restart(best_S_star_tsq, k)
                    if not current_S:
                        print(f"  PRR: Could not generate restart for k={k}. Stopping search.")
                        found_clique_for_k = None
                        break
                    if time.time() - start_time_k > self._time_limit:
                        print(f"  Timeout reached for finding k={k}. Stopping search.")
                        found_clique_for_k = None
                        break

            if found_clique_for_k:
                if not self._best_known:
                    k += 1
                    k_is_incremented = True
                    self._elite_pool.clear()
                else:
                    break
            else:
                if not k_is_incremented:
                    termination_time = time.time()
                    best_runtime_found = termination_time - start_time_overall
                break

        print(f"\n--- TSQC-PRR Complete ---")
        print(f"Max {self._gamma}-quasi-clique size: {self.best_quasi_clique_size}")
        print(f"Time to best found clique: {best_runtime_found:.2f} seconds")
        print(f"Global iterations consumed: {iteration_count_It}")

        return self.best_quasi_clique_found, best_runtime_found

    def _update_elite_pool(self, clique: set):
        """
        Add a new legal clique to the elite pool, keeping only the highest-density ones.
        """
        for S in self._elite_pool:
            if S == clique:
                return
        self._elite_pool.append(clique.copy())

        def density(S):
            k = len(S)
            max_edges = k * (k - 1) / 2
            return (self._graph.get_induced_subgraph_edges(S) / max_edges) if max_edges > 0 else 0

        self._elite_pool.sort(key=lambda S: density(S), reverse=True)
        if len(self._elite_pool) > self._elite_size:
            self._elite_pool = self._elite_pool[: self._elite_size]

    def _prr_restart(self, current_S: set, k: int) -> set:
        """
        Generate a restart by path relinking current_S toward the best elite target of the same size k.
        If no same-size elite exists, fallback to default restart.
        """
        same_k_elite = [S for S in self._elite_pool if len(S) == k]
        if not same_k_elite:
            return self._default_restart_strategy(k)

        def density(S):
            max_edges = k * (k - 1) / 2
            return (self._graph.get_induced_subgraph_edges(S) / max_edges) if max_edges > 0 else 0

        target = max(same_k_elite, key=density)
        _, best_solution = self._relinker.relink(set(current_S), set(target))
        return best_solution
