"""The TSQ algorithm."""

import random
import math
from collections import defaultdict

class TSQ:
    """
    Implements the core Tabu Search procedure (Algorithm 2 from the paper)
    to find a k-gamma-quasi-clique, starting from an initial solution.
    This represents the inner loop called by the main TSQC algorithm.
    """
    def __init__(self, graph, gamma, k, L, initial_S, current_It, update_freq_method, max_total_It):
        """
        Initializes the TSQ search for a fixed k.

        Args:
            graph (Graph): The graph object.
            gamma (float): The quasi-clique density threshold.
            k (int): The target size of the quasi-clique.
            L (int): Search depth (max iterations without improvement).
            initial_S (set): The starting solution subset.
            current_It (int): The current global iteration count (for max check and tabu).
            update_freq_method (callable): Method from TSQC to update frequency memory.
            max_total_It (int): The overall maximum iteration limit.
        """
        self._graph = graph
        self._gamma = gamma
        self._k = k
        self._L = L
        self._initial_S = initial_S.copy()
        self._start_It = current_It
        self._update_frequency = update_freq_method
        self._max_total_It = max_total_It

        self._S = None
        self._S_star = None
        self._f_S = 0
        self._f_star = 0
        self._iterations_consumed = 0
        self._tabu_until_u = defaultdict(int)
        self._tabu_until_v = defaultdict(int)


    def run(self):
        """ Executes the TSQ procedure (Algorithm 2). """
        self._S = self._initial_S.copy()
        self._S_star = self._S.copy()
        self._f_S = self._evaluate_solution(self._S) # Initial evaluation
        self._f_star = self._f_S

        consecutive_no_improvement_I = 0
        self._iterations_consumed = 0

        target_f = self._calculate_target_edges(self._k, self._gamma)
        print(f"    TSQ Start: k={self._k}, Initial f(S)={self._f_S}, Target f >= {target_f}")

        while consecutive_no_improvement_I < self._L:
            current_global_It = self._start_It + self._iterations_consumed
            if current_global_It >= self._max_total_It:
                print("    TSQ Stop: Global max iterations reached.")
                break

            degrees_in_S = self._calculate_all_degrees_relative_to_S(self._S)
            A, B, MinInS, MaxOutS = self._determine_critical_sets(self._S, degrees_in_S, current_global_It)

            u_selected, v_selected = None, None # Reset selection

            # Determine if intensification is possible and attempt it
            can_intensify = (MaxOutS - MinInS >= 0) or \
                            (MaxOutS - MinInS == -1 and any(self._graph.has_edge(u,v) for u in A for v in B))

            if can_intensify:
                best_swaps_T = self._find_best_swaps(A, B)
                u_selected, v_selected = self._intensification_select_swap(A, B, MinInS, MaxOutS, best_swaps_T)

            # If intensification failed or wasn't possible, attempt diversification
            if u_selected is None and v_selected is None:
                 # Note: best_swaps_T might need recalculation if not done above, but should be same if A,B unchanged
                 best_swaps_T = self._find_best_swaps(A, B) # Recalculate just in case (cheap)
                 u_selected, v_selected = self._diversification_select_swap(A, B, best_swaps_T, degrees_in_S)

            # Perform the swap if a valid pair was selected
            if u_selected is not None and v_selected is not None:
                # --- Perform Swap ---
                edge_exists = 1 if self._graph.has_edge(u_selected, v_selected) else 0
                delta_uv = degrees_in_S.get(v_selected, 0) - degrees_in_S.get(u_selected, 0) - edge_exists

                self._S.remove(u_selected)
                self._S.add(v_selected)
                self._f_S += delta_uv

                self._update_frequency(u_selected, v_selected, self._k)

                tabu_tenure_u, tabu_tenure_v = self._calculate_tabu_tenures(self._k, self._f_S)
                self._tabu_until_u[u_selected] = current_global_It + tabu_tenure_u
                self._tabu_until_v[v_selected] = current_global_It + tabu_tenure_v

                self._iterations_consumed += 1 # Increment iteration count *after* a successful swap/step

                # --- Check Legality and Best Solution ---
                if self._is_legal_quasi_clique(self._S, self._k, self._gamma):
                    print(f"    TSQ Found Legal Clique at It {current_global_It}! f(S) = {self._f_S}")
                    self._S_star = self._S.copy()
                    self._f_star = self._f_S
                    break # Exit TSQ run loop

                if self._f_S > self._f_star:
                    self._S_star = self._S.copy()
                    self._f_star = self._f_S
                    consecutive_no_improvement_I = 0 # Reset I
                else:
                    consecutive_no_improvement_I += 1 # Increment I

            else:
                # --- No Valid Swap Found ---
                # This should now happen much less often, potentially only if A or B is empty
                print(f"      Warning It {current_global_It}: No valid swap found (A={A}, B={B}). Skipping iteration.")
                # Increment I to prevent infinite loop if truly stuck
                consecutive_no_improvement_I += 1
                # Also increment iterations_consumed here? If skipping counts as an iteration. Let's say yes.
                self._iterations_consumed += 1


        # End of TSQ while loop
        print(f"    TSQ End: Best f* = {self._f_star} found in {self._iterations_consumed} iterations. I = {consecutive_no_improvement_I}.")
        return self._S_star, self._iterations_consumed


    # --- Intensification / Diversification Helpers ---

    def _intensification_select_swap(self, A, B, MinInS, MaxOutS, best_swaps_T):
        """ Selects swap pair (u, v) during intensification (Delta_uv >= 0). """
        u_selected, v_selected = None, None

        # Case 1: Non-edge swap with gain >= 0 (T = non-edge pairs with max gain = MaxOutS - MinInS)
        if MaxOutS - MinInS >= 0 and best_swaps_T:
             u_selected, v_selected = self._select_swap_tie_breaking(best_swaps_T)
             # print(f"DEBUG: Intensify Case 1: Selected {u_selected}, {v_selected} from T")
             return u_selected, v_selected # Return immediately

        # Case 2: Edge swap with gain >= 0 (Delta = MaxOutS - MinInS - 1 >= 0)
        if MaxOutS - MinInS - 1 >= 0:
            edge_pairs = [(u, v) for u in A for v in B if self._graph.has_edge(u, v)]
            if edge_pairs:
                 u_selected, v_selected = self._select_swap_tie_breaking(edge_pairs)
                 # print(f"DEBUG: Intensify Case 2: Selected {u_selected}, {v_selected} from edge_pairs")
                 return u_selected, v_selected # Return immediately

        # If neither case yielded a swap, intensification fails for this iteration
        # print(f"DEBUG: Intensify failed: MaxOutS-MinInS={MaxOutS - MinInS}, T empty={not best_swaps_T}, edge_pairs empty={not 'edge_pairs' in locals() or not edge_pairs}")
        return None, None # <<< REMOVED INCORRECT FALLBACK HERE


    def _select_swap_tie_breaking(self, swap_candidates):
        """ Selects a swap from candidates (default: random). Modifiable. """
        if not swap_candidates: return None, None
        return random.choice(swap_candidates)

    def _diversification_select_swap(self, A, B, best_swaps_T, degrees_in_S):
        """ Selects swap pair (u, v) during diversification."""
        u_selected, v_selected = None, None

        target_edges = self._calculate_target_edges(self._k, self._gamma)
        l_diff = max(0, target_edges - self._f_S)
        prob_P = min(((l_diff + 2) / self._k) if self._k > 0 else 0, 1.0)

        # print(f"DEBUG: Diversify: f_S={self._f_S}, target={target_edges}, l_diff={l_diff}, P={prob_P:.3f}")

        if random.random() < prob_P:
            # High perturbation
            # print("DEBUG: Diversify High Perturbation")
            if not self._S: return None, None # Cannot select u if S is empty
            u_selected = random.choice(list(self._S))

            graph_density = self._graph.density
            h_threshold = math.floor(0.85 * self._gamma * self._k) if graph_density <= 0.5 else math.floor(self._gamma * self._k)

            V_minus_S = list(self._graph.vertices - self._S)
            if not V_minus_S: return u_selected, None # Cannot select v

            # Find candidates v with degree towards S < h_threshold
            candidates_v = [v for v in V_minus_S if degrees_in_S.get(v, 0) < h_threshold]

            if candidates_v:
                v_selected = random.choice(candidates_v)
                # print(f"DEBUG: Diversify High: Found v < h: {v_selected}")
            else:
                 v_selected = random.choice(V_minus_S) # Fallback
                 # print(f"DEBUG: Diversify High: Fallback v: {v_selected}")

        else:
            # Low perturbation
            # print("DEBUG: Diversify Low Perturbation")
            if best_swaps_T: # Select from best non-improving (non-edge)
                 u_selected, v_selected = self._select_swap_tie_breaking(best_swaps_T)
                 # print(f"DEBUG: Diversify Low: Selected from T: {u_selected}, {v_selected}")
            elif A and B: # Fallback: select random non-tabu pair
                 u_selected = random.choice(list(A))
                 v_selected = random.choice(list(B))
                 # print(f"DEBUG: Diversify Low: Fallback A/B: {u_selected}, {v_selected}")
            # Else: if A or B empty, no swap possible -> handled by main loop check

        return u_selected, v_selected


    # --- Tabu List Management --- (No changes needed)

    def _calculate_tabu_tenures(self, k, f_S):
        """ Calculates adaptive tabu tenures Tu and Tv. """
        target_edges = self._calculate_target_edges(k, self._gamma)
        l_val = min(max(0, target_edges - f_S), 10)
        C_val = max(math.floor(k / 40.0) if k > 0 else 0, 6)
        Tu = math.ceil(l_val) + random.randint(0, max(0, C_val - 1))
        Tv_base = 0.6 * l_val
        Tv_rand_limit = max(0, math.floor(0.6 * C_val) - 1)
        Tv = math.ceil(Tv_base) + random.randint(0, Tv_rand_limit)
        return max(1, Tu), max(1, Tv)

    def _determine_critical_sets(self, S, degrees_in_S, current_iteration):
        """ Finds the critical sets A and B based on non-tabu vertices. """
        MinInS = float('inf')
        non_tabu_in_S = []
        for u in S:
             if current_iteration >= self._tabu_until_u.get(u, 0):
                non_tabu_in_S.append(u)
                MinInS = min(MinInS, degrees_in_S.get(u, 0))

        MaxOutS = -float('inf')
        non_tabu_out_S = []
        V_minus_S = self._graph.vertices - S
        for v in V_minus_S:
            if current_iteration >= self._tabu_until_v.get(v, 0):
                non_tabu_out_S.append(v)
                MaxOutS = max(MaxOutS, degrees_in_S.get(v, 0)) # d(v) relative to S

        A = {u for u in non_tabu_in_S if degrees_in_S.get(u, 0) == MinInS} if MinInS != float('inf') else set()
        B = {v for v in non_tabu_out_S if degrees_in_S.get(v, 0) == MaxOutS} if MaxOutS != -float('inf') else set()

        if MaxOutS == -float('inf'): MaxOutS = 0
        if MinInS == float('inf'): MinInS = 0

        return A, B, MinInS, MaxOutS

    def _find_best_swaps(self, A, B):
        """ Finds the set T of best swaps (gain = MaxOutS - MinInS, {u,v} not edge). """
        T = set()
        if A and B:
             for u in A:
                 for v in B:
                     if not self._graph.has_edge(u, v):
                         T.add((u, v))
        return list(T)


    # --- Evaluation and Helper Functions --- (No changes needed)

    def _evaluate_solution(self, subset):
        """ Calculates f(S) = number of edges in induced subgraph (for initialization). """
        if not subset or len(subset) < 2: return 0
        return self._graph.get_induced_subgraph_edges(subset)

    def _calculate_target_edges(self, k, gamma):
        """ Calculates the minimum number of edges for a legal k-gamma-quasi-clique. """
        if k < 2: return 0
        return math.ceil(gamma * k * (k - 1) / 2.0)

    def _is_legal_quasi_clique(self, subset, k, gamma):
        """ Checks if a subset is a legal k-gamma-quasi-clique using current f_S. """
        if not subset or len(subset) != k: return False
        # Use the incrementally updated value for efficiency during the run
        return self._f_S >= self._calculate_target_edges(k, gamma)

    def _calculate_all_degrees_relative_to_S(self, S):
        """ Calculates d(v) = |{u in S | {u, v} in E}| for all vertices v in V. """
        degrees = defaultdict(int)
        if not S:
             for v in self._graph.vertices: degrees[v] = 0
             return degrees

        # Calculate degrees for nodes *in* S towards S
        S_neighbors = {u: self._graph.get_neighbors(u).intersection(S) for u in S}
        for u in S:
            degrees[u] = len(S_neighbors[u])

        # Calculate degrees for nodes *outside* S towards S
        V_minus_S = self._graph.vertices - S
        for v in V_minus_S:
            degrees[v] = len(self._graph.get_neighbors(v).intersection(S)) # More concise

        return degrees