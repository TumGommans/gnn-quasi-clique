"""The TSQ algorithm."""

import random
import math
from collections import defaultdict

class TSQ:
    """
    Object that implements the core Tabu Search procedure.

    Find a k-gamma-quasi-clique, starting from an initial solution,
    representing the inner loop called by the main TSQC algorithm.
    """
    def __init__(
            self, 
            graph, 
            gamma,
            k,
            L, 
            initial_S, 
            current_It, 
            update_freq_method, 
            max_total_It, 
            rng=None,
            use_cum_saturation=False,
            use_common_neighbors=False
        ):
        """
        Initializes the TSQ search object.

        Args:
            graph:                The graph object.
            gamma:                The quasi-clique density threshold.
            k:                    The target size of the quasi-clique.
            L:                    Search depth (max iterations without improvement).
            initial_S:            The starting solution subset.
            current_It:           The current global iteration count (for max check and tabu).
            update_freq_method:   Method from TSQC to update frequency memory.
            max_total_It:         The overall maximum iteration limit.
            use_cum_saturation:   Whether to use cumulative saturation tie-breaking.
            use_common_neighbors: Whether to use common neighbors tie-breaking.
        """
        # Check that both tie-breaking mechanisms are not enabled simultaneously
        tie_breaking_methods = sum([use_cum_saturation, use_common_neighbors])
        if tie_breaking_methods > 1:
            raise ValueError("Cannot enable multiple tie-breaking mechanisms simultaneously")
        
        self._graph = graph
        self._gamma = gamma
        self._k = k
        self._L = L
        self._initial_S = initial_S.copy()
        self._start_It = current_It
        self._update_frequency = update_freq_method
        self._max_total_It = max_total_It
        self._rng = rng

        self._S = None
        self._S_star = None
        self._f_S = 0
        self._f_star = 0
        self._iterations_consumed = 0
        self._tabu_until_u = defaultdict(int)
        self._tabu_until_v = defaultdict(int)

        self.intensification_count = 0
        self.tie_breaker_count = 0

        # Cumulative saturation
        self._use_cum_saturation = use_cum_saturation
        if self._use_cum_saturation:
            # Maximum degree ∆G
            self._DeltaG = max(len(graph.get_neighbors(v)) for v in graph.vertices)
            self._cum_sat = {v: 0 for v in graph.vertices}
            # track last-change step t0(v) and TS(v)
            self._last_change = {v: 0 for v in graph.vertices}
            self._time_in_S = defaultdict(int)

        # Common neighbors tie-breaking
        self._use_common_neighbors = use_common_neighbors

    def run(self):
        """ Executes the TSQ procedure."""
        self._S = self._initial_S.copy()
        self._S_star = self._S.copy()
        self._f_S = self._evaluate_solution(self._S)
        self._f_star = self._f_S

        consecutive_no_improvement_I = 0
        self._iterations_consumed = 0

        target_f = self._calculate_target_edges(self._k, self._gamma)
        print(f"    TSQ Start: k={self._k}, Initial f(S)={self._f_S}, Target f >= {target_f}")

        ### Check whether the initial solution is already a legal clique
        if self._f_star >= target_f:
            print(f"    TSQ End: Best f* = {self._f_star} found in {self._iterations_consumed} iterations. I = {consecutive_no_improvement_I}.")
            return self._S_star, self._iterations_consumed
    
        while consecutive_no_improvement_I < self._L:
            current_global_It = self._start_It + self._iterations_consumed
            if current_global_It >= self._max_total_It:
                print("    TSQ Stop: Global max iterations reached.")
                break

            # Update cumulative saturation before each iteration
            if self._use_cum_saturation:
                self._update_cumulative_saturation(current_global_It)

            degrees_in_S = self._calculate_all_degrees_relative_to_S(self._S)
            A, B, MinInS, MaxOutS = self._determine_critical_sets(self._S, degrees_in_S, current_global_It)

            u_selected, v_selected = None, None

            # Determine if intensification is possible and execute it
            can_intensify = (MaxOutS - MinInS >= 0) or \
                            (MaxOutS - MinInS == -1 and any(self._graph.has_edge(u,v) for u in A for v in B))

            if can_intensify:
                best_swaps_T = self._find_best_swaps(A, B)
                u_selected, v_selected = self._intensification_select_swap(A, B, MinInS, MaxOutS, best_swaps_T)

            # If intensification wasn't possible, execute diversification
            if u_selected is None and v_selected is None:
                 best_swaps_T = self._find_best_swaps(A, B)
                 u_selected, v_selected = self._diversification_select_swap(A, B, best_swaps_T, degrees_in_S)

            # Perform the swap if a pair is selected by intensification/diversification
            if u_selected is not None and v_selected is not None:
                edge_exists = 1 if self._graph.has_edge(u_selected, v_selected) else 0
                delta_uv = degrees_in_S.get(v_selected, 0) - degrees_in_S.get(u_selected, 0) - edge_exists

                # Update cumulative saturation tracking before the swap
                if self._use_cum_saturation:
                    self._update_vertex_state_change(u_selected, v_selected, current_global_It)

                self._S.remove(u_selected)
                self._S.add(v_selected)
                self._f_S += delta_uv

                tabu_tenure_u, tabu_tenure_v = self._calculate_tabu_tenures(self._k, self._f_S)
                self._tabu_until_u[u_selected] = current_global_It + tabu_tenure_u
                self._tabu_until_v[v_selected] = current_global_It + tabu_tenure_v

                self._iterations_consumed += 1

                if self._is_legal_quasi_clique(self._S, self._k, self._gamma):
                    print(f"    TSQ Found Legal Clique at It {current_global_It}! f(S) = {self._f_S}")
                    self._S_star = self._S.copy()
                    self._f_star = self._f_S
                    break

                if self._f_S > self._f_star:
                    self._S_star = self._S.copy()
                    self._f_star = self._f_S
                    consecutive_no_improvement_I = 0
                else:
                    consecutive_no_improvement_I += 1
            else:
                # This should not happen, potentially only if A or B is empty
                print(f"      Warning It {current_global_It}: No valid swap found (A={A}, B={B}). Skipping iteration.")
                consecutive_no_improvement_I += 1
                self._iterations_consumed += 1

        print(f"    TSQ End: Best f* = {self._f_star} found in {self._iterations_consumed} iterations. I = {consecutive_no_improvement_I}.")
        return self._S_star, self._iterations_consumed

# ------------------------------------------------------------------------------------------------------------
# Common Neighbors Tie-Breaking Methods
# ------------------------------------------------------------------------------------------------------------

    def _calculate_common_neighbors_score(self, u, v):
        """
        Calculate the common neighbors score s1(u,v) for a swap.
        
        Args:
            u: Vertex to be removed from S
            v: Vertex to be added to S
        """
        if not self._use_common_neighbors:
            return 0
        
        S_prime = (self._S - {u}) | {v}
        common_neighbors = set()
        for w in S_prime:
            common_neighbors.update(self._graph.get_neighbors(w))
        
        common_neighbors -= S_prime
        
        return len(common_neighbors)

# ------------------------------------------------------------------------------------------------------------
# Cumulative Saturation Methods
# ------------------------------------------------------------------------------------------------------------

    def _calculate_saturation(self, v):
        """Calculate saturation of vertex v at current step."""
        if not self._use_cum_saturation:
            return 0
        
        saturation = 0
        for neighbor in self._graph.get_neighbors(v):
            if (neighbor in self._S) != (v in self._S):
                saturation += 1
        return saturation

    def _update_cumulative_saturation(self):
        """Update cumulative saturation for all vertices at current step."""
        if not self._use_cum_saturation:
            return
        
        for v in self._graph.vertices:
            current_saturation = self._calculate_saturation(v)
            
            if v in self._S:
                self._time_in_S[v] += 1
            
            self._cum_sat[v] += current_saturation
            if v in self._S:
                self._cum_sat[v] -= self._DeltaG

    def _update_vertex_state_change(self, u_removed, v_added, current_step):
        """Update tracking when vertices change state.
        
        args:
            u_removed:    the vertex removed
            v_added:      the vertex added
            current_step: the current iteration"""
        if not self._use_cum_saturation:
            return
        
        self._last_change[u_removed] = current_step
        self._last_change[v_added] = current_step
        self._time_in_S[u_removed] = 0
        self._time_in_S[v_added] = 0

    def _get_cumulative_saturation_score(self, v):
        """Get the cumulative saturation score for vertex v."""
        if not self._use_cum_saturation:
            return 0
        return self._cum_sat[v]

# ------------------------------------------------------------------------------------------------------------
# Intensification & Diversification Helpers
# ------------------------------------------------------------------------------------------------------------

    def _intensification_select_swap(self, A, B, MinInS, MaxOutS, best_swaps_T):
        """ Selects swap pair (u, v) during intensification (Delta_uv >= 0).
        
        args:
            A:       the critical set containing vertices to remove
            B:       the critical set containing vertices to add
            MinInS:  the minimum degree present in S
            MaxOutS: the maximum degree present from a vertex out of S into S
        """
        u_selected, v_selected = None, None

        if MaxOutS - MinInS >= 0 and best_swaps_T:
             u_selected, v_selected = self._select_swap_tie_breaking(best_swaps_T)
             return u_selected, v_selected
        
        if MaxOutS - MinInS - 1 >= 0:
            edge_pairs = [(u, v) for u in A for v in B if self._graph.has_edge(u, v)]
            if edge_pairs:
                 u_selected, v_selected = self._select_swap_tie_breaking(edge_pairs)
                 return u_selected, v_selected

        return None, None

    def _select_swap_tie_breaking(self, swap_candidates):
        """Selects a swap from candidates using tie-breaking mechanisms.
        
        args:
            swap_candidates: set of swaps yielding equal scores on the primary
                             scoring function
        """
        if not swap_candidates: 
            return None, None

        self.intensification_count += 1
        if len(swap_candidates) > 1:
            self.tie_breaker_count += 1

        # Common neighbors tie-breaking
        if self._use_common_neighbors and len(swap_candidates) > 1:
            best_score = float('-inf')
            best_swaps = []
            
            for u, v in swap_candidates:
                score = self._calculate_common_neighbors_score(u, v)
                
                if score > best_score:
                    best_score = score
                    best_swaps = [(u, v)]
                elif score == best_score:
                    best_swaps.append((u, v))
            
            # If we still have ties after common neighbors, break randomly
            return self._rng.choice(best_swaps) if self._rng else random.choice(best_swaps)

        # Cumulative saturation tie-breaking
        if self._use_cum_saturation and len(swap_candidates) > 1:
            best_score = float('-inf')
            best_swaps = []
            
            for u, v in swap_candidates:
                gamma_v = self._get_cumulative_saturation_score(v)
                gamma_u = self._get_cumulative_saturation_score(u)
                score = gamma_v - gamma_u
                
                if score > best_score:
                    best_score = score
                    best_swaps = [(u, v)]
                elif score == best_score:
                    best_swaps.append((u, v))
            
            return self._rng.choice(best_swaps) if self._rng else random.choice(best_swaps)
        
    def _diversification_select_swap(self, A, B, best_swaps_T, degrees_in_S):
        """ Selects swap pair (u, v) during diversification.
        
        args:
            A: the critical set containing vertices to remove
            B: the critical set containing vertices to add
            best_swaps_T: set of swaps yielding the best scores
            degrees_in_S: dictionary mapping vertices to their degrees in S
        """
        u_selected, v_selected = None, None

        target_edges = self._calculate_target_edges(self._k, self._gamma)
        l_diff = max(0, target_edges - self._f_S)
        prob_P = min(((l_diff + 2) / self._k) if self._k > 0 else 0, 0.1)

        if self._rng.random() if self._rng else random.random() < prob_P:
            if not self._S: return None, None

            u_selected = self._rng.choice(list(self._S)) if self._rng else random.choice(list(self._S))

            graph_density = self._graph.density
            h_threshold = math.floor(0.85 * self._gamma * self._k) if graph_density <= 0.5 else math.floor(self._gamma * self._k)

            V_minus_S = list(self._graph.vertices - self._S)
            if not V_minus_S: return u_selected, None

            candidates_v = [v for v in V_minus_S if degrees_in_S.get(v, 0) < h_threshold]

            if candidates_v:
                v_selected = self._rng.choice(candidates_v) if self._rng else random.choice(candidates_v)
            else:
                 v_selected = self._rng.choice(V_minus_S) if self._rng else random.choice(V_minus_S)
        else:
            if best_swaps_T:
                 u_selected, v_selected = self._select_swap_tie_breaking(best_swaps_T)
            elif A and B:
                 u_selected = self._rng.choice(list(A)) if self._rng else random.choice(list(A))
                 v_selected = self._rng.choice(list(B)) if self._rng else random.choice(list(B))

        return u_selected, v_selected

# ------------------------------------------------------------------------------------------------------------
# Tabu list management
# ------------------------------------------------------------------------------------------------------------
    
    def _calculate_tabu_tenures(self, k, f_S):
        """ Calculates adaptive tabu tenures Tu and Tv.
        
        args:
            k: the target clique size
            f_S: the evaluation function value at the current step
        """
        target_edges = self._calculate_target_edges(k, self._gamma)

        l_val = min(max(0, target_edges - f_S), 10)
        C_val = max(math.floor(k / 40.0) if k > 0 else 0, 6)

        if self._rng:
            Tu = math.ceil(l_val) + self._rng.randint(0, max(0, C_val - 1)) 
            Tv = math.ceil(0.6 * l_val) + self._rng.randint(0, max(0, math.floor(0.6 * C_val) - 1))
        else:
            Tu = math.ceil(l_val) + random.randint(0, max(0, C_val - 1))
            Tv = math.ceil(0.6 * l_val) + random.randint(0, max(0, math.floor(0.6 * C_val) - 1))
        return Tu, Tv

    def _determine_critical_sets(self, S, degrees_in_S, current_iteration):
        """ Finds the critical sets A and B based on non-tabu vertices.
        
        args:
            S: the current solution
            degrees_in_S: dictionary mapping vertices to their degrees in S
            current_iteration: the current iteration
        """
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
                MaxOutS = max(MaxOutS, degrees_in_S.get(v, 0))

        A = {u for u in non_tabu_in_S if degrees_in_S.get(u, 0) == MinInS} if MinInS != float('inf') else set()
        B = {v for v in non_tabu_out_S if degrees_in_S.get(v, 0) == MaxOutS} if MaxOutS != -float('inf') else set()

        if MaxOutS == -float('inf'): MaxOutS = 0
        if MinInS == float('inf'): MinInS = 0

        return A, B, MinInS, MaxOutS

    def _find_best_swaps(self, A, B):
        """Finds the set T of best swaps (gain = MaxOutS - MinInS, {u,v} not edge).
        
        args:
            A: the critical set containing vertices to remove
            B: the critical set containing vertices to add
        """
        T = set()
        if A and B:
             for u in A:
                 for v in B:
                     if not self._graph.has_edge(u, v):
                         T.add((u, v))
        return list(T)

# ------------------------------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------------------------------

    def _evaluate_solution(self, subset):
        """Calculates f(S) = number of edges in induced subgraph.
        
        args:
            subset: the set of vertices to compute f(S) over
        """
        if not subset or len(subset) < 2: return 0
        return self._graph.get_induced_subgraph_edges(subset)

    def _calculate_target_edges(self, k, gamma):
        """Calculates the minimum number of edges for a legal k-gamma-quasi-clique.
        
        args:
            k: the target quasi clique size
            gamma: the density threshold
        """
        if k < 2: return 0
        return math.ceil(gamma * k * (k - 1) / 2.0)

    def _is_legal_quasi_clique(self, subset, k, gamma):
        """Checks if a subset is a legal k-gamma-quasi-clique using current f_S.
        
        args:
            subset: the current solution to check
            k: the target quasi clique size
            gamma: the density threshold
        """
        if not subset or len(subset) != k: return False
        return self._f_S >= self._calculate_target_edges(k, gamma)

    def _calculate_all_degrees_relative_to_S(self, S):
        """Calculates d(v) = |{u in S | {u, v} in E}| for all vertices v in V.
        
        args:
            S: the subset to which degrees are computed
        """
        degrees = defaultdict(int)
        if not S:
             for v in self._graph.vertices: degrees[v] = 0
             return degrees

        S_neighbors = {u: self._graph.get_neighbors(u).intersection(S) for u in S}
        for u in S:
            degrees[u] = len(S_neighbors[u])

        V_minus_S = self._graph.vertices - S
        for v in V_minus_S:
            degrees[v] = len(self._graph.get_neighbors(v).intersection(S))

        return degrees
