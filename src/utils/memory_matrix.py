"""
Simplified Adaptive Memory Co-occurrence Matrix for TSQC Tie Breaking
"""

from collections import defaultdict, deque
from typing import Set, Dict, Optional
import yaml


class AdaptiveMemoryMatrix:
    """
    Simple co-occurrence matrix that learns from good solutions and helps break ties.
    """
    
    def __init__(self, num_vertices: int, config_path: Optional[str] = None):
        """
        Initialize the co-occurrence matrix.
        
        Args:
            num_vertices: Total number of vertices in the graph
            config_path: Path to YAML configuration file
        """
        self.n = num_vertices
        
        # Load configuration
        config = self._load_config(config_path)
        self.memory_decay = config['cooccurrence_matrix']['memory_decay']
        self.burn_in_iterations = config['cooccurrence_matrix']['burn_in_iterations']
        self.quality_percentile = config['cooccurrence_matrix']['quality_percentile']
        
        # Sparse matrix storage: M[i][j] = co-occurrence strength between vertices i and j
        self.M = defaultdict(lambda: defaultdict(float))
        
        # Quality tracking for learning threshold
        self.quality_history = deque(maxlen=200)  # Keep recent quality values
        self.iteration_count = 0
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file or use defaults."""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Warning: Config file {config_path} not found. Using defaults.")
        
        # Simple default configuration
        return {
            'cooccurrence_matrix': {
                'memory_decay': 0.95,
                'burn_in_iterations': 500,
                'quality_percentile': 0.8
            }
        }
    
    def update_solution_quality(self, solution: Set[int], quality: float):
        """
        Update the co-occurrence matrix based on a solution and its quality.
        Only learns from high-quality solutions.
        """
        self.iteration_count += 1
        self.quality_history.append(quality)
        
        # Don't start learning until we have enough data
        if len(self.quality_history) < 50:
            return
            
        # Calculate quality threshold (only learn from top solutions)
        sorted_qualities = sorted(self.quality_history)
        threshold_idx = int(self.quality_percentile * len(sorted_qualities))
        quality_threshold = sorted_qualities[threshold_idx]
        
        # Only update matrix for high-quality solutions
        if quality >= quality_threshold:
            self._update_matrix(solution)
    
    def _update_matrix(self, solution: Set[int]):
        """Update co-occurrence matrix for vertices that appear together in good solutions."""
        solution_list = list(solution)
        
        # Update pairwise co-occurrences
        for i, u in enumerate(solution_list):
            for j, v in enumerate(solution_list):
                if i != j:  # Don't update diagonal
                    # Strengthen co-occurrence between u and v
                    old_val = self.M[u][v]
                    self.M[u][v] = self.memory_decay * old_val + (1 - self.memory_decay)
    
    def calculate_tie_breaking_score(self, u: int, v: int, current_S: Set[int]) -> float:
        """
        Calculate tie-breaking score for swap (u out, v in).
        
        Simple scoring: How well does v fit with the remaining vertices in S?
        """
        # Skip if still in burn-in period
        if self.iteration_count < self.burn_in_iterations:
            return 0.0
        
        # Calculate how well v collaborates with vertices remaining in S
        S_without_u = current_S - {u}
        if not S_without_u:
            return 0.0
        
        # Affinity score: how often has v appeared with vertices in S?
        affinity_score = sum(self.M[v][w] for w in S_without_u) / len(S_without_u)
        
        return affinity_score
    
    def get_matrix_stats(self) -> Dict:
        """Get simple statistics about the matrix."""
        non_zero_pairs = sum(1 for i in self.M for j in self.M[i] if self.M[i][j] > 0.01)
        
        return {
            'iteration_count': self.iteration_count,
            'quality_samples': len(self.quality_history),
            'learned_pairs': non_zero_pairs,
            'burn_in_complete': self.iteration_count >= self.burn_in_iterations
        }