"""Data classes for storing state action pairs structurally."""

from dataclasses import dataclass, asdict
from typing import List
import json

@dataclass
class GraphStructure:
    num_vertices: int
    edge_index: List[List[int]]

@dataclass
class GraphFeatures:
    gamma: float
    initial_target_k: int
    current_target_k: int
    graph_density: float
    restart_count: int

@dataclass
class State:
    node_features: List[List[float]]
    graph_features: GraphFeatures

@dataclass
class Action:
    optimal_L_index: int
    optimal_L_value: int

@dataclass
class StateActionPair:
    graph_structure: GraphStructure
    state: State
    action: Action

    def to_json(self) -> str:
        """
        Serialize this GraphData to a JSON-formatted string matching the desired structure.
        """
        data_dict = asdict(self)
        try:
            return json.dumps(data_dict, separators=(",",":"), allow_nan=False)
        except ValueError as e:
            print(f"VALUE_ERROR (likely NaN/Inf): {e} in dict: {data_dict}")
            return json.dumps(data_dict, separators=(",",":"))