"""
Script for gathering representative training data for the GNN in DeepTSQC

Reads configuration from src/config/train-data.yml and runs TSQC on a variety of graphs
to collect and store optimal state-action pairs for training the GNN.
"""

import random, os, sys, yaml, time

from src.utils.graph import Graph

from src.algorithms.tsqc import TSQC
from src.algorithms.greedy_lb import GreedyLB

from src.utils.state_action_pair import (
    GraphStructure,
    State,
    Action,
    StateActionPair
)

CONFIG_PATH = "src/config/gnn/get-train-data.yml"

def load_config(path: str) -> dict:
    """
    Load YAML config from the given path.
    """
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)

def collect_train_data():
    """
    Main execution.

    Writes all state action pairs to a .jsonl file at the specified output path.
    """
    config = load_config(CONFIG_PATH)

    eco_path = config['eco_path']
    bio_path = config['bio_path']

    search_depth_grid = config['search_depth_grid']
    gamma_values = config['gamma_range']

    output_path = config['output_path']
    master_seed = config['master_seed']
    master_rng = random.Random(master_seed)

    num_runs = config['num_runs']

    seed_per_gamma = {
        gamma: master_rng.randint(1, 2**31-1) for gamma in gamma_values
    }

    time_limit = config['time_limit']
    train_data_fh = open(config['output_path'], 'a', buffering=1)

    for path in [bio_path, eco_path]:
        for name in os.listdir(path):
            filepath = os.path.join(path, name)

            print(f"Loading graph from: {filepath}")
            custom_graph_obj = Graph()
            try:
                custom_graph_obj.load_from_edgelist_file(filepath)
                print(f"Graph loaded successfully: {custom_graph_obj.num_vertices} vertices, {custom_graph_obj.num_edges} edges.")
            except FileNotFoundError:
                print(f"Error: Graph file specified in config ('{filepath}') not found.")
                sys.exit(1)
            except Exception as e:
                print(f"Failed to load graph: {e}")
                sys.exit(1)

            for gamma in gamma_values:
                for i in range(num_runs):
                    initial_k = GreedyLB(custom_graph_obj, gamma).run()
                    k = initial_k
                    start = time.time()
                    while time.time() - start <= time_limit:
                        best_depth = None
                        best_depth_index = None
                        best_runtime = float('inf')
                        
                        successful = False
                        for i, depth in enumerate(search_depth_grid):
                            tsqc = TSQC(
                                graph=custom_graph_obj,
                                gamma=gamma,
                                search_depth_L=depth,
                                time_limit=time_limit,
                                rng=random.Random(seed_per_gamma[gamma] + i),
                                best_known=True
                            )
                            quasi_clique, runtime = tsqc.solve(initial_k=k)
                            if round(runtime,1) < round(best_runtime,1) and len(quasi_clique) == k:
                                successful = True
                                best_runtime = runtime
                                best_depth = depth
                                best_depth_index = i
                        if successful:
                            json_line = store_state_action_pair(
                                custom_graph_obj, 
                                tsqc.initial_S, 
                                k,
                                gamma,
                                best_depth,
                                best_depth_index
                                )
                            train_data_fh.write(json_line + "\n")
                        k += 1

    print(f"Done! Data saved to {output_path}.")

def store_state_action_pair(
    graph: Graph, 
    subset: set, 
    k: int, 
    gamma: float, 
    L: int, 
    L_index: int
) -> StateActionPair:
    """
    Store an optimal state-action pair for training the GNN policy.
    
    args:
        graph: the current graph instance
        subset: the starting solution
        k: the target quasi clique size
        gamma: the density threshold
        L: the optimal search depth
        L_index: the index of the optimal search depth (used for the model)
    
    returns:
        StateActionPair: data object containing all the state-action information
    """
    return StateActionPair(
        graph_structure=GraphStructure(
            num_vertices=graph.num_vertices,
            edge_index=graph.edge_index
        ),
        state=State(
            node_features=graph.get_node_features(subset, k, gamma)
        ),
        action=Action(
            optimal_L_index=L_index,
            optimal_L_value=L
        )
    ).to_json()

collect_train_data()
