"""
collect_train_data.py

Reads configuration from src/config/train-data.yml, generates a heterogeneous collection
of synthetic graphs, runs TSQC on each, and streams out training (state-action) pairs
to a JSONL file.
"""
import random, os, sys, yaml

from src.utils.graph import Graph

from src.algorithms.tsqc import TSQC
from src.algorithms.greedy_lb import GreedyLB

CONFIG_PATH = "src/config/collect-train-data.yml"

def load_config(path: str) -> dict:
    """
    Load YAML config from the given path.
    """
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)

def collect_train_data():
    """
    Main execution.
    """
    config = load_config(CONFIG_PATH)

    eco_path = config['eco_path']
    bio_path = config['bio_path']

    search_depth_grid = config['search_depth_grid']
    gamma_values = config['gamma_range']

    output_path = config['output_path']
    master_seed = config['master_seed']
    master_rng = random.Random(master_seed)

    seed_per_gamma = {
        gamma: master_rng.randint(1, 2**31-1) for gamma in gamma_values
    }

    time_limit = config['time_limit']

    for path in [eco_path, bio_path]:
        if path == "data/training/biological/":
            continue
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

                initial_k = GreedyLB(custom_graph_obj, gamma).run()

                tsqc = TSQC(
                    graph=custom_graph_obj,
                    gamma=gamma,
                    collect_train_data=True,
                    time_limit=time_limit,
                    train_data_filepath=output_path,
                    rng=random.Random(seed_per_gamma[gamma]),
                    search_depth_grid=search_depth_grid
                )
                tsqc.solve(initial_k=initial_k)

    print(f"Done! Data saved to {output_path}.")

collect_train_data()
