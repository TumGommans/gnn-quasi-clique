"""main script to run the algorithm."""
import yaml
import os
import random
import sys

from utils.graph import Graph
from algorithms.tsqc import TSQC

CONFIG_FILE_PATH = "config/config.yml"

def load_config(config_path):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from '{config_path}'")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file '{config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        sys.exit(1)

def validate_config(config):
    """Basic validation for required config keys."""
    required_keys = ['graph_filepath', 'gamma', 'initial_k', 'search_depth_L', 'max_iterations_It']
    missing_keys = [key for key in required_keys if key not in config or config[key] is None]

    if missing_keys:
        print(f"Error: Missing required keys in configuration file: {', '.join(missing_keys)}")
        sys.exit(1)

    # Validate gamma range
    gamma = config['gamma']
    if not (isinstance(gamma, (int, float)) and 0 < gamma <= 1):
         print(f"Error: 'gamma' value ({gamma}) must be a number between 0 (exclusive) and 1 (inclusive).")
         sys.exit(1)

    # Add more specific validations as needed (e.g., types, positive integers)

    print("Configuration validated.")
    return True

# --- Main Execution ---
def main():
    # Load and validate configuration
    config = load_config(CONFIG_FILE_PATH)
    validate_config(config)

    # Extract parameters from config
    filepath = config['graph_filepath']
    gamma_val = config['gamma']
    initial_k_val = config['initial_k']
    search_depth_val = config['search_depth_L']
    max_iterations_val = config['max_iterations_It']
    random_seed = config.get('random_seed', None) # Use .get for optional keys

    # --- Optional: Set Random Seed ---
    if random_seed is not None and isinstance(random_seed, int):
        random.seed(random_seed)
        print(f"Random seed set to: {random_seed}")

    # --- Load Graph ---
    print(f"Loading graph from: {filepath}")
    try:
        graph = Graph()
        # Make sure graph.py uses load_from_edgelist_file or equivalent
        graph.load_from_edgelist_file(filepath)
    except FileNotFoundError:
        print(f"Error: Graph file specified in config ('{filepath}') not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load graph: {e}")
        sys.exit(1)

    if graph.num_vertices == 0:
        print("Graph is empty. Exiting.")
        return

    # --- Initialize TSQC ---
    print("Initializing TSQC...")
    tsqc_process = TSQC(graph=graph,
                        gamma=gamma_val,
                        max_iterations_It=max_iterations_val,
                        search_depth_L=search_depth_val)

    # --- Run TSQC Process ---
    print("Starting TSQC process...")
    best_clique = tsqc_process.solve(initial_k=initial_k_val)

    # --- Print Results ---
    print("\n--- Final Result ---")
    if best_clique:
        print(f"Largest {gamma_val}-quasi-clique found has size: {tsqc_process.best_quasi_clique_size}")
        # print(f"Vertices: {sorted(list(tsqc_process.best_quasi_clique_found))}")
    else:
        print(f"No {gamma_val}-quasi-clique found within limits.")

if __name__ == "__main__":
    main()