"""main script to run the algorithm."""

import yaml
import os
import json
import sys
import random
import torch

from src.utils.graph import Graph
from src.algorithms.tsqc import TSQC
from src.algorithms.tsqc_prr import PRR_TSQC
from src.algorithms.tsqc_gnn import DeepTSQC

from src.gnn.model import SearchDepthGNN


CONFIG_PATH_DIMACS = "src/config/dimacs.yml"
CONFIG_PATH_REAL_LIFE = "src/config/real-life.yml"

HYPERPARAMETER_PATH = "results/gnn-params/hyperparameters.json"
WEIGHTS_PATH = "results/gnn-params/gnn_weights.pth"

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
    required_keys = ['methods', 'filepath', 'search_depth_L', 'max_iterations_It', 'time_limit', 'master_seed', 'gamma', 'k']
    missing_keys = [key for key in required_keys if key not in config or config[key] is None]

    if missing_keys:
        print(f"Error: Missing required keys in configuration file: {', '.join(missing_keys)}")
        sys.exit(1)

    gamma_values = config['gamma']
    for gamma in gamma_values:
        if not (isinstance(gamma, (int, float)) and 0 < gamma <= 1):
            print(f"Error: 'gamma' value ({gamma}) must be a number between 0 (exclusive) and 1 (inclusive).")
            sys.exit(1)

    print("Configuration validated.")
    return True

# ------------------------------------------------------------------------------------------------------------
# Algorithm Execution
# ------------------------------------------------------------------------------------------------------------

def get_results(path):
    config = load_config(path)
    validate_config(config)

    results = {}

    methods = config['methods']
    data_path = config['filepath']
    search_depth_val = config['search_depth_L']
    max_iterations_val = config['max_iterations_It']
    time_limit = config['time_limit']

    master_seed = config['master_seed']
    random.seed(master_seed)
    seeds = [random.randint(0, 2**31 - 1) for _ in range(10)]

    gamma_values = config['gamma']
    k_values = config['k']

    name = os.path.basename(os.path.normpath(data_path))
    results_path = os.path.join("results", name)
    json_name = f"{name.replace('-', '_')}.json"

    output_dir  = os.path.join(results_path)
    output_file = os.path.join(output_dir, json_name)
    os.makedirs(output_dir, exist_ok=True)

    gnn = get_gnn_from_config()

    for name in os.listdir(data_path):
        filepath = os.path.join(data_path, name)
        instance_name, _ = os.path.splitext(name)
        
        results[instance_name] = {}
    
        print(f"Loading graph from: {filepath}")
        custom_graph_obj = Graph()
        try:
            custom_graph_obj.load_from_edgelist_file(filepath)
            print(f"Graph loaded successfully: {custom_graph_obj.num_vertices} vertices, {custom_graph_obj.num_edges} edges.")

            results[instance_name]["num_vertices"] = custom_graph_obj.num_vertices
            results[instance_name]["num_edges"] = custom_graph_obj.num_edges
            results[instance_name]["density"] = custom_graph_obj.density
        except FileNotFoundError:
            print(f"Error: Graph file specified in config ('{filepath}') not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Failed to load graph: {e}")
            sys.exit(1)

        if custom_graph_obj.num_vertices == 0:
            print("Graph is empty. Exiting.")
            return

        for gamma_val in gamma_values:
            results[instance_name][gamma_val] = {}
            for method in methods:
                results[instance_name][gamma_val][method] = {}
                results[instance_name][gamma_val][method]["objectives"] = []
                results[instance_name][gamma_val][method]["runtimes"] = []

                for i in range(10):
                    if method == "tsqc":
                        tsqc = TSQC(
                            graph=custom_graph_obj,
                            gamma=gamma_val,
                            max_iterations_It=max_iterations_val,
                            search_depth_L=search_depth_val,
                            rng=random.Random(seeds[i]),
                            best_known=True,
                            time_limit=time_limit
                        )
                    elif method == "prr_tsqc":
                        tsqc = PRR_TSQC(
                            graph=custom_graph_obj,
                            gamma=gamma_val,
                            max_iterations_It=max_iterations_val,
                            search_depth_L=search_depth_val,
                            rng=random.Random(seeds[i]),
                            best_known=True,
                            time_limit=time_limit
                        )
                    else:
                        tsqc = DeepTSQC(
                            graph=custom_graph_obj,
                            gamma=gamma_val,
                            gnn=gnn,
                            max_iterations_It=max_iterations_val,
                            rng=random.Random(seeds[i]),
                            best_known=True,
                            time_limit=time_limit
                        )                   

                    print("Starting TSQC process...")
                    best_clique_nodes, time = tsqc.solve(initial_k=k_values[gamma_val][instance_name])

                    print("\n--- Final Result ---")
                    if best_clique_nodes and len(best_clique_nodes) > 0 :
                        found_clique_size = len(best_clique_nodes)
                        if hasattr(tsqc, 'best_quasi_clique_size') and tsqc.best_quasi_clique_size > 0:
                            found_clique_size = tsqc.best_quasi_clique_size

                        print(f"Largest {gamma_val}-quasi-clique found has size: {found_clique_size}")

                        results[instance_name][gamma_val][method]["objectives"].append(found_clique_size)
                        results[instance_name][gamma_val][method]["runtimes"].append(time)
                        if tsqc.intensification_count == 0:
                            results[instance_name][gamma_val][method]["tie_breaking_proportion"] = 0
                        else:
                            results[instance_name][gamma_val][method]["tie_breaking_proportion"] = \
                            tsqc.tie_breaker_count / tsqc.intensification_count
                    else:
                        results[instance_name][gamma_val][method]["objectives"].append(0)
                        results[instance_name][gamma_val][method]["runtimes"].append(time)
                        print(f"No satisfying {gamma_val}-quasi-clique found by the TSQC algorithm within the given parameters.")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def get_gnn_from_config():
    hyperparams = load_config(HYPERPARAMETER_PATH)

    # Initialize model with best hyperparameters
    model = SearchDepthGNN(
        node_feat_dim=5,
        graph_feat_dim=5,
        hidden_dim=int(hyperparams['hidden_dim']),
        num_gin_layers=int(hyperparams['num_gin_layers']),
        mlp_layers_per_gin=int(hyperparams['mlp_layers_per_gin']),
        final_mlp_layers=int(hyperparams['final_mlp_layers']),
        readout=hyperparams['readout'],
        dropout=hyperparams['dropout'],
        activation=hyperparams['activation'],
        num_classes=3
    )

    # Load trained weights
    model.load_state_dict(torch.load('results/gnn-params/gnn_weights.pth', map_location='cpu'))
    return model

for path in (CONFIG_PATH_REAL_LIFE, CONFIG_PATH_DIMACS):
    get_results(path)