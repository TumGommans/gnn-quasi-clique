"""main script to run the algorithm."""

import yaml
import os
import random
import sys

import networkx as nx
import matplotlib.pyplot as plt

from src.utils.graph import Graph
from src.algorithms.tsqc_prr import PRR_TSQC
from src.algorithms.tsqc import TSQC

CONFIG_FILE_PATH = "src/config/single-run.yml"

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

    gamma = config['gamma']
    if not (isinstance(gamma, (int, float)) and 0 < gamma <= 1):
         print(f"Error: 'gamma' value ({gamma}) must be a number between 0 (exclusive) and 1 (inclusive).")
         sys.exit(1)

    print("Configuration validated.")
    return True

# ------------------------------------------------------------------------------------------------------------
# Visualization Function
# ------------------------------------------------------------------------------------------------------------
def visualize_quasi_clique(original_graph_obj, quasi_clique_nodes, gamma_val, graph_filepath):
    """
    Visualizes the original graph and highlights a quasi-clique.

    Args:
        original_graph_obj: Your custom Graph object.
        quasi_clique_nodes: A list or set of node identifiers in the quasi-clique.
        gamma_val: The gamma value used for the quasi-clique definition.
        graph_filepath: Path to the graph file, used for title.
    """
    print("\nAttempting to visualize the graph and quasi-clique...")

    G_nx = nx.Graph()
    all_nodes = list(original_graph_obj.vertices)
    
    extracted_edges = set()
    for u_node in original_graph_obj.vertices:
        for v_node in original_graph_obj.get_neighbors(u_node):
            if u_node < v_node:
                extracted_edges.add((u_node, v_node))
            else:
                extracted_edges.add((v_node, u_node))
    all_edges = list(extracted_edges)

    if not all_nodes:
        print("Graph has no nodes according to your Graph object, cannot visualize.")
        return

    G_nx.add_nodes_from(all_nodes)
    G_nx.add_edges_from(all_edges)

    if not G_nx.nodes():
        print("NetworkX graph is empty after attempting to add nodes/edges, cannot visualize.")
        return
    
    quasi_clique_nodes_set = set(quasi_clique_nodes)

    node_colors = []
    node_sizes = []
    for node in G_nx.nodes():
        if node in quasi_clique_nodes_set:
            node_colors.append('blue')
            node_sizes.append(150)
        else:
            node_colors.append('gray')
            node_sizes.append(50)

    edge_colors = []
    edge_widths = []
    edge_alphas = []
    for u, v in G_nx.edges():
        if u in quasi_clique_nodes_set and v in quasi_clique_nodes_set:
            edge_colors.append('blue')
            edge_widths.append(2.0)
            edge_alphas.append(1.0)
        else:
            edge_colors.append('gray')
            edge_widths.append(0.5)
            edge_alphas.append(0.7)

    print("Calculating graph layout (this may take a while for large graphs)...")
    pos = None
    if G_nx.number_of_nodes() > 0:
        try:
            if G_nx.number_of_nodes() < 200:
                 pos = nx.spring_layout(G_nx, k=0.15, iterations=20, seed=42)
            else:
                 pos = nx.kamada_kawai_layout(G_nx)
        except Exception as e:
            print(f"Layout calculation failed ({e}), falling back to random layout.")
            pos = nx.random_layout(G_nx, seed=42)
    else:
        print("No nodes in graph to calculate layout.")
        return

    plt.figure(figsize=(15, 12))
    nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G_nx, pos, edge_color=edge_colors, width=edge_widths, alpha=edge_alphas)

    if G_nx.number_of_nodes() < 100:
        nx.draw_networkx_labels(G_nx, pos, font_size=8, font_color='black')
    else:
        print("Skipping node labels for clarity due to large graph size.")

    graph_name = os.path.basename(graph_filepath)
    plt.axis('off')
    plt.tight_layout()
    
    output_filename = f"quasi_clique_visualization_{graph_name.split('.')[0]}_gamma{gamma_val}.png"
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Visualization successfully saved to: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"Error saving visualization to file: {e}")
    
    plt.show()
    print("Visualization complete.")

# ------------------------------------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------------------------------------

def main():
    config = load_config(CONFIG_FILE_PATH)
    validate_config(config)

    filepath = config['graph_filepath']
    gamma_val = config['gamma']
    initial_k_val = config['initial_k']
    search_depth_val = config['search_depth_L']
    max_iterations_val = config['max_iterations_It']
    random_seed = config.get('random_seed', None)

    if random_seed is not None and isinstance(random_seed, int):
        random.seed(random_seed)
        print(f"Random seed set to: {random_seed}")

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

    if custom_graph_obj.num_vertices == 0:
        print("Graph is empty. Exiting.")
        return

    print("Initializing TSQC...")
    tsqc_algo = PRR_TSQC(
        graph=custom_graph_obj,
        gamma=gamma_val,
        max_iterations_It=max_iterations_val,
        search_depth_L=search_depth_val,
        best_known=True
    )

    print("Starting TSQC process...")
    best_clique_nodes, _ = tsqc_algo.solve(initial_k=initial_k_val)

    print("\n--- Final Result ---")
    if best_clique_nodes and len(best_clique_nodes) > 0 :
        found_clique_size = len(best_clique_nodes)
        if hasattr(tsqc_algo, 'best_quasi_clique_size') and tsqc_algo.best_quasi_clique_size > 0:
            found_clique_size = tsqc_algo.best_quasi_clique_size

        print(f"Largest {gamma_val}-quasi-clique found has size: {found_clique_size}")
        visualize_quasi_clique(custom_graph_obj, best_clique_nodes, gamma_val, filepath)
    else:
        print(f"No satisfying {gamma_val}-quasi-clique found by the TSQC algorithm within the given parameters.")

if __name__ == "__main__":
    main()