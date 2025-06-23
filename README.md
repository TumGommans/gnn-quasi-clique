# gnn-quasi-clique

This repository contains the source code for my BSc thesis: Deep Learning-Guided Tabu Search for the Maximum Quasi-Clique Problem. The thesis paper can be found at ./thesis.pdf. All classes and methods contain docstrings, including descriptions of input arguments and return values.

## Reproducing the Results

To reproduce all the results in this research, run:

```bash
caffeinate -i bash reproduce.sh
```

This executes the `reproduce.sh` script at the root of the repository, which in turn runs:

1. **Generate training data**

   ```bash
   python -m src.scripts.get_train_data
   ```

   * Runs **TSQC** on every instance in `data/training/biological` and `data/training/ecological`.
   * For each value of *k*, considers all *L* ∈ {500, 1000, 5000}, selects the optimal *L*<sup>\*</sup>, and pairs it with the node feature matrix **X**.
   * Writes each `(X, L*)` pair to `data/training/state_action_pairs.jsonl`.

2. **Train the GNN**

   ```bash
   python -m src.scripts.train_gnn
   ```

   * Optimizes hyperparameters on the training set, then retrains on train + validation.
   * Saves model parameters and best hyperparameters in `results/gnn/gnn_weights.pth` and `hyperparameters.json`, respectively.
   * Saves test-holdout metrics and confusion matrix to `results/gnn/metrics.json`.

3. **Run experiments & collect JSON results**

   ```bash
   python -m src.scripts.get_json_results
   ```

   * Executes all experiments as described in the Experiments section.
   * Dumps JSON outputs to `results/dimacs/` and `results/real-life/`.

4. **Generate LaTeX tables**

   ```bash
   python -m src.scripts.get_tables
   ```

   Reads all JSON results, formats them into LaTeX tables, and writes:

   * `results/dimacs/dimacs.tex` -> Table 2 in the paper
   * `results/real-life/real_life.tex` -> Table 3 in the paper
   * `results/dimacs/dimacs_appendix.tex` -> Table 5 in the paper
   * `results/dimacs/real_life_appendix.tex` -> Table 6 in the paper
   * `results/gnn/hyperparameters.tex` -> Table 4 in the paper

## Development Container

I provide a Docker-based dev container under `.devcontainer/`. To get started:

1. Install **Docker Desktop** and ensure the daemon is running.
2. In VS Code, choose **Remote-Containers: Reopen in Container**.
   This installs all dependencies listed in `requirements.txt` into the container.

---

For reference, the repo is structured as follows:
```
├── data/
│   ├── dimacs/                  -> all the DIMACS graphs
│   ├── real-life/               -> all the real-life graphs
│   └── training    
│       ├── biological/          -> all the biological graphs used for collecting trainin data
│       └── ecological/          -> all the ecological graphs used for collecting trainin data
├── results/
│   ├── dimacs/                  -> results for the DIMACS graphs
│   ├── gnn/                     -> trained GNN weights and optimized hyperparameters
│   └── real-life/               -> results for the real-life graphs
└── src/
    ├── algorithms/              -> the source code for the algorithms
    ├── config/
    │   ├── gnn/                 -> config files regarding GNN scripts
    │   └── run/                 -> config files for the algorithm executions
    ├── gnn/
    ├── notebooks/               -> analysis of class distribution
    ├── scripts/                 -> execution scripts for gathering data, model training, algorithm execution and table generation
    └── utils/                   -> utilities: custom objects
