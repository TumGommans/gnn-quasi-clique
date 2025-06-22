# gnn-quasi-clique

This repository contains all the source code for my BSc Thesis: Deep Learning-Guided Tabu Search for the Maximum Quasi-Clique Problem. The thesis paper can be found at ./thesis.pdf. To reproduce all results, one can open the repo in a VSCode development container using Docker/Rancher desktop. Follow these steps:
1. Open the repository in VSCode
2. Press cmd + shift + p (macOS) / ctrl + shift + p (Windows)
3. Select 'Reopen in container'

Now, simply run the bash script at the root the repo by running the following command:
```
caffeinate -i bash reproduce.sh
```
After running this (takes some days), all the tables containing results in the paper should be written to the relevant results directories.

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
    ├── scripts/                 -> main execution scripts in which data is gathered, model is trained, algorithms are run and tables are created
    └── utils/                   -> utilities
```
