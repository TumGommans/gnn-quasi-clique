"""Script defining the training loop for the GNN, with optional manual hyperparameters."""

import json
import os
import yaml
import torch

from torch import optim
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler

from torch_geometric.loader import DataLoader

from src.gnn.model import SearchDepthGNN
from src.gnn.train import train_epoch, eval_epoch, RestartDataset

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
torch.set_num_threads(10)            
torch.set_num_interop_threads(10)  

with open('src/config/gnn-model.yml') as f:
    model_cfg = yaml.safe_load(f)

with open('src/config/train-gnn.yml') as f:
    train_cfg = yaml.safe_load(f)

use_manual = train_cfg.get('use_manual_params', False)

device = torch.device("cpu")

full_ds = RestartDataset(train_cfg['data_path'])

if use_manual:
    print("Manual hyperparameter training enabled. \n ...")
    cfg = dict(model_cfg)

    # Build model
    model = SearchDepthGNN(
        node_feat_dim=cfg['node_feat_dim'],
        graph_feat_dim=cfg['graph_feat_dim'],
        hidden_dim=int(cfg['hidden_dim']),
        num_gin_layers=int(cfg['num_gin_layers']),
        mlp_layers_per_gin=int(cfg['mlp_layers_per_gin']),
        final_mlp_layers=int(cfg['final_mlp_layers']),
        readout=cfg['readout'],
        dropout=cfg['dropout'],
        activation=cfg['activation'],
        num_classes=cfg['num_classes']
    ).to(device)

    # Split dataset
    train_len = int(train_cfg['train_val_split'] * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_labels  = [ int(full_ds[i].y) for i in train_ds.indices ]
    train_weights = [ full_ds.class_weights[l].item() for l in train_labels ]
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg['batch_size']),
        sampler=train_sampler,
        num_workers=10,      
        pin_memory=True    
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg['batch_size']),
        shuffle=False,
        num_workers=10,  
        pin_memory=True 
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    # Training loop
    for epoch in range(int(cfg['epochs'])):
        loss = train_epoch(model, train_loader, optimizer, device, full_ds.class_weights)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{cfg['epochs']}, loss={loss:.4f}")

    # In-sample accuracy
    model.eval()
    train_acc = eval_epoch(model, train_loader, device)
    print(f"In-sample (training) accuracy: {train_acc:.4f}")

else:
    # Hyperparameter tuning with Hyperopt
    from hyperopt import fmin, space_eval, tpe, hp, Trials

    def objective(params):
        cfg = params
        model = SearchDepthGNN(
            node_feat_dim=model_cfg['node_feat_dim'],
            graph_feat_dim=model_cfg['graph_feat_dim'],
            hidden_dim=int(cfg['hidden_dim']),
            num_gin_layers=int(cfg['num_gin_layers']),
            mlp_layers_per_gin=int(cfg['mlp_layers_per_gin']),
            final_mlp_layers=int(cfg['final_mlp_layers']),
            readout=cfg['readout'],
            dropout=cfg['dropout'],
            activation=cfg['activation'],
            num_classes=model_cfg['num_classes']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
        train_len = int(train_cfg['train_val_split'] * len(full_ds))
        val_len = len(full_ds) - train_len
        train_ds, val_ds = random_split(full_ds, [train_len, val_len])

        train_labels  = [ int(full_ds[i].y) for i in train_ds.indices ]
        train_weights = [ full_ds.class_weights[l].item() for l in train_labels ]
        train_sampler = WeightedRandomSampler(
            train_weights,
            num_samples=len(train_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg['batch_size']),
            sampler=train_sampler
        )
        val_loader = DataLoader(val_ds,
            batch_size=int(cfg['batch_size']),
            shuffle=False
        )

        best_val = 0.0
        for _ in range(int(cfg['epochs'])):
            train_epoch(model, train_loader, optimizer, device, full_ds.class_weights)
            best_val = max(best_val, eval_epoch(model, val_loader, device))

        return -best_val

    space = {
        'num_gin_layers': hp.choice('num_gin_layers', train_cfg['hyperopt_space']['num_gin_layers']),
        'hidden_dim': hp.choice('hidden_dim', train_cfg['hyperopt_space']['hidden_dim']),
        'mlp_layers_per_gin': hp.choice('mlp_layers_per_gin', train_cfg['hyperopt_space']['mlp_layers_per_gin']),
        'final_mlp_layers': hp.choice('final_mlp_layers', train_cfg['hyperopt_space']['final_mlp_layers']),
        'readout': hp.choice('readout', train_cfg['hyperopt_space']['readout']),
        'activation': hp.choice('activation', train_cfg['hyperopt_space']['activation']),
        'dropout': hp.uniform('dropout', train_cfg['hyperopt_space']['dropout']['min'], train_cfg['hyperopt_space']['dropout']['max']),
        'lr': hp.loguniform('lr', train_cfg['hyperopt_space']['lr']['min'], train_cfg['hyperopt_space']['lr']['max']),
        'batch_size': hp.choice('batch_size', train_cfg['hyperopt_space']['batch_size']),
        'epochs': hp.choice('epochs', train_cfg['hyperopt_space']['epochs']),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=train_cfg['max_evals'],
        trials=trials
    )

    best_params = space_eval(space, best)
    os.makedirs("results/gnn-params", exist_ok=True)
    with open("results/gnn-params/hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Optimal hyperparameters successfully written to results.")

    # Retrain on full dataset
    final_cfg = best_params
    final_model = SearchDepthGNN(
        node_feat_dim=model_cfg['node_feat_dim'],
        graph_feat_dim=model_cfg['graph_feat_dim'],
        hidden_dim=int(final_cfg['hidden_dim']),
        num_gin_layers=int(final_cfg['num_gin_layers']),
        mlp_layers_per_gin=int(final_cfg['mlp_layers_per_gin']),
        final_mlp_layers=int(final_cfg['final_mlp_layers']),
        readout=final_cfg['readout'],
        dropout=final_cfg['dropout'],
        activation=final_cfg['activation'],
        num_classes=model_cfg['num_classes']
    ).to(device)

    optim_final = optim.Adam(final_model.parameters(), lr=final_cfg['lr'])

    full_labels  = [ int(full_ds[i].y) for i in range(len(full_ds)) ]
    full_weights = [ full_ds.class_weights[l].item() for l in full_labels ]
    full_sampler = WeightedRandomSampler(
        full_weights,
        num_samples=len(full_weights),
        replacement=True
    )
    full_loader = DataLoader(
        full_ds,
        batch_size=int(final_cfg['batch_size']),
        sampler=full_sampler
    )

    print("Retraining on full dataset with best hyperparameters…")
    for epoch in range(int(final_cfg['epochs'])):
        loss = train_epoch(final_model, full_loader, optim_final, device, full_ds.class_weights)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{final_cfg['epochs']}, loss={loss:.4f}")

    out_path = train_cfg['output_path']
    torch.save(final_model.state_dict(), out_path)
    print(f"Saved final model weights to {out_path}")
