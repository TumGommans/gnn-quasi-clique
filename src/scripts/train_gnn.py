"""Script defining the training loop for the GNN, with optional manual hyperparameters."""

import json
import os
import yaml
import torch
import numpy as np

from typing import Any

from torch import optim
from torch.utils.data import random_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix
)

from torch_geometric.loader import DataLoader

from hyperopt import fmin, space_eval, tpe, hp, Trials

from src.gnn.model import SearchDepthGNN
from src.gnn.train import train_epoch, RestartDataset

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"

torch.set_num_threads(10)            
torch.set_num_interop_threads(10)  

def evaluate_model_metrics(
    model: SearchDepthGNN, 
    loader: DataLoader, 
    device: str, 
    num_classes: int
) -> dict:
    """
    Evaluate model and return comprehensive metrics.
    
    args:
        model: the trained GNN
        loader: the data loader containing all batches
        device: the device to run it on
        num_classes: the number of classes
    
    returns:
        dict: contains the evaluation metrics and confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }

with open('src/config/gnn/gnn-model.yml') as f:
    model_cfg = yaml.safe_load(f)

with open('src/config/gnn/train-gnn.yml') as f:
    train_cfg = yaml.safe_load(f)

use_manual = train_cfg.get('use_manual_params', False)

device = torch.device("cpu")
full_ds = RestartDataset(train_cfg['data_path'])

# Create 80/10/10 split
train_len = int(0.8 * len(full_ds))
val_len = int(0.1 * len(full_ds))
test_len = len(full_ds) - train_len - val_len
train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])

# Create test loader for final evaluation
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

if use_manual:
    print("Manual hyperparameter training enabled.")
    print("Loading optimal hyperparameters from results/gnn/hyperparameters.json...")
    
    with open('results/gnn/hyperparameters.json', 'r') as f:
        cfg = json.load(f)
    
    print(f"Loaded hyperparameters: {cfg}")
    
    train_val_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
    print(f"Training on {len(train_val_ds)} samples (90% of data)")
    
    # Build model with hyperparameters
    model = SearchDepthGNN(
        node_feat_dim=train_cfg['node_feat_dim'],
        hidden_dim=int(cfg['hidden_dim']),
        num_gin_layers=int(cfg['num_gin_layers']),
        mlp_layers_per_gin=int(cfg['mlp_layers_per_gin']),
        final_mlp_layers=int(cfg['final_mlp_layers']),
        readout=cfg['readout'],
        dropout=cfg['dropout'],
        activation=cfg['activation'],
        num_classes=train_cfg['num_classes']
    ).to(device)

    train_val_labels = []
    for i in train_ds.indices:
        train_val_labels.append(int(full_ds[i].y))
    for i in val_ds.indices:
        train_val_labels.append(int(full_ds[i].y))
    
    train_val_weights = [full_ds.class_weights[l].item() for l in train_val_labels]
    train_val_loader = DataLoader(
        train_val_ds,
        batch_size=int(cfg['batch_size']),
        shuffle=True,
        num_workers=0,      
        pin_memory=True    
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    # Training loop
    print(f"Training for {cfg['epochs']} epochs...")
    for epoch in range(int(cfg['epochs'])):
        loss = train_epoch(model, train_val_loader, optimizer, device, full_ds.class_weights)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{cfg['epochs']}, loss={loss:.4f}")

    # Final evaluation on test set
    print("Evaluating final model on test set...")
    test_metrics = evaluate_model_metrics(model, test_loader, device, train_cfg['num_classes'])
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    
    # Save metrics
    os.makedirs("results/gnn", exist_ok=True)
    with open("results/gnn/metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
    print("Test metrics saved to results/gnn/metrics.json")

    # Save final model
    out_path = train_cfg['output_path']
    torch.save(model.state_dict(), out_path)
    print(f"Saved final model weights to {out_path}")

else:
    def objective(params: Any) -> float:
        """Objective function to optimize hyperparameters over.
        
        args:
            params: single configuration of hyperparamers

        returns:
            -best_f1: negative F1 (due to minimization)        
        """
        cfg = params
        model = SearchDepthGNN(
            node_feat_dim=train_cfg['node_feat_dim'],
            hidden_dim=int(cfg['hidden_dim']),
            num_gin_layers=int(cfg['num_gin_layers']),
            mlp_layers_per_gin=int(cfg['mlp_layers_per_gin']),
            final_mlp_layers=int(cfg['final_mlp_layers']),
            readout=cfg['readout'],
            dropout=cfg['dropout'],
            activation=cfg['activation'],
            num_classes=train_cfg['num_classes']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg['batch_size']),
            shuffle=True
        )
        val_loader = DataLoader(val_ds,
            batch_size=int(cfg['batch_size']),
            shuffle=False
        )

        best_f1 = 0.0
        for _ in range(int(cfg['epochs'])):
            train_epoch(model, train_loader, optimizer, device, full_ds.class_weights)
            val_metrics = evaluate_model_metrics(model, val_loader, device, train_cfg['num_classes'])
            best_f1 = max(best_f1, val_metrics['f1'])

        return -best_f1

    # Define the search space
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
    
    # Call the minimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=train_cfg['max_evals'],
        trials=trials
    )

    best_params = space_eval(space, best)
    os.makedirs("results/gnn", exist_ok=True)
    with open("results/gnn/hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Optimal hyperparameters successfully written to results.")

    # Retrain on train+val (90%) with best hyperparameters
    print("Retraining on 90% of data (train+val) with best hyperparameters…")
    train_val_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
    
    final_cfg = best_params
    final_model = SearchDepthGNN(
        node_feat_dim=train_cfg['node_feat_dim'],
        hidden_dim=int(final_cfg['hidden_dim']),
        num_gin_layers=int(final_cfg['num_gin_layers']),
        mlp_layers_per_gin=int(final_cfg['mlp_layers_per_gin']),
        final_mlp_layers=int(final_cfg['final_mlp_layers']),
        readout=final_cfg['readout'],
        dropout=final_cfg['dropout'],
        activation=final_cfg['activation'],
        num_classes=train_cfg['num_classes']
    ).to(device)

    optim_final = optim.Adam(final_model.parameters(), lr=final_cfg['lr'])
    
    train_val_loader = DataLoader(
        train_val_ds,
        batch_size=int(final_cfg['batch_size']),
        shuffle=True
    )

    for epoch in range(int(final_cfg['epochs'])):
        loss = train_epoch(final_model, train_val_loader, optim_final, device, full_ds.class_weights)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{final_cfg['epochs']}, loss={loss:.4f}")

    # Final evaluation on test set
    print("Evaluating final model on test set...")
    test_metrics = evaluate_model_metrics(final_model, test_loader, device, train_cfg['num_classes'])
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    
    # Save metrics
    with open("results/gnn/metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
    print("Test metrics saved to results/gnn/metrics.json")

    # Save final model
    out_path = train_cfg['output_path']
    torch.save(final_model.state_dict(), out_path)
    print(f"Saved final model weights to {out_path}")