"""Script defining training functionalities of the GNN."""

import json, torch
import torch.nn.functional as F

from collections import Counter
from src.gnn.model import SearchDepthGNN

from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class RestartDataset(Dataset):
    """
    Object to make tailored dataset for restarts. 
    
    Initially built for each restart of TSQ, but now used it for each run of TSQC given (k, gamma)
    """

    def __init__(self, path: str):
        """initializes an object, given the data path."""
        self.path = path
        self.offsets = []
        counter = Counter()
        with open(path, 'rb') as f:
            offset = f.tell()
            for idx, raw in enumerate(f):
                self.offsets.append(offset)
                obj = json.loads(raw)
                counter[int(obj['action']['optimal_L_index'])] += 1
                offset = f.tell()

        total = sum(counter.values())
        self.class_weights = torch.tensor([
            total / (len(counter) * counter[i])
            for i in range(len(counter))
        ], dtype=torch.float)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            f.seek(self.offsets[idx])
            obj = json.loads(f.readline())
        x = torch.tensor(obj['state']['node_features'], dtype=torch.float)
        ei = torch.tensor(obj['graph_structure']['edge_index'], dtype=torch.long)
        y = torch.tensor(obj['action']['optimal_L_index'], dtype=torch.long)
        return Data(x=x, edge_index=ei, y=y)

def train_epoch(
    model: SearchDepthGNN, 
    loader: DataLoader, 
    opt: Adam, 
    device: str, 
    class_weights: Tensor
) -> float:
    """Function to train an epoch.
    
    args:
        model: the GNN architecture from SearchDepthGNN
        loader: the train data loader object containing the batches, over which gradients are computed
        opt: the optimizer, I used Adam
        device: the device to run computations on (CPU/CUDA)
        class_weights: the class weights, used to mitigate the slight imbalance
    
    returns:
        float: the average loss of this epoch
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, batch.y, weight=class_weights.to(device))
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def eval_epoch(
    model: SearchDepthGNN, 
    loader: DataLoader, 
    device: str
) -> float:
    """Function to evaluate an epoch.

    args:
        model: the GNN architecture from SearchDepthGNN
        loader: the data loader object containing the batches, over which predictions are computed
        device: the device to run computations on (CPU/CUDA)
    
    returns:    
        float: fraction of correct predictions
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
    return correct / len(loader.dataset)
