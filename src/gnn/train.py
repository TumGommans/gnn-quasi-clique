"""Script defining training functionalities of the GNN."""

import json, torch
import torch.nn.functional as F

from collections import Counter

from torch.utils.data import Dataset
from torch_geometric.data import Data

class RestartDataset(Dataset):
    def __init__(self, path):
        # 1) count total lines
        with open(path, 'rb') as f:
            total_lines = sum(1 for _ in f)
        half_lines = total_lines # * 0.1 # // 2

        self.path = path
        self.offsets = []
        counter = Counter()
        with open(path, 'rb') as f:
            offset = f.tell()
            for idx, raw in enumerate(f):
                if idx >= half_lines:
                    break

                self.offsets.append(offset)
                obj = json.loads(raw)
                counter[int(obj['action']['optimal_L_index'])] += 1
                offset = f.tell()

        # compute class weights on just that half
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
        gf = obj['state']['graph_features']
        u = torch.tensor([
            gf['gamma'],
            gf['initial_target_k'],
            gf['current_target_k'],
            gf['graph_density'],
            gf['restart_count']
        ], dtype=torch.float).unsqueeze(0)
        y = torch.tensor(obj['action']['optimal_L_index'], dtype=torch.long)
        return Data(x=x, edge_index=ei, u=u, y=y)

def train_epoch(model, loader, opt, device, class_weights):
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

def eval_epoch(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
    return correct / len(loader.dataset)
