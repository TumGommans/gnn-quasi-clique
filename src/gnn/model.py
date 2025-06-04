"""Script defining the GNN architecture to predict the search depth."""

import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GINConv
from torch_geometric.nn.norm import GraphNorm

from src.gnn.constants import ACTIVATIONS, READOUTS

class SearchDepthGNN(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int,
        num_gin_layers: int,
        mlp_layers_per_gin: int,
        final_mlp_layers: int,  
        dropout: float,
        readout: str,
        activation: str,
        num_classes: int
    ):
        super().__init__()

        self.act = ACTIVATIONS[activation]
        self.pool = READOUTS[readout]

        self.gin_layers = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(num_gin_layers):
            mlp = []
            dims = [in_dim] + [hidden_dim] * mlp_layers_per_gin
            for i in range(len(dims) - 1):
                mlp.append(nn.Linear(dims[i], dims[i+1]))
                mlp.append(self.act)
            self.gin_layers.append(GINConv(
                nn.Sequential(*mlp)
            ))
            in_dim = hidden_dim

        self.bn_layers = nn.ModuleList([
            GraphNorm(hidden_dim)
            for _ in range(num_gin_layers)
        ])

        # self.bn_final = GraphNorm(hidden_dim+graph_feat_dim)

        # Final MLP
        mlp = []
        dims = [hidden_dim] + [hidden_dim] * final_mlp_layers + [num_classes]
        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                mlp.append(self.act)
                mlp.append(nn.Dropout(dropout))
        self.final_mlp = nn.Sequential(*mlp)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.gin_layers):
            x = conv(x, edge_index)         
            x = self.bn_layers[i](x) 
            x = self.act(x)
        x = self.pool(x, batch)
        # x = torch.cat([x, u], dim=1)

        # x = self.bn_final(x)

        out = self.final_mlp(x)
        return F.log_softmax(out, dim=-1)
