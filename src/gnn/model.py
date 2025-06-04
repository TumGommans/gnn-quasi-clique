"""Script defining the GNN architecture to predict the search depth."""

import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GINConv
from torch_geometric.nn.norm import GraphNorm

from src.gnn.constants import ACTIVATIONS, READOUTS

class SearchDepthGNN(nn.Module):
    """GNN model to predict the optimal search depth for TSQC."""
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
        """inititializes an instance.
        
        args:
            node_feat_dim:      node feature dimension
            hidden_dim:         the number of neurons in the hidden layers
            num_gin_layers:     the number of GIN layers throughout graph embedding
            mlp_layers_per_gin: the number of Multi-Layer Perceptron layers per GIN module
            final_mlp_layers:   the number of Multi-Layer Perceptron layers to convert pooled graph embedding into scalar prediction
            dropout:            the dropout rate
            readout:            the readout function to obtain a graph embedding
            activation:         the non-linear activation function after each layer
            num_classes:        the (discrete!) number of options for the search depth
        """
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
        """Forward pass throug the architecture.
        
        args:
            data: Data() object containing the batch data required.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.gin_layers):
            x = conv(x, edge_index)         
            x = self.bn_layers[i](x) 
            x = self.act(x)
        x = self.pool(x, batch)

        out = self.final_mlp(x)
        return F.log_softmax(out, dim=-1)
