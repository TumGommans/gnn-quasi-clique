"""Script containing constant objects for the GNN."""

from torch import nn
from torch_geometric.nn import (
    global_add_pool, 
    global_mean_pool, 
    global_max_pool
)

ACTIVATIONS = {
    'ReLU': nn.ReLU(),
    'ELU': nn.ELU(),
    'LeakyReLU': nn.LeakyReLU(),
}

READOUTS = {
     'sum': global_add_pool,
     'mean': global_mean_pool,
     'max': global_max_pool
}
