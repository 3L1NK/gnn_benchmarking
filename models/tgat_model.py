# models/tgat_model.py

import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import TGAT


class TGATModel(nn.Module):
    """
    Temporal GNN for evolving graphs.

    Input:
      - sequence of node features over time
      - sequence of edge_index over time
      - sequence of edge weights over time

    We use TGAT from torch_geometric_temporal and a linear head on top.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        heads: int = 4,
    ):
        super().__init__()

        self.tgat = TGAT(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, time):
        """
        x: [num_nodes, input_dim]
        edge_index: [2, num_edges]
        edge_weight: [num_edges] or None
        time: [num_nodes] time stamps as float or long
        """
        h = self.tgat(x, edge_index, time, edge_weight)
        out = self.head(h).squeeze(-1)
        return out
