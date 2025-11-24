import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import TGCN


class TemporalGCNModel(nn.Module):
    """
    Simple Temporal GCN model using TGCN from torch_geometric_temporal.
    Works as a drop-in replacement for TGAT in your training pipeline.

    Input:
        x           node features
        edge_index  graph edges
        edge_weight graph edge weights
        time_index  (ignored, only for API compatibility)

    Output:
        logits for binary classification (buy or sell)
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.tgcn = TGCN(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight, time_index=None):
        # temporal graph conv
        h = self.tgcn(x, edge_index, edge_weight)
        h = self.dropout(h)
        # output logits
        out = self.linear(h).squeeze(-1)
        return out
