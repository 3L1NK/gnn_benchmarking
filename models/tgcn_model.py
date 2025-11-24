# models/tgcn_model.py

import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import TGCN


class TemporalGCNModel(nn.Module):
    """
    Simple temporal GCN model using the TGCN cell.

    At each time step t it takes:
      x_t           node features at day t, shape [N_t, F]
      edge_index_t  edges at day t
      edge_weight_t edge weights
      h_prev        previous hidden state, shape [N_prev, H] or None

    It returns:
      logits_t      node logits at day t, shape [N_t]
      h_t           new hidden state, shape [N_t, H]
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.tgcn = TGCN(in_channels=input_dim, out_channels=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, h_prev=None):
        # TGCN keeps its own hidden state if h_prev is passed in
        h_t = self.tgcn(x, edge_index, edge_weight, h_prev)
        h_t = self.dropout(h_t)
        logits = self.out(h_t).squeeze(-1)
        return logits, h_t
