# models/gnn_model.py

import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv


class StaticGNN(nn.Module):
    """
    Generic node level regression model for static graphs.

    Supports:
      - GCN
      - GAT
    """

    def __init__(
        self,
        gnn_type: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        heads: int = 4,
    ):
        super().__init__()

        gnn_type = gnn_type.lower()
        if gnn_type not in {"gcn", "gat"}:
            raise ValueError(f"StaticGNN only supports gcn and gat, got {gnn_type}")

        self.gnn_type = gnn_type
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        convs = []
        in_dim = input_dim

        for layer in range(num_layers):
            if self.gnn_type == "gcn":
                conv = GCNConv(in_dim, hidden_dim)
                out_dim = hidden_dim
            else:  # gat
                # multi head GAT, concat=False keeps feature dim = hidden_dim
                conv = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
                out_dim = hidden_dim

            convs.append(conv)
            in_dim = out_dim

        self.convs = nn.ModuleList(convs)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            # edge_weight is only used by GCNConv, GATConv ignores it
            if self.gnn_type == "gcn" and edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        out = self.head(x).squeeze(-1)
        return out
