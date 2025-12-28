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
        use_residual: bool = True,
    ):
        super().__init__()

        gnn_type = gnn_type.lower()
        if gnn_type not in {"gcn", "gat"}:
            raise ValueError(f"StaticGNN only supports gcn and gat, got {gnn_type}")

        self.gnn_type = gnn_type
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

        convs = []
        bns = []
        res_projs = []
        in_dim = input_dim

        # keep shallow to avoid over-smoothing
        num_layers = max(1, min(num_layers, 2))
        if self.gnn_type == "gat":
            heads = max(1, min(heads, 2))

        for layer in range(num_layers):
            if self.gnn_type == "gcn":
                conv = GCNConv(in_dim, hidden_dim)
                out_dim = hidden_dim
            else:  # gat
                # multi head GAT, concat=False keeps feature dim = hidden_dim
                # Note: PyG's GATConv does not accept/use an `edge_weight` tensor;
                # any edge weighting is therefore ignored by the attention implementation.
                # If you need weighted attention, implement custom attention or
                # construct edge features used by the convolution.
                conv = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
                out_dim = hidden_dim

            convs.append(conv)
            # batchnorm to stabilize training across days / batches
            bns.append(nn.BatchNorm1d(out_dim))

            # residual projection if dims differ
            if use_residual and in_dim != out_dim:
                res_projs.append(nn.Linear(in_dim, out_dim))
            else:
                res_projs.append(nn.Identity())

            in_dim = out_dim

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.res_projs = nn.ModuleList(res_projs)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            # edge_weight is only used by GCNConv, GATConv ignores it
            if self.gnn_type == "gcn" and edge_weight is not None:
                out = conv(x, edge_index, edge_weight)
            else:
                out = conv(x, edge_index)

            if self.use_residual:
                res = self.res_projs[i](x)
                out = out + res

            # BatchNorm expects shape [N, C]
            out = self.bns[i](out)
            x = self.activation(out)
            x = self.dropout(x)

        out = self.head(x).squeeze(-1)
        return out
