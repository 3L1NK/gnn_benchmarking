import torch
from torch import nn
from torch_geometric.nn import GATConv


class StaticTGAT(nn.Module):
    """
    Lightweight TGAT-like model that exposes the same `forward(x, edge_index, edge_weight=None)`
    signature expected by the training loop. This implementation uses stacked GATConv
    layers (temporal aspects are not modelled explicitly here to remain compatible
    with the snapshot-per-day training loop). If you later add a true temporal
    TGAT cell, swap it in here while keeping the same forward signature.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, heads: int = 2, dropout: float = 0.0, use_residual: bool = True):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

        # keep shallow to avoid oversmoothing
        num_layers = max(1, min(num_layers, 3))
        heads = max(1, min(heads, 4))

        convs = []
        bns = []
        res_projs = []
        in_dim = input_dim

        for layer in range(num_layers):
            conv = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
            out_dim = hidden_dim
            convs.append(conv)
            bns.append(nn.BatchNorm1d(out_dim))
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
        # Note: PyG's GATConv ignores `edge_weight`. If you want to include
        # multiplicative edge signals in attention, implement a custom attention layer.
        for i, conv in enumerate(self.convs):
            out = conv(x, edge_index)
            if self.use_residual:
                res = self.res_projs[i](x)
                out = out + res
            out = self.bns[i](out)
            x = self.activation(out)
            x = self.dropout(x)

        out = self.head(x).squeeze(-1)
        return out
