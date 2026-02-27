import torch
from torch import nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=128, hidden_dim_2=64, dropout=0.0):
        super().__init__()
        hidden_dim_1 = int(hidden_dim_1)
        hidden_dim_2 = int(hidden_dim_2)
        dropout = float(dropout)
        layers = [
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim_1, hidden_dim_2),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim_2, 1))
        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
