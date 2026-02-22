import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]   # last time step
        return self.fc(last_hidden).squeeze(-1)
