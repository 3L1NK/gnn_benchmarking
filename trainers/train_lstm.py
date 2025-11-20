from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.lstm_model import LSTMModel
from utils.seeds import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.metrics import mse, rank_ic, hit_rate
from utils.backtest import backtest_long_short
from utils.plot import plot_equity_curve


class LSTMDataset(Dataset):
    def __init__(self, df, feat_cols, lookback, horizon):
        self.df = df.sort_values(["ticker", "date"]).copy()
        self.feat_cols = feat_cols
        self.lookback = lookback
        self.horizon = horizon

        self.df["target"] = self.df.groupby("ticker")["log_ret_1d"].shift(-horizon)
        self.df = self.df.dropna(subset=["target"])

        self.index = []
        for ticker, g in self.df.groupby("ticker"):
            g = g.reset_index(drop=True)
            for i in range(lookback, len(g)):
                self.index.append((ticker, g.index[i]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ticker, row_idx = self.index[idx]
        g = self.df[self.df["ticker"] == ticker].reset_index(drop=True)
        pos = g.index.get_loc(row_idx)
        start = pos - self.lookback
        end = pos

        seq = g.loc[start:end - 1, self.feat_cols].values.astype(float)
        target = float(g.loc[end, "target"])

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def train_lstm(config):
    set_seed(42)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    df = load_price_panel(
        config["data"]["price_file"], config["data"]["start_date"], config["data"]["end_date"]
    )
    df, feat_cols = add_technical_features(df)

    val_start = pd.to_datetime(config["training"]["val_start"])
    test_start = pd.to_datetime(config["training"]["test_start"])
    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] < val_start]
    val_df = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test_df = df[df["date"] >= test_start]

    lookback = config["data"]["lookback_window"]
    horizon = config["data"]["target_horizon"]

    train_ds = LSTMDataset(train_df, feat_cols, lookback, horizon)
    val_ds = LSTMDataset(val_df, feat_cols, lookback, horizon)
    test_ds = LSTMDataset(test_df, feat_cols, lookback, horizon)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    model = LSTMModel(
        input_dim=len(feat_cols),
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    bad_epochs = 0
    patience = config["training"]["patience"]
    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_lstm.pt"

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(loss_fn(pred, y).item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"[LSTM] Epoch {epoch} train {mean_train:.5f} val {mean_val:.5f}")

        if mean_val < best_val:
            best_val = mean_val
            bad_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # collect predictions at last step of each sequence for test_df
    rows = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y_np = y.numpy()
            # we do not know exact date here. if needed you can extend dataset to return date and ticker
            # for now we just compute aggregate metrics
            for yp, yt in zip(pred, y_np):
                rows.append({"pred": float(yp), "realized_ret": float(yt)})

    test_df_small = pd.DataFrame(rows)
    print("Test MSE", mse(test_df_small["pred"], test_df_small["realized_ret"]))
