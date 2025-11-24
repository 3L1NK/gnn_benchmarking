# trainers/train_lstm.py

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


# -------------------------------------------------
# DATASET (OPTIMIZED)
# -------------------------------------------------

class LSTMDataset(Dataset):
    def __init__(self, df, feat_cols, lookback, horizon):
        self.lookback = lookback
        self.horizon = horizon
        self.feat_cols = feat_cols

        # sort and clean
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df = df.dropna(subset=feat_cols + ["log_ret_1d"]).reset_index(drop=True)

        # shift target
        df["target"] = df.groupby("ticker")["log_ret_1d"].shift(-horizon)
        df = df.dropna(subset=["target"]).reset_index(drop=True)

        # group once instead of every batch
        self.groups = {
            t: g.reset_index(drop=True)
            for t, g in df.groupby("ticker")
        }

        # build (ticker, index) location list
        self.index_list = []
        for t, g in self.groups.items():
            for pos in range(lookback, len(g)):
                self.index_list.append((t, pos))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        t, pos = self.index_list[idx]
        g = self.groups[t]

        start = pos - self.lookback
        end = pos

        seq = g.loc[start:end - 1, self.feat_cols].values.astype(float)
        target = float(g.loc[end, "target"])

        # convert date to str for safe DataLoader batching
        date = str(g.loc[end, "date"])
        ticker = str(t)

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            date,
            ticker,
        )


# -------------------------------------------------
# TRAINING
# -------------------------------------------------

def train_lstm(config):
    set_seed(42)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

    # load data
    df = load_price_panel(
        config["data"]["price_file"],
        config["data"]["start_date"],
        config["data"]["end_date"],
    )
    df, feat_cols = add_technical_features(df)

    df = df.dropna(subset=list(feat_cols) + ["log_ret_1d"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # splits
    val_start = pd.to_datetime(config["training"]["val_start"])
    test_start = pd.to_datetime(config["training"]["test_start"])

    train_df = df[df["date"] < val_start]
    val_df   = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test_df  = df[df["date"] >= test_start]

    lookback = config["data"]["lookback_window"]
    horizon  = config["data"]["target_horizon"]

    train_ds = LSTMDataset(train_df, feat_cols, lookback, horizon)
    val_ds   = LSTMDataset(val_df, feat_cols, lookback, horizon)
    test_ds  = LSTMDataset(test_df, feat_cols, lookback, horizon)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Test samples:", len(test_ds))

    # model
    model = LSTMModel(
        input_dim=len(feat_cols),
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    patience = config["training"]["patience"]
    bad_epochs = 0

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_lstm.pt"

    # -------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_losses = []

        for x, y, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_losses.append(loss_fn(pred, y).item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        print(f"[LSTM] Epoch {epoch} train {mean_train:.5f} val {mean_val:.5f}")

        # early stopping
        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), model_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    # -------------------------------------------------
    # TEST + BACKTEST
    # -------------------------------------------------

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []
    with torch.no_grad():
        for x, y, dates, tickers in test_loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            y_np = y.numpy()

            for p, r, d, t in zip(preds, y_np, dates, tickers):
                rows.append({
                    "date": pd.to_datetime(d),
                    "ticker": t,
                    "pred": float(p),
                    "realized_ret": float(r),
                })

    pred_df = pd.DataFrame(rows).sort_values("date")
    pred_df.to_csv(out_dir / "lstm_predictions.csv", index=False)

    # IC, HIT
    daily_metrics = []
    for d, g in pred_df.groupby("date"):
        if g["pred"].nunique() < 2 or g["realized_ret"].nunique() < 2:
            ic = np.nan
        else:
            ic = rank_ic(g["pred"], g["realized_ret"])

        hit = hit_rate(g["pred"], g["realized_ret"], top_k=config["evaluation"]["top_k"])
        daily_metrics.append({"date": d, "ic": ic, "hit": hit})

    daily_metrics = pd.DataFrame(daily_metrics).sort_values("date")
    print("LSTM mean IC:", daily_metrics["ic"].dropna().mean())
    print("LSTM mean hit:", daily_metrics["hit"].dropna().mean())

    # backtest
    curve, daily_ret, stats = backtest_long_short(
        pred_df,
        config["evaluation"]["top_k"],
        config["evaluation"]["transaction_cost_bps"],
        config["evaluation"]["risk_free_rate"],
    )

    curve.to_csv(out_dir / "lstm_equity_curve.csv", header=["value"])
    plot_equity_curve(curve, "LSTM long short", out_dir / "lstm_equity_curve.png")

    print("LSTM backtest stats:", stats)
