# trainers/train_lstm.py

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.lstm_model import LSTMModel
from utils.seeds import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.metrics import rank_ic, hit_rate
from utils.backtest import backtest_long_only
from utils.plot import (
    plot_equity_curve,
    plot_daily_ic,
    plot_ic_hist,
)


# -------------------------------------------------
# DATASET
# -------------------------------------------------

class LSTMDataset(Dataset):
    def __init__(self, df, feat_cols, lookback, horizon):
        self.lookback = lookback
        self.horizon = horizon
        self.feat_cols = feat_cols

        # sort and clean
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df = df.dropna(subset=feat_cols + ["log_ret_1d"]).reset_index(drop=True)

        # shift target horizon days ahead
        df["target"] = df.groupby("ticker")["log_ret_1d"].shift(-horizon)
        df = df.dropna(subset=["target"]).reset_index(drop=True)

        # group once
        self.groups = {
            t: g.reset_index(drop=True)
            for t, g in df.groupby("ticker")
        }

        # locations for __getitem__
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

        date = str(g.loc[end, "date"])
        ticker = str(t)

        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            date,
            ticker,
        )


# -------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------

def train_lstm(config):
    set_seed(42)

    device_str = config["training"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # -----------------------
    # 1. Load and prepare data
    # -----------------------
    df = load_price_panel(
        config["data"]["price_file"],
        config["data"]["start_date"],
        config["data"]["end_date"],
    )
    df, feat_cols = add_technical_features(df)

    df = df.dropna(subset=list(feat_cols) + ["log_ret_1d"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

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
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Test samples:", len(test_ds))

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_lstm.pt"

    # -----------------------
    # 2. Hyperparameter choice
    #    either from YAML or simple grid search
    # -----------------------
    tune_cfg = config.get("tuning", {"enabled": False})
    use_tuning = tune_cfg.get("enabled", False)
    print("LSTM tuning enabled:", use_tuning)

    if not use_tuning:
        # use fixed values from YAML
        hidden_dim = config["model"]["hidden_dim"]
        num_layers = config["model"]["num_layers"]
        dropout    = config["model"]["dropout"]
        lr         = config["training"]["lr"]

        best_params = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
        }
        print("Using fixed LSTM parameters:", best_params)

    else:
        # small grid defined in code, like xgb_raw
        param_grid = {
            "hidden_dim": [32, 64],
            "num_layers": [1, 2],
            "dropout":    [0.0, 0.2],
            "lr":         [0.0003, 0.0005],
        }

        best_val = float("inf")
        best_params = None

        max_epochs_tune = min(10, config["training"]["max_epochs"])
        print("Starting LSTM hyperparameter search")

        for values in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), values))
            print("Trying params:", params)

            model = LSTMModel(
                input_dim=len(feat_cols),
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
            loss_fn = torch.nn.MSELoss()

            # short training for tuning
            for epoch in range(max_epochs_tune):
                model.train()
                for x, y, _, _ in train_loader:
                    x = x.to(device)
                    y = y.to(device)

                    optimizer.zero_grad()
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    loss.backward()
                    optimizer.step()

            # one validation pass
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y, _, _ in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    val_losses.append(loss_fn(pred, y).item())

            mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
            print("Params", params, "val_loss", mean_val)

            if mean_val < best_val:
                best_val = mean_val
                best_params = params

        print("Best LSTM parameters:", best_params, "with val_loss", best_val)

    # -----------------------
    # 3. Final training with early stopping
    # -----------------------
    hidden_dim = best_params["hidden_dim"]
    num_layers = best_params["num_layers"]
    dropout    = best_params["dropout"]
    lr         = best_params["lr"]

    model = LSTMModel(
        input_dim=len(feat_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    patience = config["training"]["patience"]
    bad_epochs = 0

    max_epochs = config["training"]["max_epochs"]

    for epoch in range(max_epochs):
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

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        print(f"[LSTM] Epoch {epoch} train {mean_train:.5f} val {mean_val:.5f}")

        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), model_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    # -----------------------
    # 4. Test predictions
    # -----------------------
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

    # -----------------------
    # 5. IC / hit metrics
    # -----------------------
    daily_metrics = []
    for d, g in pred_df.groupby("date"):
        ic = rank_ic(g["pred"], g["realized_ret"])
        hit = hit_rate(
            g["pred"],
            g["realized_ret"],
            top_k=config["evaluation"]["top_k"],
        )
        daily_metrics.append({"date": d, "ic": ic, "hit": hit})

    daily_metrics = pd.DataFrame(daily_metrics).sort_values("date")
    daily_metrics.to_csv(out_dir / "lstm_daily_metrics.csv", index=False)

    print("LSTM mean IC:", daily_metrics["ic"].mean())
    print("LSTM mean hit:", daily_metrics["hit"].mean())

    plot_daily_ic(
        daily_metrics,
        out_dir / "lstm_ic_timeseries.png",
    )
    plot_ic_hist(
        daily_metrics,
        out_dir / "lstm_ic_histogram.png",
    )

    # -----------------------
    # 6. Long only backtest, daily rebalancing
    # -----------------------
    curve, daily_ret, stats = backtest_long_only(
        pred_df,
        top_k=config["evaluation"]["top_k"],
        transaction_cost_bps=config["evaluation"]["transaction_cost_bps"],
        risk_free_rate=config["evaluation"]["risk_free_rate"],
    )

    curve.to_csv(out_dir / "lstm_equity_curve.csv", header=["value"])
    plot_equity_curve(curve, "LSTM long only", out_dir / "lstm_equity_curve.png")

    print("LSTM backtest stats:", stats)
