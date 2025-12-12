# trainers/train_lstm.py

from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.lstm_model import LSTMModel
from utils.seeds import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.metrics import rank_ic, hit_rate
from utils.backtest import backtest_long_only, backtest_buy_and_hold
from utils.plot import (
    plot_equity_curve,
    plot_equity_comparison,
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
        self.group_arrays = {}
        self.targets = {}
        self.index_list = []
        self.meta_dates = []
        self.meta_tickers = []

        for t, g in df.groupby("ticker"):
            g = g.reset_index(drop=True)
            if len(g) <= lookback:
                continue

            x = g[feat_cols].values.astype("float32")  # fast, no pandas
            y = g["target"].values.astype("float32")

            self.group_arrays[t] = x
            self.targets[t] = y

            for pos in range(lookback, len(g)):
                self.index_list.append((t, pos))
                self.meta_dates.append(g.loc[pos, "date"])
                self.meta_tickers.append(t)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        t, pos = self.index_list[idx]
        start = pos - self.lookback
        end = pos

        x = self.group_arrays[t][start:end]
        y = self.targets[t][end]

        return (
            torch.from_numpy(x),
            torch.tensor(y),
        )



# -------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------

def train_lstm(config):
    set_seed(42)

    device_str = config["training"]["device"]
    use_cuda = torch.cuda.is_available() and device_str.startswith("cuda")
    device = torch.device(device_str if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"[lstm] device={device}, cuda_available={torch.cuda.is_available()}")
    if use_cuda:
        try:
            print(f"[lstm] cuda device name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # -----------------------
    # 1. Load and prepare data (prefer cached features)
    # -----------------------
    cache_path = Path("data/processed/feature_cache.parquet")
    cache_cols = cache_path.with_suffix(".cols.json")

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if cache_cols.exists():
            with cache_cols.open("r") as f:
                feat_cols = json.load(f)
        else:
            base_cols = {"ticker", "date", "close", "volume", "target"}
            feat_cols = [c for c in df.columns if c not in base_cols]
        print(f"[lstm] loaded cached features from {cache_path}")
    else:
        df = load_price_panel(
            config["data"]["price_file"],
            config["data"]["start_date"],
            config["data"]["end_date"],
        )
        df, feat_cols = add_technical_features(df)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        with cache_cols.open("w") as f:
            json.dump(feat_cols, f)
        print(f"[lstm] computed features and cached to {cache_path}")

    df = df.dropna(subset=list(feat_cols) + ["log_ret_1d"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    val_start = pd.to_datetime(config["training"]["val_start"])
    test_start = pd.to_datetime(config["training"]["test_start"])

    train_df = df[df["date"] < val_start]
    val_df   = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test_df  = df[df["date"] >= test_start]

    lookback = config["data"]["lookback_window"]
    horizon  = config["data"]["target_horizon"]

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    train_ds = LSTMDataset(train_df, feat_cols, lookback, horizon)
    val_ds   = LSTMDataset(val_df, feat_cols, lookback, horizon)
    test_ds  = LSTMDataset(test_df, feat_cols, lookback, horizon)

    # Cache index lists to avoid recomputing between grid runs
    index_cache = {
        "train": (train_ds, cache_dir / "train_index.pt"),
        "val": (val_ds, cache_dir / "val_index.pt"),
        "test": (test_ds, cache_dir / "test_index.pt"),
    }
    for name, (ds, path) in index_cache.items():
        if path.exists():
            try:
                ds.index_list = torch.load(path)
                print(f"[lstm] loaded cached {name} indices from {path}")
            except Exception:
                pass
        else:
            try:
                torch.save(ds.index_list, path)
                print(f"[lstm] cached {name} indices to {path}")
            except Exception:
                pass

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=16,
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
                for x, y in train_loader:
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
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    val_losses.append(loss_fn(pred, y).detach())

            mean_val = float(torch.stack(val_losses).mean().item()) if val_losses else float("inf")
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

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_losses.append(loss_fn(pred, y).detach())

        mean_train = float(torch.stack(train_losses).mean().item()) if train_losses else float("nan")
        mean_val = float(torch.stack(val_losses).mean().item()) if val_losses else float("inf")
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
    all_preds = []
    all_real = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            y_np = y.numpy()
            all_preds.extend(preds)
            all_real.extend(y_np)

    for i, (p, r) in enumerate(zip(all_preds, all_real)):
        rows.append({
            "date": pd.to_datetime(test_ds.meta_dates[i]),
            "ticker": test_ds.meta_tickers[i],
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
    # 6. Long only backtest, daily rebalancing + buy-and-hold comparison
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

    # Buy-and-hold on the same test window for comparison
    bh_df = pred_df[["date", "ticker", "realized_ret"]].rename(
        columns={"realized_ret": "log_ret_1d"}
    )
    eq_bh, ret_bh, stats_bh = backtest_buy_and_hold(
        bh_df,
        risk_free_rate=config["evaluation"]["risk_free_rate"],
    )
    eq_bh.to_csv(out_dir / "lstm_buy_and_hold_equity_curve.csv", header=["value"])

    plot_equity_curve(
        eq_bh,
        "Buy and Hold",
        out_dir / "lstm_buy_and_hold_equity_curve.png",
    )

    # Combined comparison plot
    plot_equity_comparison(
        curve,
        eq_bh,
        "LSTM vs Buy and Hold",
        out_dir / "lstm_equity_comparison.png",
    )

    print("Buy-and-hold stats:", stats_bh)
