"""
trainers/train_lstm.py

LSTM trainer with:
- Cached sequence construction (no runtime feature building)
- Optional hyperparameter search
- Mixed precision and GPU-friendly data loading
- Rebased equity plots for fair buy-and-hold comparison
"""

import time
from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd
import torch
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler

from models.lstm_model import LSTMModel
from utils.seeds import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.metrics import rank_ic, hit_rate
from utils.backtest import backtest_long_only
from utils.baseline import get_global_buy_and_hold
from utils.plot import (
    plot_equity_curve,
    plot_equity_comparison,
    plot_daily_ic,
    plot_ic_hist,
)
from utils.cache import cache_load, cache_save, cache_key, cache_path
from utils.device import get_device, default_num_workers
from utils.sanity import check_tensor


class TensorLSTMDataset(Dataset):
    def __init__(self, X, y, dates, tickers):
        self.X = X
        self.y = y
        self.dates = dates
        self.tickers = tickers

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _build_sequences(df, feat_cols, lookback, horizon):
    """
    Convert per-ticker time series into fixed-length sequences and targets.
    Assumes df is already cleaned and contains log_ret_1d plus features.
    """
    # We walk each ticker once: slice fixed lookback windows and take the horizon-ahead return as target.
    X_list, y_list, dates, tickers = [], [], [], []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) <= lookback:
            continue
        x_arr = g[feat_cols].values.astype("float32")
        y_arr = g["log_ret_1d"].shift(-horizon).values.astype("float32")
        for pos in range(lookback, len(g) - horizon):
            X_list.append(x_arr[pos - lookback:pos])
            y_list.append(y_arr[pos])
            dates.append(pd.to_datetime(g.loc[pos, "date"]))
            tickers.append(t)
    if not X_list:
        raise ValueError("No sequences built; check data coverage and lookback/horizon.")
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y, dates, tickers


def _prepare_cached_sequences(config, df, feat_cols, split_masks):
    """
    Build or load cached train/val/test tensors for the LSTM.
    Returns dict with {split: {X, y, dates, tickers}}.
    """
    cache_id = cache_key(
        {
            "model": "lstm",
            "data": config["data"],
            "training": config["training"],
            "lookback": config["data"]["lookback_window"],
        },
        dataset_version="lstm_sequences",
        extra_files=[config["data"]["price_file"]],
    )
    cache_file = cache_path("lstm_sequences", cache_id)
    rebuild = config.get("cache", {}).get("rebuild", False)
    if not rebuild:
        cached = cache_load(cache_file)
        if cached is not None:
            print(f"[lstm] loaded sequences from cache {cache_file}")
            return cached

    lookback = config["data"]["lookback_window"]
    horizon = config["data"]["target_horizon"]

    # Build sequences once on the full dataframe, then assign sequences
    # to train/val/test based on the sequence prediction date. This mirrors
    # a realistic pipeline where each sequence uses full available history.
    X_all, y_all, dates_all, tickers_all = _build_sequences(df, feat_cols, lookback, horizon)

    dates_ser = pd.to_datetime(pd.Series(dates_all))
    val_start = pd.to_datetime(config["training"]["val_start"])
    test_start = pd.to_datetime(config["training"]["test_start"])

    seq_train_mask = dates_ser < val_start
    seq_val_mask = (dates_ser >= val_start) & (dates_ser < test_start)
    seq_test_mask = dates_ser >= test_start

    splits = {}
    for name, mask in [("train", seq_train_mask), ("val", seq_val_mask), ("test", seq_test_mask)]:
        idxs = mask[mask].index.values
        if len(idxs) == 0:
            raise ValueError(f"[lstm] No sequences for split {name}; check date ranges and lookback/horizon.")
        X = X_all[idxs]
        y = y_all[idxs]
        dates = [dates_all[i] for i in idxs]
        tickers = [tickers_all[i] for i in idxs]
        splits[name] = {"X": X, "y": y, "dates": dates, "tickers": tickers}
        issues = check_tensor(f"{name}_X", X) + check_tensor(f"{name}_y", y)
        if issues:
            raise ValueError(f"[lstm] Sanity failed for {name}: {'; '.join(issues)}")

    cache_save(cache_file, splits)
    print(f"[lstm] saved sequences to cache {cache_file}")
    return splits


def train_lstm(config):
    """
    Main entrypoint for LSTM training.
    Workflow:
      1) Load/calc features, split by date
      2) Build/load cached sequences (no runtime feature work)
      3) Optional hyperparameter search
      4) Train with AMP, GPU-friendly loaders, grad clipping
      5) Evaluate, backtest, and plot with rebased curves
    """
    set_seed(42)

    device = get_device(config["training"]["device"])
    use_cuda = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    print(f"[lstm] device={device}, cuda_available={torch.cuda.is_available()}")
    if use_cuda:
        try:
            print(f"[lstm] cuda device name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    rebuild = config.get("cache", {}).get("rebuild", False)

    # 1. Load and prepare data (prefer cached features)
    cache_path_feat = Path("data/processed/feature_cache.parquet")
    cache_cols = cache_path_feat.with_suffix(".cols.json")

    if cache_path_feat.exists() and not rebuild:
        df = pd.read_parquet(cache_path_feat)
        if cache_cols.exists():
            with cache_cols.open("r") as f:
                feat_cols = json.load(f)
        else:
            base_cols = {"ticker", "date", "close", "volume", "target"}
            feat_cols = [c for c in df.columns if c not in base_cols]
        print(f"[lstm] loaded cached features from {cache_path_feat}")
    else:
        df = load_price_panel(
            config["data"]["price_file"],
            config["data"]["start_date"],
            config["data"]["end_date"],
        )
        df, feat_cols = add_technical_features(df)
        print(df.head())
        cache_path_feat.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path_feat)
        with cache_cols.open("w") as f:
            json.dump(feat_cols, f)
        print(f"[lstm] computed features and cached to {cache_path_feat}")

    df = df.dropna(subset=list(feat_cols) + ["log_ret_1d"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    val_start = pd.to_datetime(config["training"]["val_start"])
    test_start = pd.to_datetime(config["training"]["test_start"])

    train_mask = df["date"] < val_start
    val_mask = (df["date"] >= val_start) & (df["date"] < test_start)
    test_mask = df["date"] >= test_start

    splits = _prepare_cached_sequences(config, df, feat_cols, (train_mask, val_mask, test_mask))

    # 2. Hyperparameter choice (unchanged)
    tune_cfg = config.get("tuning", {"enabled": False})
    use_tuning = tune_cfg.get("enabled", False)
    print("LSTM tuning enabled:", use_tuning)

    if not use_tuning:
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

            X_train, y_train = splits["train"]["X"], splits["train"]["y"]
            X_val, y_val = splits["val"]["X"], splits["val"]["y"]

            train_ds = TensorLSTMDataset(X_train, y_train, [], [])
            val_ds   = TensorLSTMDataset(X_val, y_val, [], [])
            bs = max(2, config["training"]["batch_size"])
            num_workers = default_num_workers()
            loader_kwargs = {
                "batch_size": bs,
                "shuffle": True,
                "num_workers": num_workers,
                "pin_memory": use_cuda,
                "persistent_workers": num_workers > 0,
            }
            train_loader = DataLoader(train_ds, **loader_kwargs)
            val_loader = DataLoader(val_ds, **{**loader_kwargs, "shuffle": False})

            scaler = GradScaler(device_type="cuda" if use_cuda else "cpu")
            for epoch in range(max_epochs_tune):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    with amp.autocast(device_type="cuda", enabled=use_cuda):
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                    scaler.scale(loss).backward()
                    clip_val = config["training"].get("gradient_clip", 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                    scaler.step(optimizer)
                    scaler.update()

                model.eval()
                val_losses = []
                with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_cuda):
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        pred = model(xb)
                        val_losses.append(loss_fn(pred, yb).detach())

                mean_val = float(torch.stack(val_losses).mean().item()) if val_losses else float("inf")
                print("Params", params, "val_loss", mean_val)

                if mean_val < best_val:
                    best_val = mean_val
                    best_params = params

        print("Best LSTM parameters:", best_params, "with val_loss", best_val)

    # 3. Final training
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

    bs = max(2, config["training"]["batch_size"])
    num_workers = default_num_workers()
    loader_kwargs = {
        "batch_size": bs,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        TensorLSTMDataset(splits["train"]["X"], splits["train"]["y"], [], []),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TensorLSTMDataset(splits["val"]["X"], splits["val"]["y"], [], []),
        **{**loader_kwargs, "shuffle": False},
    )
    test_loader = DataLoader(
        TensorLSTMDataset(splits["test"]["X"], splits["test"]["y"], [], []),
        **{**loader_kwargs, "shuffle": False},
    )

    best_val = float("inf")
    bad_epochs = 0
    patience = config["training"]["patience"]
    scaler = GradScaler(device_type="cuda" if use_cuda else "cpu")

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_lstm.pt"

    max_epochs = config["training"]["max_epochs"]

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            # AMP for faster training on GPU; GradScaler handles scaling/backward.
            with amp.autocast(device_type="cuda", enabled=use_cuda):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            clip_val = config["training"].get("gradient_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.detach())
        data_time = 0.0
        compute_time = time.time() - t0

        model.eval()
        val_losses = []
        with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_cuda):
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).detach())

        mean_train = float(torch.stack(train_losses).mean().item()) if train_losses else float("nan")
        mean_val = float(torch.stack(val_losses).mean().item()) if val_losses else float("inf")
        print(f"[LSTM] Epoch {epoch} train {mean_train:.5f} val {mean_val:.5f} (compute {compute_time:.2f}s)")

        if mean_val < best_val:
            best_val = mean_val
            torch.save(model.state_dict(), model_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    # 4. Test predictions
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []
    all_preds = []
    all_real = []
    with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_cuda):
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            preds = model(xb).cpu().numpy()
            y_np = yb.numpy()
            all_preds.extend(preds)
            all_real.extend(y_np)

    dates = splits["test"]["dates"]
    tickers = splits["test"]["tickers"]
    for i, (p, r) in enumerate(zip(all_preds, all_real)):
        rows.append({
            "date": pd.to_datetime(dates[i]),
            "ticker": tickers[i],
            "pred": float(p),
            "realized_ret": float(r),
        })

    pred_df = pd.DataFrame(rows).sort_values("date")
    pred_df.to_csv(out_dir / "lstm_predictions.csv", index=False)

    # 5. IC / hit metrics
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

    # 6. Long only backtest, daily rebalancing + buy-and-hold comparison
    curve, daily_ret, stats = backtest_long_only(
        pred_df,
        top_k=config["evaluation"]["top_k"],
        transaction_cost_bps=config["evaluation"]["transaction_cost_bps"],
        risk_free_rate=config["evaluation"]["risk_free_rate"],
        rebalance_freq=5,
    )
    print("LSTM backtest stats:", stats)

    # Global buy-and-hold baseline (precomputed, model independent)
    eq_bh_full, ret_bh_full, stats_bh = get_global_buy_and_hold(
        config,
        rebuild=config.get("cache", {}).get("rebuild", False),
        align_start_date=config["training"]["test_start"],
    )
    print("[baseline] global buy-and-hold stats", stats_bh)

    # align baseline to test window and rebase both curves to start at 1.0 for plotting
    start_d = max(curve.index.min(), pred_df["date"].min())
    end_d = pred_df["date"].max()
    eq_bh = eq_bh_full.loc[(eq_bh_full.index >= start_d) & (eq_bh_full.index <= end_d)]

    def _rebase(series, start):
        series = series.loc[series.index >= start]
        if series.empty:
            return series
        return series / series.iloc[0]

    curve_rebased = _rebase(curve, start_d)
    eq_bh_rebased = _rebase(eq_bh, start_d)

    curve_rebased.to_csv(out_dir / "lstm_equity_curve.csv", header=["value"])
    plot_equity_curve(curve_rebased, "LSTM long only", out_dir / "lstm_equity_curve.png")

    eq_bh_rebased.to_csv(out_dir / "lstm_buy_and_hold_equity_curve.csv", header=["value"])
    plot_equity_curve(
        eq_bh_rebased,
        "Buy and Hold",
        out_dir / "lstm_buy_and_hold_equity_curve.png",
    )

    # Combined comparison plot
    plot_equity_comparison(
        curve_rebased,
        eq_bh_rebased,
        "LSTM vs Buy and Hold",
        out_dir / "lstm_equity_comparison.png",
    )
