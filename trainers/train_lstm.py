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
import json

import numpy as np
import pandas as pd
import torch
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

from models.sequence.lstm import LSTMModel
from utils.seeds import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.cache import cache_load, cache_save, cache_key, cache_path
from utils.device import get_device, default_num_workers
from utils.sanity import check_tensor
from utils.targets import build_target
from utils.preprocessing import scale_features
from utils.splits import split_time
from utils.predictions import sanitize_predictions
from utils.eval_runner import evaluate_and_report
from utils.artifacts import resolve_output_dirs
from utils.tuning import enumerate_param_candidates, score_prediction_objective


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


def _build_sequences(df, feat_cols, lookback, target_col):
    """
    Convert per-ticker time series into fixed-length sequences and targets.
    Assumes df is already cleaned and contains log_ret_1d plus features.
    """
    # We walk each ticker once: slice fixed lookback windows and take the precomputed target as label.
    X_list, y_list, dates, tickers = [], [], [], []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) <= lookback:
            continue
        x_arr = g[feat_cols].values.astype("float32")
        y_arr = g[target_col].values.astype("float32")
        for pos in range(lookback, len(g)):
            X_list.append(x_arr[pos - lookback:pos])
            y_list.append(y_arr[pos])
            dates.append(pd.to_datetime(g.loc[pos, "date"]))
            tickers.append(t)
    if not X_list:
        raise ValueError("No sequences built; check data coverage and lookback/horizon.")
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return X, y, dates, tickers


def _prepare_cached_sequences(config, df, feat_cols, target_col, debug=False):
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
            "preprocess": config.get("preprocess", {"scale_features": True, "scaler": "standard"}),
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

    # Build sequences once on the full dataframe, then assign sequences
    # to train/val/test based on the sequence prediction date. This mirrors
    # a realistic pipeline where each sequence uses full available history.
    X_all, y_all, dates_all, tickers_all = _build_sequences(df, feat_cols, lookback, target_col=target_col)

    seq_train_mask, seq_val_mask, seq_test_mask, _ = split_time(
        dates_all,
        config,
        label="lstm_sequences",
        debug=debug,
    )

    splits = {}
    for name, mask in [("train", seq_train_mask), ("val", seq_val_mask), ("test", seq_test_mask)]:
        idxs = np.where(mask)[0]
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
    debug = bool(config.get("debug", {}).get("leakage", False))

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

    df, target_col = build_target(df, config, target_col="target")
    df = df.dropna(subset=list(feat_cols) + [target_col]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    train_mask, _, _, _ = split_time(df["date"], config, label="lstm", debug=debug)
    pre_cfg = config.get("preprocess", {})
    do_scale = bool(pre_cfg.get("scale_features", True))
    df_scaled, _ = scale_features(df, feat_cols, train_mask, label="lstm", debug=debug, scale=do_scale)

    splits = _prepare_cached_sequences(config, df_scaled, feat_cols, target_col, debug=debug)

    # 2. Hyperparameter choice (unchanged)
    tune_cfg = config.get("tuning", {"enabled": False})
    use_tuning = tune_cfg.get("enabled", False)
    print("LSTM tuning enabled:", use_tuning)

    fixed_params = {
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "num_layers": int(config["model"]["num_layers"]),
        "dropout": float(config["model"]["dropout"]),
        "lr": float(config["training"]["lr"]),
    }
    if not use_tuning:
        best_params = fixed_params
        print("Using fixed LSTM parameters:", best_params)
    else:
        default_grid = {
            "hidden_dim": [32, 64],
            "num_layers": [1, 2],
            "dropout": [0.0, 0.2],
            "lr": [0.0003, 0.0005],
        }
        param_grid = tune_cfg.get("param_grid", default_grid)
        tuning_objective = str(tune_cfg.get("objective", "val_rmse"))
        sample_mode = str(tune_cfg.get("sample_mode", "grid"))
        max_trials = int(tune_cfg.get("max_trials", 0) or 0)
        tuning_seed = int(tune_cfg.get("seed", config.get("seed", 42)))
        candidates = enumerate_param_candidates(
            fixed_params=fixed_params,
            param_grid=param_grid,
            sample_mode=sample_mode,
            max_trials=max_trials,
            seed=tuning_seed,
        )
        max_epochs_tune = int(tune_cfg.get("tune_max_epochs", min(10, config["training"]["max_epochs"])))
        tune_patience = int(tune_cfg.get("tune_patience", max(2, min(5, config["training"]["patience"]))))
        print(
            f"Starting LSTM hyperparameter search: objective={tuning_objective} "
            f"sample_mode={sample_mode} candidates={len(candidates)}"
        )

        best_params = None
        best_score = float("-inf")

        X_train, y_train = splits["train"]["X"], splits["train"]["y"]
        X_val, y_val = splits["val"]["X"], splits["val"]["y"]
        y_val_np = y_val.detach().cpu().numpy()
        val_dates = splits["val"]["dates"]
        val_tickers = splits["val"]["tickers"]

        train_ds = TensorLSTMDataset(X_train, y_train, [], [])
        val_ds = TensorLSTMDataset(X_val, y_val, [], [])
        bs = max(2, config["training"]["batch_size"])
        num_workers = default_num_workers()
        loader_kwargs = {
            "batch_size": bs,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": use_cuda,
            "persistent_workers": num_workers > 0,
        }
        train_loader_tune = DataLoader(train_ds, **loader_kwargs)
        val_loader_tune = DataLoader(val_ds, **{**loader_kwargs, "shuffle": False})

        for params in candidates:
            print("Trying params:", params)
            model = LSTMModel(
                input_dim=len(feat_cols),
                hidden_dim=int(params["hidden_dim"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
            loss_fn = torch.nn.MSELoss()
            scaler = GradScaler(enabled=use_cuda)
            local_best_val = float("inf")
            local_bad_epochs = 0
            for _ in range(max_epochs_tune):
                model.train()
                for xb, yb in train_loader_tune:
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
                    for xb, yb in val_loader_tune:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        pred = model(xb)
                        val_losses.append(loss_fn(pred, yb).detach())
                mean_val = float(torch.stack(val_losses).mean().item()) if val_losses else float("inf")
                if mean_val < local_best_val:
                    local_best_val = mean_val
                    local_bad_epochs = 0
                else:
                    local_bad_epochs += 1
                    if local_bad_epochs >= tune_patience:
                        break

            model.eval()
            val_preds = []
            with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_cuda):
                for xb, _ in val_loader_tune:
                    xb = xb.to(device, non_blocking=True)
                    val_preds.extend(model(xb).detach().cpu().numpy().tolist())
            score_payload = score_prediction_objective(
                objective=tuning_objective,
                y_true=y_val_np,
                preds=val_preds,
                dates=val_dates,
                tickers=val_tickers,
                top_k=int(config["evaluation"].get("top_k", 20)),
                transaction_cost_bps=float(config["evaluation"].get("transaction_cost_bps", 5.0)),
                risk_free_rate=float(config["evaluation"].get("risk_free_rate", 0.0)),
                rebalance_freq=int(config["evaluation"].get("primary_rebalance_freq", 1)),
            )
            metric_name = score_payload["metric_name"]
            metric_value = score_payload["metric_value"]
            score = score_payload["score"]
            print("Params", params, metric_name, metric_value, "score", score)
            if score > best_score:
                best_score = score
                best_params = params

        print("Best LSTM parameters:", best_params, "best_score", best_score)

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
    scaler = GradScaler(enabled=use_cuda)

    out_dirs = resolve_output_dirs(config, model_type=config.get("model", {}).get("type", "lstm"))
    out_dir = out_dirs.canonical
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_lstm.pt"

    max_epochs = config["training"]["max_epochs"]

    train_start = time.time()
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
    train_seconds = time.time() - train_start

    # 4. Test predictions
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []
    all_preds = []
    all_real = []
    infer_start = time.time()
    with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_cuda):
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            preds = model(xb).cpu().numpy()
            y_np = yb.numpy()
            all_preds.extend(preds)
            all_real.extend(y_np)
    inference_seconds = time.time() - infer_start

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

    pred_df = sanitize_predictions(pred_df, strict_unique=True)
    summary = evaluate_and_report(
        config=config,
        pred_df=pred_df,
        out_dirs=out_dirs,
        run_name="lstm",
        model_name=config["model"]["type"],
        model_family=config["model"]["family"],
        edge_type="none",
        directed=False,
        graph_window="",
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
    )
    print("LSTM primary policy stats:", summary.get("stats", {}))
