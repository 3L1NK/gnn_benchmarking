from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.baseline_mlp import BaselineMLP
from trainers.common import prepare_panel
from utils.artifacts import resolve_output_dirs
from utils.device import default_num_workers, get_device
from utils.eval_runner import evaluate_and_report
from utils.predictions import sanitize_predictions
from utils.preprocessing import scale_features
from utils.seeds import set_seed
from utils.splits import split_time
from utils.targets import build_target
from utils.tuning import enumerate_param_candidates, score_prediction_objective


def _prepare_data(config: dict, feat_cols: list[str]) -> tuple[pd.DataFrame, str, np.ndarray, np.ndarray, np.ndarray]:
    debug = bool(config.get("debug", {}).get("leakage", False))
    df, target_col = build_target(config["_mlp_panel"], config, target_col="target")
    df = df.dropna(subset=list(feat_cols) + [target_col]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    train_mask, val_mask, test_mask, _ = split_time(df["date"], config, label="mlp", debug=debug)
    pre_cfg = config.get("preprocess", {})
    do_scale = bool(pre_cfg.get("scale_features", True))
    df_scaled, _ = scale_features(df, feat_cols, train_mask, label="mlp", debug=debug, scale=do_scale)
    return df_scaled, target_col, train_mask, val_mask, test_mask


def _fit_one(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[dict, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = torch.nn.MSELoss()

    best_state: dict | None = None
    best_val = float("inf")
    bad_epochs = 0

    for _ in range(int(max_epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_state, best_val


def train_mlp(config: dict) -> None:
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("training", {}).get("device", "cpu"))
    use_cuda = device.type == "cuda"
    num_workers = default_num_workers()

    panel, feat_cols = prepare_panel(config, prefer_cached_feature_panel=True)
    config["_mlp_panel"] = panel
    df_scaled, target_col, train_mask, val_mask, test_mask = _prepare_data(config, feat_cols)
    config.pop("_mlp_panel", None)

    x = df_scaled[feat_cols].to_numpy(dtype=np.float32, copy=True)
    y = df_scaled[target_col].to_numpy(dtype=np.float32, copy=True)

    x_train = torch.tensor(x[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[train_mask], dtype=torch.float32)
    x_val = torch.tensor(x[val_mask], dtype=torch.float32)
    y_val = torch.tensor(y[val_mask], dtype=torch.float32)
    x_test = torch.tensor(x[test_mask], dtype=torch.float32)
    y_test = torch.tensor(y[test_mask], dtype=torch.float32)

    batch_size = int(config.get("training", {}).get("batch_size", 512))
    loader_kwargs = {
        "batch_size": max(32, batch_size),
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(TensorDataset(x_val, y_val), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False, **loader_kwargs)

    model_cfg = config.get("model", {})
    fixed_params = {
        "hidden_dim_1": int(model_cfg.get("hidden_dim_1", 128)),
        "hidden_dim_2": int(model_cfg.get("hidden_dim_2", 64)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "lr": float(config.get("training", {}).get("lr", 5e-4)),
        "weight_decay": float(config.get("training", {}).get("weight_decay", 1e-5)),
    }
    tune_cfg = config.get("tuning", {})
    use_tuning = bool(tune_cfg.get("enabled", False))
    best_params = dict(fixed_params)

    if use_tuning:
        param_grid = tune_cfg.get(
            "param_grid",
            {
                "hidden_dim_1": [64, 128],
                "hidden_dim_2": [32, 64],
                "dropout": [0.0, 0.1, 0.2],
                "lr": [3e-4, 5e-4, 1e-3],
                "weight_decay": [0.0, 1e-5, 5e-5],
            },
        )
        candidates = enumerate_param_candidates(
            fixed_params=fixed_params,
            param_grid=param_grid,
            sample_mode=str(tune_cfg.get("sample_mode", "random")),
            max_trials=int(tune_cfg.get("max_trials", 12) or 12),
            seed=int(tune_cfg.get("seed", config.get("seed", 42))),
        )
        tune_epochs = int(tune_cfg.get("tune_max_epochs", min(12, int(config.get("training", {}).get("max_epochs", 40)))))
        tune_patience = int(tune_cfg.get("tune_patience", 4))
        objective = str(tune_cfg.get("objective", "val_backtest_sharpe_annualized"))
        eval_cfg = config.get("evaluation", {})
        val_dates = df_scaled.loc[val_mask, "date"].to_numpy()
        val_tickers = df_scaled.loc[val_mask, "ticker"].to_numpy()
        y_val_np = y[val_mask]

        best_score = float("-inf")
        for params in candidates:
            candidate = dict(fixed_params)
            candidate.update(params)
            m = BaselineMLP(
                input_dim=len(feat_cols),
                hidden_dim_1=int(candidate["hidden_dim_1"]),
                hidden_dim_2=int(candidate["hidden_dim_2"]),
                dropout=float(candidate["dropout"]),
            ).to(device)
            _fit_one(
                model=m,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=float(candidate["lr"]),
                weight_decay=float(candidate["weight_decay"]),
                max_epochs=tune_epochs,
                patience=tune_patience,
                device=device,
            )
            preds = []
            m.eval()
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    preds.extend(m(xb).detach().cpu().numpy().tolist())
            score_payload = score_prediction_objective(
                objective=objective,
                y_true=y_val_np,
                preds=preds,
                dates=val_dates,
                tickers=val_tickers,
                top_k=int(eval_cfg.get("top_k", 20)),
                transaction_cost_bps=float(eval_cfg.get("transaction_cost_bps", 0.0)),
                risk_free_rate=float(eval_cfg.get("risk_free_rate", 0.0)),
                rebalance_freq=int(eval_cfg.get("primary_rebalance_freq", 1)),
            )
            score = float(score_payload["score"])
            if score > best_score:
                best_score = score
                best_params = candidate

    out_dirs = resolve_output_dirs(config, model_type=config.get("model", {}).get("type", "mlp"))
    out_dir = out_dirs.canonical
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BaselineMLP(
        input_dim=len(feat_cols),
        hidden_dim_1=int(best_params["hidden_dim_1"]),
        hidden_dim_2=int(best_params["hidden_dim_2"]),
        dropout=float(best_params["dropout"]),
    ).to(device)

    train_start = time.time()
    best_state, _ = _fit_one(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        max_epochs=int(config.get("training", {}).get("max_epochs", 40)),
        patience=int(config.get("training", {}).get("patience", 6)),
        device=device,
    )
    train_seconds = time.time() - train_start
    model_path = out_dir / "best_mlp.pt"
    torch.save(best_state, model_path)

    model.eval()
    preds = []
    reals = []
    infer_start = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yp = model(xb).detach().cpu().numpy()
            preds.extend(yp.tolist())
            reals.extend(yb.detach().cpu().numpy().tolist())
    inference_seconds = time.time() - infer_start

    test_rows = df_scaled.loc[test_mask, ["date", "ticker"]].reset_index(drop=True)
    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(test_rows["date"]),
            "ticker": test_rows["ticker"].astype(str),
            "pred": np.asarray(preds, dtype=float),
            "realized_ret": np.asarray(reals, dtype=float),
        }
    ).sort_values("date")
    pred_df = sanitize_predictions(pred_df, strict_unique=True)
    pred_df.to_csv(out_dir / "mlp_predictions.csv", index=False)

    summary = evaluate_and_report(
        config=config,
        pred_df=pred_df,
        out_dirs=out_dirs,
        run_name="mlp",
        model_name=config.get("model", {}).get("type", "mlp"),
        model_family=config.get("model", {}).get("family", "mlp"),
        edge_type="none",
        directed=False,
        graph_window="",
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
    )
    print("MLP primary policy stats:", summary.get("stats", {}))

