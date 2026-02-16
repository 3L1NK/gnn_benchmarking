"""
Assert that target labels align across model data pipelines.

Run from repo root:
  python scripts/check_target_consistency.py
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.train_gnn import _build_snapshots_and_targets
from trainers.train_lstm import _build_sequences
from trainers.train_xgboost import _build_feature_panel
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.targets import build_target
from utils.config_normalize import load_config as load_normalized_config

PROJECT_ROOT = REPO_ROOT
def load_config(config_path: Path) -> dict:
    return load_normalized_config(config_path, PROJECT_ROOT)


def _target_cfg(cfg: dict) -> dict:
    data = cfg.get("data", {})
    return {
        "target_type": data.get("target_type"),
        "target_name": data.get("target_name"),
        "target_horizon": data.get("target_horizon"),
    }


def _assert_targets_match(name: str, model_df: pd.DataFrame, target_df: pd.DataFrame, tol: float = 1e-9) -> None:
    model_df = model_df.copy()
    target_df = target_df.copy()
    model_df["date"] = pd.to_datetime(model_df["date"])
    target_df["date"] = pd.to_datetime(target_df["date"])
    merged = model_df.merge(target_df, on=["date", "ticker"], how="inner", suffixes=("_model", "_target"))
    if merged.empty:
        raise AssertionError(f"[{name}] no overlapping (date, ticker) pairs to compare.")
    if len(merged) != len(model_df):
        raise AssertionError(f"[{name}] only {len(merged)} of {len(model_df)} rows matched target_df.")
    max_diff = (merged["target_model"] - merged["target_target"]).abs().max()
    if not pd.isna(max_diff) and max_diff > tol:
        raise AssertionError(f"[{name}] target mismatch: max abs diff {max_diff:.3e} exceeds {tol}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-config", default="configs/runs/core/xgb_raw.yaml")
    parser.add_argument("--lstm-config", default="configs/runs/core/lstm.yaml")
    parser.add_argument("--gnn-config", default="configs/runs/core/gcn_corr_only.yaml")
    args = parser.parse_args()

    cfg_xgb = load_config(Path(args.xgb_config))
    cfg_lstm = load_config(Path(args.lstm_config))
    cfg_gnn = load_config(Path(args.gnn_config))

    target_cfgs = [_target_cfg(cfg) for cfg in (cfg_xgb, cfg_lstm, cfg_gnn)]
    if len({tuple(sorted(c.items())) for c in target_cfgs}) != 1:
        raise ValueError(f"Target configs differ across models: {target_cfgs}")

    price_files = {cfg["data"]["price_file"] for cfg in (cfg_xgb, cfg_lstm, cfg_gnn)}
    if len(price_files) != 1:
        raise ValueError(f"Configs must use the same price_file to compare targets: {price_files}")
    price_file = price_files.pop()

    start = max(pd.to_datetime(cfg["data"]["start_date"]) for cfg in (cfg_xgb, cfg_lstm, cfg_gnn))
    end = min(pd.to_datetime(cfg["data"]["end_date"]) for cfg in (cfg_xgb, cfg_lstm, cfg_gnn))
    if start >= end:
        raise ValueError("No overlapping date range across configs.")

    for cfg in (cfg_xgb, cfg_lstm, cfg_gnn):
        cfg["data"]["start_date"] = start.strftime("%Y-%m-%d")
        cfg["data"]["end_date"] = end.strftime("%Y-%m-%d")

    df = load_price_panel(price_file, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    df, feat_cols = add_technical_features(df)
    df, target_col = build_target(df, cfg_xgb, target_col="target")
    target_df = df[["date", "ticker", target_col]].rename(columns={target_col: "target"})

    # XGBoost panel
    xgb_df, _ = _build_feature_panel(cfg_xgb)
    xgb_df = xgb_df[["date", "ticker", "target"]].rename(columns={"target": "target"})
    _assert_targets_match("xgboost", xgb_df, target_df)

    # LSTM sequences
    lookback = cfg_lstm["data"]["lookback_window"]
    df_clean = df.dropna(subset=list(feat_cols) + [target_col]).reset_index(drop=True)
    X, y, dates, tickers = _build_sequences(df_clean, feat_cols, lookback, target_col=target_col)
    lstm_df = pd.DataFrame({"date": dates, "ticker": tickers, "target": y.numpy().astype(float)})
    _assert_targets_match("lstm", lstm_df, target_df)

    # GNN snapshots
    snapshots, _, _ = _build_snapshots_and_targets(cfg_gnn)
    rows = []
    for g in snapshots:
        if not hasattr(g, "valid_mask"):
            continue
        mask = g.valid_mask.cpu().numpy()
        y_vals = g.y.cpu().numpy()
        for t, m, y_val in zip(g.tickers, mask, y_vals):
            if not m:
                continue
            rows.append({"date": pd.to_datetime(g.date), "ticker": t, "target": float(y_val)})
    gnn_df = pd.DataFrame(rows)
    _assert_targets_match("gnn", gnn_df, target_df)

    print("Target consistency check passed for xgboost, lstm, and gnn.")


if __name__ == "__main__":
    main()
