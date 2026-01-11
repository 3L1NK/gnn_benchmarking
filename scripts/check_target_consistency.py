"""
Assert that target labels align across model data pipelines.

Run from repo root:
  python scripts/check_target_consistency.py
"""
import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from trainers.train_gnn import _build_snapshots_and_targets
from trainers.train_lstm import _build_sequences
from trainers.train_xgboost import _build_feature_panel
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.targets import build_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _deep_update(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    include_path = cfg.pop("include", None)
    if include_path:
        if not isinstance(include_path, list):
            include_path = [include_path]
        base = {}
        for inc in include_path:
            inc_path = Path(inc)
            if not inc_path.is_absolute():
                candidate = (config_path.parent / inc_path).resolve()
                if candidate.exists():
                    inc_path = candidate
                else:
                    inc_path = (PROJECT_ROOT / inc_path).resolve()
            if not inc_path.exists():
                raise FileNotFoundError(f"Included config '{inc_path}' does not exist")
            base = _deep_update(base, deepcopy(load_config(inc_path)))
        cfg = _deep_update(base, cfg)

    # Backwards compatibility mapping for older configs that still use graph_edges.
    if "graph" not in cfg and "graph_edges" in cfg:
        ge = cfg.get("graph_edges", {}) or {}
        graph = {}
        graph["use_corr"] = bool(ge.get("use_correlation", ge.get("use_corr", False)))
        graph["use_sector"] = bool(ge.get("use_sector", False))
        if ge.get("use_industry", False):
            graph["use_sector"] = True
        graph["use_granger"] = bool(cfg.get("granger", {}).get("enabled", False))
        if "corr_top_k" in ge:
            graph["corr_top_k"] = int(ge.get("corr_top_k", 10))
        if "corr_min_periods" in ge:
            graph["corr_min_periods"] = int(ge.get("corr_min_periods", 0))
        if "sector_weight" in ge:
            graph["sector_weight"] = float(ge.get("sector_weight", 0.2))
        if "industry_weight" in ge:
            graph["industry_weight"] = float(ge.get("industry_weight", 0.1))
        if "w_corr" in ge:
            graph["w_corr"] = float(ge.get("w_corr", 1.0))
        if "w_sector" in ge:
            graph["w_sector"] = float(ge.get("w_sector", 0.2))
        if "w_granger" in ge:
            graph["w_granger"] = float(ge.get("w_granger", 0.2))
        if "sector_top_k" in ge:
            graph["sector_top_k"] = int(ge.get("sector_top_k", 5))
        if "granger_top_k" in ge:
            graph["granger_top_k"] = int(ge.get("granger_top_k", 5))
        cfg["graph"] = graph

    return cfg


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
    parser.add_argument("--xgb-config", default="configs/xgb_raw.yaml")
    parser.add_argument("--lstm-config", default="configs/lstm.yaml")
    parser.add_argument("--gnn-config", default="configs/gcn/gcn.yaml")
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
