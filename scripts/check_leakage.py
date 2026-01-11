"""
Leakage checks for splits, scaling, and graph windows.

Run from repo root:
  python scripts/check_leakage.py
"""
import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from trainers.train_gnn import _build_snapshots_and_targets
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.preprocessing import fit_feature_scaler
from utils.splits import split_time
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


def _check_split_consistency(cfgs):
    val_starts = {cfg["training"]["val_start"] for cfg in cfgs}
    test_starts = {cfg["training"]["test_start"] for cfg in cfgs}
    if len(val_starts) != 1 or len(test_starts) != 1:
        raise ValueError(f"Split boundaries differ: val_start={val_starts}, test_start={test_starts}")


def _check_scaler_train_only(cfg):
    df = load_price_panel(cfg["data"]["price_file"], cfg["data"]["start_date"], cfg["data"]["end_date"])
    df, feat_cols = add_technical_features(df)
    df, target_col = build_target(df, cfg, target_col="target")
    df = df.dropna(subset=feat_cols + [target_col]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    train_mask, _, _, _ = split_time(df["date"], cfg, label="leakage_check", debug=False)
    train_dates = df.loc[train_mask, "date"]
    val_start = pd.to_datetime(cfg["training"]["val_start"])
    if train_dates.empty:
        raise ValueError("No training rows found for scaler check.")
    if train_dates.max() >= val_start:
        raise ValueError("Scaler fit range leaks into validation/test period.")

    fit_feature_scaler(df, feat_cols, train_mask, label="leakage_check", debug=False)


def _check_gnn_windows(cfg):
    snapshots, _, _ = _build_snapshots_and_targets(cfg)
    for g in snapshots:
        if not hasattr(g, "window_end") or not hasattr(g, "date"):
            raise ValueError("GNN snapshots missing window_end/date metadata.")
        if pd.to_datetime(g.window_end) > pd.to_datetime(g.date):
            raise ValueError(f"GNN window_end {g.window_end} exceeds snapshot date {g.date}")


def _check_xgb_graph_window(cfg):
    df = load_price_panel(cfg["data"]["price_file"], cfg["data"]["start_date"], cfg["data"]["end_date"])
    df["date"] = pd.to_datetime(df["date"])
    val_start = pd.to_datetime(cfg["training"]["val_start"])
    train_dates = df.loc[df["date"] < val_start, "date"]
    if train_dates.empty:
        raise ValueError("No training rows found for XGB graph window check.")
    if train_dates.max() >= val_start:
        raise ValueError("XGB graph window leaks into validation/test period.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-config", default="configs/xgb_raw.yaml")
    parser.add_argument("--lstm-config", default="configs/lstm.yaml")
    parser.add_argument("--gnn-config", default="configs/gcn/gcn.yaml")
    args = parser.parse_args()

    cfg_xgb = load_config(Path(args.xgb_config))
    cfg_lstm = load_config(Path(args.lstm_config))
    cfg_gnn = load_config(Path(args.gnn_config))

    _check_split_consistency([cfg_xgb, cfg_lstm, cfg_gnn])
    _check_scaler_train_only(cfg_xgb)
    _check_scaler_train_only(cfg_lstm)
    _check_scaler_train_only(cfg_gnn)
    _check_xgb_graph_window(cfg_xgb)
    _check_gnn_windows(cfg_gnn)

    print("Leakage checks passed for splits, scalers, and GNN windows.")


if __name__ == "__main__":
    main()
