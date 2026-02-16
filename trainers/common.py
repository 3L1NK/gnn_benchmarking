from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.preprocessing import scale_features
from utils.splits import split_time
from utils.targets import build_target


DEFAULT_FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_20d", "log_ret_1d",
    "mom_3d", "mom_10", "mom_21d",
    "vol_5d", "vol_20d", "vol_60d",
    "drawdown_20d",
    "volume_pct_change", "vol_z_5", "vol_z_20",
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
]


def prepare_panel(config: dict, *, prefer_cached_feature_panel: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    df = load_price_panel(price_file, start, end)
    feat_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]

    if not feat_cols or not prefer_cached_feature_panel:
        df, generated = add_technical_features(df)
        feat_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns] or list(generated)

    return df, feat_cols


def build_target_and_splits(
    df: pd.DataFrame,
    feat_cols: List[str],
    config: dict,
    *,
    label: str,
    debug: bool = False,
    target_col: str = "target",
):
    df, target_col = build_target(df, config, target_col=target_col)
    df = df.dropna(subset=list(feat_cols) + [target_col]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    train_mask, val_mask, test_mask, split_meta = split_time(df["date"], config, label=label, debug=debug)
    pre_cfg = config.get("preprocess", {})
    do_scale = bool(pre_cfg.get("scale_features", True))
    df_scaled, scaler = scale_features(df, feat_cols, train_mask, label=label, debug=debug, scale=do_scale)
    return df_scaled, target_col, train_mask, val_mask, test_mask, split_meta, scaler


def fit_model(*args, **kwargs):
    raise NotImplementedError("Family-specific trainers implement this.")


def predict(*args, **kwargs):
    raise NotImplementedError("Family-specific trainers implement this.")


def evaluate_and_report(*args, **kwargs):
    from utils.eval_runner import evaluate_and_report as _eval

    return _eval(*args, **kwargs)
