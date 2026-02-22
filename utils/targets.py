from __future__ import annotations

from typing import Tuple

import pandas as pd


_TARGET_NAME_TO_COLUMN = {
    "log_return": "log_ret_1d",
}


def build_target(df: pd.DataFrame, config: dict, target_col: str = "target") -> Tuple[pd.DataFrame, str]:
    """Attach a single supervised target column based on config data keys."""
    data_cfg = config.get("data", {})
    target_type = data_cfg.get("target_type", "regression")
    target_name = data_cfg.get("target_name", "log_return")
    horizon = int(data_cfg.get("target_horizon", 1))

    if target_type != "regression":
        raise ValueError(f"Unsupported target_type={target_type}; only 'regression' is supported.")
    if horizon < 1:
        raise ValueError(f"target_horizon must be >= 1, got {horizon}.")

    base_col = _TARGET_NAME_TO_COLUMN.get(target_name)
    if base_col is None:
        raise ValueError(f"Unsupported target_name={target_name}; expected one of {sorted(_TARGET_NAME_TO_COLUMN)}.")
    if base_col not in df.columns:
        raise ValueError(f"Expected column '{base_col}' to build target '{target_name}'.")

    out = df.copy()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out[target_col] = out.groupby("ticker")[base_col].shift(-horizon)
    out = out.dropna(subset=[target_col]).reset_index(drop=True)
    return out, target_col
