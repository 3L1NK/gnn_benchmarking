from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import backtest_long_only
from .metrics import rank_ic
from .predictions import sanitize_predictions


SUPPORTED_TUNING_OBJECTIVES = {
    "val_rmse",
    "val_ic",
    "val_backtest_sharpe_annualized",
}


def _as_list(values) -> List:
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def enumerate_param_candidates(
    *,
    fixed_params: Mapping[str, object] | None = None,
    param_grid: Mapping[str, Iterable[object]] | None = None,
    sample_mode: str = "grid",
    max_trials: int = 0,
    seed: int = 42,
) -> List[Dict[str, object]]:
    fixed = dict(fixed_params or {})
    grid = dict(param_grid or {})
    if not grid:
        return [fixed]

    keys = list(grid.keys())
    value_lists = [_as_list(grid[k]) for k in keys]
    combos = []
    for values in product(*value_lists):
        overrides = dict(zip(keys, values))
        combos.append({**fixed, **overrides})

    if not combos:
        return [fixed]

    sample_mode_norm = str(sample_mode or "grid").strip().lower()
    if sample_mode_norm == "random":
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(combos)).tolist()
        ordered = [combos[i] for i in order]
    else:
        ordered = combos

    if int(max_trials or 0) > 0:
        return ordered[: int(max_trials)]
    return ordered


def _mean_daily_rank_ic(pred_df: pd.DataFrame) -> float:
    rows = []
    for _, g in pred_df.groupby("date", sort=True):
        rows.append(rank_ic(g["pred"], g["realized_ret"]))
    if not rows:
        return float("nan")
    return float(pd.Series(rows, dtype=float).mean())


def score_prediction_objective(
    *,
    objective: str,
    y_true: Sequence[float],
    preds: Sequence[float],
    dates: Sequence,
    tickers: Sequence,
    top_k: int,
    transaction_cost_bps: float,
    risk_free_rate: float,
    rebalance_freq: int = 1,
) -> Dict[str, float]:
    objective_norm = str(objective or "val_rmse").strip().lower()
    if objective_norm not in SUPPORTED_TUNING_OBJECTIVES:
        raise ValueError(
            f"Unsupported tuning objective '{objective_norm}'. "
            f"Expected one of {sorted(SUPPORTED_TUNING_OBJECTIVES)}."
        )

    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "ticker": pd.Series(tickers).astype(str).values,
            "pred": np.asarray(preds, dtype=float),
            "realized_ret": np.asarray(y_true, dtype=float),
        }
    )
    pred_df = sanitize_predictions(pred_df, strict_unique=True)
    y = pred_df["realized_ret"].to_numpy(dtype=float)
    p = pred_df["pred"].to_numpy(dtype=float)

    if objective_norm == "val_rmse":
        value = float(np.sqrt(np.mean((y - p) ** 2))) if len(pred_df) else float("nan")
        score = -value if np.isfinite(value) else float("-inf")
        return {"score": float(score), "metric_value": float(value), "metric_name": "rmse"}

    if objective_norm == "val_ic":
        value = _mean_daily_rank_ic(pred_df)
        score = value if np.isfinite(value) else float("-inf")
        return {"score": float(score), "metric_value": float(value), "metric_name": "mean_rank_ic"}

    # val_backtest_sharpe_annualized
    _, _, stats = backtest_long_only(
        pred_df,
        top_k=int(top_k),
        transaction_cost_bps=float(transaction_cost_bps),
        risk_free_rate=float(risk_free_rate),
        rebalance_freq=int(rebalance_freq),
    )
    value = float(stats.get("sharpe_annualized", float("nan")))
    if not np.isfinite(value):
        sharpe_daily = float(stats.get("sharpe", float("nan")))
        value = float(sharpe_daily * np.sqrt(252.0)) if np.isfinite(sharpe_daily) else float("nan")
    score = value if np.isfinite(value) else float("-inf")
    return {"score": float(score), "metric_value": float(value), "metric_name": "sharpe_annualized"}

