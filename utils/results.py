from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .metrics import rank_ic


RESULT_FIELDS = [
    "experiment_id",
    "timestamp",
    "seed",
    "model_name",
    "model_family",
    "edge_type",
    "directed",
    "graph_window",
    "target_type",
    "target_horizon",
    "lookback_window",
    "protocol_version",
    "split_train_end",
    "split_val_start",
    "split_test_start",
    "rebalance_freq",
    "baseline_version",
    "target_policy_hash",
    "prediction_rows",
    "prediction_unique_pairs",
    "prediction_rmse",
    "prediction_mae",
    "prediction_rank_ic",
    "portfolio_final_value",
    "portfolio_cumulative_return",
    "portfolio_annualized_return",
    "portfolio_annualized_volatility",
    "portfolio_sharpe",
    "portfolio_sharpe_daily",
    "portfolio_sharpe_annualized",
    "portfolio_sortino_annualized",
    "portfolio_max_drawdown",
    "portfolio_turnover",
    "runtime_train_seconds",
    "runtime_inference_seconds",
    "run_tag",
    "out_dir",
    "artifact_prefix",
]


def _format_window(start, end) -> str:
    if start is None or end is None:
        return ""
    try:
        s = pd.to_datetime(start).date()
        e = pd.to_datetime(end).date()
        return f"{s}..{e}"
    except Exception:
        return ""


def edge_type_from_graph_cfg(graph_cfg: dict) -> str:
    parts = []
    if graph_cfg.get("use_corr"):
        parts.append("corr")
    if graph_cfg.get("use_sector"):
        parts.append("sector")
    if graph_cfg.get("use_granger"):
        parts.append("granger")
    return "+".join(parts) if parts else "none"


def prediction_metrics(pred_df: Optional[pd.DataFrame], daily_metrics: Optional[pd.DataFrame]) -> Dict[str, float]:
    rmse = float("nan")
    mae = float("nan")
    rank_ic_mean = float("nan")

    if pred_df is not None and not pred_df.empty:
        diff = pred_df["pred"].to_numpy(dtype=float) - pred_df["realized_ret"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))

    if daily_metrics is not None and not daily_metrics.empty and "ic" in daily_metrics:
        rank_ic_mean = float(daily_metrics["ic"].mean())
    elif pred_df is not None and not pred_df.empty:
        rank_ic_mean = float(rank_ic(pred_df["pred"], pred_df["realized_ret"]))

    return {"rmse": rmse, "mae": mae, "rank_ic": rank_ic_mean}


def build_experiment_result(
    config: dict,
    *,
    model_name: str,
    model_family: str,
    edge_type: str,
    directed: bool,
    graph_window: str,
    pred_df: Optional[pd.DataFrame],
    daily_metrics: Optional[pd.DataFrame],
    stats: dict,
    train_seconds: float,
    inference_seconds: float,
    protocol_fields: Optional[dict] = None,
    prediction_rows: Optional[int] = None,
    prediction_unique_pairs: Optional[int] = None,
    run_tag: Optional[str] = None,
    out_dir: Optional[str] = None,
    artifact_prefix: Optional[str] = None,
) -> dict:
    pm = prediction_metrics(pred_df, daily_metrics)
    data_cfg = config.get("data", {})
    run_tag = run_tag or config.get("experiment_name", model_name)
    seed = int(config.get("seed", 42))
    rebalance_for_id = ""
    if protocol_fields and protocol_fields.get("rebalance_freq") is not None:
        rebalance_for_id = f"_reb{int(protocol_fields.get('rebalance_freq'))}"

    result = {
        "experiment_id": f"{run_tag}_{int(time.time())}{rebalance_for_id}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "seed": seed,
        "run_tag": run_tag,
        "out_dir": str(out_dir or config.get("evaluation", {}).get("out_dir", "")),
        "artifact_prefix": str(artifact_prefix or model_name),
        "model_name": model_name,
        "model_family": model_family,
        "edge_type": edge_type,
        "directed": bool(directed),
        "graph_window": graph_window,
        "target_type": data_cfg.get("target_type", ""),
        "target_horizon": int(data_cfg.get("target_horizon", 0)) if data_cfg.get("target_horizon") is not None else 0,
        "lookback_window": int(data_cfg.get("lookback_window", 0)) if data_cfg.get("lookback_window") is not None else 0,
        "protocol_version": "",
        "split_train_end": "",
        "split_val_start": str(config.get("training", {}).get("val_start", "")),
        "split_test_start": str(config.get("training", {}).get("test_start", "")),
        "rebalance_freq": 0,
        "baseline_version": "",
        "target_policy_hash": "",
        "prediction_rows": int(prediction_rows) if prediction_rows is not None else (int(len(pred_df)) if pred_df is not None else 0),
        "prediction_unique_pairs": int(prediction_unique_pairs)
        if prediction_unique_pairs is not None
        else (
            int(pred_df.drop_duplicates(["date", "ticker"]).shape[0])
            if pred_df is not None and not pred_df.empty and {"date", "ticker"}.issubset(pred_df.columns)
            else 0
        ),
        "prediction_rmse": pm["rmse"],
        "prediction_mae": pm["mae"],
        "prediction_rank_ic": pm["rank_ic"],
        "portfolio_final_value": float(stats.get("final_value", float("nan"))),
        "portfolio_cumulative_return": float(stats.get("cumulative_return", float("nan"))),
        "portfolio_annualized_return": float(stats.get("annualized_return", float("nan"))),
        "portfolio_annualized_volatility": float(stats.get("annualized_volatility", float("nan"))),
        "portfolio_sharpe": float(stats.get("sharpe", float("nan"))),
        "portfolio_sharpe_daily": float(stats.get("sharpe", float("nan"))),
        "portfolio_sharpe_annualized": float(stats.get("sharpe_annualized", float("nan"))),
        "portfolio_sortino_annualized": float(stats.get("sortino_annualized", float("nan"))),
        "portfolio_max_drawdown": float(stats.get("max_drawdown", float("nan"))),
        "portfolio_turnover": float(stats.get("avg_turnover", float("nan"))),
        "runtime_train_seconds": float(train_seconds),
        "runtime_inference_seconds": float(inference_seconds),
    }

    if protocol_fields:
        result.update({k: protocol_fields.get(k, result.get(k, "")) for k in [
            "protocol_version",
            "split_train_end",
            "split_val_start",
            "split_test_start",
            "rebalance_freq",
            "baseline_version",
            "target_policy_hash",
        ]})

    # Ensure all fields exist
    for key in RESULT_FIELDS:
        if key not in result:
            result[key] = ""
    return result


def save_experiment_result(result: dict, results_path: Optional[Path] = None) -> Path:
    if results_path is None:
        results_path = Path("results") / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # enforce stable key ordering
    ordered = {k: result.get(k, "") for k in RESULT_FIELDS}
    with results_path.open("a") as f:
        f.write(json.dumps(ordered) + "\n")
    return results_path
