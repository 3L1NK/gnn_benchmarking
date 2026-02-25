from __future__ import annotations

import hashlib
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
    "split_id",
    "config_hash",
    "run_key",
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


def _short_hash(payload: Dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _config_signature(config: dict) -> Dict[str, object]:
    data_cfg = config.get("data", {}) or {}
    eval_cfg = config.get("evaluation", {}) or {}
    exp_cfg = config.get("experiment", {}) or {}

    data_signature = {
        "target_type": data_cfg.get("target_type", ""),
        "target_name": data_cfg.get("target_name", ""),
        "target_horizon": data_cfg.get("target_horizon", ""),
        "lookback_window": data_cfg.get("lookback_window", ""),
        "corr_window": data_cfg.get("corr_window", ""),
        "corr_threshold": data_cfg.get("corr_threshold", ""),
        "min_history": data_cfg.get("min_history", ""),
    }
    eval_signature = {
        "top_k": eval_cfg.get("top_k", ""),
        "transaction_cost_bps": eval_cfg.get("transaction_cost_bps", ""),
        "risk_free_rate": eval_cfg.get("risk_free_rate", ""),
        "backtest_policies": eval_cfg.get("backtest_policies", ""),
        "primary_rebalance_freq": eval_cfg.get("primary_rebalance_freq", ""),
    }
    return {
        "model": config.get("model", {}) or {},
        "graph": config.get("graph", {}) or {},
        "granger": config.get("granger", {}) or {},
        "training": config.get("training", {}) or {},
        "tuning": config.get("tuning", {}) or {},
        "loss": config.get("loss", {}) or {},
        "data": data_signature,
        "evaluation": eval_signature,
        "experiment": {
            "protocol_version": exp_cfg.get("protocol_version", ""),
            "enforce_protocol": exp_cfg.get("enforce_protocol", ""),
        },
    }


def config_hash_from_config(config: dict) -> str:
    return _short_hash(_config_signature(config))


def split_id_from_fields(protocol_version: object, split_train_end: object, split_val_start: object, split_test_start: object, target_policy_hash: object) -> str:
    payload = {
        "protocol_version": str(protocol_version or ""),
        "split_train_end": str(split_train_end or ""),
        "split_val_start": str(split_val_start or ""),
        "split_test_start": str(split_test_start or ""),
        "target_policy_hash": str(target_policy_hash or ""),
    }
    return _short_hash(payload)


def run_key_from_fields(
    model_family: object,
    model_name: object,
    edge_type: object,
    seed: object,
    target_type: object,
    target_horizon: object,
    rebalance_freq: object,
    split_id: object,
    config_hash: object,
) -> str:
    payload = {
        "model_family": str(model_family or ""),
        "model_name": str(model_name or ""),
        "edge_type": str(edge_type or ""),
        "seed": int(seed) if seed is not None and str(seed) != "" else "",
        "target_type": str(target_type or ""),
        "target_horizon": int(target_horizon) if target_horizon is not None and str(target_horizon) != "" else "",
        "rebalance_freq": int(rebalance_freq) if rebalance_freq is not None and str(rebalance_freq) != "" else "",
        "split_id": str(split_id or ""),
        "config_hash": str(config_hash or ""),
    }
    return _short_hash(payload)


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

    sharpe_daily = float(stats.get("sharpe", float("nan")))
    sharpe_annualized = float(stats.get("sharpe_annualized", float("nan")))
    if not np.isfinite(sharpe_annualized) and np.isfinite(sharpe_daily):
        sharpe_annualized = float(sharpe_daily * np.sqrt(252.0))

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
        "split_id": "",
        "config_hash": config_hash_from_config(config),
        "run_key": "",
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
        "portfolio_sharpe": sharpe_daily,
        "portfolio_sharpe_daily": sharpe_daily,
        "portfolio_sharpe_annualized": sharpe_annualized,
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

    result["split_id"] = split_id_from_fields(
        result.get("protocol_version"),
        result.get("split_train_end"),
        result.get("split_val_start"),
        result.get("split_test_start"),
        result.get("target_policy_hash"),
    )
    result["run_key"] = run_key_from_fields(
        result.get("model_family"),
        result.get("model_name"),
        result.get("edge_type"),
        result.get("seed"),
        result.get("target_type"),
        result.get("target_horizon"),
        result.get("rebalance_freq"),
        result.get("split_id"),
        result.get("config_hash"),
    )

    # Ensure all fields exist
    for key in RESULT_FIELDS:
        if key not in result:
            result[key] = ""
    return result


def save_experiment_result(result: dict, results_path: Optional[Path] = None) -> Path:
    if results_path is None:
        results_path = Path("results") / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: result.get(k, "") for k in RESULT_FIELDS}

    existing_rows = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_rows.append({k: row.get(k, "") for k in RESULT_FIELDS})

    incoming_key = str(ordered.get("run_key", "") or "")
    incoming_ts = pd.to_datetime(ordered.get("timestamp", ""), errors="coerce")
    replaced = False
    if incoming_key:
        for idx, row in enumerate(existing_rows):
            row_key = str(row.get("run_key", "") or "")
            if row_key != incoming_key:
                continue
            row_ts = pd.to_datetime(row.get("timestamp", ""), errors="coerce")
            should_replace = pd.isna(row_ts) or (not pd.isna(incoming_ts) and incoming_ts >= row_ts)
            if should_replace:
                existing_rows[idx] = ordered
            replaced = True
            break
    if not replaced:
        existing_rows.append(ordered)

    def _ts_sort_key(row: dict) -> tuple[int, pd.Timestamp]:
        ts = pd.to_datetime(row.get("timestamp", ""), errors="coerce")
        if pd.isna(ts):
            return (1, pd.Timestamp.min)
        return (0, ts)

    existing_rows = sorted(existing_rows, key=_ts_sort_key)
    tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in existing_rows:
            f.write(json.dumps({k: row.get(k, "") for k in RESULT_FIELDS}) + "\n")
    tmp_path.replace(results_path)
    return results_path
