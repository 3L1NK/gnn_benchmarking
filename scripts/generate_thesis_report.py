#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from utils.prediction_audit import assert_no_prediction_artifact_issues
from utils.metrics import annualized_sharpe_from_returns
from utils.backtest import backtest_long_only, backtest_long_short
from utils.baseline import compute_buy_and_hold_fixed_shares, compute_equal_weight_rebalanced, write_baseline_curve_csv
from utils.config_normalize import load_config
from utils.data_loading import load_price_panel


KEY_METRICS = [
    "prediction_rmse",
    "prediction_mae",
    "prediction_rank_ic",
    "portfolio_annualized_return",
    "portfolio_sharpe",
    "portfolio_sharpe_daily",
    "portfolio_sharpe_annualized",
    "portfolio_sortino_annualized",
    "portfolio_max_drawdown",
    "portfolio_turnover",
]

MODEL_ALIAS = {
    "xgb_raw": "XGB",
    "xgb_node2vec": "XGB+Node2Vec",
    "lstm": "LSTM",
    "gcn": "GCN",
    "gat": "GAT",
    "tgcn_static": "TGCN-static",
    "tgat_static": "TGAT-static",
}

EDGE_ALIAS = {
    "corr+sector+granger": "corr+sec+gr",
    "corr_sector_granger": "corr+sec+gr",
    "node2vec_correlation": "n2v-corr",
    "corr": "corr",
    "sector": "sector",
    "granger": "granger",
}

COST_SWEEP_BPS = [0, 5, 10]
ACTIVE_TRADING_DAYS = 252.0
BUY_HOLD_LABEL = "Buy and hold (fixed shares)"
EQW_LABEL = "Equal weight (rebalanced, all assets)"
TOPK_LABEL = "Top K long-only (equal weight within Top K)"
LONG_SHORT_LABEL = "Top 3 long, bottom 3 short (market-neutral)"


def _reb_label(freq: int) -> str:
    return f"reb={int(freq)}"


def _safe_read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    df = pd.read_json(path, lines=True)
    if df.empty:
        raise ValueError("results file exists but has no rows")
    return df


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _run_label(row: pd.Series) -> str:
    model_name = str(row.get("model_name", "")).strip().lower()
    edge_raw = row.get("edge_type", "")
    base = MODEL_ALIAS.get(model_name, model_name.upper() if model_name else "Model")
    edge = str(edge_raw).strip().lower() if pd.notna(edge_raw) else ""
    if edge and edge not in {"none", "nan"}:
        edge_label = EDGE_ALIAS.get(edge, edge.replace("_", "+"))
        return f"{base} ({edge_label})"
    return base


def _find_latest(path_pattern: str) -> Optional[Path]:
    candidates = list(Path(".").glob(path_pattern))
    if not candidates:
        return None
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _classify_category(row: pd.Series) -> str:
    model = str(row.get("model_name", "")).lower()
    family = str(row.get("model_family", "")).lower()
    edge = str(row.get("edge_type", "")).lower()

    if model in {"tgcn_static", "tgat_static", "tgcn", "tgat"}:
        return "static_temporal_labeled"
    if family == "gnn" and model in {"gcn", "gat"}:
        return "static_gnn"
    if model in {"xgb_node2vec", "graphlasso_linear", "graphlasso_xgb", "granger_xgb"}:
        return "graph_feature"
    if family in {"xgboost", "lstm"} and (model in {"xgb_raw", "lstm"} or edge in {"none", "", "nan"}):
        return "non_graph"
    return "other"


def _pick_latest_rows(master: pd.DataFrame) -> pd.DataFrame:
    """Pick latest row for each logical run key to avoid duplicate historical reruns in plots."""
    df = master.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT
    group_key_cols: List[str] = []
    if "run_key" in df.columns and df["run_key"].astype("string").fillna("").str.len().gt(0).any():
        df["__group_run_key"] = df["run_key"].astype("string").fillna("__NA__")
        group_key_cols = ["__group_run_key"]
    else:
        key_cols = ["run_tag", "model_name", "edge_type", "rebalance_freq", "target_policy_hash"]
        for col in key_cols:
            if col not in df.columns:
                df[col] = pd.NA
            gcol = f"__group_{col}"
            if col == "rebalance_freq":
                df[gcol] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
            else:
                df[gcol] = df[col].astype("string").fillna("__NA__")
            group_key_cols.append(gcol)
    df = df.sort_values("timestamp")
    latest = df.groupby(group_key_cols, as_index=False, dropna=False).tail(1).reset_index(drop=True)
    latest = latest.drop(columns=group_key_cols, errors="ignore")
    return latest


def _best_by_group(df: pd.DataFrame, group_cols: List[str], score_col: str = "portfolio_sharpe_annualized") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.sort_values(score_col, ascending=False)
    return ranked.groupby(group_cols, as_index=False).head(1).reset_index(drop=True)


def _fill_annualized_sharpe(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["portfolio_sharpe"] = pd.to_numeric(out.get("portfolio_sharpe"), errors="coerce")
    if "portfolio_sharpe_daily" not in out.columns:
        out["portfolio_sharpe_daily"] = out["portfolio_sharpe"]
    out["portfolio_sharpe_daily"] = pd.to_numeric(out["portfolio_sharpe_daily"], errors="coerce")
    if "portfolio_sharpe_annualized" not in out.columns:
        out["portfolio_sharpe_annualized"] = pd.NA
    out["portfolio_sharpe_annualized"] = pd.to_numeric(out["portfolio_sharpe_annualized"], errors="coerce")

    missing_ann = out["portfolio_sharpe_annualized"].isna()
    if missing_ann.any():
        out.loc[missing_ann, "portfolio_sharpe_annualized"] = (
            out.loc[missing_ann, "portfolio_sharpe_daily"] * np.sqrt(252.0)
        )

    # Final fallback from daily return artifacts for rows that remain missing.
    still_missing_idx = out.index[out["portfolio_sharpe_annualized"].isna()].tolist()
    for idx in still_missing_idx:
        row = out.loc[idx]
        freq = int(pd.to_numeric(row.get("rebalance_freq"), errors="coerce")) if pd.notna(row.get("rebalance_freq")) else 1
        metrics_path = _daily_metrics_path(row, freq)
        if not metrics_path.exists():
            continue
        try:
            m = pd.read_csv(metrics_path)
        except Exception:
            continue
        if "daily_return" not in m.columns:
            continue
        sharpe = annualized_sharpe_from_returns(pd.to_numeric(m["daily_return"], errors="coerce").dropna().values)
        if np.isfinite(sharpe):
            out.at[idx, "portfolio_sharpe_annualized"] = float(sharpe)
            if pd.isna(out.at[idx, "portfolio_sharpe_daily"]):
                out.at[idx, "portfolio_sharpe_daily"] = float(sharpe / np.sqrt(252.0))
            if pd.isna(out.at[idx, "portfolio_sharpe"]):
                out.at[idx, "portfolio_sharpe"] = float(sharpe / np.sqrt(252.0))
    return out


def _assert_no_nan_metrics(df: pd.DataFrame, metric_cols: List[str]) -> None:
    for col in metric_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required metric column '{col}' in report dataframe.")
        series = pd.to_numeric(df[col], errors="coerce")
        n_nan = int(series.isna().sum())
        if n_nan > 0:
            raise ValueError(f"Found {n_nan} NaN values in required metric column '{col}'.")


def _read_curve_csv(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.shape[1] < 2:
        return None
    date_col = df.columns[0]
    val_col = df.columns[1]
    out = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values, index=pd.to_datetime(df[date_col], errors="coerce"))
    out = out.dropna()
    out = out[~out.index.isna()]
    out = out.sort_index()
    if out.empty:
        return None
    return out


def _model_curve_path(row: pd.Series, freq: int) -> Path:
    out_dir_raw = row.get("out_dir", "")
    out_dir_str = _safe_str(out_dir_raw).strip()
    out_dir = Path(out_dir_str) if out_dir_str else Path("experiments")
    if out_dir == Path("experiments"):
        run_tag = _safe_str(row.get("run_tag", "")).strip()
        if run_tag:
            candidate = out_dir / run_tag
            if candidate.exists():
                out_dir = candidate
    prefix = _safe_str(row.get("artifact_prefix", "")).strip() or _safe_str(row.get("model_name", "")).strip()
    if not prefix:
        prefix = "model"
    use_discovery_fallback = out_dir == Path("experiments")
    direct = out_dir / f"{prefix}_equity_curve_reb{freq}.csv"
    if direct.exists():
        return direct
    # Backward-compatible fallback
    legacy = out_dir / f"{prefix}_equity_curve.csv"
    if legacy.exists():
        return legacy
    if use_discovery_fallback:
        discovered = _find_latest(f"experiments/**/{prefix}_equity_curve_reb{freq}.csv")
        if discovered is not None:
            return discovered
        discovered = _find_latest(f"experiments/**/{prefix}_equity_curve.csv")
        if discovered is not None:
            return discovered
    return direct


def _daily_metrics_path(row: pd.Series, freq: int) -> Path:
    out_dir_raw = row.get("out_dir", "")
    out_dir_str = _safe_str(out_dir_raw).strip()
    out_dir = Path(out_dir_str) if out_dir_str else Path("experiments")
    if out_dir == Path("experiments"):
        run_tag = _safe_str(row.get("run_tag", "")).strip()
        if run_tag:
            candidate = out_dir / run_tag
            if candidate.exists():
                out_dir = candidate
    prefix = _safe_str(row.get("artifact_prefix", "")).strip() or _safe_str(row.get("model_name", "")).strip()
    if not prefix:
        prefix = "model"
    use_discovery_fallback = out_dir == Path("experiments")
    direct = out_dir / f"{prefix}_daily_metrics_reb{freq}.csv"
    if direct.exists():
        return direct
    legacy = out_dir / f"{prefix}_daily_metrics.csv"
    if legacy.exists():
        return legacy
    if use_discovery_fallback:
        discovered = _find_latest(f"experiments/**/{prefix}_daily_metrics_reb{freq}.csv")
        if discovered is not None:
            return discovered
        discovered = _find_latest(f"experiments/**/{prefix}_daily_metrics.csv")
        if discovered is not None:
            return discovered
    return direct


def _prediction_path(row: pd.Series) -> Path:
    out_dir_raw = row.get("out_dir", "")
    out_dir_str = _safe_str(out_dir_raw).strip()
    out_dir = Path(out_dir_str) if out_dir_str else Path("experiments")
    if out_dir == Path("experiments"):
        run_tag = _safe_str(row.get("run_tag", "")).strip()
        if run_tag:
            candidate = out_dir / run_tag
            if candidate.exists():
                out_dir = candidate
    prefix = _safe_str(row.get("artifact_prefix", "")).strip() or _safe_str(row.get("model_name", "")).strip()
    if not prefix:
        prefix = "model"
    use_discovery_fallback = out_dir == Path("experiments")
    direct = out_dir / f"{prefix}_predictions.csv"
    if direct.exists():
        return direct
    if use_discovery_fallback:
        discovered = _find_latest(f"experiments/**/{prefix}_predictions.csv")
        if discovered is not None:
            return discovered
    return direct


def _read_prediction_df(row: pd.Series) -> Optional[pd.DataFrame]:
    p = _prediction_path(row)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    required = {"date", "ticker", "pred", "realized_ret"}
    if not required.issubset(df.columns):
        return None
    out = df[list(required)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str)
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    out["realized_ret"] = pd.to_numeric(out["realized_ret"], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "pred", "realized_ret"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    if out.empty:
        return None
    return out


def _select_price_col(df: pd.DataFrame) -> str:
    for c in ["adj_close", "close", "price", "last", "px_last"]:
        if c in df.columns:
            return c
    raise ValueError("No price column found for baseline computation.")


def _baseline_paths(row: pd.Series, freq: int) -> Tuple[Path, Path]:
    out_dir_raw = row.get("out_dir", "")
    out_dir_str = _safe_str(out_dir_raw).strip()
    out_dir = Path(out_dir_str) if out_dir_str else Path("experiments")
    if out_dir == Path("experiments"):
        run_tag = _safe_str(row.get("run_tag", "")).strip()
        if run_tag:
            candidate = out_dir / run_tag
            if candidate.exists():
                out_dir = candidate
    use_discovery_fallback = out_dir == Path("experiments")
    bh = out_dir / "buy_and_hold_equity_curve.csv"
    eqw = out_dir / f"equal_weight_equity_curve_reb{freq}.csv"
    if use_discovery_fallback and not bh.exists():
        found = _find_latest("experiments/**/buy_and_hold_equity_curve.csv")
        if found is not None:
            bh = found
    if use_discovery_fallback and not eqw.exists():
        found = _find_latest(f"experiments/**/equal_weight_equity_curve_reb{freq}.csv")
        if found is not None:
            eqw = found
    return bh, eqw


def _summary_path(row: pd.Series) -> Optional[Path]:
    out_dir_raw = row.get("out_dir", "")
    out_dir_str = _safe_str(out_dir_raw).strip()
    out_dir = Path(out_dir_str) if out_dir_str else Path("experiments")
    run_tag = _safe_str(row.get("run_tag", "")).strip()
    prefix = _safe_str(row.get("artifact_prefix", "")).strip()
    model_name = _safe_str(row.get("model_name", "")).strip()

    candidates = []
    if run_tag:
        candidates.append(out_dir / f"{run_tag}_summary.json")
    if prefix:
        candidates.append(out_dir / f"{prefix}_summary.json")
    if model_name:
        candidates.append(out_dir / f"{model_name}_summary.json")

    for c in candidates:
        if c.exists():
            return c

    patterns = []
    if run_tag:
        patterns.append(f"experiments/**/{run_tag}_summary.json")
    if prefix:
        patterns.append(f"experiments/**/{prefix}_summary.json")
    if model_name:
        patterns.append(f"experiments/**/{model_name}_summary.json")
    for p in patterns:
        found = _find_latest(p)
        if found is not None:
            return found
    return None


def _summary_payload_for_row(row: pd.Series) -> Dict[str, Any]:
    sp = _summary_path(row)
    if sp is None or not sp.exists():
        return {}
    try:
        payload = json.loads(sp.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _baseline_context_for_row(row: pd.Series) -> Dict[str, object]:
    out = {
        "global_start_date": "",
        "global_end_date": "",
        "global_final_value": np.nan,
        "test_start_date": "",
        "test_end_date": "",
        "test_final_value": np.nan,
        "test_rebased": True,
        "test_annualized_return": np.nan,
    }

    payload = _summary_payload_for_row(row)
    if payload:
        try:
            ctx = payload.get("baseline_context", {})
            gbh = ctx.get("global_buy_and_hold", {})
            tbh = ctx.get("test_window_buy_and_hold", {})
            bh_stats = payload.get("buy_and_hold_stats", {})
            out.update(
                {
                    "global_start_date": _safe_str(gbh.get("start_date", "")),
                    "global_end_date": _safe_str(gbh.get("end_date", "")),
                    "global_final_value": pd.to_numeric(gbh.get("final_value"), errors="coerce"),
                    "test_start_date": _safe_str(tbh.get("start_date", "")),
                    "test_end_date": _safe_str(tbh.get("end_date", "")),
                    "test_final_value": pd.to_numeric(tbh.get("final_value"), errors="coerce"),
                    "test_rebased": bool(tbh.get("rebased", True)),
                    "test_annualized_return": pd.to_numeric(bh_stats.get("annualized_return"), errors="coerce"),
                }
            )
        except Exception:
            pass

    # Fallback for legacy summaries without baseline_context.
    if not out["test_start_date"] or not np.isfinite(float(out["test_final_value"] if pd.notna(out["test_final_value"]) else np.nan)):
        freq = int(pd.to_numeric(row.get("rebalance_freq"), errors="coerce"))
        bh_path, _ = _baseline_paths(row, freq)
        bh = _read_curve_csv(bh_path)
        if bh is not None:
            out["test_start_date"] = str(bh.index.min().date())
            out["test_end_date"] = str(bh.index.max().date())
            out["test_final_value"] = float(bh.iloc[-1])
            if len(bh) > 1 and float(bh.iloc[0]) > 0:
                years = max((len(bh) - 1) / 252.0, 1e-9)
                out["test_annualized_return"] = float((float(bh.iloc[-1]) / float(bh.iloc[0])) ** (1.0 / years) - 1.0)

    if not out["global_start_date"] or not np.isfinite(float(out["global_final_value"] if pd.notna(out["global_final_value"]) else np.nan)):
        global_bh = _read_curve_csv(Path("data/processed/baselines/buy_and_hold_global.csv"))
        if global_bh is not None:
            out["global_start_date"] = str(global_bh.index.min().date())
            out["global_end_date"] = str(global_bh.index.max().date())
            out["global_final_value"] = float(global_bh.iloc[-1])

    return out


def _stats_from_curve(curve: Optional[pd.Series]) -> Dict[str, float]:
    out = {
        "portfolio_final_value": float("nan"),
        "portfolio_annualized_return": float("nan"),
        "portfolio_max_drawdown": float("nan"),
        "portfolio_sharpe_annualized": float("nan"),
    }
    if curve is None or curve.empty:
        return out

    s = curve.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if s.empty:
        return out

    out["portfolio_final_value"] = float(s.iloc[-1])
    dd = s / s.cummax() - 1.0
    out["portfolio_max_drawdown"] = float(dd.min()) if not dd.empty else float("nan")

    if len(s) > 1 and float(s.iloc[0]) > 0.0:
        daily_ret = s.pct_change().dropna()
        years = max((len(s) - 1) / 252.0, 1e-9)
        out["portfolio_annualized_return"] = float((float(s.iloc[-1]) / float(s.iloc[0])) ** (1.0 / years) - 1.0)
        sharpe = annualized_sharpe_from_returns(daily_ret.values)
        if np.isfinite(sharpe):
            out["portfolio_sharpe_annualized"] = float(sharpe)

    return out


def _extract_baseline_stats(payload: Dict[str, Any], freq: int, baseline_key: str) -> Dict[str, float]:
    stats_by = payload.get("stats_by_rebalance_freq", {})
    by_freq = stats_by.get(str(int(freq)), {}) if isinstance(stats_by, dict) else {}
    candidate = {}
    if isinstance(by_freq, dict):
        candidate = by_freq.get(baseline_key, {}) or {}
    if not candidate and baseline_key in payload:
        candidate = payload.get(baseline_key, {}) or {}
    if not isinstance(candidate, dict):
        candidate = {}
    return {
        "portfolio_final_value": pd.to_numeric(candidate.get("final_value"), errors="coerce"),
        "portfolio_annualized_return": pd.to_numeric(candidate.get("annualized_return"), errors="coerce"),
        "portfolio_max_drawdown": pd.to_numeric(candidate.get("max_drawdown"), errors="coerce"),
        "portfolio_sharpe_annualized": pd.to_numeric(candidate.get("sharpe_annualized"), errors="coerce"),
    }


def _build_baseline_policy_comparison(latest_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "strategy_name",
        "strategy_kind",
        "rebalance_freq",
        "rebalance_label",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_final_value",
        "test_start_date",
        "test_end_date",
        "source_run_tag",
    ]
    if latest_df.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, object]] = []
    freqs = sorted(pd.to_numeric(latest_df["rebalance_freq"], errors="coerce").dropna().astype(int).unique().tolist())
    for freq in freqs:
        subset = latest_df[latest_df["rebalance_freq"] == int(freq)].copy()
        if subset.empty:
            continue
        row = subset.sort_values("portfolio_sharpe_annualized", ascending=False).iloc[0]
        payload = _summary_payload_for_row(row)
        ctx = _baseline_context_for_row(row)
        bh_path, eqw_path = _baseline_paths(row, int(freq))
        bh_curve = _read_curve_csv(bh_path)
        eqw_curve = _read_curve_csv(eqw_path)

        bh_stats = _extract_baseline_stats(payload, int(freq), "buy_and_hold_stats")
        eqw_stats = _extract_baseline_stats(payload, int(freq), "equal_weight_stats")
        bh_fallback = _stats_from_curve(bh_curve)
        eqw_fallback = _stats_from_curve(eqw_curve)
        for k, v in bh_fallback.items():
            if pd.isna(bh_stats.get(k)):
                bh_stats[k] = v
        for k, v in eqw_fallback.items():
            if pd.isna(eqw_stats.get(k)):
                eqw_stats[k] = v

        common = {
            "rebalance_freq": int(freq),
            "rebalance_label": _reb_label(int(freq)),
            "test_start_date": ctx.get("test_start_date", ""),
            "test_end_date": ctx.get("test_end_date", ""),
            "source_run_tag": _safe_str(row.get("run_tag", "")),
        }
        rows.append(
            {
                "strategy_name": BUY_HOLD_LABEL,
                "strategy_kind": "baseline_buy_and_hold",
                **common,
                **bh_stats,
            }
        )
        rows.append(
            {
                "strategy_name": EQW_LABEL,
                "strategy_kind": "baseline_equal_weight",
                **common,
                **eqw_stats,
            }
        )

    out = pd.DataFrame(rows)
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols].copy()
    out = out.sort_values(["rebalance_freq", "strategy_kind"]).reset_index(drop=True)
    return out


def _write_baseline_policy_comparison_csv(latest_df: pd.DataFrame, out_dir: Path) -> None:
    baseline_df = _build_baseline_policy_comparison(latest_df)
    baseline_df.to_csv(out_dir / "baseline_policy_comparison.csv", index=False)


def _audit_equal_weight_rebalance(latest_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    series_by_freq: Dict[int, pd.Series] = {}
    path_by_freq: Dict[int, str] = {}
    diag_by_freq: Dict[int, Dict[str, object]] = {}
    for freq in sorted(pd.to_numeric(latest_df["rebalance_freq"], errors="coerce").dropna().astype(int).unique().tolist()):
        subset = latest_df[latest_df["rebalance_freq"] == int(freq)].copy()
        if subset.empty:
            continue
        row = subset.sort_values("portfolio_sharpe_annualized", ascending=False).iloc[0]
        _, eqw_path = _baseline_paths(row, int(freq))
        eqw_curve = _read_curve_csv(eqw_path)
        path_by_freq[int(freq)] = str(eqw_path)
        if eqw_curve is not None:
            series_by_freq[int(freq)] = eqw_curve.copy()
            rows.append(
                {
                    "rebalance_freq": int(freq),
                    "source_run_tag": _safe_str(row.get("run_tag", "")),
                    "eqw_path": str(eqw_path),
                    "start_date": str(eqw_curve.index.min().date()),
                    "end_date": str(eqw_curve.index.max().date()),
                    "n_points": int(len(eqw_curve)),
                    "final_value": float(eqw_curve.iloc[-1]),
                    "status": "ok",
                    "detail": "",
                }
            )
        else:
            rows.append(
                {
                    "rebalance_freq": int(freq),
                    "source_run_tag": _safe_str(row.get("run_tag", "")),
                    "eqw_path": str(eqw_path),
                    "start_date": "",
                    "end_date": "",
                    "n_points": 0,
                    "final_value": np.nan,
                    "status": "missing",
                    "detail": "equal-weight curve not found",
                }
            )

    # Enrich diagnostics from canonical baseline logic using the same window.
    try:
        cfg = load_config("configs/runs/core/xgb_raw.yaml", REPO_ROOT)
        price_df = load_price_panel(
            cfg["data"]["price_file"],
            cfg["data"]["start_date"],
            cfg["data"]["end_date"],
        )
        price_col = _select_price_col(price_df)
        universe = None
        universe_file = cfg["data"].get("universe_file")
        if universe_file and Path(universe_file).exists():
            u = pd.read_csv(universe_file)
            if "ticker" in u.columns:
                universe = u["ticker"].dropna().unique().tolist()

        all_dates: List[pd.Timestamp] = []
        for s in series_by_freq.values():
            if s is not None and not s.empty:
                all_dates.extend(pd.to_datetime(s.index).tolist())
        if all_dates:
            start_date = str(min(all_dates).date())
            end_date = str(max(all_dates).date())
            for freq in sorted(series_by_freq.keys()):
                _, _, _, diag = compute_equal_weight_rebalanced(
                    price_df,
                    price_col=price_col,
                    start_date=start_date,
                    end_date=end_date,
                    universe=universe,
                    initial_value=1.0,
                    rebalance_freq=int(freq),
                    risk_free_rate=0.0,
                    return_diagnostics=True,
                )
                diag_by_freq[int(freq)] = diag
    except Exception:
        diag_by_freq = {}

    first_curve_diff_date = ""
    if 1 in series_by_freq and 5 in series_by_freq:
        s1 = series_by_freq[1].copy()
        s5 = series_by_freq[5].copy()
        for d in s1.index.intersection(s5.index):
            if not np.isclose(float(s1.loc[d]), float(s5.loc[d]), rtol=1e-12, atol=1e-12):
                first_curve_diff_date = str(pd.to_datetime(d).date())
                break

    first_weights_differ_date = ""
    if 1 in diag_by_freq and 5 in diag_by_freq:
        w1 = diag_by_freq[1].get("close_weights")
        w5 = diag_by_freq[5].get("close_weights")
        if isinstance(w1, pd.DataFrame) and isinstance(w5, pd.DataFrame):
            idx = w1.index.intersection(w5.index)
            cols = w1.columns.intersection(w5.columns)
            for d in idx:
                a = pd.to_numeric(w1.loc[d, cols], errors="coerce").fillna(0.0).values
                b = pd.to_numeric(w5.loc[d, cols], errors="coerce").fillna(0.0).values
                if not np.allclose(a, b, rtol=1e-12, atol=1e-12):
                    first_weights_differ_date = str(pd.to_datetime(d).date())
                    break

    for r in rows:
        freq = int(pd.to_numeric(r.get("rebalance_freq"), errors="coerce"))
        if freq in diag_by_freq:
            d = diag_by_freq[freq]
            r["rebalance_dates_count"] = int(d.get("rebalance_count", 0))
            r["turnover_mean"] = float(d.get("turnover_mean", np.nan))
            r["turnover_median"] = float(d.get("turnover_median", np.nan))
            r["turnover_max"] = float(d.get("turnover_max", np.nan))
        else:
            r["rebalance_dates_count"] = np.nan
            r["turnover_mean"] = np.nan
            r["turnover_median"] = np.nan
            r["turnover_max"] = np.nan
        r["first_curve_diff_date_reb1_vs_reb5"] = first_curve_diff_date
        r["first_weights_diff_date_reb1_vs_reb5"] = first_weights_differ_date

    # Hard gate for reb=1 vs reb=5 identical series.
    if 1 in series_by_freq and 5 in series_by_freq:
        s1 = series_by_freq[1].copy()
        s5 = series_by_freq[5].copy()
        identical = bool(s1.index.equals(s5.index) and np.allclose(s1.values, s5.values, rtol=1e-12, atol=1e-12))
        if identical and len(s1) > 3:
            reb1_count = int(diag_by_freq.get(1, {}).get("rebalance_count", 0))
            reb5_count = int(diag_by_freq.get(5, {}).get("rebalance_count", 0))
            reb1_turnover = float(diag_by_freq.get(1, {}).get("turnover_mean", np.nan))
            reb5_turnover = float(diag_by_freq.get(5, {}).get("turnover_mean", np.nan))
            detail = (
                "EQW rebalance integrity failed: reb=1 and reb=5 equal-weight series are identical. "
                f"reb1_path={path_by_freq.get(1, '')}, reb5_path={path_by_freq.get(5, '')}, "
                f"reb1_rebalance_dates={reb1_count}, reb5_rebalance_dates={reb5_count}, "
                f"reb1_turnover_mean={reb1_turnover:.6f}, reb5_turnover_mean={reb5_turnover:.6f}, "
                f"first_weights_diff_date={first_weights_differ_date or 'none'}"
            )
            rows.append(
                {
                    "rebalance_freq": -1,
                    "source_run_tag": "",
                    "eqw_path": "",
                    "start_date": "",
                    "end_date": "",
                    "n_points": int(len(s1)),
                    "final_value": float(s1.iloc[-1]),
                    "status": "fail",
                    "detail": detail,
                }
            )
            audit_df = pd.DataFrame(rows)
            audit_df.to_csv(out_dir / "equal_weight_rebalance_audit.csv", index=False)
            raise ValueError(detail)

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(out_dir / "equal_weight_rebalance_audit.csv", index=False)
    return audit_df


def _parse_graph_window_end(graph_window: object) -> Optional[pd.Timestamp]:
    text = _safe_str(graph_window).strip()
    if not text:
        return None
    parts = text.split("..")
    if len(parts) != 2:
        return None
    end_ts = pd.to_datetime(parts[1], errors="coerce")
    if pd.isna(end_ts):
        return None
    return end_ts


def _audit_graph_time_awareness(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for _, row in master.iterrows():
        run_tag = _safe_str(row.get("run_tag", ""))
        model = _safe_str(row.get("model_name", ""))
        edge = _safe_str(row.get("edge_type", "")).strip().lower()
        graph_window = _safe_str(row.get("graph_window", ""))
        split_train_end = pd.to_datetime(row.get("split_train_end"), errors="coerce")
        window_end = _parse_graph_window_end(graph_window)
        pred_df = _read_prediction_df(row)
        max_pred_ts = None
        if pred_df is not None and not pred_df.empty:
            max_pred_ts = pd.to_datetime(pred_df["date"], errors="coerce").dropna().max()

        is_graph_model = model in {"gcn", "gat", "tgcn_static", "tgat_static", "xgb_node2vec", "graphlasso_linear", "granger_xgb", "graphlasso_xgb"} or edge not in {"", "none", "nan"}
        status = "pass"
        detail = "non-graph model"
        mode = "n/a"
        if is_graph_model:
            if window_end is None:
                status = "fail"
                detail = "graph model missing graph_window end date metadata"
            elif pd.isna(split_train_end):
                status = "fail"
                detail = "split_train_end missing"
            else:
                if pd.to_datetime(window_end) <= pd.to_datetime(split_train_end):
                    mode = "frozen_train"
                    detail = f"window_end={window_end.date()} <= split_train_end={split_train_end.date()}"
                elif max_pred_ts is not None and pd.to_datetime(window_end) <= pd.to_datetime(max_pred_ts):
                    mode = "rolling_prediction_time"
                    detail = f"window_end={window_end.date()} <= max_prediction_date={pd.to_datetime(max_pred_ts).date()}"
                else:
                    mode = "invalid"
                    status = "fail"
                    if max_pred_ts is None:
                        detail = (
                            f"window_end={window_end.date()} exceeds split_train_end={split_train_end.date()} "
                            "and no prediction dates available for rolling-time validation"
                        )
                    else:
                        detail = (
                            f"window_end={window_end.date()} exceeds both split_train_end={split_train_end.date()} "
                            f"and max_prediction_date={pd.to_datetime(max_pred_ts).date()}"
                        )

        if status == "fail":
            failures.append(f"{run_tag}: {detail}")
        rows.append(
            {
                "run_tag": run_tag,
                "model_name": model,
                "edge_type": edge,
                "graph_window": graph_window,
                "max_timestamp_used": _safe_str(window_end.date() if window_end is not None else ""),
                "split_train_end": _safe_str(row.get("split_train_end", "")),
                "max_prediction_date": _safe_str(pd.to_datetime(max_pred_ts).date() if max_pred_ts is not None else ""),
                "audit_mode": mode,
                "status": status,
                "detail": detail,
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(out_dir / "graph_time_awareness_audit.csv", index=False)
    if failures:
        raise ValueError("Graph time-awareness audit failed:\n - " + "\n - ".join(failures[:20]))
    return audit_df


def _annualized_tracking_error(active_returns: pd.Series) -> float:
    if active_returns.empty:
        return float("nan")
    std = float(active_returns.std())
    if not np.isfinite(std):
        return float("nan")
    return float(std * np.sqrt(ACTIVE_TRADING_DAYS))


def _information_ratio(active_returns: pd.Series) -> float:
    if active_returns.empty:
        return float("nan")
    mean = float(active_returns.mean())
    std = float(active_returns.std())
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float((mean / std) * np.sqrt(ACTIVE_TRADING_DAYS))


def _compute_active_metrics(model_curve: pd.Series, eqw_curve: pd.Series) -> Dict[str, object]:
    m = model_curve.copy().sort_index()
    e = eqw_curve.copy().sort_index()
    idx = m.index.intersection(e.index)
    if idx.empty:
        return {}
    m = m.loc[idx]
    e = e.loc[idx]
    rel = (m / e).dropna()
    if rel.empty:
        return {}

    m_ret = m.pct_change().dropna()
    e_ret = e.pct_change().dropna()
    ridx = m_ret.index.intersection(e_ret.index)
    if ridx.empty:
        return {}
    active = (m_ret.loc[ridx] - e_ret.loc[ridx]).dropna()
    if active.empty:
        return {}

    return {
        "relative_curve": rel,
        "active_returns": active,
        "active_ann_return": float(active.mean() * ACTIVE_TRADING_DAYS),
        "tracking_error_ann": _annualized_tracking_error(active),
        "information_ratio": _information_ratio(active),
        "relative_final_value": float(rel.iloc[-1]),
        "relative_max_drawdown": float((rel / rel.cummax() - 1.0).min()),
    }


def _write_alpha_vs_equal_weight(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [
        "run_tag",
        "model_name",
        "edge_type",
        "rebalance_freq",
        "rebalance_label",
        "active_return_annualized",
        "tracking_error_annualized",
        "information_ratio_annualized",
        "relative_final_value",
        "relative_max_drawdown",
        "active_return_mean_daily_arithmetic",
        "active_return_type",
        "annualization_days",
    ]
    rows: List[Dict[str, object]] = []
    rel_rows: List[Dict[str, object]] = []
    for _, row in master.iterrows():
        freq = int(pd.to_numeric(row.get("rebalance_freq"), errors="coerce"))
        model_curve = _read_curve_csv(_model_curve_path(row, freq))
        _, eqw_path = _baseline_paths(row, freq)
        eqw_curve = _read_curve_csv(eqw_path)
        if model_curve is None or eqw_curve is None:
            continue
        payload = _compute_active_metrics(model_curve, eqw_curve)
        if not payload:
            continue
        rel_curve = payload.pop("relative_curve")
        active = payload.pop("active_returns")
        rows.append(
            {
                "run_tag": _safe_str(row.get("run_tag", "")),
                "model_name": _safe_str(row.get("model_name", "")),
                "edge_type": _safe_str(row.get("edge_type", "")),
                "rebalance_freq": freq,
                "rebalance_label": _reb_label(freq),
                "active_return_annualized": payload["active_ann_return"],
                "tracking_error_annualized": payload["tracking_error_ann"],
                "information_ratio_annualized": payload["information_ratio"],
                "relative_final_value": payload["relative_final_value"],
                "relative_max_drawdown": payload["relative_max_drawdown"],
                "active_return_mean_daily_arithmetic": float(active.mean()),
                "active_return_type": "arithmetic",
                "annualization_days": int(ACTIVE_TRADING_DAYS),
            }
        )
        rel_rows.extend(
            [
                {
                    "run_tag": _safe_str(row.get("run_tag", "")),
                    "rebalance_freq": freq,
                    "date": d,
                    "relative_wealth": float(v),
                }
                for d, v in rel_curve.items()
            ]
        )

        # Hard check: relative wealth is pointwise Model / EQW.
        if not np.allclose(
            rel_curve.values,
            (model_curve.loc[rel_curve.index] / eqw_curve.loc[rel_curve.index]).values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ):
            raise ValueError(f"Relative wealth check failed for run_tag={_safe_str(row.get('run_tag', ''))}, reb={freq}")

    alpha_df = pd.DataFrame(rows, columns=cols)
    if not alpha_df.empty:
        alpha_df = alpha_df.sort_values(["rebalance_freq", "information_ratio_annualized"], ascending=[True, False])
    alpha_df.to_csv(out_dir / "alpha_vs_equal_weight.csv", index=False)
    rel_df = pd.DataFrame(rel_rows, columns=["run_tag", "rebalance_freq", "date", "relative_wealth"])
    rel_df.to_csv(out_dir / "relative_wealth_vs_equal_weight.csv", index=False)

    if not alpha_df.empty:
        plt.figure(figsize=(10, 7))
        d = alpha_df.dropna(subset=["information_ratio_annualized", "active_return_annualized"]).copy()
        if not d.empty:
            plt.scatter(d["active_return_annualized"], d["information_ratio_annualized"], alpha=0.8, s=55)
            for _, r in d.iterrows():
                plt.text(float(r["active_return_annualized"]), float(r["information_ratio_annualized"]), str(r["run_tag"]), fontsize=7)
            plt.xlabel("Active Return (Annualized, Arithmetic)")
            plt.ylabel("Information Ratio (Annualized)")
            plt.title("Active Return vs Information Ratio (vs Equal Weight)")
            plt.grid(alpha=0.3, linestyle="--")
            plt.tight_layout()
            plt.savefig(out_dir / "active_ir_vs_active_return.png", dpi=220)
            plt.close()

    if not rel_df.empty:
        plt.figure(figsize=(12, 6.5))
        keep = (
            alpha_df.sort_values(["rebalance_freq", "information_ratio_annualized"], ascending=[True, False])
            .groupby("rebalance_freq", as_index=False)
            .head(3)
        )
        for _, r in keep.iterrows():
            sub = rel_df[(rel_df["run_tag"] == r["run_tag"]) & (rel_df["rebalance_freq"] == r["rebalance_freq"])].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("date")
            plt.plot(pd.to_datetime(sub["date"]), sub["relative_wealth"], label=f"{r['run_tag']} (reb={int(r['rebalance_freq'])})")
        plt.axhline(1.0, linestyle="--", linewidth=1.2, color="black")
        plt.ylabel("Relative Wealth (Model / Equal Weight)")
        plt.xlabel("Date")
        plt.title("Relative Wealth vs Equal Weight")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(fontsize=8, frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "relative_wealth_key_models.png", dpi=220)
        plt.close()
    return alpha_df


def _cost_label(bps: int) -> str:
    return "gross (0 bps)" if int(bps) == 0 else f"net ({int(bps)} bps)"


def _write_cost_sensitivity_long_only(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [
        "run_tag",
        "model_name",
        "edge_type",
        "rebalance_freq",
        "rebalance_label",
        "cost_bps",
        "cost_label",
        "strategy_label",
        "portfolio_final_value",
        "portfolio_annualized_return",
        "portfolio_sharpe_annualized",
        "portfolio_max_drawdown",
        "portfolio_turnover",
        "n_points",
        "turnover_definition",
        "cost_formula",
    ]
    rows: List[Dict[str, object]] = []
    pred_cache: Dict[str, Optional[pd.DataFrame]] = {}
    for _, row in master.iterrows():
        key = f"{_safe_str(row.get('run_tag',''))}|{_safe_str(row.get('artifact_prefix',''))}|{_safe_str(row.get('out_dir',''))}"
        if key not in pred_cache:
            pred_cache[key] = _read_prediction_df(row)
        pred_df = pred_cache[key]
        if pred_df is None or pred_df.empty:
            continue
        freq = int(pd.to_numeric(row.get("rebalance_freq"), errors="coerce"))
        for bps in COST_SWEEP_BPS:
            eq, _, stats = backtest_long_only(
                pred_df,
                top_k=20,
                transaction_cost_bps=float(bps),
                risk_free_rate=0.0,
                rebalance_freq=freq,
            )
            rows.append(
                {
                    "run_tag": _safe_str(row.get("run_tag", "")),
                    "model_name": _safe_str(row.get("model_name", "")),
                    "edge_type": _safe_str(row.get("edge_type", "")),
                    "rebalance_freq": freq,
                    "rebalance_label": _reb_label(freq),
                    "cost_bps": int(bps),
                    "cost_label": _cost_label(int(bps)),
                    "strategy_label": TOPK_LABEL,
                    "portfolio_final_value": float(stats.get("final_value", np.nan)),
                    "portfolio_annualized_return": float(stats.get("annualized_return", np.nan)),
                    "portfolio_sharpe_annualized": float(stats.get("sharpe_annualized", np.nan)),
                    "portfolio_max_drawdown": float(stats.get("max_drawdown", np.nan)),
                    "portfolio_turnover": float(stats.get("avg_turnover", np.nan)),
                    "n_points": int(len(eq)),
                    "turnover_definition": "sum(abs(w_t - w_{t-1}))",
                    "cost_formula": "cost_t=(bps/10000)*turnover_t on rebalance dates",
                }
            )
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(out_dir / "cost_sensitivity_long_only.csv", index=False)
    return out


def _write_long_short_tables(master: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_all: List[Dict[str, object]] = []
    pred_rows = master[master["rebalance_freq"] == 5].copy()
    if pred_rows.empty:
        pred_rows = master.copy()
    pred_rows = pred_rows.drop_duplicates(subset=["run_tag"], keep="first")
    for _, row in pred_rows.iterrows():
        pred_df = _read_prediction_df(row)
        if pred_df is None or pred_df.empty:
            continue
        for reb in [5, 21]:
            for bps in COST_SWEEP_BPS:
                eq, _, stats = backtest_long_short(
                    pred_df,
                    top_k=3,
                    bottom_k=3,
                    transaction_cost_bps=float(bps),
                    risk_free_rate=0.0,
                    rebalance_freq=int(reb),
                    long_leg_gross=0.5,
                    short_leg_gross=0.5,
                )
                rows_all.append(
                    {
                        "run_tag": _safe_str(row.get("run_tag", "")),
                        "model_name": _safe_str(row.get("model_name", "")),
                        "edge_type": _safe_str(row.get("edge_type", "")),
                        "rebalance_freq": int(reb),
                        "rebalance_label": _reb_label(int(reb)),
                        "cost_bps": int(bps),
                        "cost_label": _cost_label(int(bps)),
                        "strategy_label": LONG_SHORT_LABEL,
                        "long_k": 3,
                        "short_k": 3,
                        "long_leg_gross": 0.5,
                        "short_leg_gross": 0.5,
                        "portfolio_final_value": float(stats.get("final_value", np.nan)),
                        "portfolio_annualized_return": float(stats.get("annualized_return", np.nan)),
                        "portfolio_sharpe_annualized": float(stats.get("sharpe_annualized", np.nan)),
                        "portfolio_max_drawdown": float(stats.get("max_drawdown", np.nan)),
                        "portfolio_turnover": float(stats.get("avg_turnover", np.nan)),
                        "n_points": int(len(eq)),
                        "turnover_definition": "sum(abs(w_t - w_{t-1}))",
                        "cost_formula": "cost_t=(bps/10000)*turnover_t on rebalance dates",
                    }
                )

    cols = [
        "run_tag",
        "model_name",
        "edge_type",
        "rebalance_freq",
        "rebalance_label",
        "cost_bps",
        "cost_label",
        "strategy_label",
        "long_k",
        "short_k",
        "long_leg_gross",
        "short_leg_gross",
        "portfolio_final_value",
        "portfolio_annualized_return",
        "portfolio_sharpe_annualized",
        "portfolio_max_drawdown",
        "portfolio_turnover",
        "n_points",
        "turnover_definition",
        "cost_formula",
        "long_sum_target",
        "short_sum_target",
    ]
    all_df = pd.DataFrame(rows_all, columns=cols)
    all_df.to_csv(out_dir / "cost_sensitivity_long_short.csv", index=False)
    base_df = all_df[all_df["cost_bps"] == 0].copy() if "cost_bps" in all_df.columns else pd.DataFrame(columns=cols)
    base_df.to_csv(out_dir / "long_short_top3_bottom3.csv", index=False)

    if not base_df.empty:
        plt.figure(figsize=(12, 6.5))
        top = base_df[base_df["rebalance_freq"] == 5].sort_values("portfolio_sharpe_annualized", ascending=False).head(5)
        for _, r in top.iterrows():
            pred_row = pred_rows[pred_rows["run_tag"] == r["run_tag"]].iloc[0]
            pred_df = _read_prediction_df(pred_row)
            if pred_df is None:
                continue
            eq, _, _ = backtest_long_short(
                pred_df,
                top_k=3,
                bottom_k=3,
                transaction_cost_bps=0.0,
                risk_free_rate=0.0,
                rebalance_freq=5,
                long_leg_gross=0.5,
                short_leg_gross=0.5,
            )
            plt.plot(eq.index, eq.values, label=str(r["run_tag"]))
        plt.title(f"{LONG_SHORT_LABEL} equity curves (reb=5, gross)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(fontsize=8, frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / "long_short_equity_top3_bottom3.png", dpi=220)
        plt.close()
    return base_df, all_df


def _write_monthly_rebalance_subset(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    selected = {
        "xgb_node2vec_corr_tuned_all",
        "xgb_raw_tuned_all",
        "gat_corr_sector_granger_tuned_all",
    }
    rows: List[Dict[str, object]] = []
    subset = master[master["run_tag"].isin(selected)].copy()
    subset = subset.drop_duplicates(subset=["run_tag"], keep="first")
    for _, row in subset.iterrows():
        pred_df = _read_prediction_df(row)
        if pred_df is None or pred_df.empty:
            continue
        for reb in [1, 5, 21]:
            # Reuse canonical run metrics when available; only recompute missing policies.
            stats = None
            n_points = 0
            canonical = master[
                (master["run_tag"] == _safe_str(row.get("run_tag", "")))
                & (pd.to_numeric(master["rebalance_freq"], errors="coerce") == int(reb))
            ]
            if not canonical.empty:
                c = canonical.sort_values("timestamp").iloc[-1]
                stats = {
                    "final_value": float(pd.to_numeric(c.get("portfolio_final_value"), errors="coerce")),
                    "annualized_return": float(pd.to_numeric(c.get("portfolio_annualized_return"), errors="coerce")),
                    "sharpe_annualized": float(pd.to_numeric(c.get("portfolio_sharpe_annualized"), errors="coerce")),
                    "max_drawdown": float(pd.to_numeric(c.get("portfolio_max_drawdown"), errors="coerce")),
                    "avg_turnover": float(pd.to_numeric(c.get("portfolio_turnover"), errors="coerce")),
                }
                n_points = int(pd.to_datetime(pred_df["date"], errors="coerce").dropna().nunique())
            else:
                eq, _, stats = backtest_long_only(
                    pred_df,
                    top_k=20,
                    transaction_cost_bps=0.0,
                    risk_free_rate=0.0,
                    rebalance_freq=int(reb),
                )
                n_points = int(len(eq))
            rows.append(
                {
                    "strategy_name": _safe_str(row.get("run_tag", "")),
                    "strategy_kind": "model_topk_long_only",
                    "strategy_label": TOPK_LABEL,
                    "rebalance_freq": int(reb),
                    "rebalance_label": _reb_label(int(reb)),
                    "cost_bps": 0,
                    "cost_label": _cost_label(0),
                    "portfolio_final_value": float(stats.get("final_value", np.nan)),
                    "portfolio_annualized_return": float(stats.get("annualized_return", np.nan)),
                    "portfolio_sharpe_annualized": float(stats.get("sharpe_annualized", np.nan)),
                    "portfolio_max_drawdown": float(stats.get("max_drawdown", np.nan)),
                    "portfolio_turnover": float(stats.get("avg_turnover", np.nan)),
                    "n_points": n_points,
                    "turnover_definition": "sum(abs(w_t - w_{t-1}))",
                    "cost_formula": "cost_t=(bps/10000)*turnover_t on rebalance dates",
                    "long_sum_target": 0.5,
                    "short_sum_target": -0.5,
                }
            )

    try:
        cfg = load_config("configs/runs/core/xgb_raw.yaml", REPO_ROOT)
        price_df = load_price_panel(
            cfg["data"]["price_file"],
            cfg["data"]["start_date"],
            cfg["data"]["end_date"],
        )
        # Align baseline subset to the same prediction horizon window used by selected models.
        pred_window_start = None
        pred_window_end = None
        for _, row in subset.iterrows():
            p = _read_prediction_df(row)
            if p is None or p.empty:
                continue
            p0 = pd.to_datetime(p["date"], errors="coerce").dropna()
            if p0.empty:
                continue
            s = p0.min()
            e = p0.max()
            pred_window_start = s if pred_window_start is None else min(pred_window_start, s)
            pred_window_end = e if pred_window_end is None else max(pred_window_end, e)
        window_start = str(pred_window_start.date()) if pred_window_start is not None else cfg["training"]["test_start"]
        window_end = str(pred_window_end.date()) if pred_window_end is not None else cfg["data"]["end_date"]

        price_col = _select_price_col(price_df)
        universe = None
        universe_file = cfg["data"].get("universe_file")
        if universe_file and Path(universe_file).exists():
            u = pd.read_csv(universe_file)
            if "ticker" in u.columns:
                universe = u["ticker"].dropna().unique().tolist()
        bh_eq, _, bh_stats = compute_buy_and_hold_fixed_shares(
            price_df,
            price_col=price_col,
            start_date=window_start,
            end_date=window_end,
            universe=universe,
            initial_value=1.0,
            risk_free_rate=0.0,
        )
        for reb in [1, 5, 21]:
            eqw_eq, _, eqw_stats = compute_equal_weight_rebalanced(
                price_df,
                price_col=price_col,
                start_date=window_start,
                end_date=window_end,
                universe=universe,
                initial_value=1.0,
                rebalance_freq=int(reb),
                risk_free_rate=0.0,
            )
            rows.append(
                {
                    "strategy_name": EQW_LABEL,
                    "strategy_kind": "baseline_equal_weight",
                    "strategy_label": EQW_LABEL,
                    "rebalance_freq": int(reb),
                    "rebalance_label": _reb_label(int(reb)),
                    "cost_bps": 0,
                    "cost_label": _cost_label(0),
                    "portfolio_final_value": float(eqw_stats.get("final_value", np.nan)),
                    "portfolio_annualized_return": float(eqw_stats.get("annualized_return", np.nan)),
                    "portfolio_sharpe_annualized": float(eqw_stats.get("sharpe_annualized", np.nan)),
                    "portfolio_max_drawdown": float(eqw_stats.get("max_drawdown", np.nan)),
                    "portfolio_turnover": float(eqw_stats.get("avg_turnover", np.nan)),
                    "n_points": int(len(eqw_eq)),
                }
            )
            rows.append(
                {
                    "strategy_name": BUY_HOLD_LABEL,
                    "strategy_kind": "baseline_buy_and_hold",
                    "strategy_label": BUY_HOLD_LABEL,
                    "rebalance_freq": int(reb),
                    "rebalance_label": _reb_label(int(reb)),
                    "cost_bps": 0,
                    "cost_label": _cost_label(0),
                    "portfolio_final_value": float(bh_stats.get("final_value", np.nan)),
                    "portfolio_annualized_return": float(bh_stats.get("annualized_return", np.nan)),
                    "portfolio_sharpe_annualized": float(bh_stats.get("sharpe_annualized", np.nan)),
                    "portfolio_max_drawdown": float(bh_stats.get("max_drawdown", np.nan)),
                    "portfolio_turnover": 0.0,
                    "n_points": int(len(bh_eq)),
                }
            )
    except Exception as exc:
        rows.append(
            {
                "strategy_name": "baseline_computation_error",
                "strategy_kind": "diagnostic",
                "rebalance_freq": -1,
                "cost_bps": 0,
                "cost_label": _cost_label(0),
                "portfolio_final_value": np.nan,
                "portfolio_annualized_return": np.nan,
                "portfolio_sharpe_annualized": np.nan,
                "portfolio_max_drawdown": np.nan,
                "portfolio_turnover": np.nan,
                "n_points": 0,
                "detail": str(exc),
            }
        )
    cols = [
        "strategy_name",
        "strategy_kind",
        "strategy_label",
        "rebalance_freq",
        "rebalance_label",
        "cost_bps",
        "cost_label",
        "portfolio_final_value",
        "portfolio_annualized_return",
        "portfolio_sharpe_annualized",
        "portfolio_max_drawdown",
        "portfolio_turnover",
        "n_points",
        "detail",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(out_dir / "monthly_rebalance_subset.csv", index=False)
    return out


def _write_lookback_sensitivity_subset(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    targets = [
        ("xgb_node2vec", "node2vec_correlation", "xgb_node2vec_corr"),
        ("xgb_raw", "none", "xgb_raw"),
        ("gat", "corr+sector+granger", "gat_corr_sector_granger"),
    ]
    lookbacks = [14, 30, 60]
    rows: List[Dict[str, object]] = []
    m = master.copy()
    m["lookback_window"] = pd.to_numeric(m.get("lookback_window", pd.Series(dtype=float)), errors="coerce")
    m["rebalance_freq"] = pd.to_numeric(m.get("rebalance_freq", pd.Series(dtype=float)), errors="coerce")
    for model_name, edge_type, run_prefix in targets:
        df = m[
            (m["model_name"].astype(str) == model_name)
            & (m["edge_type"].astype(str) == edge_type)
            & (m["rebalance_freq"] == 5)
        ].copy()
        for lb in lookbacks:
            sub = df[df["lookback_window"] == int(lb)].copy()
            if sub.empty:
                rows.append(
                    {
                        "run_prefix": run_prefix,
                        "model_name": model_name,
                        "edge_type": edge_type,
                        "rebalance_freq": 5,
                        "lookback_window": int(lb),
                        "status": "missing_rerun_required",
                        "portfolio_sharpe_annualized": np.nan,
                        "portfolio_annualized_return": np.nan,
                        "portfolio_max_drawdown": np.nan,
                        "portfolio_final_value": np.nan,
                    }
                )
                continue
            row = sub.sort_values("portfolio_sharpe_annualized", ascending=False).iloc[0]
            rows.append(
                {
                    "run_prefix": run_prefix,
                    "model_name": model_name,
                    "edge_type": edge_type,
                    "rebalance_freq": 5,
                    "lookback_window": int(lb),
                    "status": "available",
                    "portfolio_sharpe_annualized": float(row.get("portfolio_sharpe_annualized", np.nan)),
                    "portfolio_annualized_return": float(row.get("portfolio_annualized_return", np.nan)),
                    "portfolio_max_drawdown": float(row.get("portfolio_max_drawdown", np.nan)),
                    "portfolio_final_value": float(row.get("portfolio_final_value", np.nan)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "lookback_sensitivity_subset.csv", index=False)
    return out


def _write_corr_window_sensitivity_subset(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    targets = [
        ("gat", "corr", "gat_corr_only"),
        ("gcn", "corr", "gcn_corr_only"),
        ("xgb_node2vec", "node2vec_correlation", "xgb_node2vec_corr"),
    ]
    corr_windows = [30, 60, 120]
    rows: List[Dict[str, object]] = []
    m = master.copy()
    m["corr_window"] = pd.to_numeric(m.get("corr_window", pd.Series(dtype=float)), errors="coerce")
    m["rebalance_freq"] = pd.to_numeric(m.get("rebalance_freq", pd.Series(dtype=float)), errors="coerce")
    for model_name, edge_type, run_prefix in targets:
        for cw in corr_windows:
            sub = m[
                (m["model_name"].astype(str) == model_name)
                & (m["edge_type"].astype(str) == edge_type)
                & (m["rebalance_freq"] == 5)
                & (m["corr_window"] == int(cw))
            ].copy()
            if sub.empty:
                rows.append(
                    {
                        "model_name": model_name,
                        "edge_type": edge_type,
                        "run_prefix": run_prefix,
                        "rebalance_freq": 5,
                        "corr_window": int(cw),
                        "status": "missing_rerun_required",
                        "portfolio_sharpe_annualized": np.nan,
                        "portfolio_annualized_return": np.nan,
                        "portfolio_max_drawdown": np.nan,
                        "portfolio_final_value": np.nan,
                    }
                )
                continue
            row = sub.sort_values("portfolio_sharpe_annualized", ascending=False).iloc[0]
            rows.append(
                {
                    "model_name": model_name,
                    "edge_type": edge_type,
                    "run_prefix": run_prefix,
                    "rebalance_freq": 5,
                    "corr_window": int(cw),
                    "status": "available",
                    "portfolio_sharpe_annualized": float(row.get("portfolio_sharpe_annualized", np.nan)),
                    "portfolio_annualized_return": float(row.get("portfolio_annualized_return", np.nan)),
                    "portfolio_max_drawdown": float(row.get("portfolio_max_drawdown", np.nan)),
                    "portfolio_final_value": float(row.get("portfolio_final_value", np.nan)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "corr_window_sensitivity_subset.csv", index=False)
    return out


def _rolling_cv_fold_year(row: pd.Series) -> Optional[int]:
    test_start = pd.to_datetime(row.get("split_test_start"), errors="coerce")
    if not pd.isna(test_start):
        return int(test_start.year)
    tag = _safe_str(row.get("run_tag", ""))
    m = re.search(r"(?:y|year)(20\d{2})", tag)
    if m:
        return int(m.group(1))
    return None


def _curve_date_bounds(series: Optional[pd.Series]) -> Tuple[str, str]:
    if series is None or series.empty:
        return "", ""
    idx = pd.to_datetime(series.index, errors="coerce")
    idx = idx[~pd.isna(idx)]
    if len(idx) == 0:
        return "", ""
    return str(idx.min().date()), str(idx.max().date())


def _metrics_date_bounds(row: pd.Series, freq: int) -> Tuple[str, str]:
    path = _daily_metrics_path(row, int(freq))
    if not path.exists():
        return "", ""
    try:
        m = pd.read_csv(path)
    except Exception:
        return "", ""
    if "date" not in m.columns or m.empty:
        return "", ""
    d = pd.to_datetime(m["date"], errors="coerce").dropna()
    if d.empty:
        return "", ""
    return str(d.min().date()), str(d.max().date())


def _write_rolling_cv_summary(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [
        "entry_name",
        "strategy_kind",
        "run_tag",
        "model_name",
        "edge_type",
        "rebalance_freq",
        "fold_year",
        "fold_label",
        "configured_test_start",
        "observed_test_start",
        "observed_test_end",
        "summary_scope",
        "n_folds",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_turnover",
    ]

    if master.empty:
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(out_dir / "rolling_cv_summary.csv", index=False)
        return empty

    cv_mask = master["run_tag"].astype("string").str.contains("rolling_cv", case=False, na=False)
    cv_df = master[cv_mask].copy()
    if cv_df.empty:
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(out_dir / "rolling_cv_summary.csv", index=False)
        return empty

    cv_df["rebalance_freq"] = pd.to_numeric(cv_df.get("rebalance_freq"), errors="coerce")
    cv_df = cv_df[cv_df["rebalance_freq"] == 5].copy()
    if cv_df.empty:
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(out_dir / "rolling_cv_summary.csv", index=False)
        return empty

    cv_df["fold_year"] = cv_df.apply(_rolling_cv_fold_year, axis=1)
    cv_df = cv_df[cv_df["fold_year"].notna()].copy()
    if cv_df.empty:
        empty = pd.DataFrame(columns=cols)
        empty.to_csv(out_dir / "rolling_cv_summary.csv", index=False)
        return empty
    cv_df["fold_year"] = cv_df["fold_year"].astype(int)
    cv_df["fold_label"] = cv_df["fold_year"].map(lambda y: f"test_{int(y)}")

    fold_rows: List[Dict[str, object]] = []
    numeric_cols = [
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_turnover",
    ]

    for _, row in cv_df.iterrows():
        observed_start, observed_end = _metrics_date_bounds(row, 5)
        fold_rows.append(
            {
                "entry_name": _run_label(row),
                "strategy_kind": "model_long_only_topk",
                "run_tag": _safe_str(row.get("run_tag", "")),
                "model_name": _safe_str(row.get("model_name", "")),
                "edge_type": _safe_str(row.get("edge_type", "")),
                "rebalance_freq": 5,
                "fold_year": int(row["fold_year"]),
                "fold_label": f"test_{int(row['fold_year'])}",
                "configured_test_start": _safe_str(row.get("split_test_start", "")),
                "observed_test_start": observed_start,
                "observed_test_end": observed_end,
                "summary_scope": "fold",
                "n_folds": 1,
                "portfolio_sharpe_annualized": pd.to_numeric(row.get("portfolio_sharpe_annualized"), errors="coerce"),
                "portfolio_annualized_return": pd.to_numeric(row.get("portfolio_annualized_return"), errors="coerce"),
                "portfolio_max_drawdown": pd.to_numeric(row.get("portfolio_max_drawdown"), errors="coerce"),
                "portfolio_turnover": pd.to_numeric(row.get("portfolio_turnover"), errors="coerce"),
            }
        )

    # One baseline row per fold (shared across models).
    for year, subset in cv_df.groupby("fold_year", as_index=False):
        row = subset.sort_values("run_tag").iloc[0]
        payload = _summary_payload_for_row(row)
        bh_stats = _extract_baseline_stats(payload, 5, "buy_and_hold_stats")
        eqw_stats = _extract_baseline_stats(payload, 5, "equal_weight_stats")

        bh_curve_path, eqw_curve_path = _baseline_paths(row, 5)
        bh_curve = _read_curve_csv(bh_curve_path)
        eqw_curve = _read_curve_csv(eqw_curve_path)
        bh_fallback = _stats_from_curve(bh_curve)
        eqw_fallback = _stats_from_curve(eqw_curve)
        for k, v in bh_fallback.items():
            if pd.isna(bh_stats.get(k)):
                bh_stats[k] = v
        for k, v in eqw_fallback.items():
            if pd.isna(eqw_stats.get(k)):
                eqw_stats[k] = v

        configured_test_start = _safe_str(row.get("split_test_start", ""))
        bh_start, bh_end = _curve_date_bounds(bh_curve)
        eqw_start, eqw_end = _curve_date_bounds(eqw_curve)

        fold_rows.append(
            {
                "entry_name": BUY_HOLD_LABEL,
                "strategy_kind": "baseline_buy_and_hold",
                "run_tag": "",
                "model_name": "baseline",
                "edge_type": "none",
                "rebalance_freq": 5,
                "fold_year": int(year),
                "fold_label": f"test_{int(year)}",
                "configured_test_start": configured_test_start,
                "observed_test_start": bh_start,
                "observed_test_end": bh_end,
                "summary_scope": "fold",
                "n_folds": 1,
                "portfolio_sharpe_annualized": pd.to_numeric(
                    bh_stats.get("portfolio_sharpe_annualized"), errors="coerce"
                ),
                "portfolio_annualized_return": pd.to_numeric(
                    bh_stats.get("portfolio_annualized_return"), errors="coerce"
                ),
                "portfolio_max_drawdown": pd.to_numeric(
                    bh_stats.get("portfolio_max_drawdown"), errors="coerce"
                ),
                "portfolio_turnover": 0.0,
            }
        )

        fold_rows.append(
            {
                "entry_name": EQW_LABEL,
                "strategy_kind": "baseline_equal_weight",
                "run_tag": "",
                "model_name": "baseline",
                "edge_type": "none",
                "rebalance_freq": 5,
                "fold_year": int(year),
                "fold_label": f"test_{int(year)}",
                "configured_test_start": configured_test_start,
                "observed_test_start": eqw_start,
                "observed_test_end": eqw_end,
                "summary_scope": "fold",
                "n_folds": 1,
                "portfolio_sharpe_annualized": pd.to_numeric(
                    eqw_stats.get("portfolio_sharpe_annualized"), errors="coerce"
                ),
                "portfolio_annualized_return": pd.to_numeric(
                    eqw_stats.get("portfolio_annualized_return"), errors="coerce"
                ),
                "portfolio_max_drawdown": pd.to_numeric(
                    eqw_stats.get("portfolio_max_drawdown"), errors="coerce"
                ),
                "portfolio_turnover": 0.0,
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    for col in numeric_cols:
        folds_df[col] = pd.to_numeric(folds_df[col], errors="coerce")
    if folds_df[numeric_cols].isna().any().any():
        bad = {c: int(folds_df[c].isna().sum()) for c in numeric_cols if int(folds_df[c].isna().sum()) > 0}
        raise ValueError(f"rolling_cv_summary contains NaN fold metrics: {bad}")

    agg_rows: List[Dict[str, object]] = []
    for (entry_name, strategy_kind, model_name, edge_type, rebalance_freq), g in folds_df.groupby(
        ["entry_name", "strategy_kind", "model_name", "edge_type", "rebalance_freq"],
        dropna=False,
    ):
        vals = g[numeric_cols]
        n_folds = int(len(g))
        means = vals.mean(axis=0)
        stds = vals.std(axis=0, ddof=0)
        for scope, vec in (("mean", means), ("std", stds)):
            agg_rows.append(
                {
                    "entry_name": entry_name,
                    "strategy_kind": strategy_kind,
                    "run_tag": "",
                    "model_name": model_name,
                    "edge_type": edge_type,
                    "rebalance_freq": int(rebalance_freq),
                    "fold_year": pd.NA,
                    "fold_label": scope,
                    "configured_test_start": "",
                    "observed_test_start": "",
                    "observed_test_end": "",
                    "summary_scope": scope,
                    "n_folds": n_folds,
                    "portfolio_sharpe_annualized": float(vec["portfolio_sharpe_annualized"]),
                    "portfolio_annualized_return": float(vec["portfolio_annualized_return"]),
                    "portfolio_max_drawdown": float(vec["portfolio_max_drawdown"]),
                    "portfolio_turnover": float(vec["portfolio_turnover"]),
                }
            )

    out = pd.concat([folds_df, pd.DataFrame(agg_rows)], ignore_index=True)
    out["__scope_rank"] = out["summary_scope"].map({"fold": 0, "mean": 1, "std": 2}).fillna(9)
    out = out.sort_values(
        ["__scope_rank", "strategy_kind", "entry_name", "fold_year", "run_tag"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).drop(columns=["__scope_rank"]).reset_index(drop=True)

    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[cols].copy()
    out.to_csv(out_dir / "rolling_cv_summary.csv", index=False)
    return out


def _write_professor_main_results_table(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [
        "strategy_name",
        "strategy_label",
        "type",
        "run_tag",
        "run_key",
        "rebalance_freq",
        "rebalance_label",
        "final_value",
        "annual_return",
        "annual_vol",
        "sharpe_annualized",
        "max_drawdown",
        "turnover",
    ]
    rows: List[Dict[str, object]] = []
    if master.empty:
        out = pd.DataFrame(columns=cols)
        out.to_csv(out_dir / "professor_main_results_table.csv", index=False)
        return out

    # Baseline rows from canonical baseline policy table (reb=1, reb=5).
    baseline_policy_path = out_dir / "baseline_policy_comparison.csv"
    eqw_audit_path = out_dir / "equal_weight_rebalance_audit.csv"
    eqw_turnover_map: Dict[int, float] = {}
    if eqw_audit_path.exists():
        try:
            audit = pd.read_csv(eqw_audit_path)
            for _, r in audit.iterrows():
                f = int(pd.to_numeric(r.get("rebalance_freq"), errors="coerce"))
                t = pd.to_numeric(r.get("turnover_mean"), errors="coerce")
                if f > 0 and pd.notna(t):
                    eqw_turnover_map[f] = float(t)
        except Exception:
            pass
    if baseline_policy_path.exists():
        try:
            base = pd.read_csv(baseline_policy_path)
            base = base[pd.to_numeric(base["rebalance_freq"], errors="coerce").isin([1, 5, 21])].copy()
            for _, r in base.iterrows():
                freq = int(pd.to_numeric(r.get("rebalance_freq"), errors="coerce"))
                sharpe = float(pd.to_numeric(r.get("portfolio_sharpe_annualized"), errors="coerce"))
                ann_ret = float(pd.to_numeric(r.get("portfolio_annualized_return"), errors="coerce"))
                ann_vol = abs(ann_ret / sharpe) if np.isfinite(sharpe) and abs(sharpe) > 1e-12 else float("nan")
                if not np.isfinite(sharpe):
                    sharpe = 0.0
                if not np.isfinite(ann_ret):
                    ann_ret = 0.0
                if not np.isfinite(ann_vol):
                    ann_vol = 0.0
                name = _safe_str(r.get("strategy_name", ""))
                turnover = 0.0 if name == BUY_HOLD_LABEL else float(eqw_turnover_map.get(freq, 0.0))
                strategy_key = "baseline_buy_and_hold" if name == BUY_HOLD_LABEL else "baseline_equal_weight"
                rows.append(
                    {
                        "strategy_name": name,
                        "strategy_label": name,
                        "type": "baseline",
                        "run_tag": "",
                        "run_key": f"{strategy_key}_{freq}",
                        "rebalance_freq": int(freq),
                        "rebalance_label": _reb_label(int(freq)),
                        "final_value": float(pd.to_numeric(r.get("portfolio_final_value"), errors="coerce")),
                        "annual_return": ann_ret,
                        "annual_vol": ann_vol,
                        "sharpe_annualized": sharpe,
                        "max_drawdown": float(pd.to_numeric(r.get("portfolio_max_drawdown"), errors="coerce"))
                        if pd.notna(pd.to_numeric(r.get("portfolio_max_drawdown"), errors="coerce"))
                        else 0.0,
                        "turnover": turnover,
                    }
                )
        except Exception:
            pass

    # Best 3 learned models at reb=5 from canonical master metrics (no recomputation).
    best = master[master["rebalance_freq"] == 5].sort_values("portfolio_sharpe_annualized", ascending=False).head(3)
    for _, row in best.iterrows():
        reb = int(pd.to_numeric(row.get("rebalance_freq"), errors="coerce"))
        rows.append(
            {
                "strategy_name": f"{_safe_str(row.get('run_tag', ''))} ({TOPK_LABEL})",
                "strategy_label": TOPK_LABEL,
                "type": "model",
                "run_tag": _safe_str(row.get("run_tag", "")),
                "run_key": _safe_str(row.get("run_key", "")),
                "rebalance_freq": int(reb),
                "rebalance_label": _reb_label(int(reb)),
                "final_value": float(pd.to_numeric(row.get("portfolio_final_value"), errors="coerce")),
                "annual_return": float(pd.to_numeric(row.get("portfolio_annualized_return"), errors="coerce")),
                "annual_vol": float(pd.to_numeric(row.get("portfolio_annualized_volatility"), errors="coerce")),
                "sharpe_annualized": float(pd.to_numeric(row.get("portfolio_sharpe_annualized"), errors="coerce")),
                "max_drawdown": float(pd.to_numeric(row.get("portfolio_max_drawdown"), errors="coerce")),
                "turnover": float(pd.to_numeric(row.get("portfolio_turnover"), errors="coerce")),
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.drop_duplicates(subset=["run_key", "rebalance_freq", "strategy_label"], keep="first")
        required = ["final_value", "annual_return", "annual_vol", "sharpe_annualized", "max_drawdown", "turnover"]
        bad = out[required].isna().any(axis=1)
        if bad.any():
            missing_rows = out.loc[bad, ["strategy_name", "rebalance_label"]].to_dict(orient="records")
            raise ValueError(f"Professor main table has NaN metrics: {missing_rows[:5]}")
        out = out.sort_values(["type", "strategy_name", "rebalance_freq"]).reset_index(drop=True)
    out.to_csv(out_dir / "professor_main_results_table.csv", index=False)
    return out


def _write_cost_sensitivity_summary(
    long_only_cost: pd.DataFrame,
    long_short_cost: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not long_only_cost.empty:
        lo = long_only_cost.groupby("cost_bps", as_index=False).agg(
            sharpe_mean=("portfolio_sharpe_annualized", "mean"),
            ann_return_mean=("portfolio_annualized_return", "mean"),
            max_dd_mean=("portfolio_max_drawdown", "mean"),
            turnover_mean=("portfolio_turnover", "mean"),
            n=("run_tag", "count"),
        )
        for _, r in lo.iterrows():
            rows.append(
                {
                    "strategy": "long_only_topk",
                    "cost_bps": int(r["cost_bps"]),
                    "cost_label": _cost_label(int(r["cost_bps"])),
                    "sharpe_mean": float(r["sharpe_mean"]),
                    "ann_return_mean": float(r["ann_return_mean"]),
                    "max_dd_mean": float(r["max_dd_mean"]),
                    "turnover_mean": float(r["turnover_mean"]),
                    "n_runs": int(r["n"]),
                }
            )
    if not long_short_cost.empty:
        ls = long_short_cost.groupby("cost_bps", as_index=False).agg(
            sharpe_mean=("portfolio_sharpe_annualized", "mean"),
            ann_return_mean=("portfolio_annualized_return", "mean"),
            max_dd_mean=("portfolio_max_drawdown", "mean"),
            turnover_mean=("portfolio_turnover", "mean"),
            n=("run_tag", "count"),
        )
        for _, r in ls.iterrows():
            rows.append(
                {
                    "strategy": "long_short_top3_bottom3",
                    "cost_bps": int(r["cost_bps"]),
                    "cost_label": _cost_label(int(r["cost_bps"])),
                    "sharpe_mean": float(r["sharpe_mean"]),
                    "ann_return_mean": float(r["ann_return_mean"]),
                    "max_dd_mean": float(r["max_dd_mean"]),
                    "turnover_mean": float(r["turnover_mean"]),
                    "n_runs": int(r["n"]),
                }
            )
    out = pd.DataFrame(
        rows,
        columns=[
            "strategy",
            "cost_bps",
            "cost_label",
            "sharpe_mean",
            "ann_return_mean",
            "max_dd_mean",
            "turnover_mean",
            "n_runs",
        ],
    )
    out.to_csv(out_dir / "cost_sensitivity_summary.csv", index=False)

    if not out.empty:
        plt.figure(figsize=(10, 6))
        for strategy, g in out.groupby("strategy"):
            g = g.sort_values("cost_bps")
            plt.plot(g["cost_bps"], g["sharpe_mean"], marker="o", linewidth=2.0, label=strategy)
        plt.xticks(COST_SWEEP_BPS)
        plt.xlabel("Transaction Cost (bps)")
        plt.ylabel("Mean Annualized Sharpe")
        plt.title("Cost Sensitivity Summary")
        plt.grid(alpha=0.3, linestyle="--")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "cost_sensitivity_summary.png", dpi=220)
        plt.close()
    return out


def _write_audit_status(eqw_audit: pd.DataFrame, graph_audit: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not eqw_audit.empty:
        eqw_fail = int((eqw_audit.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "fail").sum())
        eqw_missing = int((eqw_audit.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "missing").sum())
        rows.append(
            {
                "audit_name": "equal_weight_rebalance_integrity",
                "status": "fail" if eqw_fail > 0 else "pass",
                "fail_rows": eqw_fail,
                "warning_rows": eqw_missing,
                "detail": "fail if reb=1 and reb=5 EQW series are identical",
            }
        )
    if not graph_audit.empty:
        graph_fail = int((graph_audit.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "fail").sum())
        rows.append(
            {
                "audit_name": "graph_time_awareness",
                "status": "fail" if graph_fail > 0 else "pass",
                "fail_rows": graph_fail,
                "warning_rows": 0,
                "detail": "fail if graph max timestamp exceeds policy bound",
            }
        )
    out = pd.DataFrame(rows, columns=["audit_name", "status", "fail_rows", "warning_rows", "detail"])
    out.to_csv(out_dir / "audit_status.csv", index=False)
    return out


def _write_strategy_definitions_markdown(out_dir: Path) -> None:
    text = (
        "# Strategy Definitions\n\n"
        f"- `{BUY_HOLD_LABEL}`: equal dollars at t0, fixed shares, no rebalance.\n"
        f"- `{EQW_LABEL}`: set 1/N on rebalance dates, hold and let weights drift between rebalances.\n"
        f"- `{TOPK_LABEL}`: select Top K by model score on rebalance dates, equal weight within Top K, hold between rebalances.\n"
        f"- `{LONG_SHORT_LABEL}`: top 3 long and bottom 3 short, equal within legs, scaled to +0.5 / -0.5.\n\n"
        "Rebalance notation: `reb=1`, `reb=5`, `reb=21`.\n\n"
        "Turnover/cost convention:\n"
        "- `turnover_t = sum_i |w_{i,t} - w_{i,t-1}|`\n"
        "- `cost_t = (bps/10000) * turnover_t` on rebalance dates.\n"
    )
    (out_dir / "strategy_definitions.md").write_text(text, encoding="utf-8")


def _refresh_equal_weight_artifacts(master: pd.DataFrame) -> None:
    """Refresh per-run equal-weight baseline curves using current baseline logic."""
    if master.empty:
        return
    try:
        cfg = load_config("configs/runs/core/xgb_raw.yaml", REPO_ROOT)
        price_df = load_price_panel(
            cfg["data"]["price_file"],
            cfg["data"]["start_date"],
            cfg["data"]["end_date"],
        )
        price_col = _select_price_col(price_df)
        universe = None
        universe_file = cfg["data"].get("universe_file")
        if universe_file and Path(universe_file).exists():
            u = pd.read_csv(universe_file)
            if "ticker" in u.columns:
                universe = u["ticker"].dropna().unique().tolist()
    except Exception:
        return

    eq_cache: Dict[Tuple[int, str, str], Optional[pd.Series]] = {}
    for _, row in master.iterrows():
        freq = int(pd.to_numeric(row.get("rebalance_freq"), errors="coerce"))
        pred = _read_prediction_df(row)
        if pred is None or pred.empty:
            continue
        start_date = str(pd.to_datetime(pred["date"], errors="coerce").dropna().min().date())
        end_date = str(pd.to_datetime(pred["date"], errors="coerce").dropna().max().date())
        key = (freq, start_date, end_date)
        if key not in eq_cache:
            try:
                eq, _, _ = compute_equal_weight_rebalanced(
                    price_df,
                    price_col=price_col,
                    start_date=start_date,
                    end_date=end_date,
                    universe=universe,
                    initial_value=1.0,
                    rebalance_freq=freq,
                    risk_free_rate=0.0,
                )
                eq_cache[key] = eq
            except Exception:
                eq_cache[key] = None
        eq_series = eq_cache.get(key)
        if eq_series is None:
            continue
        out_dir_raw = _safe_str(row.get("out_dir", "")).strip()
        out_dir = Path(out_dir_raw) if out_dir_raw else Path("experiments")
        run_tag = _safe_str(row.get("run_tag", "")).strip()
        if out_dir == Path("experiments") and run_tag:
            candidate = out_dir / run_tag
            if candidate.exists():
                out_dir = candidate
        write_baseline_curve_csv(eq_series, out_dir / f"equal_weight_equity_curve_reb{freq}.csv", "equal_weight")


def _write_baseline_context_csv(latest_df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    if latest_df.empty:
        pd.DataFrame(rows).to_csv(out_dir / "baseline_context.csv", index=False)
        return

    for freq in sorted(pd.to_numeric(latest_df["rebalance_freq"], errors="coerce").dropna().astype(int).unique()):
        subset = latest_df[latest_df["rebalance_freq"] == int(freq)].copy()
        if subset.empty:
            continue
        row = subset.sort_values("portfolio_sharpe_annualized", ascending=False).iloc[0]
        ctx = _baseline_context_for_row(row)
        rows.append(
            {
                "rebalance_freq": int(freq),
                "source_run_tag": _safe_str(row.get("run_tag", "")),
                "global_start_date": ctx["global_start_date"],
                "global_end_date": ctx["global_end_date"],
                "global_final_value": ctx["global_final_value"],
                "test_start_date": ctx["test_start_date"],
                "test_end_date": ctx["test_end_date"],
                "test_final_value": ctx["test_final_value"],
                "test_rebased": bool(ctx["test_rebased"]),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "baseline_context.csv", index=False)


def _build_decision_ranking(latest_df: pd.DataFrame) -> pd.DataFrame:
    if latest_df.empty:
        return latest_df.copy()

    ranked = latest_df.copy()
    ctx_df = pd.DataFrame([_baseline_context_for_row(r) for _, r in ranked.iterrows()])
    ranked = pd.concat([ranked.reset_index(drop=True), ctx_df.reset_index(drop=True)], axis=1)

    ranked["delta_final_value_vs_test_bh"] = ranked["portfolio_final_value"] - ranked["test_final_value"]
    ranked["ratio_final_value_vs_test_bh"] = ranked["portfolio_final_value"] / ranked["test_final_value"]
    ranked["delta_annualized_return_vs_test_bh"] = ranked["portfolio_annualized_return"] - ranked["test_annualized_return"]

    ranked = ranked.sort_values(
        [
            "portfolio_sharpe_annualized",
            "portfolio_max_drawdown",
            "prediction_rank_ic",
            "portfolio_turnover",
            "run_tag",
            "model_name",
            "edge_type",
        ],
        ascending=[False, False, False, True, True, True, True],
    ).reset_index(drop=True)
    ranked["decision_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    keep_cols = [
        "decision_rank",
        "run_tag",
        "model_name",
        "edge_type",
        "rebalance_freq",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "prediction_rank_ic",
        "portfolio_turnover",
        "portfolio_final_value",
        "test_final_value",
        "delta_final_value_vs_test_bh",
        "ratio_final_value_vs_test_bh",
        "delta_annualized_return_vs_test_bh",
        "test_start_date",
        "test_end_date",
    ]
    return ranked[keep_cols].copy()


def _plot_risk_frontier(df: pd.DataFrame, out_dir: Path, freq: int) -> int:
    if df.empty:
        return 0
    d = df.dropna(subset=["portfolio_annualized_return", "portfolio_max_drawdown", "portfolio_sharpe_annualized"]).copy()
    if d.empty:
        return 0
    d["label"] = d.apply(_run_label, axis=1)
    sharpes = pd.to_numeric(d["portfolio_sharpe_annualized"], errors="coerce").fillna(0.0)
    sizes = (sharpes.clip(lower=0.0) + 0.05) * 900.0

    plt.figure(figsize=(10, 7))
    plt.scatter(
        d["portfolio_max_drawdown"],
        d["portfolio_annualized_return"],
        s=sizes,
        alpha=0.6,
        color="#1f77b4",
        edgecolors="white",
        linewidths=0.8,
    )
    for _, r in d.iterrows():
        plt.text(float(r["portfolio_max_drawdown"]), float(r["portfolio_annualized_return"]), str(r["label"]), fontsize=7)
    plt.xlabel("Max Drawdown")
    plt.ylabel("Annualized Return")
    plt.title(f"Risk Frontier (reb={freq})")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_dir / f"risk_frontier_reb{freq}.png", dpi=220)
    plt.close()
    return int(len(d))


def _plot_bubble_risk_return(df: pd.DataFrame, out_dir: Path, freq: int) -> int:
    if df.empty:
        return 0
    d = df.dropna(
        subset=["portfolio_annualized_return", "portfolio_annualized_volatility", "portfolio_sharpe_annualized"]
    ).copy()
    if d.empty:
        return 0

    d["label"] = d.apply(_run_label, axis=1)
    d["model_family"] = d.get("model_family", pd.Series(dtype=str)).astype(str).str.lower().fillna("other")
    d["edge_type"] = d.get("edge_type", pd.Series(dtype=str)).astype(str).fillna("none")
    sharpes = pd.to_numeric(d["portfolio_sharpe_annualized"], errors="coerce").fillna(0.0)
    d["bubble_size"] = (sharpes.clip(lower=0.0) + 0.05) * 850.0

    families = sorted(d["model_family"].unique().tolist())
    family_colors = {f: c for f, c in zip(families, plt.cm.tab10(np.linspace(0, 1, max(len(families), 1))))}
    markers_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h"]
    edge_types = sorted(d["edge_type"].unique().tolist())
    edge_markers = {e: markers_cycle[i % len(markers_cycle)] for i, e in enumerate(edge_types)}

    plt.figure(figsize=(11, 7))
    for _, r in d.iterrows():
        plt.scatter(
            float(r["portfolio_annualized_volatility"]),
            float(r["portfolio_annualized_return"]),
            s=float(r["bubble_size"]),
            c=[family_colors.get(str(r["model_family"]), "#1f77b4")],
            marker=edge_markers.get(str(r["edge_type"]), "o"),
            alpha=0.65,
            edgecolors="white",
            linewidths=0.8,
        )
    for _, r in d.iterrows():
        plt.text(
            float(r["portfolio_annualized_volatility"]),
            float(r["portfolio_annualized_return"]),
            str(r["label"]),
            fontsize=7,
        )

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title(f"Return-Vol Bubble Plot (reb={freq})")
    plt.grid(alpha=0.3, linestyle="--")

    family_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=family_colors[f], label=f, markersize=8)
        for f in families
    ]
    edge_handles = [
        plt.Line2D([0], [0], marker=edge_markers[e], color="black", linestyle="None", label=f"edge={e}", markersize=7)
        for e in edge_types
    ]
    if family_handles:
        leg1 = plt.legend(handles=family_handles, title="Model Family", loc="upper left", frameon=False)
        plt.gca().add_artist(leg1)
    if edge_handles:
        plt.legend(handles=edge_handles, title="Edge Type", loc="lower right", frameon=False, ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"bubble_risk_return_reb{freq}.png", dpi=220)
    plt.close()
    return int(len(d))


def _plot_metric_bars(df: pd.DataFrame, out_dir: Path, freq: int) -> None:
    if df.empty:
        return
    top = df.sort_values("portfolio_sharpe_annualized", ascending=False).head(12).copy()
    if top.empty:
        return
    top["label"] = top.apply(_run_label, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    top.plot(x="label", y="portfolio_sharpe_annualized", kind="bar", ax=axes[0], color="#1f77b4", legend=False)
    axes[0].set_title(f"Annualized Sharpe (reb={freq})")
    axes[0].set_xlabel("Run")
    axes[0].set_ylabel("Annualized Sharpe")

    top.plot(x="label", y="portfolio_annualized_return", kind="bar", ax=axes[1], color="#2ca02c", legend=False)
    axes[1].set_title(f"Annualized Return (reb={freq})")
    axes[1].set_xlabel("Run")
    axes[1].set_ylabel("Annualized Return")

    top.plot(x="label", y="portfolio_max_drawdown", kind="bar", ax=axes[2], color="#d62728", legend=False)
    axes[2].set_title(f"Max Drawdown (reb={freq})")
    axes[2].set_xlabel("Run")
    axes[2].set_ylabel("Max Drawdown")

    for ax in axes:
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_metrics_reb{freq}.png", dpi=180)
    plt.close(fig)


def _plot_ic_vs_sharpe(df: pd.DataFrame, out_dir: Path, freq: int) -> int:
    if df.empty:
        return 0
    d = df.dropna(subset=["prediction_rank_ic", "portfolio_sharpe_annualized"]).copy()
    if d.empty:
        return 0
    d["label"] = d.apply(_run_label, axis=1)

    plt.figure(figsize=(10, 7))
    plt.scatter(d["prediction_rank_ic"], d["portfolio_sharpe_annualized"], alpha=0.85, s=50)
    for _, r in d.iterrows():
        plt.text(float(r["prediction_rank_ic"]), float(r["portfolio_sharpe_annualized"]), str(r["label"]), fontsize=7)
    plt.xlabel("Mean Rank IC")
    plt.ylabel("Portfolio Sharpe (Annualized)")
    plt.title(f"IC vs Annualized Sharpe (reb={freq})")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_dir / f"ic_vs_sharpe_reb{freq}.png", dpi=220)
    plt.close()
    return int(len(d))


def _plot_ic_bar(df: pd.DataFrame, out_dir: Path, freq: int) -> None:
    if df.empty:
        return
    d = df.sort_values("prediction_rank_ic", ascending=False).head(12).copy()
    if d.empty:
        return
    d["label"] = d.apply(_run_label, axis=1)

    plt.figure(figsize=(12, 5))
    plt.bar(d["label"], d["prediction_rank_ic"], color="#4c78a8")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Rank IC")
    plt.title(f"IC by Run (reb={freq})")
    plt.tight_layout()
    plt.savefig(out_dir / f"ic_bar_reb{freq}.png", dpi=180)
    plt.close()


def _plot_ic_distribution_boxplot(df: pd.DataFrame, out_dir: Path, freq: int, max_runs: int = 12) -> None:
    if df.empty:
        return
    top = df.sort_values("portfolio_sharpe_annualized", ascending=False).head(max_runs).copy()
    if top.empty:
        return

    series_list = []
    labels = []
    for _, row in top.iterrows():
        p = _daily_metrics_path(row, freq)
        if not p.exists():
            continue
        try:
            daily = pd.read_csv(p)
        except Exception:
            continue
        if "ic" not in daily.columns:
            continue
        ic = pd.to_numeric(daily["ic"], errors="coerce").dropna()
        if ic.empty:
            continue
        series_list.append(ic.values)
        labels.append(_run_label(row))

    if not series_list:
        return

    plt.figure(figsize=(max(10, len(labels) * 0.8), 6))
    plt.boxplot(series_list, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Daily IC")
    plt.title(f"IC Distribution by Run (reb={freq})")
    plt.tight_layout()
    plt.savefig(out_dir / f"ic_distribution_boxplot_reb{freq}.png", dpi=180)
    plt.close()


def _plot_equity_panels(latest_df: pd.DataFrame, out_dir: Path, frequencies: List[int]) -> None:
    if latest_df.empty or not frequencies:
        return

    latest_df = latest_df.copy()
    latest_df["category"] = latest_df.apply(_classify_category, axis=1)
    key_runs = _best_by_group(
        latest_df[latest_df["category"].isin({"non_graph", "graph_feature", "static_gnn", "static_temporal_labeled"})],
        ["rebalance_freq", "category"],
    )
    if key_runs.empty:
        return

    def _draw_equity_ax(ax: plt.Axes, subset: pd.DataFrame, freq: int) -> None:
        if subset.empty:
            ax.set_title(f"Key Equity Curves (reb={freq})")
            ax.axis("off")
            return

        baseline_drawn = False
        baseline_window = ""
        curve_count = 0
        for _, row in subset.iterrows():
            curve = _read_curve_csv(_model_curve_path(row, int(freq)))
            if curve is None:
                continue
            curve_count += 1
            ax.plot(curve.index, curve.values, label=_run_label(row), linewidth=2.0)
            if not baseline_drawn:
                bh_path, eqw_path = _baseline_paths(row, int(freq))
                bh = _read_curve_csv(bh_path)
                eqw = _read_curve_csv(eqw_path)
                if bh is not None:
                    ax.plot(
                        bh.index,
                        bh.values,
                        label=BUY_HOLD_LABEL,
                        linestyle="--",
                        linewidth=1.8,
                    )
                    baseline_window = f"{bh.index.min().date()}..{bh.index.max().date()}"
                if eqw is not None:
                    ax.plot(eqw.index, eqw.values, label=EQW_LABEL, linestyle=":", linewidth=1.8)
                baseline_drawn = True

        if baseline_window:
            ax.set_title(f"Key Equity Curves (reb={freq})\nTest window: {baseline_window}")
        else:
            ax.set_title(f"Key Equity Curves (reb={freq})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.3, linestyle="--")
        if curve_count > 0:
            ax.legend(fontsize=9, ncol=2, frameon=False)

    # Backward-compatible combined panel.
    ncols = len(frequencies)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 5.5), squeeze=False)
    for ax, freq in zip(axes.ravel().tolist(), frequencies):
        subset = key_runs[key_runs["rebalance_freq"] == int(freq)].copy()
        _draw_equity_ax(ax, subset, int(freq))
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curves_key_models.png", dpi=220)
    plt.close(fig)

    # Readable thesis variants: one larger figure per rebalance policy.
    for freq in frequencies:
        subset = key_runs[key_runs["rebalance_freq"] == int(freq)].copy()
        fig_single, ax_single = plt.subplots(figsize=(11, 6.5))
        _draw_equity_ax(ax_single, subset, int(freq))
        fig_single.tight_layout()
        fig_single.savefig(out_dir / f"equity_curves_key_models_reb{int(freq)}.png", dpi=220)
        plt.close(fig_single)


def _build_run_matrix(latest_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run_id",
        "run_key",
        "run_tag",
        "model_family",
        "model_name",
        "edge_type",
        "corr_window",
        "lookback_window",
        "seed",
        "target_type",
        "target_horizon",
        "rebalance_freq",
        "split_id",
        "config_hash",
    ]
    out = latest_df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    out["inclusion_status"] = "included"
    out = out[cols + ["inclusion_status"]].copy()
    out["rebalance_freq"] = pd.to_numeric(out["rebalance_freq"], errors="coerce").astype("Int64")
    out = out.sort_values(
        ["rebalance_freq", "model_family", "model_name", "edge_type", "seed", "run_tag", "run_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def generate_reports(results_path: Path, out_dir: Path, *, expected_runs: Optional[int] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_strategy_definitions_markdown(out_dir)
    raw = _safe_read_results(results_path)
    raw = _ensure_columns(
        raw,
        KEY_METRICS
        + [
            "run_id",
            "experiment_id",
            "timestamp",
            "model_name",
            "model_family",
            "edge_type",
            "corr_window",
            "rebalance_freq",
            "run_tag",
            "out_dir",
            "artifact_prefix",
            "target_policy_hash",
            "split_id",
            "config_hash",
            "run_key",
        ],
    )
    raw = _fill_annualized_sharpe(raw)
    raw["rebalance_freq"] = pd.to_numeric(raw["rebalance_freq"], errors="coerce").fillna(1).astype(int)

    sort_cols = [
        "rebalance_freq",
        "portfolio_sharpe_annualized",
        "portfolio_max_drawdown",
        "prediction_rank_ic",
        "portfolio_turnover",
        "run_key",
        "run_tag",
    ]
    for col in sort_cols:
        if col not in raw.columns:
            raw[col] = pd.NA
    raw_sorted = raw.sort_values(
        sort_cols,
        ascending=[True, False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    raw_sorted.to_csv(out_dir / "master_comparison_raw.csv", index=False)

    latest = _pick_latest_rows(raw_sorted)
    latest["category"] = latest.apply(_classify_category, axis=1)
    latest = latest.sort_values(
        sort_cols,
        ascending=[True, False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    _assert_no_nan_metrics(
        latest,
        [
            "portfolio_sharpe_annualized",
            "portfolio_annualized_return",
            "portfolio_max_drawdown",
            "portfolio_turnover",
            "prediction_rank_ic",
        ],
    )

    run_matrix = _build_run_matrix(latest)
    run_matrix.to_csv(out_dir / "run_matrix.csv", index=False)
    if expected_runs is not None and len(run_matrix) != int(expected_runs):
        raise ValueError(
            f"Run matrix size mismatch: expected {int(expected_runs)} rows, found {len(run_matrix)} rows."
        )

    master = latest.copy()
    master.to_csv(out_dir / "master_comparison.csv", index=False)

    _refresh_equal_weight_artifacts(master)

    # Hard gates before secondary report artifacts.
    eqw_audit = _audit_equal_weight_rebalance(master, out_dir)
    graph_audit = _audit_graph_time_awareness(master, out_dir)
    _write_audit_status(eqw_audit, graph_audit, out_dir)

    family_summary = _best_by_group(
        master[master["category"].isin({"non_graph", "graph_feature", "static_gnn", "static_temporal_labeled"})],
        ["rebalance_freq", "category"],
    )
    family_summary = family_summary.sort_values(["rebalance_freq", "category"]).reset_index(drop=True)
    family_summary.to_csv(out_dir / "family_summary.csv", index=False)

    edge_df = master[
        (master["model_name"].isin(["gcn", "gat"]))
        & master["edge_type"].notna()
        & ~master["edge_type"].astype("string").str.lower().isin({"", "nan", "none"})
    ].copy()
    if edge_df.empty:
        edge_agg = pd.DataFrame(columns=["rebalance_freq", "edge_type", "model_count"] + KEY_METRICS)
    else:
        edge_agg = (
            edge_df.groupby(["rebalance_freq", "edge_type"], as_index=False)
            .agg(
                model_count=("model_name", "count"),
                prediction_rmse=("prediction_rmse", "mean"),
                prediction_mae=("prediction_mae", "mean"),
                prediction_rank_ic=("prediction_rank_ic", "mean"),
                portfolio_annualized_return=("portfolio_annualized_return", "mean"),
                portfolio_sharpe=("portfolio_sharpe", "mean"),
                portfolio_sharpe_daily=("portfolio_sharpe_daily", "mean"),
                portfolio_sharpe_annualized=("portfolio_sharpe_annualized", "mean"),
                portfolio_sortino_annualized=("portfolio_sortino_annualized", "mean"),
                portfolio_max_drawdown=("portfolio_max_drawdown", "mean"),
                portfolio_turnover=("portfolio_turnover", "mean"),
            )
            .sort_values(["rebalance_freq", "portfolio_sharpe_annualized"], ascending=[True, False])
            .reset_index(drop=True)
        )
    edge_agg.to_csv(out_dir / "edge_ablation_summary.csv", index=False)

    decision_ranking = _build_decision_ranking(master)
    decision_ranking.to_csv(out_dir / "decision_ranking.csv", index=False)
    _write_baseline_context_csv(master, out_dir)
    _write_baseline_policy_comparison_csv(master, out_dir)
    _write_alpha_vs_equal_weight(master, out_dir)
    cost_lo = _write_cost_sensitivity_long_only(master, out_dir)
    _, cost_ls = _write_long_short_tables(master, out_dir)
    _write_cost_sensitivity_summary(cost_lo, cost_ls, out_dir)
    _write_monthly_rebalance_subset(master, out_dir)
    _write_lookback_sensitivity_subset(master, out_dir)
    _write_corr_window_sensitivity_subset(master, out_dir)
    _write_rolling_cv_summary(master, out_dir)
    _write_professor_main_results_table(master, out_dir)

    freqs = sorted(master["rebalance_freq"].dropna().unique().tolist())
    for freq in freqs:
        freq_df = master[master["rebalance_freq"] == int(freq)].copy()
        if freq_df.empty:
            continue
        _plot_metric_bars(freq_df, out_dir, int(freq))
        _plot_ic_bar(freq_df, out_dir, int(freq))
        _plot_ic_distribution_boxplot(freq_df, out_dir, int(freq))
        expected_ic_points = int(
            freq_df.dropna(subset=["prediction_rank_ic", "portfolio_sharpe_annualized"]).shape[0]
        )
        actual_ic_points = _plot_ic_vs_sharpe(freq_df, out_dir, int(freq))
        if actual_ic_points != expected_ic_points:
            raise ValueError(
                f"IC-vs-Sharpe point mismatch for rebalance {int(freq)}: expected {expected_ic_points}, got {actual_ic_points}."
            )
        if expected_ic_points > 1 and actual_ic_points <= 1:
            raise ValueError(f"IC-vs-Sharpe plot for rebalance {int(freq)} has <=1 point unexpectedly.")

        expected_risk_points = int(
            freq_df.dropna(
                subset=["portfolio_annualized_return", "portfolio_max_drawdown", "portfolio_sharpe_annualized"]
            ).shape[0]
        )
        actual_risk_points = _plot_risk_frontier(freq_df, out_dir, int(freq))
        if actual_risk_points != expected_risk_points:
            raise ValueError(
                f"Risk-frontier point mismatch for rebalance {int(freq)}: expected {expected_risk_points}, got {actual_risk_points}."
            )
        if expected_risk_points > 1 and actual_risk_points <= 1:
            raise ValueError(f"Risk-frontier plot for rebalance {int(freq)} has <=1 point unexpectedly.")
        _plot_bubble_risk_return(freq_df, out_dir, int(freq))

    _plot_equity_panels(master, out_dir, freqs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready benchmark tables and plots.")
    parser.add_argument("--results", type=str, default="results/results_tuned_all.jsonl")
    parser.add_argument("--out", type=str, default="results/reports/thesis_tuned_all")
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument(
        "--prediction-root",
        action="append",
        default=["experiments_tuned_all", "experiments"],
        help="Prediction artifact root to audit before report generation. Can be repeated.",
    )
    parser.add_argument(
        "--skip-prediction-audit",
        action="store_true",
        help="Skip fail-fast audit for duplicate (date,ticker) prediction rows.",
    )
    args = parser.parse_args()

    if not args.skip_prediction_audit:
        files = assert_no_prediction_artifact_issues(args.prediction_root)
        print(f"[report] prediction audit passed across {len(files)} file(s)")

    expected_runs = args.expected_runs
    if expected_runs is None and "results_tuned_all" in str(args.results):
        expected_runs = 45
    generate_reports(Path(args.results), Path(args.out), expected_runs=expected_runs)
    print(f"[report] wrote thesis artifacts to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
