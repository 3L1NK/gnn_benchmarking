#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.prediction_audit import assert_no_prediction_artifact_issues


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
    key_cols = ["run_tag", "model_name", "edge_type", "rebalance_freq", "target_policy_hash"]
    group_key_cols: List[str] = []
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
    direct = out_dir / f"{prefix}_equity_curve_reb{freq}.csv"
    if direct.exists():
        return direct
    # Backward-compatible fallback
    legacy = out_dir / f"{prefix}_equity_curve.csv"
    if legacy.exists():
        return legacy
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
    direct = out_dir / f"{prefix}_daily_metrics_reb{freq}.csv"
    if direct.exists():
        return direct
    legacy = out_dir / f"{prefix}_daily_metrics.csv"
    if legacy.exists():
        return legacy
    discovered = _find_latest(f"experiments/**/{prefix}_daily_metrics_reb{freq}.csv")
    if discovered is not None:
        return discovered
    discovered = _find_latest(f"experiments/**/{prefix}_daily_metrics.csv")
    if discovered is not None:
        return discovered
    return direct


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
    bh = out_dir / "buy_and_hold_equity_curve.csv"
    eqw = out_dir / f"equal_weight_equity_curve_reb{freq}.csv"
    if not bh.exists():
        found = _find_latest("experiments/**/buy_and_hold_equity_curve.csv")
        if found is not None:
            bh = found
    if not eqw.exists():
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

    sp = _summary_path(row)
    if sp is not None and sp.exists():
        try:
            payload = json.loads(sp.read_text())
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


def _plot_risk_frontier(df: pd.DataFrame, out_dir: Path, freq: int) -> None:
    if df.empty:
        return
    d = df.dropna(subset=["portfolio_annualized_return", "portfolio_max_drawdown", "portfolio_sharpe_annualized"]).copy()
    if d.empty:
        return
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


def _plot_ic_vs_sharpe(df: pd.DataFrame, out_dir: Path, freq: int) -> None:
    if df.empty:
        return
    d = df.dropna(subset=["prediction_rank_ic", "portfolio_sharpe_annualized"]).copy()
    if d.empty:
        return
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
                    ax.plot(bh.index, bh.values, label="buy_and_hold (test rebased)", linestyle="--", linewidth=1.8)
                    baseline_window = f"{bh.index.min().date()}..{bh.index.max().date()}"
                if eqw is not None:
                    ax.plot(eqw.index, eqw.values, label="equal_weight", linestyle=":", linewidth=1.8)
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


def generate_reports(results_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _safe_read_results(results_path)
    raw = _ensure_columns(
        raw,
        KEY_METRICS
        + [
            "experiment_id",
            "timestamp",
            "model_name",
            "model_family",
            "edge_type",
            "rebalance_freq",
            "run_tag",
            "out_dir",
            "artifact_prefix",
            "target_policy_hash",
        ],
    )
    raw["portfolio_sharpe"] = pd.to_numeric(raw["portfolio_sharpe"], errors="coerce")
    if "portfolio_sharpe_daily" not in raw.columns:
        raw["portfolio_sharpe_daily"] = raw["portfolio_sharpe"]
    raw["portfolio_sharpe_daily"] = pd.to_numeric(raw["portfolio_sharpe_daily"], errors="coerce")
    if "portfolio_sharpe_annualized" not in raw.columns:
        raw["portfolio_sharpe_annualized"] = raw["portfolio_sharpe_daily"] * np.sqrt(252.0)
    raw["portfolio_sharpe_annualized"] = pd.to_numeric(raw["portfolio_sharpe_annualized"], errors="coerce")
    raw["rebalance_freq"] = pd.to_numeric(raw["rebalance_freq"], errors="coerce").fillna(1).astype(int)

    master = raw.sort_values(["rebalance_freq", "portfolio_sharpe_annualized"], ascending=[True, False]).reset_index(drop=True)
    master.to_csv(out_dir / "master_comparison.csv", index=False)

    latest = _pick_latest_rows(master)
    latest["category"] = latest.apply(_classify_category, axis=1)

    family_summary = _best_by_group(
        latest[latest["category"].isin({"non_graph", "graph_feature", "static_gnn", "static_temporal_labeled"})],
        ["rebalance_freq", "category"],
    )
    family_summary = family_summary.sort_values(["rebalance_freq", "category"]).reset_index(drop=True)
    family_summary.to_csv(out_dir / "family_summary.csv", index=False)

    edge_df = latest[(latest["model_name"].isin(["gcn", "gat"])) & latest["edge_type"].notna()].copy()
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

    decision_ranking = _build_decision_ranking(latest)
    decision_ranking.to_csv(out_dir / "decision_ranking.csv", index=False)
    _write_baseline_context_csv(latest, out_dir)

    freqs = sorted(latest["rebalance_freq"].dropna().unique().tolist())
    for freq in freqs:
        freq_df = latest[latest["rebalance_freq"] == int(freq)].copy()
        if freq_df.empty:
            continue
        _plot_metric_bars(freq_df, out_dir, int(freq))
        _plot_ic_bar(freq_df, out_dir, int(freq))
        _plot_ic_distribution_boxplot(freq_df, out_dir, int(freq))
        _plot_ic_vs_sharpe(freq_df, out_dir, int(freq))
        _plot_risk_frontier(freq_df, out_dir, int(freq))

    _plot_equity_panels(latest, out_dir, freqs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready benchmark tables and plots.")
    parser.add_argument("--results", type=str, default="results/results.jsonl")
    parser.add_argument("--out", type=str, default="results/reports/thesis")
    parser.add_argument(
        "--prediction-root",
        action="append",
        default=["experiments"],
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

    generate_reports(Path(args.results), Path(args.out))
    print(f"[report] wrote thesis artifacts to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
