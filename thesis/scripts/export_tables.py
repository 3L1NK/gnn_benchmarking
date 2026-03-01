#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import pandas as pd

DEFAULT_TUNED_REPORT_DIR = REPO_ROOT / "results" / "reports" / "thesis_tuned_all"
DEFAULT_CORE_REPORT_DIR = REPO_ROOT / "results" / "reports" / "thesis"
_report_dir_env = os.getenv("THESIS_REPORT_DIR")
if _report_dir_env:
    REPORT_DIR = Path(_report_dir_env)
else:
    REPORT_DIR = DEFAULT_TUNED_REPORT_DIR
OUTPUT_DIR = REPO_ROOT / "thesis" / "tables" / "generated"

MODEL_NAME_MAP = {
    "xgb_raw": "XGB",
    "xgb_node2vec": "XGB+Node2Vec",
    "lstm": "LSTM",
    "gcn": "GCN",
    "gat": "GAT",
    "tgcn_static": "TGCN-static",
    "tgat_static": "TGAT-static",
}

FAMILY_MAP = {
    "xgboost": "XGB",
    "lstm": "LSTM",
    "gnn": "GNN",
}

CATEGORY_MAP = {
    "graph_feature": "Graph-feature",
    "non_graph": "Non-graph",
    "static_gnn": "Static GNN",
    "static_temporal_labeled": "Static temporal",
}

EDGE_MAP = {
    "node2vec_correlation": "n2v-corr",
    "corr+sector+granger": "corr+sec+gr",
    "corr_sector_granger": "corr+sec+gr",
    "corr": "corr",
    "sector": "sector",
    "granger": "granger",
    "none": "none",
}


def _load_csv(name: str) -> pd.DataFrame:
    if not REPORT_DIR.exists():
        raise FileNotFoundError(
            f"Report directory does not exist: {REPORT_DIR}. "
            "Generate report first with scripts/generate_thesis_report.py."
        )
    path = REPORT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required report file: {path}")
    return pd.read_csv(path)


def _load_csv_optional(name: str) -> pd.DataFrame:
    if not REPORT_DIR.exists():
        return pd.DataFrame()
    path = REPORT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _metric_col(df: pd.DataFrame) -> str:
    if "portfolio_sharpe_annualized" in df.columns:
        return "portfolio_sharpe_annualized"
    if "portfolio_sharpe_daily" in df.columns:
        return "portfolio_sharpe_daily"
    if "portfolio_sharpe" in df.columns:
        return "portfolio_sharpe"
    raise ValueError("No Sharpe metric column found in input data")


def _edge_label(value: object) -> str:
    edge = str(value).strip().lower()
    if not edge or edge in {"nan"}:
        return ""
    return EDGE_MAP.get(edge, edge.replace("_", "+"))


def _model_name_label(value: object) -> str:
    model = str(value).strip().lower()
    if not model or model in {"nan"}:
        return "Model"
    return MODEL_NAME_MAP.get(model, model.upper())


def _family_label(value: object) -> str:
    family = str(value).strip().lower()
    if not family or family in {"nan"}:
        return ""
    return FAMILY_MAP.get(family, family)


def _category_label(value: object) -> str:
    cat = str(value).strip().lower()
    if not cat or cat in {"nan"}:
        return ""
    return CATEGORY_MAP.get(cat, cat)


def _model_label(row: pd.Series) -> str:
    model = _model_name_label(row.get("model_name", ""))
    edge = _edge_label(row.get("edge_type", ""))
    label = model
    if edge and edge != "none":
        label = f"{label} ({edge})"
    return label


def _strip_tuning_suffix(label: object) -> str:
    text = str(label).strip()
    for suffix in ("_tuned_all", "_tuned"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _run_tag_to_label(label: object) -> str:
    tag = _strip_tuning_suffix(label)
    if tag == "xgb_raw":
        return "XGB"
    if tag == "xgb_node2vec_corr":
        return "XGB+Node2Vec (corr)"
    if tag == "lstm":
        return "LSTM"
    if tag.startswith("gcn_"):
        edge = _edge_label(tag.replace("gcn_", "").replace("_only", ""))
        return f"GCN ({edge})" if edge else "GCN"
    if tag.startswith("gat_"):
        edge = _edge_label(tag.replace("gat_", "").replace("_only", ""))
        return f"GAT ({edge})" if edge else "GAT"
    if tag.startswith("tgcn_static_"):
        edge = _edge_label(tag.replace("tgcn_static_", "").replace("_only", ""))
        return f"TGCN-static ({edge})" if edge else "TGCN-static"
    if tag.startswith("tgat_static_"):
        edge = _edge_label(tag.replace("tgat_static_", "").replace("_only", ""))
        return f"TGAT-static ({edge})" if edge else "TGAT-static"
    return tag.replace("_", "-")


def _reb_to_label(value: object) -> str:
    try:
        return f"reb={int(pd.to_numeric(value, errors='coerce'))}"
    except Exception:
        return str(value)


def _write_table(df: pd.DataFrame, out_name: str, caption: str, label: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabular = df.to_latex(index=False, escape=True, float_format="%.4f")
    latex = (
        "\\begin{table}[htbp]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{tabular.rstrip()}\n"
        "}\n"
        "\\end{table}\n"
    )
    out_path = OUTPUT_DIR / out_name
    out_path.write_text("% Auto-generated by thesis/scripts/export_tables.py\n" + latex, encoding="utf-8")


def _write_longtable(df: pd.DataFrame, out_name: str, caption: str, label: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(
        index=False,
        escape=True,
        float_format="%.4f",
        longtable=True,
        caption=caption,
        label=label,
    )
    out_path = OUTPUT_DIR / out_name
    out_path.write_text(
        "% Auto-generated by thesis/scripts/export_tables.py\n"
        "\\begingroup\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        f"{latex}\n"
        "\\endgroup\n",
        encoding="utf-8",
    )


def _write_no_data_table(out_name: str, caption: str, label: str) -> None:
    _write_table(
        pd.DataFrame({"Status": ["not_available"]}),
        out_name=out_name,
        caption=caption,
        label=label,
    )


def _short_hash(value: object, n: int = 12) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text[:n]


def _export_top_models(master: pd.DataFrame) -> None:
    sharpe_col = _metric_col(master)
    for freq in (1, 5):
        subset = master.copy()
        if "rebalance_freq" in subset.columns:
            subset = subset[subset["rebalance_freq"] == freq]
        if subset.empty:
            continue

        subset = subset.sort_values(sharpe_col, ascending=False).head(8)
        family_col = subset["model_family"] if "model_family" in subset.columns else pd.Series([""] * len(subset), index=subset.index)
        table = pd.DataFrame(
            {
                "Model": subset.apply(_model_label, axis=1),
                "Family": family_col.map(_family_label),
                "Rank IC": subset.get("prediction_rank_ic", pd.NA),
                "Sharpe": subset.get(sharpe_col, pd.NA),
                "Max DD": subset.get("portfolio_max_drawdown", pd.NA),
                "Final": subset.get("portfolio_final_value", pd.NA),
            }
        )
        _write_table(
            table,
            out_name=f"top_models_reb{freq}.tex",
            caption=f"Top models by Sharpe (rebalance\\_freq={freq}).",
            label=f"tab:top-models-reb{freq}",
        )


def _export_family_summary(family: pd.DataFrame) -> None:
    sharpe_col = _metric_col(family)
    category_col = "category" if "category" in family.columns else "model_family"

    subset = family.copy()
    keep = [
        category_col,
        "rebalance_freq",
        "run_tag",
        "prediction_rank_ic",
        sharpe_col,
        "portfolio_final_value",
    ]
    keep = [col for col in keep if col in subset.columns]
    subset = subset[keep]
    subset = subset.rename(
        columns={
            category_col: "Category",
            "rebalance_freq": "Rebalance",
            "run_tag": "Model",
            "prediction_rank_ic": "Rank IC",
            sharpe_col: "Sharpe",
            "portfolio_final_value": "Final",
        }
    )
    if "Category" in subset.columns:
        subset["Category"] = subset["Category"].map(_category_label)
    if "Model" in subset.columns:
        subset["Model"] = subset["Model"].map(_run_tag_to_label)
    if "Rebalance" in subset.columns:
        subset["Rebalance"] = subset["Rebalance"].map(_reb_to_label)

    _write_table(
        subset,
        out_name="family_summary.tex",
        caption="Best runs by model category and rebalance policy.",
        label="tab:family-summary",
    )


def _export_edge_ablation(edge: pd.DataFrame) -> None:
    sharpe_col = _metric_col(edge)
    keep = [
        "rebalance_freq",
        "edge_type",
        "model_count",
        "prediction_rank_ic",
        sharpe_col,
        "portfolio_max_drawdown",
    ]
    keep = [col for col in keep if col in edge.columns]

    subset = edge[keep].rename(
        columns={
            "rebalance_freq": "Rebalance",
            "edge_type": "Edge",
            "model_count": "N",
            "prediction_rank_ic": "Mean IC",
            sharpe_col: "Mean Sharpe",
            "portfolio_max_drawdown": "Mean MDD",
        }
    )
    if "Edge" in subset.columns:
        subset["Edge"] = subset["Edge"].map(_edge_label)
    if "Rebalance" in subset.columns:
        subset["Rebalance"] = subset["Rebalance"].map(_reb_to_label)

    _write_table(
        subset,
        out_name="edge_ablation.tex",
        caption="Edge-type ablation summary from thesis report output.",
        label="tab:edge-ablation",
    )


def _export_run_matrix(run_matrix: pd.DataFrame) -> None:
    keep = [
        "run_tag",
        "model_family",
        "model_name",
        "edge_type",
        "seed",
        "rebalance_freq",
        "split_id",
        "config_hash",
        "inclusion_status",
    ]
    keep = [c for c in keep if c in run_matrix.columns]
    subset = run_matrix[keep].copy()
    subset = subset.rename(
        columns={
            "run_tag": "Run",
            "model_family": "Family",
            "model_name": "Model",
            "edge_type": "Edge",
            "seed": "Seed",
            "rebalance_freq": "Reb",
            "split_id": "Split ID",
            "config_hash": "Config Hash",
            "inclusion_status": "Status",
        }
    )
    if "Family" in subset.columns:
        subset["Family"] = subset["Family"].map(_family_label)
    if "Edge" in subset.columns:
        subset["Edge"] = subset["Edge"].map(_edge_label)
    if "Model" in subset.columns:
        subset["Model"] = subset["Model"].map(_model_name_label)
    if "Run" in subset.columns:
        subset["Run"] = subset["Run"].map(_run_tag_to_label)
    if "Split ID" in subset.columns:
        subset["Split ID"] = subset["Split ID"].map(_short_hash)
    if "Config Hash" in subset.columns:
        subset["Config Hash"] = subset["Config Hash"].map(_short_hash)
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)

    _write_table(
        subset,
        out_name="run_matrix.tex",
        caption="Canonical tuned-all run matrix used by all thesis report summaries.",
        label="tab:run-matrix",
    )
    _write_table(
        subset,
        out_name="run_matrix_appendix.tex",
        caption="Canonical tuned-all run matrix (appendix copy for audit completeness).",
        label="tab:run-matrix-appendix",
    )


def _export_decision_ranking(decision: pd.DataFrame) -> None:
    keep = [
        "decision_rank",
        "run_tag",
        "rebalance_freq",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "prediction_rank_ic",
        "portfolio_max_drawdown",
        "portfolio_turnover",
        "portfolio_final_value",
        "ratio_final_value_vs_test_bh",
    ]
    keep = [c for c in keep if c in decision.columns]
    subset = decision[keep].copy()
    subset = subset.rename(
        columns={
            "decision_rank": "Rank",
            "run_tag": "Run",
            "rebalance_freq": "Reb",
            "portfolio_sharpe_annualized": "Sharpe",
            "portfolio_annualized_return": "Ann Return",
            "prediction_rank_ic": "Rank IC",
            "portfolio_max_drawdown": "Max DD",
            "portfolio_turnover": "Turnover",
            "portfolio_final_value": "Final",
            "ratio_final_value_vs_test_bh": "Final/Test BH",
        }
    )
    if "Run" in subset.columns:
        subset["Run"] = subset["Run"].map(_run_tag_to_label)
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    _write_longtable(
        subset,
        out_name="decision_ranking_full.tex",
        caption="Full deterministic decision ranking across canonical tuned-all runs.",
        label="tab:decision-ranking-full",
    )


def _export_master_comparison(master: pd.DataFrame) -> None:
    keep = [
        "run_tag",
        "model_family",
        "edge_type",
        "rebalance_freq",
        "prediction_rank_ic",
        "prediction_rmse",
        "prediction_mae",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_annualized_volatility",
        "portfolio_max_drawdown",
        "portfolio_turnover",
        "portfolio_final_value",
    ]
    keep = [c for c in keep if c in master.columns]
    subset = master[keep].copy()
    subset = subset.rename(
        columns={
            "run_tag": "Run",
            "model_family": "Family",
            "edge_type": "Edge",
            "rebalance_freq": "Reb",
            "prediction_rank_ic": "Rank IC",
            "prediction_rmse": "RMSE",
            "prediction_mae": "MAE",
            "portfolio_sharpe_annualized": "Sharpe",
            "portfolio_annualized_return": "Ann Return",
            "portfolio_annualized_volatility": "Ann Vol",
            "portfolio_max_drawdown": "Max DD",
            "portfolio_turnover": "Turnover",
            "portfolio_final_value": "Final",
        }
    )
    if "Run" in subset.columns:
        subset["Run"] = subset["Run"].map(_run_tag_to_label)
    if "Family" in subset.columns:
        subset["Family"] = subset["Family"].map(_family_label)
    if "Edge" in subset.columns:
        subset["Edge"] = subset["Edge"].map(_edge_label)
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)

    for freq in (1, 5):
        freq_df = subset[subset["Reb"] == freq].copy()
        if freq_df.empty:
            continue
        freq_df = freq_df.sort_values(["Sharpe", "Rank IC", "Turnover"], ascending=[False, False, True])
        _write_longtable(
            freq_df,
            out_name=f"master_comparison_reb{freq}_full.tex",
            caption=f"Full canonical master comparison for rebalance\\_freq={freq}.",
            label=f"tab:master-comparison-reb{freq}-full",
        )


def _export_runtime_summary(master: pd.DataFrame) -> None:
    required = {"model_family", "rebalance_freq", "runtime_train_seconds", "runtime_inference_seconds"}
    if not required.issubset(master.columns):
        return
    summary = (
        master.groupby(["model_family", "rebalance_freq"], as_index=False)[
            ["runtime_train_seconds", "runtime_inference_seconds"]
        ].agg(["mean", "median", "min", "max"])
    )
    summary.columns = [
        "Family",
        "Reb",
        "Train Mean (s)",
        "Train Median (s)",
        "Train Min (s)",
        "Train Max (s)",
        "Infer Mean (s)",
        "Infer Median (s)",
        "Infer Min (s)",
        "Infer Max (s)",
    ]
    summary["Family"] = summary["Family"].map(_family_label)
    summary = summary.sort_values(["Reb", "Train Mean (s)"], ascending=[True, False])
    _write_table(
        summary,
        out_name="runtime_summary.tex",
        caption="Runtime summary by model family and rebalance policy.",
        label="tab:runtime-summary",
    )


def _export_baseline_context(baseline: pd.DataFrame) -> None:
    keep = [
        "rebalance_freq",
        "global_start_date",
        "global_end_date",
        "global_final_value",
        "test_start_date",
        "test_end_date",
        "test_final_value",
    ]
    keep = [c for c in keep if c in baseline.columns]
    if not keep:
        return
    subset = baseline[keep].copy()
    subset = subset.rename(
        columns={
            "rebalance_freq": "Reb",
            "global_start_date": "Global Start",
            "global_end_date": "Global End",
            "global_final_value": "Global Final",
            "test_start_date": "Test Start",
            "test_end_date": "Test End",
            "test_final_value": "Test Final",
        }
    )
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    _write_table(
        subset,
        out_name="baseline_context.tex",
        caption="Buy-and-hold baseline context used for report-relative interpretation.",
        label="tab:baseline-context",
    )


def _export_baseline_policy_comparison(baseline_policy: pd.DataFrame) -> None:
    if baseline_policy.empty:
        return
    keep = [
        "strategy_name",
        "rebalance_freq",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_final_value",
        "test_start_date",
        "test_end_date",
    ]
    keep = [c for c in keep if c in baseline_policy.columns]
    if not keep:
        return
    subset = baseline_policy[keep].copy()
    subset = subset.rename(
        columns={
            "strategy_name": "Strategy",
            "rebalance_freq": "Reb",
            "portfolio_sharpe_annualized": "Sharpe",
            "portfolio_annualized_return": "Ann Return",
            "portfolio_max_drawdown": "Max DD",
            "portfolio_final_value": "Final",
            "test_start_date": "Test Start",
            "test_end_date": "Test End",
        }
    )
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    _write_table(
        subset,
        out_name="baseline_policy_comparison.tex",
        caption="Baseline benchmark metrics in the test window (Buy and hold fixed shares, Equal weight rebalanced all assets).",
        label="tab:baseline-policy-comparison",
    )


def _export_model_vs_baseline(master: pd.DataFrame, baseline_policy: pd.DataFrame) -> None:
    if master.empty or baseline_policy.empty:
        return
    sharpe_col = _metric_col(master)
    freqs = sorted(pd.to_numeric(master.get("rebalance_freq"), errors="coerce").dropna().astype(int).unique().tolist())
    for freq in freqs:
        model_subset = master[pd.to_numeric(master.get("rebalance_freq"), errors="coerce") == int(freq)].copy()
        if model_subset.empty:
            continue
        model_subset = model_subset.sort_values(sharpe_col, ascending=False).head(8)
        model_rows = pd.DataFrame(
            {
                "Strategy": model_subset.get("run_tag", pd.Series(dtype=object)).map(_run_tag_to_label),
                "Type": "Model",
                "Reb": int(freq),
                "Sharpe": model_subset.get(sharpe_col, pd.NA),
                "Ann Return": model_subset.get("portfolio_annualized_return", pd.NA),
                "Max DD": model_subset.get("portfolio_max_drawdown", pd.NA),
                "Final": model_subset.get("portfolio_final_value", pd.NA),
            }
        )
        baseline_subset = baseline_policy[
            pd.to_numeric(baseline_policy.get("rebalance_freq"), errors="coerce") == int(freq)
        ].copy()
        baseline_rows = pd.DataFrame(
            {
                "Strategy": baseline_subset.get("strategy_name", pd.Series(dtype=object)),
                "Type": "Baseline",
                "Reb": int(freq),
                "Sharpe": baseline_subset.get("portfolio_sharpe_annualized", pd.NA),
                "Ann Return": baseline_subset.get("portfolio_annualized_return", pd.NA),
                "Max DD": baseline_subset.get("portfolio_max_drawdown", pd.NA),
                "Final": baseline_subset.get("portfolio_final_value", pd.NA),
            }
        )
        table = pd.concat([model_rows, baseline_rows], ignore_index=True)
        if "Reb" in table.columns:
            table["Reb"] = table["Reb"].map(_reb_to_label)
        _write_table(
            table,
            out_name=f"model_vs_baseline_reb{int(freq)}.tex",
            caption=f"Top model runs and baseline benchmarks in one view (rebalance\\_freq={int(freq)}).",
            label=f"tab:model-vs-baseline-reb{int(freq)}",
        )


def _export_alpha_vs_equal_weight(alpha_df: pd.DataFrame) -> None:
    if alpha_df.empty:
        _write_no_data_table(
            out_name="alpha_vs_equal_weight.tex",
            caption="Active-vs-equal-weight diagnostics (arithmetic active returns, annualized with 252 trading days).",
            label="tab:alpha-vs-equal-weight",
        )
        return
    subset = alpha_df.copy()
    keep = [
        "run_tag",
        "rebalance_freq",
        "active_return_annualized",
        "tracking_error_annualized",
        "information_ratio_annualized",
        "relative_final_value",
        "relative_max_drawdown",
    ]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.sort_values(
        ["rebalance_freq", "information_ratio_annualized", "active_return_annualized"],
        ascending=[True, False, False],
    )
    subset = subset.groupby("rebalance_freq", as_index=False).head(8)
    subset = subset.rename(
        columns={
            "run_tag": "Run",
            "rebalance_freq": "Reb",
            "active_return_annualized": "Active Return",
            "tracking_error_annualized": "Tracking Error",
            "information_ratio_annualized": "IR",
            "relative_final_value": "Rel Final",
            "relative_max_drawdown": "Rel Max DD",
        }
    )
    if "Run" in subset.columns:
        subset["Run"] = subset["Run"].map(_run_tag_to_label)
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    _write_table(
        subset,
        out_name="alpha_vs_equal_weight.tex",
        caption="Active-vs-equal-weight diagnostics (arithmetic active returns, annualized with 252 trading days).",
        label="tab:alpha-vs-equal-weight",
    )


def _export_long_short_top3_bottom3(long_short: pd.DataFrame) -> None:
    if long_short.empty:
        _write_no_data_table(
            out_name="long_short_top3_bottom3.tex",
            caption="Market-neutral long-short results (top 3 long, bottom 3 short; gross, cost bps = 0).",
            label="tab:long-short-top3-bottom3",
        )
        return
    subset = long_short.copy()
    keep = [
        "run_tag",
        "rebalance_freq",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_final_value",
        "portfolio_turnover",
        "long_k",
        "short_k",
        "long_leg_gross",
        "short_leg_gross",
    ]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.sort_values(["rebalance_freq", "portfolio_sharpe_annualized"], ascending=[True, False]).head(12)
    subset = subset.rename(
        columns={
            "run_tag": "Run",
            "rebalance_freq": "Reb",
            "portfolio_sharpe_annualized": "Sharpe",
            "portfolio_annualized_return": "Ann Return",
            "portfolio_max_drawdown": "Max DD",
            "portfolio_final_value": "Final",
            "portfolio_turnover": "Turnover",
            "long_k": "Long K",
            "short_k": "Short K",
            "long_leg_gross": "Long Sum",
            "short_leg_gross": "Short Sum",
        }
    )
    if "Run" in subset.columns:
        subset["Run"] = subset["Run"].map(_run_tag_to_label)
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    _write_table(
        subset,
        out_name="long_short_top3_bottom3.tex",
        caption="Market-neutral long-short results (top 3 long, bottom 3 short; gross, cost bps = 0).",
        label="tab:long-short-top3-bottom3",
    )


def _export_cost_sensitivity_summary(cost_summary: pd.DataFrame) -> None:
    if cost_summary.empty:
        _write_no_data_table(
            out_name="cost_sensitivity_summary.tex",
            caption="Cost sensitivity summary for long-only and long-short strategies (0/5/10 bps).",
            label="tab:cost-sensitivity-summary",
        )
        return
    subset = cost_summary.copy()
    keep = ["strategy", "cost_bps", "cost_label", "sharpe_mean", "ann_return_mean", "max_dd_mean", "turnover_mean", "n_runs"]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.rename(
        columns={
            "strategy": "Strategy",
            "cost_bps": "Cost (bps)",
            "cost_label": "Label",
            "sharpe_mean": "Mean Sharpe",
            "ann_return_mean": "Mean Ann Return",
            "max_dd_mean": "Mean Max DD",
            "turnover_mean": "Mean Turnover",
            "n_runs": "N",
        }
    )
    _write_table(
        subset,
        out_name="cost_sensitivity_summary.tex",
        caption="Cost sensitivity summary for long-only and long-short strategies (0/5/10 bps).",
        label="tab:cost-sensitivity-summary",
    )


def _export_audit_status(audit_status: pd.DataFrame) -> None:
    if audit_status.empty:
        _write_no_data_table(
            out_name="audit_status.tex",
            caption="Fail-fast audit status for equal-weight rebalance integrity and graph time-awareness.",
            label="tab:audit-status",
        )
        return
    subset = audit_status.copy()
    keep = ["audit_name", "status", "fail_rows", "warning_rows", "detail"]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.rename(
        columns={
            "audit_name": "Audit",
            "status": "Status",
            "fail_rows": "Fails",
            "warning_rows": "Warnings",
            "detail": "Detail",
        }
    )
    _write_table(
        subset,
        out_name="audit_status.tex",
        caption="Fail-fast audit status for equal-weight rebalance integrity and graph time-awareness.",
        label="tab:audit-status",
    )


def _export_monthly_rebalance_subset(monthly: pd.DataFrame) -> None:
    if monthly.empty:
        _write_no_data_table(
            out_name="monthly_rebalance_subset.tex",
            caption="Subset comparison for monthly rebalancing extension (rebalance\\_freq=21) with gross costs.",
            label="tab:monthly-rebalance-subset",
        )
        return
    subset = monthly.copy()
    keep = [
        "strategy_name",
        "strategy_kind",
        "rebalance_freq",
        "cost_label",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_final_value",
        "portfolio_turnover",
    ]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.rename(
        columns={
            "strategy_name": "Strategy",
            "strategy_kind": "Kind",
            "rebalance_freq": "Reb",
            "cost_label": "Cost",
            "portfolio_sharpe_annualized": "Sharpe",
            "portfolio_annualized_return": "Ann Return",
            "portfolio_max_drawdown": "Max DD",
            "portfolio_final_value": "Final",
            "portfolio_turnover": "Turnover",
        }
    )
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    _write_table(
        subset,
        out_name="monthly_rebalance_subset.tex",
        caption="Subset comparison for monthly rebalancing extension (rebalance\\_freq=21) with gross costs.",
        label="tab:monthly-rebalance-subset",
    )


def _export_lookback_sensitivity_subset(lookback: pd.DataFrame) -> None:
    if lookback.empty:
        _write_no_data_table(
            out_name="lookback_sensitivity_subset.tex",
            caption="Lookback sensitivity subset (14/30/60); missing entries require retraining.",
            label="tab:lookback-sensitivity-subset",
        )
        return
    subset = lookback.copy()
    if "run_tag" not in subset.columns and "run_prefix" in subset.columns:
        subset["run_tag"] = subset["run_prefix"]
    keep = [
        "run_tag",
        "model_name",
        "edge_type",
        "rebalance_freq",
        "lookback_window",
        "status",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_final_value",
    ]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.rename(
        columns={
            "run_tag": "Run",
            "model_name": "Model",
            "edge_type": "Edge",
            "rebalance_freq": "Reb",
            "lookback_window": "Lookback",
            "status": "Status",
            "portfolio_sharpe_annualized": "Sharpe",
            "portfolio_annualized_return": "Ann Return",
            "portfolio_max_drawdown": "Max DD",
            "portfolio_final_value": "Final",
        }
    )
    if "Run" in subset.columns:
        subset["Run"] = subset["Run"].map(_run_tag_to_label)
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    _write_table(
        subset,
        out_name="lookback_sensitivity_subset.tex",
        caption="Lookback sensitivity subset (14/30/60); missing entries require retraining.",
        label="tab:lookback-sensitivity-subset",
    )


def _export_corr_window_sensitivity_subset(corr_window: pd.DataFrame) -> None:
    if corr_window.empty:
        _write_no_data_table(
            out_name="corr_window_sensitivity_subset.tex",
            caption="Correlation-window sensitivity subset (30/60/120); missing entries require retraining.",
            label="tab:corr-window-sensitivity-subset",
        )
        return
    subset = corr_window.copy()
    if "run_tag" not in subset.columns and "run_prefix" in subset.columns:
        subset["run_tag"] = subset["run_prefix"]
    keep = [
        "run_tag",
        "model_name",
        "edge_type",
        "rebalance_freq",
        "corr_window",
        "status",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_final_value",
    ]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.rename(
        columns={
            "run_tag": "Run",
            "model_name": "Model",
            "edge_type": "Edge",
            "rebalance_freq": "Reb",
            "corr_window": "Corr Window",
            "status": "Status",
            "portfolio_sharpe_annualized": "Sharpe",
            "portfolio_annualized_return": "Ann Return",
            "portfolio_max_drawdown": "Max DD",
            "portfolio_final_value": "Final",
        }
    )
    if "Run" in subset.columns:
        subset["Run"] = subset["Run"].map(_run_tag_to_label)
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    _write_table(
        subset,
        out_name="corr_window_sensitivity_subset.tex",
        caption="Correlation-window sensitivity subset (30/60/120); missing entries require retraining.",
        label="tab:corr-window-sensitivity-subset",
    )


def _export_rolling_cv_table(rolling_cv: pd.DataFrame) -> None:
    if rolling_cv.empty:
        _write_no_data_table(
            out_name="rolling_cv_table.tex",
            caption="Rolling walk-forward CV summary (test years 2020--2024, rebalance\\_freq=5, gross costs).",
            label="tab:rolling-cv-table",
        )
        return
    subset = rolling_cv.copy()
    keep = [
        "entry_name",
        "strategy_kind",
        "summary_scope",
        "fold_label",
        "rebalance_freq",
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_turnover",
        "n_folds",
    ]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.rename(
        columns={
            "entry_name": "Entry",
            "strategy_kind": "Kind",
            "summary_scope": "Scope",
            "fold_label": "Fold",
            "rebalance_freq": "Reb",
            "portfolio_sharpe_annualized": "Sharpe",
            "portfolio_annualized_return": "Ann Return",
            "portfolio_max_drawdown": "Max DD",
            "portfolio_turnover": "Turnover",
            "n_folds": "N Folds",
        }
    )
    if "Reb" in subset.columns:
        subset["Reb"] = subset["Reb"].map(_reb_to_label)
    subset = subset.sort_values(["Scope", "Kind", "Entry", "Fold"], kind="mergesort").reset_index(drop=True)
    _write_table(
        subset,
        out_name="rolling_cv_table.tex",
        caption="Rolling walk-forward CV summary (test years 2020--2024, rebalance\\_freq=5, gross costs).",
        label="tab:rolling-cv-table",
    )


def _export_professor_main_results_table(df: pd.DataFrame) -> None:
    if df.empty:
        _write_no_data_table(
            out_name="professor_main_results_table.tex",
            caption="Professor-facing main table with baselines and best learned models.",
            label="tab:professor-main-results",
        )
        return
    subset = df.copy()
    keep = [
        "strategy_name",
        "strategy_label",
        "type",
        "rebalance_label",
        "final_value",
        "annual_return",
        "annual_vol",
        "sharpe_annualized",
        "max_drawdown",
        "turnover",
    ]
    keep = [c for c in keep if c in subset.columns]
    subset = subset[keep].copy()
    subset = subset.rename(
        columns={
            "strategy_name": "Strategy",
            "strategy_label": "Policy",
            "type": "Type",
            "rebalance_label": "Rebalance",
            "final_value": "Final",
            "annual_return": "Ann Return",
            "annual_vol": "Ann Vol",
            "sharpe_annualized": "Sharpe",
            "max_drawdown": "Max DD",
            "turnover": "Turnover",
        }
    )
    _write_table(
        subset,
        out_name="professor_main_results_table.tex",
        caption="Professor-facing main table with Buy and hold, Equal weight, and best learned models.",
        label="tab:professor-main-results",
    )


def main() -> None:
    master = _load_csv("master_comparison.csv")
    family = _load_csv("family_summary.csv")
    edge = _load_csv("edge_ablation_summary.csv")
    run_matrix = _load_csv("run_matrix.csv")
    decision = _load_csv("decision_ranking.csv")
    baseline = _load_csv("baseline_context.csv")
    baseline_policy = _load_csv_optional("baseline_policy_comparison.csv")
    alpha_vs_eqw = _load_csv_optional("alpha_vs_equal_weight.csv")
    long_short = _load_csv_optional("long_short_top3_bottom3.csv")
    cost_summary = _load_csv_optional("cost_sensitivity_summary.csv")
    audit_status = _load_csv_optional("audit_status.csv")
    monthly_subset = _load_csv_optional("monthly_rebalance_subset.csv")
    lookback_subset = _load_csv_optional("lookback_sensitivity_subset.csv")
    corr_window_subset = _load_csv_optional("corr_window_sensitivity_subset.csv")
    rolling_cv_summary = _load_csv_optional("rolling_cv_summary.csv")
    professor_main = _load_csv_optional("professor_main_results_table.csv")

    _export_top_models(master)
    _export_family_summary(family)
    _export_edge_ablation(edge)
    _export_run_matrix(run_matrix)
    _export_decision_ranking(decision)
    _export_master_comparison(master)
    _export_runtime_summary(master)
    _export_baseline_context(baseline)
    _export_baseline_policy_comparison(baseline_policy)
    _export_model_vs_baseline(master, baseline_policy)
    _export_alpha_vs_equal_weight(alpha_vs_eqw)
    _export_long_short_top3_bottom3(long_short)
    _export_cost_sensitivity_summary(cost_summary)
    _export_audit_status(audit_status)
    _export_monthly_rebalance_subset(monthly_subset)
    _export_lookback_sensitivity_subset(lookback_subset)
    _export_corr_window_sensitivity_subset(corr_window_subset)
    _export_rolling_cv_table(rolling_cv_summary)
    _export_professor_main_results_table(professor_main)

    print(f"Using report directory: {REPORT_DIR}")
    print(f"Wrote LaTeX tables to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
