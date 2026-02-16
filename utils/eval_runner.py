from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .artifacts import OutputDirs
from .backtest import backtest_long_only
from .baseline import (
    BASELINE_VERSION,
    get_buy_and_hold_for_window,
    get_equal_weight_for_window,
    write_baseline_curve_csv,
)
from .metrics import rank_ic, hit_rate, sharpe_ratio
from .plot import (
    plot_daily_ic,
    plot_ic_hist,
    plot_equity_curve,
    plot_equity_comparison,
)
from .predictions import sanitize_predictions, prediction_row_stats
from .protocol import protocol_from_config
from .results import build_experiment_result, save_experiment_result


def _compute_daily_ic_hit(pred_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    for d, g in pred_df.groupby("date"):
        ic = rank_ic(g["pred"], g["realized_ret"])
        hit = hit_rate(g["pred"], g["realized_ret"], top_k=top_k)
        rows.append({"date": pd.to_datetime(d), "ic": ic, "hit": hit})
    if not rows:
        return pd.DataFrame(columns=["date", "ic", "hit"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _write_series_with_legacy_copy(series: pd.Series, canonical_path: Path, legacy_path: Optional[Path] = None) -> None:
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(canonical_path, header=["value"])
    if legacy_path is not None:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(legacy_path, header=["value"])


def evaluate_and_report(
    *,
    config: dict,
    pred_df: pd.DataFrame,
    out_dirs: OutputDirs,
    run_name: str,
    model_name: str,
    model_family: str,
    edge_type: str,
    directed: bool,
    graph_window: str,
    train_seconds: float,
    inference_seconds: float,
    bh_full_curve=None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pred_df = sanitize_predictions(pred_df, strict_unique=True)
    pred_stats = prediction_row_stats(pred_df)

    protocol = protocol_from_config(config, baseline_version=BASELINE_VERSION)
    policies = list(protocol.backtest_policies)
    primary_freq = int(protocol.primary_rebalance_freq)

    out_dir = out_dirs.canonical
    legacy_dir = out_dirs.legacy

    # canonical prediction artifact
    pred_path = out_dir / f"{run_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    if legacy_dir is not None:
        pred_df.to_csv(legacy_dir / f"{model_name}_predictions.csv", index=False)

    daily_metrics_ic = _compute_daily_ic_hit(pred_df, top_k=int(config["evaluation"]["top_k"]))
    plot_daily_ic(daily_metrics_ic, out_dir / f"{run_name}_ic_timeseries.png")
    plot_ic_hist(daily_metrics_ic, out_dir / f"{run_name}_ic_histogram.png")
    if legacy_dir is not None:
        plot_daily_ic(daily_metrics_ic, legacy_dir / f"{model_name}_ic_timeseries.png")
        plot_ic_hist(daily_metrics_ic, legacy_dir / f"{model_name}_ic_histogram.png")

    start_d, end_d = pred_df["date"].min(), pred_df["date"].max()
    eq_bh, _, stats_bh = get_buy_and_hold_for_window(
        config,
        start_date=start_d,
        end_date=end_d,
        rebuild=config.get("cache", {}).get("rebuild", False),
        eq_full=bh_full_curve,
    )

    # Buy-and-hold curve is policy-independent.
    write_baseline_curve_csv(eq_bh, out_dir / "buy_and_hold_equity_curve.csv", "buy_and_hold")
    if legacy_dir is not None:
        write_baseline_curve_csv(eq_bh, legacy_dir / "buy_and_hold_equity_curve.csv", "buy_and_hold")

    results_path = Path(config.get("evaluation", {}).get("results_path", "results/results.jsonl"))
    stats_by_freq: Dict[int, Dict[str, Any]] = {}
    rolling_window = int(config.get("evaluation", {}).get("rolling_window", 63))

    for freq in policies:
        equity_curve, daily_ret, stats = backtest_long_only(
            pred_df,
            top_k=int(config["evaluation"]["top_k"]),
            transaction_cost_bps=float(config["evaluation"]["transaction_cost_bps"]),
            risk_free_rate=float(config["evaluation"]["risk_free_rate"]),
            rebalance_freq=int(freq),
        )

        eq_series = equity_curve.copy()
        eq_series.index = pd.to_datetime(eq_series.index)
        daily_ret_series = pd.Series(daily_ret, index=eq_series.index[: len(daily_ret)])
        dd = eq_series / eq_series.cummax() - 1.0

        metrics = daily_metrics_ic.merge(
            pd.DataFrame(
                {
                    "date": daily_ret_series.index,
                    "daily_return": daily_ret_series.values,
                    "drawdown": dd.reindex(daily_ret_series.index).values,
                }
            ),
            on="date",
            how="left",
        )

        metrics_path = out_dir / f"{run_name}_daily_metrics_reb{freq}.csv"
        metrics.to_csv(metrics_path, index=False)
        if legacy_dir is not None:
            metrics.to_csv(legacy_dir / f"{model_name}_daily_metrics_reb{freq}.csv", index=False)

        eq_csv = out_dir / f"{run_name}_equity_curve_reb{freq}.csv"
        _write_series_with_legacy_copy(
            eq_series,
            eq_csv,
            legacy_dir / f"{model_name}_equity_curve_reb{freq}.csv" if legacy_dir is not None else None,
        )

        eq_png = out_dir / f"{run_name}_equity_curve_reb{freq}.png"
        plot_equity_curve(eq_series, f"{run_name} long only (reb={freq})", eq_png)
        if legacy_dir is not None:
            plot_equity_curve(eq_series, f"{model_name.upper()} long only (reb={freq})", legacy_dir / f"{model_name}_equity_curve_reb{freq}.png")

        eq_eqw, _, stats_eqw = get_equal_weight_for_window(
            config,
            start_date=start_d,
            end_date=end_d,
            rebalance_freq=int(freq),
            rebuild=config.get("cache", {}).get("rebuild", False),
        )
        write_baseline_curve_csv(eq_eqw, out_dir / f"equal_weight_equity_curve_reb{freq}.csv", "equal_weight")
        if legacy_dir is not None:
            write_baseline_curve_csv(eq_eqw, legacy_dir / f"equal_weight_equity_curve_reb{freq}.csv", "equal_weight")

        comp_bh = out_dir / f"{run_name}_vs_buy_and_hold_reb{freq}.png"
        comp_eqw = out_dir / f"{run_name}_vs_equal_weight_reb{freq}.png"
        plot_equity_comparison(eq_series, eq_bh, f"{run_name} vs Buy and Hold (reb={freq})", comp_bh)
        plot_equity_comparison(eq_series, eq_eqw, f"{run_name} vs Equal Weight (reb={freq})", comp_eqw)
        if legacy_dir is not None:
            plot_equity_comparison(eq_series, eq_bh, f"{model_name.upper()} vs Buy and Hold (reb={freq})", legacy_dir / f"{model_name}_vs_buy_and_hold_reb{freq}.png")
            plot_equity_comparison(eq_series, eq_eqw, f"{model_name.upper()} vs Equal Weight (reb={freq})", legacy_dir / f"{model_name}_vs_equal_weight_reb{freq}.png")

        ic_series = metrics.set_index("date")["ic"].dropna()
        rolling = pd.DataFrame({"date": metrics["date"]}).copy()
        rolling["rolling_sharpe"] = daily_ret_series.rolling(rolling_window).apply(
            lambda x: sharpe_ratio(x, float(config["evaluation"]["risk_free_rate"])) if len(x) > 1 else np.nan,
            raw=False,
        ).values
        rolling["rolling_ic_mean"] = ic_series.reindex(metrics["date"]).rolling(rolling_window).mean().values
        rolling.to_csv(out_dir / f"{run_name}_rolling_metrics_reb{freq}.csv", index=False)
        if legacy_dir is not None:
            rolling.to_csv(legacy_dir / f"{model_name}_rolling_metrics_reb{freq}.csv", index=False)

        # backward-compatible primary files without policy suffix
        if int(freq) == primary_freq:
            metrics.to_csv(out_dir / f"{run_name}_daily_metrics.csv", index=False)
            _write_series_with_legacy_copy(
                eq_series,
                out_dir / f"{run_name}_equity_curve.csv",
                legacy_dir / f"{model_name}_equity_curve.csv" if legacy_dir is not None else None,
            )
            plot_equity_curve(eq_series, f"{run_name} long only", out_dir / f"{run_name}_equity_curve.png")
            plot_equity_curve(eq_bh, "Buy and Hold", out_dir / f"{run_name}_buy_and_hold_equity_curve.png")
            write_baseline_curve_csv(eq_eqw, out_dir / "equal_weight_equity_curve.csv", "equal_weight")
            plot_equity_comparison(eq_series, eq_bh, f"{run_name}: Model vs Buy & Hold", out_dir / f"{run_name}_vs_buy_and_hold.png")
            rolling.to_csv(out_dir / f"{run_name}_rolling_metrics.csv", index=False)
            if legacy_dir is not None:
                metrics.to_csv(legacy_dir / f"{model_name}_daily_metrics.csv", index=False)
                plot_equity_curve(eq_series, f"{model_name.upper()} long only", legacy_dir / f"{model_name}_equity_curve.png")
                plot_equity_curve(eq_bh, "Buy and Hold", legacy_dir / f"{model_name}_buy_and_hold_equity_curve.png")
                write_baseline_curve_csv(eq_eqw, legacy_dir / "equal_weight_equity_curve.csv", "equal_weight")
                plot_equity_comparison(eq_series, eq_bh, f"{model_name.upper()} vs Buy and Hold", legacy_dir / f"{model_name}_equity_comparison.png")
                rolling.to_csv(legacy_dir / f"{model_name}_rolling_metrics.csv", index=False)

        ic_mean = float(ic_series.mean()) if not ic_series.empty else float("nan")
        ic_std = float(ic_series.std()) if not ic_series.empty else float("nan")
        ic_tstat = float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_series.size > 1 and ic_std > 0 else float("nan")
        vol = float(np.std(daily_ret_series.values) * np.sqrt(252)) if not daily_ret_series.empty else float("nan")
        max_dd = float(dd.min()) if not dd.empty else float("nan")

        stats_by_freq[int(freq)] = {
            "stats": stats,
            "buy_and_hold_stats": stats_bh,
            "equal_weight_stats": stats_eqw,
            "ic_mean": ic_mean,
            "ic_tstat": ic_tstat,
            "volatility": vol,
            "max_drawdown": max_dd,
            "daily_metrics_path": str(metrics_path),
        }

        result = build_experiment_result(
            config,
            model_name=model_name,
            model_family=model_family,
            edge_type=edge_type,
            directed=bool(directed),
            graph_window=graph_window,
            pred_df=pred_df,
            daily_metrics=metrics,
            stats=stats,
            train_seconds=float(train_seconds),
            inference_seconds=float(inference_seconds),
            protocol_fields=protocol.as_result_fields(rebalance_freq=int(freq)),
            prediction_rows=pred_stats["prediction_rows"],
            prediction_unique_pairs=pred_stats["prediction_unique_pairs"],
            run_tag=config.get("experiment_name", run_name),
            out_dir=str(out_dir),
            artifact_prefix=run_name,
        )
        save_experiment_result(result, results_path)

    primary_stats = stats_by_freq[primary_freq]
    summary: Dict[str, Any] = {
        "run_tag": config.get("experiment_name", run_name),
        "model_type": model_name,
        "protocol_version": protocol.protocol_version,
        "target_policy_hash": protocol.target_policy_hash,
        "stats": primary_stats["stats"],
        "buy_and_hold_stats": primary_stats["buy_and_hold_stats"],
        "equal_weight_stats": primary_stats["equal_weight_stats"],
        "ic_mean": primary_stats["ic_mean"],
        "ic_tstat": primary_stats["ic_tstat"],
        "volatility": primary_stats["volatility"],
        "max_drawdown": primary_stats["max_drawdown"],
        "stats_by_rebalance_freq": {str(k): v for k, v in stats_by_freq.items()},
        "prediction_rows": pred_stats["prediction_rows"],
        "prediction_unique_pairs": pred_stats["prediction_unique_pairs"],
    }
    if extra_summary:
        summary.update(extra_summary)

    run_tag = config.get("experiment_name", run_name)
    with (out_dir / f"{run_tag}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    if run_tag != run_name:
        with (out_dir / f"{run_name}_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

    if legacy_dir is not None:
        with (legacy_dir / f"{run_tag}_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

    return summary
