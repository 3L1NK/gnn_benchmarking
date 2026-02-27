import json

import pandas as pd

from scripts.generate_thesis_report import generate_reports


def _write_curve(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values, columns=["date", "value"]).to_csv(path, index=False)


def _result_row(*, run_tag: str, run_key: str, freq: int, out_dir: str):
    return {
        "experiment_id": f"{run_tag}_{freq}",
        "timestamp": "2026-01-01T00:00:00",
        "seed": 42,
        "model_name": "xgb_raw",
        "model_family": "xgboost",
        "edge_type": "none",
        "directed": False,
        "graph_window": "",
        "target_type": "regression",
        "target_horizon": 1,
        "lookback_window": 60,
        "protocol_version": "v1_thesis_core",
        "split_train_end": "2015-12-31",
        "split_val_start": "2016-01-01",
        "split_test_start": "2020-01-01",
        "rebalance_freq": freq,
        "baseline_version": "price_bh_v2_eqw_v2",
        "target_policy_hash": "abc123",
        "split_id": "split123",
        "config_hash": "cfg123",
        "run_key": run_key,
        "prediction_rows": 2,
        "prediction_unique_pairs": 2,
        "prediction_rmse": 0.1,
        "prediction_mae": 0.08,
        "prediction_rank_ic": 0.01,
        "portfolio_final_value": 1.4 if freq == 1 else 1.7,
        "portfolio_cumulative_return": 0.4 if freq == 1 else 0.7,
        "portfolio_annualized_return": 0.10 if freq == 1 else 0.13,
        "portfolio_annualized_volatility": 0.2,
        "portfolio_sharpe": 0.03 if freq == 1 else 0.04,
        "portfolio_sharpe_daily": 0.03 if freq == 1 else 0.04,
        "portfolio_sharpe_annualized": 0.50 if freq == 1 else 0.64,
        "portfolio_sortino_annualized": 0.9,
        "portfolio_max_drawdown": -0.3 if freq == 1 else -0.28,
        "portfolio_turnover": 0.4 if freq == 1 else 0.14,
        "runtime_train_seconds": 1.0,
        "runtime_inference_seconds": 0.1,
        "run_tag": run_tag,
        "out_dir": out_dir,
        "artifact_prefix": "run_a",
    }


def test_generate_reports_writes_baseline_policy_comparison(tmp_path):
    exp_dir = tmp_path / "experiments" / "run_a"
    report_dir = tmp_path / "report"
    results_path = tmp_path / "results.jsonl"

    _write_curve(exp_dir / "run_a_equity_curve_reb1.csv", [["2020-01-02", 1.0], ["2024-12-27", 1.4]])
    _write_curve(exp_dir / "run_a_equity_curve_reb5.csv", [["2020-01-02", 1.0], ["2024-12-27", 1.7]])
    _write_curve(exp_dir / "buy_and_hold_equity_curve.csv", [["2020-01-02", 1.0], ["2024-12-27", 5.049]])
    _write_curve(exp_dir / "equal_weight_equity_curve_reb1.csv", [["2020-01-02", 1.0], ["2024-12-27", 2.1]])
    _write_curve(exp_dir / "equal_weight_equity_curve_reb5.csv", [["2020-01-02", 1.0], ["2024-12-27", 2.6]])

    pd.DataFrame(
        {
            "date": ["2020-01-02", "2024-12-27"],
            "ic": [0.01, 0.02],
            "hit": [0.5, 1.0],
            "daily_return": [0.01, 0.02],
            "drawdown": [0.0, -0.1],
        }
    ).to_csv(exp_dir / "run_a_daily_metrics_reb1.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2020-01-02", "2024-12-27"],
            "ic": [0.01, 0.02],
            "hit": [0.5, 1.0],
            "daily_return": [0.01, 0.02],
            "drawdown": [0.0, -0.1],
        }
    ).to_csv(exp_dir / "run_a_daily_metrics_reb5.csv", index=False)

    summary = {
        "baseline_context": {
            "test_window_buy_and_hold": {
                "start_date": "2020-01-02",
                "end_date": "2024-12-27",
                "final_value": 5.049,
                "rebased": True,
            }
        },
        "stats_by_rebalance_freq": {
            "1": {
                "buy_and_hold_stats": {
                    "final_value": 5.049,
                    "annualized_return": 0.35,
                    "max_drawdown": -0.32,
                    "sharpe_annualized": 1.05,
                },
                "equal_weight_stats": {
                    "final_value": 2.1,
                    "annualized_return": 0.16,
                    "max_drawdown": -0.28,
                    "sharpe_annualized": 0.62,
                },
            },
            "5": {
                "buy_and_hold_stats": {
                    "final_value": 5.049,
                    "annualized_return": 0.35,
                    "max_drawdown": -0.32,
                    "sharpe_annualized": 1.05,
                },
                "equal_weight_stats": {
                    "final_value": 2.6,
                    "annualized_return": 0.20,
                    "max_drawdown": -0.25,
                    "sharpe_annualized": 0.71,
                },
            },
        },
    }
    (exp_dir / "run_a_summary.json").write_text(json.dumps(summary))

    rows = [
        _result_row(run_tag="run_a", run_key="rk1", freq=1, out_dir=str(exp_dir)),
        _result_row(run_tag="run_a", run_key="rk5", freq=5, out_dir=str(exp_dir)),
    ]
    pd.DataFrame(rows).to_json(results_path, orient="records", lines=True)

    generate_reports(results_path, report_dir, expected_runs=2)

    baseline = pd.read_csv(report_dir / "baseline_policy_comparison.csv")
    assert len(baseline) == 4
    assert set(baseline["strategy_name"].tolist()) == {
        "Buy and hold (fixed shares)",
        "Equal weight (rebalanced, all assets)",
    }

    eqw_reb5 = baseline[
        (baseline["strategy_name"] == "Equal weight (rebalanced, all assets)") & (baseline["rebalance_freq"] == 5)
    ].iloc[0]
    assert abs(float(eqw_reb5["portfolio_sharpe_annualized"]) - 0.71) < 1e-9
    assert abs(float(eqw_reb5["portfolio_final_value"]) - 2.6) < 1e-9
