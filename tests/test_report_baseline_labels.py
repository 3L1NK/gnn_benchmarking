import json

import matplotlib
import matplotlib.axes
import pandas as pd

matplotlib.use("Agg", force=True)

from scripts.generate_thesis_report import generate_reports


def _write_curve(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values, columns=["date", "value"]).to_csv(path, index=False)


def test_report_outputs_rebased_label_and_window_dates(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments" / "run_a"
    out_dir = tmp_path / "report"
    results_path = tmp_path / "results.jsonl"

    _write_curve(
        exp_dir / "gcn_equity_curve_reb1.csv",
        [["2020-01-02", 1.0], ["2024-12-27", 1.8]],
    )
    _write_curve(
        exp_dir / "buy_and_hold_equity_curve.csv",
        [["2020-01-02", 1.0], ["2024-12-27", 5.049]],
    )
    _write_curve(
        exp_dir / "equal_weight_equity_curve_reb1.csv",
        [["2020-01-02", 1.0], ["2024-12-27", 2.0]],
    )
    pd.DataFrame(
        {
            "date": ["2020-01-02", "2024-12-27"],
            "ic": [0.01, 0.02],
            "hit": [0.5, 1.0],
            "daily_return": [0.01, 0.02],
            "drawdown": [0.0, -0.1],
        }
    ).to_csv(exp_dir / "gcn_daily_metrics_reb1.csv", index=False)

    summary_payload = {
        "baseline_context": {
            "global_buy_and_hold": {
                "start_date": "2000-03-29",
                "end_date": "2024-12-30",
                "final_value": 51.178,
            },
            "test_window_buy_and_hold": {
                "start_date": "2020-01-02",
                "end_date": "2024-12-27",
                "final_value": 5.049,
                "rebased": True,
            },
        },
        "buy_and_hold_stats": {
            "annualized_return": 0.35,
        },
    }
    (exp_dir / "run_a_summary.json").write_text(json.dumps(summary_payload))

    row = {
        "experiment_id": "exp_1",
        "timestamp": "2026-01-01T00:00:00",
        "seed": 42,
        "model_name": "gcn",
        "model_family": "gnn",
        "edge_type": "corr",
        "directed": False,
        "graph_window": "2000-01-01..2019-12-31",
        "target_type": "regression",
        "target_horizon": 1,
        "lookback_window": 60,
        "protocol_version": "v1_thesis_core",
        "split_train_end": "2015-12-31",
        "split_val_start": "2016-01-01",
        "split_test_start": "2020-01-01",
        "rebalance_freq": 1,
        "baseline_version": "price_bh_v2_eqw_v1",
        "target_policy_hash": "abc123",
        "prediction_rows": 4,
        "prediction_unique_pairs": 4,
        "prediction_rmse": 0.1,
        "prediction_mae": 0.08,
        "prediction_rank_ic": 0.02,
        "portfolio_final_value": 1.8,
        "portfolio_cumulative_return": 0.8,
        "portfolio_annualized_return": 0.15,
        "portfolio_annualized_volatility": 0.2,
        "portfolio_sharpe": 0.3,
        "portfolio_sharpe_daily": 0.3,
        "portfolio_sharpe_annualized": 0.3 * (252.0 ** 0.5),
        "portfolio_sortino_annualized": 0.8,
        "portfolio_max_drawdown": -0.2,
        "portfolio_turnover": 0.25,
        "runtime_train_seconds": 1.0,
        "runtime_inference_seconds": 0.1,
        "run_tag": "run_a",
        "out_dir": str(exp_dir),
        "artifact_prefix": "gcn",
    }
    pd.DataFrame([row]).to_json(results_path, orient="records", lines=True)

    captured_labels = []
    captured_titles = []
    original_plot = matplotlib.axes.Axes.plot
    original_set_title = matplotlib.axes.Axes.set_title

    def wrapped_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        if label is not None:
            captured_labels.append(str(label))
        return original_plot(self, *args, **kwargs)

    def wrapped_set_title(self, title, *args, **kwargs):
        captured_titles.append(str(title))
        return original_set_title(self, title, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", wrapped_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", wrapped_set_title)

    generate_reports(results_path, out_dir)

    baseline_context = pd.read_csv(out_dir / "baseline_context.csv")
    assert baseline_context.loc[0, "global_start_date"] == "2000-03-29"
    assert baseline_context.loc[0, "global_end_date"] == "2024-12-30"
    assert baseline_context.loc[0, "test_start_date"] == "2020-01-02"
    assert baseline_context.loc[0, "test_end_date"] == "2024-12-27"

    assert "buy_and_hold (test rebased)" in captured_labels
    assert any("Test window: 2020-01-02..2024-12-27" in t for t in captured_titles)
