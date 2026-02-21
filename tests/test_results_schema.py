import pandas as pd
import numpy as np

from utils.results import build_experiment_result, RESULT_FIELDS


def test_build_experiment_result_includes_protocol_fields():
    cfg = {
        "seed": 42,
        "data": {
            "target_type": "regression",
            "target_horizon": 1,
            "lookback_window": 60,
        },
        "training": {
            "val_start": "2016-01-01",
            "test_start": "2020-01-01",
        },
    }
    pred = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["A", "B"],
            "pred": [0.1, -0.1],
            "realized_ret": [0.05, -0.03],
        }
    )
    daily = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "ic": [0.1], "hit": [0.5]})
    stats = {
        "final_value": 1.1,
        "cumulative_return": 0.1,
        "annualized_return": 0.05,
        "annualized_volatility": 0.2,
        "sharpe": 0.2,
        "sharpe_annualized": float(0.2 * np.sqrt(252.0)),
        "sortino_annualized": 1.5,
        "max_drawdown": -0.1,
        "avg_turnover": 0.2,
    }
    result = build_experiment_result(
        cfg,
        model_name="xgb_raw",
        model_family="xgboost",
        edge_type="none",
        directed=False,
        graph_window="",
        pred_df=pred,
        daily_metrics=daily,
        stats=stats,
        train_seconds=1.0,
        inference_seconds=0.1,
        protocol_fields={
            "protocol_version": "v1_thesis_core",
            "split_train_end": "2015-12-31",
            "split_val_start": "2016-01-01",
            "split_test_start": "2020-01-01",
            "rebalance_freq": 1,
            "baseline_version": "price_bh_v2_eqw_v1",
            "target_policy_hash": "abc123",
        },
        prediction_rows=2,
        prediction_unique_pairs=2,
        run_tag="xgb_raw_run",
        out_dir="experiments/xgb_raw",
        artifact_prefix="xgb_raw",
    )
    for f in RESULT_FIELDS:
        assert f in result
    assert result["protocol_version"] == "v1_thesis_core"
    assert result["rebalance_freq"] == 1
    assert result["prediction_rows"] == 2
    assert result["prediction_unique_pairs"] == 2
    assert result["run_tag"] == "xgb_raw_run"
    assert result["out_dir"] == "experiments/xgb_raw"
    assert result["artifact_prefix"] == "xgb_raw"
    assert result["portfolio_sharpe_daily"] == result["portfolio_sharpe"]
    assert result["portfolio_sharpe_annualized"] == stats["sharpe_annualized"]
    assert result["portfolio_sortino_annualized"] == stats["sortino_annualized"]


def test_build_experiment_result_backward_compatible_without_new_ratio_fields():
    cfg = {
        "seed": 42,
        "data": {
            "target_type": "regression",
            "target_horizon": 1,
            "lookback_window": 60,
        },
        "training": {
            "val_start": "2016-01-01",
            "test_start": "2020-01-01",
        },
    }
    pred = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["A"],
            "pred": [0.1],
            "realized_ret": [0.05],
        }
    )
    daily = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "ic": [0.1], "hit": [1.0]})
    legacy_stats = {
        "final_value": 1.02,
        "cumulative_return": 0.02,
        "annualized_return": 0.02,
        "annualized_volatility": 0.1,
        "sharpe": 0.3,
        "max_drawdown": -0.05,
    }
    result = build_experiment_result(
        cfg,
        model_name="xgb_raw",
        model_family="xgboost",
        edge_type="none",
        directed=False,
        graph_window="",
        pred_df=pred,
        daily_metrics=daily,
        stats=legacy_stats,
        train_seconds=1.0,
        inference_seconds=0.1,
    )
    assert result["portfolio_sharpe"] == legacy_stats["sharpe"]
    assert result["portfolio_sharpe_daily"] == legacy_stats["sharpe"]
    assert pd.isna(result["portfolio_sharpe_annualized"])
    assert pd.isna(result["portfolio_sortino_annualized"])
