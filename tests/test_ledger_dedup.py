import json

import numpy as np
import pandas as pd

from utils.results import build_experiment_result, save_experiment_result


def _sample_result(final_value: float):
    cfg = {
        "seed": 42,
        "data": {
            "target_type": "regression",
            "target_horizon": 1,
            "lookback_window": 60,
            "target_name": "log_return",
        },
        "model": {"family": "xgboost", "type": "xgb_raw"},
        "graph": {"use_corr": False, "use_sector": False, "use_granger": False},
        "training": {
            "val_start": "2016-01-01",
            "test_start": "2020-01-01",
        },
        "evaluation": {
            "top_k": 20,
            "transaction_cost_bps": 5,
            "risk_free_rate": 0.0,
            "backtest_policies": [1, 5],
            "primary_rebalance_freq": 1,
        },
        "experiment": {"protocol_version": "v1_thesis_core", "enforce_protocol": True},
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
        "final_value": final_value,
        "cumulative_return": 0.1,
        "annualized_return": 0.05,
        "annualized_volatility": 0.2,
        "sharpe": 0.2,
        "sharpe_annualized": float(0.2 * np.sqrt(252.0)),
        "sortino_annualized": 1.5,
        "max_drawdown": -0.1,
        "avg_turnover": 0.2,
    }
    return build_experiment_result(
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
            "baseline_version": "price_bh_v2_eqw_v2",
            "target_policy_hash": "abc123",
        },
        prediction_rows=2,
        prediction_unique_pairs=2,
        run_tag="xgb_raw_run",
        out_dir="experiments/xgb_raw",
        artifact_prefix="xgb_raw",
    )


def test_save_experiment_result_upserts_by_run_key(tmp_path):
    results_path = tmp_path / "results.jsonl"

    first = _sample_result(final_value=1.1)
    second = _sample_result(final_value=1.5)
    first["timestamp"] = "2026-01-01T00:00:00"
    second["timestamp"] = "2026-01-02T00:00:00"

    assert first["run_key"] == second["run_key"]

    save_experiment_result(first, results_path)
    save_experiment_result(second, results_path)

    rows = [json.loads(x) for x in results_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1
    assert float(rows[0]["portfolio_final_value"]) == 1.5
