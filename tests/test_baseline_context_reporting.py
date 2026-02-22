import importlib
import sys
import types

import matplotlib
import numpy as np
import pandas as pd

from utils.artifacts import OutputDirs

matplotlib.use("Agg", force=True)


def test_summary_includes_global_and_test_baseline_context(tmp_path, monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda *args, **kwargs: None
    fake_torch.save = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    eval_runner = importlib.import_module("utils.eval_runner")

    eq_global = pd.Series(
        [1.0, 51.178],
        index=pd.to_datetime(["2000-03-29", "2024-12-30"]),
        name="buy_and_hold",
    )
    eq_test = pd.Series(
        [1.0, 5.049],
        index=pd.to_datetime(["2020-01-02", "2024-12-27"]),
        name="buy_and_hold",
    )
    eq_eqw = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["2020-01-02", "2024-12-27"]),
        name="equal_weight",
    )

    monkeypatch.setattr(
        eval_runner,
        "get_global_buy_and_hold",
        lambda *args, **kwargs: (
            eq_global,
            np.array([0.0], dtype=float),
            {
                "final_value": 51.178,
                "cumulative_return": 50.178,
                "annualized_return": 0.18,
                "annualized_volatility": 0.2,
                "sharpe": 0.7,
                "sharpe_annualized": 0.7 * np.sqrt(252.0),
                "sortino_annualized": 1.0,
                "max_drawdown": -0.5,
                "avg_turnover": 0.0,
            },
        ),
    )
    monkeypatch.setattr(
        eval_runner,
        "get_buy_and_hold_for_window",
        lambda *args, **kwargs: (
            eq_test,
            np.array([0.0], dtype=float),
            {
                "final_value": 5.049,
                "cumulative_return": 4.049,
                "annualized_return": 0.35,
                "annualized_volatility": 0.25,
                "sharpe": 0.8,
                "sharpe_annualized": 0.8 * np.sqrt(252.0),
                "sortino_annualized": 1.2,
                "max_drawdown": -0.3,
                "avg_turnover": 0.0,
            },
        ),
    )
    monkeypatch.setattr(
        eval_runner,
        "get_equal_weight_for_window",
        lambda *args, **kwargs: (
            eq_eqw,
            np.array([0.0], dtype=float),
            {
                "final_value": 2.0,
                "cumulative_return": 1.0,
                "annualized_return": 0.15,
                "annualized_volatility": 0.22,
                "sharpe": 0.4,
                "sharpe_annualized": 0.4 * np.sqrt(252.0),
                "sortino_annualized": 0.7,
                "max_drawdown": -0.2,
                "avg_turnover": 0.1,
            },
        ),
    )
    monkeypatch.setattr(
        eval_runner,
        "backtest_long_only",
        lambda *args, **kwargs: (
            pd.Series(
                [1.0, 1.1],
                index=pd.to_datetime(["2020-01-02", "2024-12-27"]),
                name="model",
            ),
            [0.0, 0.10],
            {
                "final_value": 1.1,
                "cumulative_return": 0.1,
                "annualized_return": 0.05,
                "annualized_volatility": 0.15,
                "sharpe": 0.3,
                "sharpe_annualized": 0.3 * np.sqrt(252.0),
                "sortino_annualized": 0.6,
                "max_drawdown": -0.1,
                "avg_turnover": 0.2,
            },
        ),
    )
    monkeypatch.setattr(eval_runner, "plot_daily_ic", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_runner, "plot_ic_hist", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_runner, "plot_equity_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_runner, "plot_equity_comparison", lambda *args, **kwargs: None)

    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2024-12-27", "2024-12-27"]),
            "ticker": ["A", "B", "A", "B"],
            "pred": [0.2, 0.1, 0.3, 0.05],
            "realized_ret": [0.01, -0.01, 0.02, -0.02],
        }
    )

    config = {
        "experiment_name": "toy_run",
        "seed": 42,
        "data": {
            "target_type": "regression",
            "target_name": "log_return",
            "target_horizon": 1,
            "lookback_window": 60,
        },
        "training": {
            "val_start": "2016-01-01",
            "test_start": "2020-01-01",
        },
        "evaluation": {
            "top_k": 1,
            "transaction_cost_bps": 5.0,
            "risk_free_rate": 0.0,
            "backtest_policies": [1],
            "primary_rebalance_freq": 1,
            "results_path": str(tmp_path / "results.jsonl"),
            "out_dir": str(tmp_path / "exp"),
        },
        "experiment": {
            "protocol_version": "v1_thesis_core",
        },
    }

    out_dirs = OutputDirs(canonical=tmp_path / "exp", legacy=None)
    out_dirs.canonical.mkdir(parents=True, exist_ok=True)
    summary = eval_runner.evaluate_and_report(
        config=config,
        pred_df=pred_df,
        out_dirs=out_dirs,
        run_name="toy_model",
        model_name="toy_model",
        model_family="xgboost",
        edge_type="none",
        directed=False,
        graph_window="",
        train_seconds=1.0,
        inference_seconds=0.1,
    )

    ctx = summary["baseline_context"]
    assert ctx["global_buy_and_hold"]["start_date"] == "2000-03-29"
    assert ctx["global_buy_and_hold"]["end_date"] == "2024-12-30"
    assert np.isclose(ctx["global_buy_and_hold"]["final_value"], 51.178)
    assert ctx["test_window_buy_and_hold"]["start_date"] == "2020-01-02"
    assert ctx["test_window_buy_and_hold"]["end_date"] == "2024-12-27"
    assert np.isclose(ctx["test_window_buy_and_hold"]["final_value"], 5.049)
    assert ctx["test_window_buy_and_hold"]["rebased"] is True

    bh_csv = pd.read_csv(tmp_path / "exp" / "buy_and_hold_equity_curve.csv")
    assert np.isclose(float(bh_csv["buy_and_hold"].iloc[0]), 1.0)

    for module_name in ["utils.eval_runner", "utils.baseline", "utils.cache", "utils.plot"]:
        sys.modules.pop(module_name, None)
