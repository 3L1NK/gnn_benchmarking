import importlib
import sys
import types

import numpy as np
import pandas as pd

def test_xgb_trainer_uses_configured_tuning_objective_and_selects_best_params(tmp_path, monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.load = lambda *args, **kwargs: None
    fake_torch.save = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_graphs = types.ModuleType("utils.graphs")
    fake_graphs.rolling_corr_edges = lambda *args, **kwargs: []
    fake_graphs.graphical_lasso_precision = lambda *args, **kwargs: (None, None, None, {})
    fake_graphs.granger_edges = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "utils.graphs", fake_graphs)

    tx = importlib.import_module("trainers.train_xgboost")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2015-12-31",
                    "2015-12-31",
                    "2017-01-03",
                    "2017-01-03",
                    "2020-01-02",
                    "2020-01-02",
                ]
            ),
            "ticker": ["A", "B", "A", "B", "A", "B"],
            "target": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03],
            "f1": [1.0, 0.5, 1.2, 0.6, 1.4, 0.7],
            "f2": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        }
    )

    config = {
        "seed": 42,
        "model": {
            "family": "xgboost",
            "type": "xgb_raw",
            "params": {"n_estimators": 10},
        },
        "data": {
            "price_file": "dummy_prices.parquet",
            "target_type": "regression",
            "target_horizon": 1,
            "lookback_window": 60,
        },
        "training": {
            "val_start": "2016-01-01",
            "test_start": "2020-01-01",
            "lr": 0.001,
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
        "preprocess": {"scale_features": False},
        "cache": {"rebuild": True},
        "tuning": {
            "enabled": True,
            "objective": "val_backtest_sharpe_annualized",
            "sample_mode": "grid",
            "max_trials": 0,
            "seed": 42,
            "param_grid": {"max_depth": [2, 5]},
        },
    }

    monkeypatch.setattr(tx, "_build_feature_panel", lambda cfg: (df.copy(), ["f1", "f2"]))
    monkeypatch.setattr(tx, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(tx, "scale_features", lambda *args, **kwargs: (args[0], None))
    monkeypatch.setattr(tx, "cache_load", lambda *args, **kwargs: None)
    monkeypatch.setattr(tx, "cache_save", lambda *args, **kwargs: None)
    monkeypatch.setattr(tx, "cache_key", lambda *args, **kwargs: "cache_key")
    monkeypatch.setattr(tx, "cache_path", lambda *args, **kwargs: tmp_path / "cache.pkl")
    monkeypatch.setattr(
        tx,
        "get_global_buy_and_hold",
        lambda *args, **kwargs: (
            pd.Series([1.0, 1.1], index=pd.to_datetime(["2020-01-02", "2020-01-03"])),
            np.array([0.1]),
            {"final_value": 1.1},
        ),
    )
    monkeypatch.setattr(tx.XGBoostTrainer, "_evaluate_and_backtest", lambda self, *args, **kwargs: None)

    created_params = []
    observed_objectives = []

    class FakeXGBRegressor:
        def __init__(self, **kwargs):
            self.params = dict(kwargs)
            created_params.append(dict(kwargs))

        def fit(self, X, y, eval_set=None, verbose=False, **kwargs):
            return self

        def predict(self, X):
            return np.full(len(X), float(self.params.get("max_depth", 0)))

    def fake_score_prediction_objective(**kwargs):
        observed_objectives.append(kwargs["objective"])
        metric_value = float(np.mean(kwargs["preds"]))
        return {
            "score": metric_value,
            "metric_value": metric_value,
            "metric_name": "fake_metric",
        }

    monkeypatch.setattr(tx, "XGBRegressor", FakeXGBRegressor)
    monkeypatch.setattr(tx, "score_prediction_objective", fake_score_prediction_objective)

    trainer = tx.XGBoostTrainer(config)
    trainer.train_xgb_raw()

    assert observed_objectives
    assert all(obj == "val_backtest_sharpe_annualized" for obj in observed_objectives)
    assert created_params[-1]["max_depth"] == 5
    assert created_params[-1]["n_estimators"] == 10

    for module_name in ["trainers.train_xgboost", "utils.graphs", "utils.cache"]:
        sys.modules.pop(module_name, None)
