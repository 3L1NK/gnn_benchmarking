import pandas as pd
import pytest

from scripts.generate_thesis_report import generate_reports


def _write_curve(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values, columns=["date", "value"]).to_csv(path, index=False)


def _row(*, run_tag: str, run_key: str, freq: int, model_name: str, edge_type: str, graph_window: str, split_train_end: str, out_dir: str):
    return {
        "experiment_id": f"{run_tag}_{freq}",
        "timestamp": "2026-01-01T00:00:00",
        "seed": 42,
        "model_name": model_name,
        "model_family": "gnn" if model_name in {"gcn", "gat", "tgcn_static", "tgat_static"} else "xgboost",
        "edge_type": edge_type,
        "directed": False,
        "graph_window": graph_window,
        "target_type": "regression",
        "target_horizon": 1,
        "lookback_window": 60,
        "protocol_version": "v1_thesis_core",
        "split_train_end": split_train_end,
        "split_val_start": "2016-01-01",
        "split_test_start": "2020-01-01",
        "rebalance_freq": freq,
        "baseline_version": "price_bh_v2_eqw_v2",
        "target_policy_hash": "abc123",
        "split_id": "split123",
        "config_hash": "cfg123",
        "run_key": run_key,
        "prediction_rows": 4,
        "prediction_unique_pairs": 4,
        "prediction_rmse": 0.1,
        "prediction_mae": 0.08,
        "prediction_rank_ic": 0.01,
        "portfolio_final_value": 1.2,
        "portfolio_cumulative_return": 0.2,
        "portfolio_annualized_return": 0.12,
        "portfolio_annualized_volatility": 0.2,
        "portfolio_sharpe": 0.03,
        "portfolio_sharpe_daily": 0.03,
        "portfolio_sharpe_annualized": 0.03 * (252.0 ** 0.5),
        "portfolio_sortino_annualized": 0.9,
        "portfolio_max_drawdown": -0.2,
        "portfolio_turnover": 0.2,
        "runtime_train_seconds": 1.0,
        "runtime_inference_seconds": 0.1,
        "run_tag": run_tag,
        "out_dir": out_dir,
        "artifact_prefix": run_tag,
    }


def test_eqw_rebalance_gate_fails_on_identical_reb1_reb5(tmp_path):
    exp_dir = tmp_path / "experiments" / "run_a"
    out_dir = tmp_path / "report"
    results_path = tmp_path / "results.jsonl"

    identical = [
        ["2020-01-02", 1.0],
        ["2020-01-03", 1.05],
        ["2020-01-06", 1.10],
        ["2020-01-07", 1.15],
    ]
    _write_curve(exp_dir / "equal_weight_equity_curve_reb1.csv", identical)
    _write_curve(exp_dir / "equal_weight_equity_curve_reb5.csv", identical)

    rows = [
        _row(
            run_tag="run_a",
            run_key="rk1",
            freq=1,
            model_name="xgb_raw",
            edge_type="none",
            graph_window="",
            split_train_end="2015-12-31",
            out_dir=str(exp_dir),
        ),
        _row(
            run_tag="run_a",
            run_key="rk5",
            freq=5,
            model_name="xgb_raw",
            edge_type="none",
            graph_window="",
            split_train_end="2015-12-31",
            out_dir=str(exp_dir),
        ),
    ]
    pd.DataFrame(rows).to_json(results_path, orient="records", lines=True)

    with pytest.raises(ValueError, match="EQW rebalance integrity failed"):
        generate_reports(results_path, out_dir, expected_runs=2)

    audit = pd.read_csv(out_dir / "equal_weight_rebalance_audit.csv")
    assert (audit["status"].astype(str).str.lower() == "fail").any()


def test_graph_time_awareness_gate_fails_when_window_exceeds_train_end(tmp_path):
    exp_dir = tmp_path / "experiments" / "run_gcn"
    out_dir = tmp_path / "report"
    results_path = tmp_path / "results.jsonl"

    _write_curve(
        exp_dir / "equal_weight_equity_curve_reb1.csv",
        [["2020-01-02", 1.0], ["2020-01-03", 1.02], ["2020-01-06", 1.03], ["2020-01-07", 1.01]],
    )

    row = _row(
        run_tag="run_gcn",
        run_key="rk_gcn_1",
        freq=1,
        model_name="gcn",
        edge_type="corr",
        graph_window="2000-01-01..2019-12-31",
        split_train_end="2015-12-31",
        out_dir=str(exp_dir),
    )
    pd.DataFrame([row]).to_json(results_path, orient="records", lines=True)

    with pytest.raises(ValueError, match="Graph time-awareness audit failed"):
        generate_reports(results_path, out_dir, expected_runs=1)

    audit = pd.read_csv(out_dir / "graph_time_awareness_audit.csv")
    assert (audit["status"].astype(str).str.lower() == "fail").any()
