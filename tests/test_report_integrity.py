import pandas as pd

from scripts.generate_thesis_report import generate_reports


def _base_row(*, run_tag: str, run_key: str, model_name: str, model_family: str, edge_type: str, freq: int, sharpe_ann, sharpe_daily, ic: float, ann_ret: float, mdd: float, turnover: float, out_dir: str, ts: str):
    return {
        "experiment_id": f"{run_tag}_{freq}",
        "timestamp": ts,
        "seed": 42,
        "model_name": model_name,
        "model_family": model_family,
        "edge_type": edge_type,
        "directed": False,
        "graph_window": "2000-01-01..2015-12-31",
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
        "prediction_rows": 100,
        "prediction_unique_pairs": 100,
        "prediction_rmse": 0.1,
        "prediction_mae": 0.08,
        "prediction_rank_ic": ic,
        "portfolio_final_value": 1.2,
        "portfolio_cumulative_return": 0.2,
        "portfolio_annualized_return": ann_ret,
        "portfolio_annualized_volatility": 0.2,
        "portfolio_sharpe": sharpe_daily,
        "portfolio_sharpe_daily": sharpe_daily,
        "portfolio_sharpe_annualized": sharpe_ann,
        "portfolio_sortino_annualized": 1.0,
        "portfolio_max_drawdown": mdd,
        "portfolio_turnover": turnover,
        "runtime_train_seconds": 1.0,
        "runtime_inference_seconds": 0.1,
        "run_tag": run_tag,
        "out_dir": out_dir,
        "artifact_prefix": run_tag,
    }


def test_generate_reports_dedup_and_plot_sanity(tmp_path):
    report_out = tmp_path / "report"
    results_path = tmp_path / "results.jsonl"
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        # duplicate logical run: older + newer timestamp; latest should be kept
        _base_row(
            run_tag="xgb_raw_tuned_all",
            run_key="rk_xgb_1",
            model_name="xgb_raw",
            model_family="xgboost",
            edge_type="none",
            freq=1,
            sharpe_ann=0.55,
            sharpe_daily=0.55 / (252.0 ** 0.5),
            ic=0.011,
            ann_ret=0.14,
            mdd=-0.31,
            turnover=0.42,
            out_dir=str(exp_dir),
            ts="2026-01-01T00:00:00",
        ),
        _base_row(
            run_tag="xgb_raw_tuned_all",
            run_key="rk_xgb_1",
            model_name="xgb_raw",
            model_family="xgboost",
            edge_type="none",
            freq=1,
            sharpe_ann=0.60,
            sharpe_daily=0.60 / (252.0 ** 0.5),
            ic=0.012,
            ann_ret=0.15,
            mdd=-0.30,
            turnover=0.40,
            out_dir=str(exp_dir),
            ts="2026-01-02T00:00:00",
        ),
        _base_row(
            run_tag="gcn_corr_only_tuned_all",
            run_key="rk_gcn_1",
            model_name="gcn",
            model_family="gnn",
            edge_type="corr",
            freq=1,
            sharpe_ann=None,
            sharpe_daily=0.48 / (252.0 ** 0.5),
            ic=0.009,
            ann_ret=0.12,
            mdd=-0.29,
            turnover=0.43,
            out_dir=str(exp_dir),
            ts="2026-01-02T00:00:00",
        ),
        _base_row(
            run_tag="xgb_raw_tuned_all",
            run_key="rk_xgb_5",
            model_name="xgb_raw",
            model_family="xgboost",
            edge_type="none",
            freq=5,
            sharpe_ann=0.72,
            sharpe_daily=0.72 / (252.0 ** 0.5),
            ic=0.010,
            ann_ret=0.18,
            mdd=-0.33,
            turnover=0.14,
            out_dir=str(exp_dir),
            ts="2026-01-02T00:00:00",
        ),
        _base_row(
            run_tag="gcn_corr_only_tuned_all",
            run_key="rk_gcn_5",
            model_name="gcn",
            model_family="gnn",
            edge_type="corr",
            freq=5,
            sharpe_ann=0.63,
            sharpe_daily=0.63 / (252.0 ** 0.5),
            ic=0.008,
            ann_ret=0.16,
            mdd=-0.34,
            turnover=0.13,
            out_dir=str(exp_dir),
            ts="2026-01-02T00:00:00",
        ),
    ]

    pd.DataFrame(rows).to_json(results_path, orient="records", lines=True)
    generate_reports(results_path, report_out, expected_runs=4)

    master = pd.read_csv(report_out / "master_comparison.csv")
    run_matrix = pd.read_csv(report_out / "run_matrix.csv")

    assert len(master) == 4
    assert len(run_matrix) == 4
    assert master["run_key"].nunique() == 4
    assert master["portfolio_sharpe_annualized"].notna().all()
