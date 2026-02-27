# Thesis Technical Guide

## 1. Purpose of This Document

This document is the deep technical reference for the thesis benchmarking repository.
It explains:

- how the repository works end-to-end,
- which code paths produce which thesis artifacts,
- what changed in the latest cycle,
- why buy-and-hold can appear as either about `5.x` or `51.x`,
- how to interpret results for thesis decisions.

Use this together with the root `README.md`.

## 2. Thesis Framing and Research Questions

The thesis benchmark is structured to answer these practical questions:

1. Do graph-based models outperform non-graph baselines for next-day return prediction under a fixed protocol?
2. Which edge construction signal is most useful (`corr`, `sector`, `granger`, or combinations)?
3. Do prediction improvements convert into better portfolio outcomes after transaction costs?
4. Under risk-adjusted evaluation, which models are preferred for deployment-style decisions?
5. Can targeted medium-budget retuning improve the core winners without unacceptable drawdown deterioration?

## 3. End-to-End Execution Flow

### 3.1 Entry Points

- Single run:

```bash
python train.py --config configs/runs/core/gcn_corr_only.yaml
```

- Full core thesis matrix:

```bash
python scripts/run_core_thesis_matrix.py --fresh-results --with-report
```

- Targeted retune matrix:

```bash
python scripts/run_targeted_retune_matrix.py --fresh-results --with-report
```

### 3.2 Control Flow

The main pipeline is:

1. `train.py`
2. `models/registry.py` (`run_model` dispatch by model family)
3. trainer function:
   - `trainers/train_gnn.py`
   - `trainers/train_xgboost.py`
   - `trainers/train_lstm.py`
4. shared evaluation and artifact writing in `utils/eval_runner.py`
5. consolidated thesis reporting via `scripts/generate_thesis_report.py`

### 3.3 Protocol Guard

`train.py` calls `utils.protocol.assert_canonical_protocol(config)`.
This enforces canonical thesis settings when `experiment.enforce_protocol: true`.

## 4. Canonical Protocol (`v1_thesis_core`)

The canonical benchmark settings are:

- target: next-day log return regression (`target_horizon=1`, `target_name=log_return`),
- train: dates `< 2016-01-01`,
- validation: `2016-01-01` to `2019-12-31`,
- test: dates `>= 2020-01-01` (first trading day is `2020-01-02`),
- portfolio policy: long-only top-k, transaction cost included,
- rebalance policies: `rebalance_freq=1` and `rebalance_freq=5`,
- primary policy: `rebalance_freq=1`.

Core constants are defined in `utils/protocol.py`.

## 5. Baseline Methodology and the `5.x` vs `51.x` Question

## 5.1 Why Both Values Exist

Both values are correct and expected.

- Global buy-and-hold is measured on the full baseline horizon and is around `51.178x`.
- Test-window buy-and-hold is sliced to the test window and rebased to `1.0` at test start, and is around `5.049x`.

These are different because:

1. they use different date windows,
2. the test-window series is explicitly rebased.

## 5.2 Exact Windows

- Global buy-and-hold window: `2000-03-29` to `2024-12-30`, final around `51.178x`.
- Test rebased buy-and-hold window: `2020-01-02` to `2024-12-27`, final around `5.049x`.

## 5.3 Where It Is Reported

`utils/eval_runner.py` now reports and stores both contexts.
It prints log lines in this shape:

- `BH(test rebased): 5.049x (+404.9%) [2020-01-02..2024-12-27]`
- `BH(global): 51.178x [2000-03-29..2024-12-30]`

And writes `baseline_context` into each run summary JSON.

## 6. Model Families and Their Thesis Role

### 6.1 Non-graph baselines

- `xgb_raw`
- `lstm`

Purpose: establish non-graph prediction and portfolio baselines.

### 6.2 Graph-feature baseline

- `xgb_node2vec_corr`

Purpose: isolate structural graph embedding value without end-to-end message passing.

### 6.3 Static GNN edge ablations

- `gcn_*`
- `gat_*`

Purpose: test which edge signal contributes most under identical protocol.

### 6.4 Static temporal-labeled baselines

- `tgcn_static_corr_only`
- `tgat_static_corr_only`

Purpose: include architecture labels while preserving the static protocol scope.

## 7. Metrics and Decision Ranking

## 7.1 Prediction-level metrics

- `prediction_rmse`
- `prediction_mae`
- `prediction_rank_ic`

## 7.2 Portfolio-level metrics

- `portfolio_final_value`
- `portfolio_cumulative_return`
- `portfolio_annualized_return`
- `portfolio_annualized_volatility`
- `portfolio_sharpe` (legacy compatibility)
- `portfolio_sharpe_daily` (explicit daily alias)
- `portfolio_sharpe_annualized`
- `portfolio_sortino_annualized`
- `portfolio_max_drawdown`
- `portfolio_turnover`

## 7.3 Decision Rank Order

The ranking logic is deterministic and risk-adjusted first:

1. `portfolio_sharpe_annualized` descending,
2. `portfolio_max_drawdown` descending (less negative is better),
3. `prediction_rank_ic` descending,
4. `portfolio_turnover` ascending.

The compact ranking output is `results/reports/thesis_tuned_all/decision_ranking.csv`.

## 8. Public Interfaces and Schemas Updated

## 8.1 Summary JSON contract (`utils/eval_runner.py`)

Added:

- `baseline_context.global_buy_and_hold.start_date`
- `baseline_context.global_buy_and_hold.end_date`
- `baseline_context.global_buy_and_hold.final_value`
- `baseline_context.test_window_buy_and_hold.start_date`
- `baseline_context.test_window_buy_and_hold.end_date`
- `baseline_context.test_window_buy_and_hold.final_value`
- `baseline_context.test_window_buy_and_hold.rebased`

## 8.2 Results row schema (`utils/results.py`)

Added:

- `portfolio_sharpe_daily`
- `portfolio_sharpe_annualized`
- `portfolio_sortino_annualized`

Kept for compatibility:

- `portfolio_sharpe`

## 8.3 Tuning schema used by trainers

Supported keys:

- `tuning.enabled`
- `tuning.objective` in `{val_rmse, val_ic, val_backtest_sharpe_annualized}`
- `tuning.max_trials`
- `tuning.sample_mode` in `{grid, random}`
- `tuning.seed`
- `tuning.param_grid`

## 8.4 Plot labeling API (`utils/plot.py`)

`plot_equity_comparison(...)` now supports explicit baseline labeling and subtitle.

## 9. Artifact Map (What Gets Written and Why)

| Artifact | Path | Produced by | Purpose |
|---|---|---|---|
| run results ledger | `results/results_tuned_all.jsonl` | `utils/eval_runner.py` | unified row-per-run-per-policy results table |
| retune results ledger | `results/results_retune.jsonl` | retune configs + eval runner | isolated retune output |
| run summary | `experiments/<run_tag>/<run_tag>_summary.json` | `utils/eval_runner.py` | stats + baseline_context + per-policy detail |
| model predictions | `experiments/<run_tag>/<model>_predictions.csv` | trainer + eval runner | raw prediction audit and diagnostics |
| daily metrics | `experiments/<run_tag>/<model>_daily_metrics_reb{1,5}.csv` | eval runner | IC/hit/returns/drawdown timeline |
| model equity | `experiments/<run_tag>/<model>_equity_curve_reb{1,5}.csv` | eval runner | policy-specific equity path |
| buy-and-hold window baseline | `experiments/<run_tag>/buy_and_hold_equity_curve.csv` | eval runner | sliced/rebased baseline for fair test-window comparison |
| equal-weight baseline (all assets) | `experiments/<run_tag>/equal_weight_equity_curve_reb{1,5}.csv` | eval runner | additional reference baseline |
| master report table | `results/reports/thesis_tuned_all/master_comparison.csv` | `scripts/generate_thesis_report.py` | full ranked comparison table |
| family summary | `results/reports/thesis_tuned_all/family_summary.csv` | report script | best-by-family compact view |
| edge ablation summary | `results/reports/thesis_tuned_all/edge_ablation_summary.csv` | report script | edge-type aggregated performance |
| baseline context report | `results/reports/thesis_tuned_all/baseline_context.csv` | report script | explicit global vs test baseline window context |
| decision ranking | `results/reports/thesis_tuned_all/decision_ranking.csv` | report script | deterministic risk-adjusted ranking |
| risk frontier | `results/reports/thesis_tuned_all/risk_frontier_reb1.png`, `risk_frontier_reb5.png` | report script | annualized return vs drawdown; point size by annualized Sharpe |
| retune deltas | `results/reports/retune_comparison/retune_delta_summary.csv` | `scripts/compare_core_vs_retune.py` | retune-minus-core deltas |
| retune winners | `results/reports/retune_comparison/retune_winners.csv` | compare script | models that pass fixed winner rule |

## 10. How Tables and Figures Are Built

1. Each training run writes prediction/equity/metrics artifacts under its `out_dir`.
2. `utils/eval_runner.py` appends standardized rows to results JSONL.
3. `scripts/generate_thesis_report.py` reads JSONL and resolves run artifacts.
4. It emits thesis CSV summaries and plot files in `results/reports/thesis_tuned_all`.

Command:

```bash
python scripts/generate_thesis_report.py --results results/results_tuned_all.jsonl --out results/reports/thesis_tuned_all --expected-runs 28
```

## 11. Targeted Retune Workflow

## 11.1 Target set

- `gcn_granger_only`
- `gcn_sector_only`
- `xgb_node2vec_corr`
- `lstm`

Retune configs are in `configs/runs/retune_medium/`.

## 11.2 Run command

```bash
python scripts/run_targeted_retune_matrix.py --fresh-results --with-report
```

Default report output is `results/reports/thesis_retune`.

## 11.3 Compare core vs retune

```bash
python scripts/compare_core_vs_retune.py \
  --core results/results.jsonl \
  --retune results/results_retune.jsonl \
  --out results/reports/retune_comparison
```

## 11.4 Winner rule

A retuned run is a winner only if both are true:

1. `portfolio_sharpe_annualized` improves versus core baseline,
2. `portfolio_max_drawdown` does not worsen by more than `0.02` absolute.

## 12. Reproducibility Runbook

## 12.1 Minimal single-run sanity check

```bash
python train.py --config configs/runs/core/xgb_raw.yaml
```

## 12.2 Full core thesis matrix + report

```bash
python scripts/run_core_thesis_matrix.py --fresh-results --with-report
```

## 12.3 Targeted retune matrix + retune report

```bash
python scripts/run_targeted_retune_matrix.py --fresh-results --with-report
```

## 12.4 Compare core and retune outcomes

```bash
python scripts/compare_core_vs_retune.py
```

## 13. How This Repository Answers the Thesis

| Thesis question | Artifact(s) to inspect | Metric evidence | Decision interpretation |
|---|---|---|---|
| Are graph models better than non-graph baselines? | `results/reports/thesis_tuned_all/family_summary.csv`, `master_comparison.csv` | `portfolio_sharpe_annualized`, `portfolio_max_drawdown`, `prediction_rank_ic` | If graph family dominates on Sharpe with acceptable drawdown, graph approach is supported. |
| Which edge signal works best? | `results/reports/thesis_tuned_all/edge_ablation_summary.csv` | edge-type mean Sharpe/return/drawdown | Higher Sharpe and better drawdown for an edge type indicates stronger signal utility. |
| Do prediction gains monetize after costs? | `master_comparison.csv`, per-run `*_daily_metrics_reb*.csv` | IC vs realized annualized return and drawdown | Positive IC without portfolio improvement implies translation failure after portfolio construction/costs. |
| Which model is best for decision use? | `results/reports/thesis_tuned_all/decision_ranking.csv`, `risk_frontier_reb*.png` | deterministic ranking order and frontier position | Top-ranked models combine risk-adjusted return quality and operational frictions. |
| Does medium retune improve top models safely? | `results/reports/retune_comparison/retune_delta_summary.csv`, `retune_winners.csv` | delta Sharpe and drawdown constraint | Winner rows identify safe improvements; empty winners implies no robust improvement in this pass. |

## 14. What Changed in This Cycle

This cycle introduced and validated:

1. explicit baseline-context reporting (global and test-rebased BH),
2. annualized risk metrics in results schema while keeping legacy compatibility,
3. shared tuning objective framework across GNN/XGB/LSTM,
4. decision-first thesis report outputs (`decision_ranking.csv`, `baseline_context.csv`, risk frontiers),
5. targeted retune matrix tooling and core-vs-retune comparison scripts,
6. regression tests for schema, baseline reporting, labeling, metric scaling, and tuning objective selection,
7. Python 3.9 compatibility fix in plotting type annotations.

## 15. Current Status Note

As of this update, code paths support all new fields and report artifacts.

If your existing report CSVs were generated before these changes, they can be stale.
Regenerate using:

```bash
python scripts/generate_thesis_report.py --results results/results_tuned_all.jsonl --out results/reports/thesis_tuned_all --expected-runs 28
```

For retune outputs:

```bash
python scripts/generate_thesis_report.py --results results/results_retune.jsonl --out results/reports/thesis_retune
```

## 16. Troubleshooting and Common Confusions

1. `5.x` vs `51.x` buy-and-hold mismatch:
   - this is expected (test-window rebased versus global full horizon).
2. Missing `portfolio_sharpe_annualized` in old rows:
   - regenerate from latest pipeline or allow compare script fallback from legacy `portfolio_sharpe`.
3. No retune winners:
   - expected if improvements fail winner constraints; this is a valid outcome.
4. Run output path confusion:
   - trust `out_dir` and `artifact_prefix` columns in `results*.jsonl`.
5. Rebalance policy confusion:
   - each run logs two policy rows (`rebalance_freq=1` and `5`), with `1` as primary.

## 17. Documentation Validation Checklist

Use this checklist when updating this document:

1. every command references a script/config that exists in this repo,
2. every artifact path named here is either generated by listed commands or explicitly marked as derived,
3. baseline explanation includes exact windows and rebasing semantics,
4. ranking order in this doc matches `scripts/generate_thesis_report.py`,
5. winner rule here matches `scripts/compare_core_vs_retune.py`.

## 18. Acceptance Criteria for Documentation Quality

A new researcher should be able to:

1. run one core model,
2. run the core matrix and generate thesis report,
3. locate and interpret the key thesis CSV/PNG outputs,
4. explain the baseline `5.x` vs `51.x` distinction correctly,
5. determine whether retune improved core models under the fixed winner rule.
