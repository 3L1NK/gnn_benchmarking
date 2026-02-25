# GNN Benchmarking for Multi Asset Return Prediction

This repository contains the implementation and experiments for my **Master’s Thesis** on benchmarking Graph Neural Networks (GNNs) and traditional ML models (e.g. XGBoost) for **stock return prediction**.  
It includes the data preprocessing, graph construction, feature embedding (Node2Vec), model training, and evaluation workflow.

---

## Thesis Guide

For the full technical documentation (pipeline internals, artifact map, thesis-question mapping, baseline `5.x` vs `51.x` explanation, retune workflow, and change audit), use:

- `docs/THESIS_README.md`

## 📂 Project Structure


This project implements the main experiments for the thesis:

- Graph neural networks on rolling correlation and graphical lasso graphs
- Non graph neural baselines (LSTM, MLP)
- Representation baselines (node2vec embeddings plus XGBoost)
- Portfolio level evaluation with long-only top-k strategy
- Model code grouped by family under `models/graph/`, `models/sequence/`, and `models/baselines/`

To run an experiment on the HU GPU server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py --config configs/runs/core/gcn_corr_only.yaml
```

To run the core thesis matrix:

```bash
python scripts/run_core_thesis_matrix.py --fresh-results --with-report
```

## Config Layout

- `configs/templates/`: shared protocol and model templates
- `configs/runs/core/`: thesis comparison matrix (canonical runs only)
- `configs/runs/ablation/`: controlled ablation studies
- `configs/runs/exploratory/`: non-core and development experiments
- `configs/legacy/`: archived legacy model-zoo configs

Naming convention:
- `{family_or_model}_{edge_scope_or_variant}.yaml`
- examples: `gcn_corr_only.yaml`, `xgb_node2vec_corr.yaml`, `xgb_graphlasso.yaml`
- legacy paths are still accepted through alias resolution in `utils/config_aliases.py`

## Canonical Benchmark Protocol

- Target: next-day log-return regression (`target_horizon=1`, `target_name=log_return`)
- Splits: train `< 2016-01-01`, val `2016-01-01..2019-12-31`, test `>= 2020-01-01`
- Portfolio: long-only top-k equal-weight with transaction costs
- Backtest policies: both daily (`rebalance_freq=1`) and weekly (`rebalance_freq=5`)
- Temporal variants are reported as static baselines (`tgcn_static`, `tgat_static`) unless a true temporal training loop is explicitly introduced.
- Config guard: `experiment.enforce_protocol: true` fails fast when a run drifts from this protocol.

## Buy-and-Hold Baseline

The canonical buy-and-hold baseline is computed from prices (not log returns) using fixed shares:
equal-weight at t0 across a fixed universe, no rebalancing, portfolio value is the sum of
shares_i * price_i,t. Missing prices are forward-filled per ticker with a max gap (default 5
trading days). Tickers without a price on the global start date are dropped, and any ticker
with remaining gaps after forward-fill is removed so the universe is fixed. The calendar is
aligned by taking the intersection of dates across remaining tickers.

For evaluation, the baseline is sliced to the prediction window and rebased to 1.0 at the
first date of that window (single consistent code path across LSTM/GNN/XGB). The global
baseline is cached and also written to `data/processed/baselines/buy_and_hold_global.csv`.
Use the cache rebuild flag to recompute if upstream price data changes.

An additional canonical equal-weight rebalanced baseline is produced for each rebalance
policy and cached from price data as well.

## One-Command Thesis Report

After running experiments:

```bash
python scripts/generate_thesis_report.py --results results/results.jsonl --out results/reports/thesis
```

This generates:
- `master_comparison.csv`
- `family_summary.csv`
- `edge_ablation_summary.csv`
- `equity_curves_key_models.png`
- `ic_distribution_boxplot_reb1.png`, `ic_distribution_boxplot_reb5.png`
- thesis plots under `results/reports/thesis/`

## Write Thesis in LaTeX (In-Repo)

A thesis writing workspace is available under:

- `thesis/`

From there you can:

```bash
cd thesis
make pdf
```

This will regenerate LaTeX tables from `results/reports/thesis/*.csv` and compile `thesis/main.tex`.
