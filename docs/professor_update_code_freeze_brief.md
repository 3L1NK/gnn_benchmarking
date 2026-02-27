# Professor Update (Code Freeze Brief)

## Scope Completed
- Baseline and strategy naming are standardized in code, plots, CSV, and LaTeX:
  - `Buy and hold (fixed shares)`
  - `Equal weight (rebalanced, all assets)`
  - `Top K long-only (equal weight within Top K)`
  - `Top 3 long, bottom 3 short (market-neutral)`
- Hard gates remain active:
  - equal-weight rebalance integrity gate
  - graph time-awareness gate
- Canonical comparison is gross (`0 bps`), with robustness sensitivity at `0/5/10 bps`.

## Baseline Construction (Explicit)
- `Buy and hold (fixed shares)`: equal dollars at `t0`, fixed shares, no rebalance.
- `Equal weight (rebalanced, all assets)`: set `1/N` at rebalance dates, hold and drift between rebalances.
- `Top K long-only (equal weight within Top K)`: rank by model signal, pick Top K, equal-weight selected assets, hold between rebalances.
- `Top 3 long, bottom 3 short (market-neutral)`: equal-weight within each leg, scale to long `+0.5`, short `-0.5`, hold between rebalances.

## Interpretation for 2020-01-02 to 2024-12-27
In this window, `Buy and hold (fixed shares)` has annualized Sharpe `1.227` and is unusually strong, consistent with market drift and winner compounding. A model with Sharpe around `0.8` can still be statistically respectable in absolute risk-adjusted terms, yet underperform passive buy-and-hold wealth when the regime strongly rewards holding winners and penalizes rebalance-driven turnover. For this reason, we treat buy-and-hold as contextual and report `active vs equal-weight` plus `market-neutral long-short` diagnostics as the primary evidence of signal usefulness.

## Reporting Artifacts
- Main professor table (baseline + best learned models):
  - `results/reports/thesis_tuned_all/professor_main_results_table.csv`
  - `thesis/tables/generated/professor_main_results_table.tex`
- Baseline policy table:
  - `results/reports/thesis_tuned_all/baseline_policy_comparison.csv`
- Hard-gate diagnostics:
  - `results/reports/thesis_tuned_all/equal_weight_rebalance_audit.csv`
  - `results/reports/thesis_tuned_all/graph_time_awareness_audit.csv`
