# Thesis Report Reproduction (Tuned-All Canonical)

## Source of Truth
- Ledger: `results/results_tuned_all.jsonl`
- Report outputs: `results/reports/thesis_tuned_all/`
- LaTeX tables: `thesis/tables/generated/`
- Thesis PDF: `thesis/main.pdf`

## One-Command Build
```bash
./.venv_gnn/bin/python scripts/build_thesis_report.py --build-pdf
```

## Step-by-Step Build
1. Run tuned matrix (if needed):
```bash
./.venv_gnn/bin/python scripts/run_all_models_tuned_matrix.py --fresh-results --with-report --budget medium
```

2. Deduplicate ledger:
```bash
./.venv_gnn/bin/python scripts/deduplicate_results_ledger.py --results results/results_tuned_all.jsonl
```

3. Regenerate canonical report:
```bash
./.venv_gnn/bin/python scripts/generate_thesis_report.py \
  --results results/results_tuned_all.jsonl \
  --out results/reports/thesis_tuned_all \
  --expected-runs 26
```

4. Export LaTeX tables:
```bash
THESIS_REPORT_DIR=results/reports/thesis_tuned_all \
  ./.venv_gnn/bin/python thesis/scripts/export_tables.py
```

5. Compile PDF:
```bash
cd thesis
make pdf PYTHON=../.venv_gnn/bin/python
```

## Validation Checklist
- `run_matrix.csv` has exactly 26 rows.
- `master_comparison.csv` has no NaN in `portfolio_sharpe_annualized`, `portfolio_annualized_return`, `portfolio_max_drawdown`, `portfolio_turnover`, `prediction_rank_ic`.
- `run_key` is unique in `master_comparison.csv`.
- `ic_vs_sharpe_reb1.png`, `ic_vs_sharpe_reb5.png`, `risk_frontier_reb1.png`, `risk_frontier_reb5.png` show multiple points (for this matrix).
- `thesis/references.bib` contains no placeholder `Unknown` metadata.

## Fast Audits
```bash
./.venv_gnn/bin/python scripts/audit_bibliography.py
./.venv_gnn/bin/python -m pytest -q tests/test_ledger_dedup.py tests/test_report_integrity.py tests/test_bibliography_audit.py
```
