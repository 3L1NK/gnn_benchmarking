# Thesis LaTeX Workspace

This directory is a LaTeX writing workspace that lives inside the benchmarking repo.
It is prewired to use artifacts generated under `results/reports/thesis/`.

## Structure

- `main.tex`: thesis entry point
- `metadata.tex`: title-page metadata (name, advisor, date)
- `preamble.tex`: shared packages and formatting
- `chapters/`: chapter content stubs
- `appendix/`: appendix material
- `references.bib`: BibTeX database
- `scripts/export_tables.py`: converts report CSVs to LaTeX tables
- `tables/generated/`: auto-generated `.tex` tables included by chapters

## Typical Workflow

1. Regenerate thesis report artifacts (from repo root):

```bash
python scripts/generate_thesis_report.py --results results/results.jsonl --out results/reports/thesis
```

2. Regenerate LaTeX tables:

```bash
python thesis/scripts/export_tables.py
```

By default, table export reads from `results/reports/thesis_tuned_all/` if that folder exists; otherwise it falls back to `results/reports/thesis/`.  
Override explicitly when needed:

```bash
THESIS_REPORT_DIR=results/reports/thesis python thesis/scripts/export_tables.py
```

3. Build PDF (from `thesis/`):

```bash
make pdf
```

## Build Commands

From `thesis/`:

- `make pdf`: export tables, then compile once
- `make watch`: export tables, then watch-and-rebuild
- `make clean`: remove LaTeX build artifacts
- `make report`: refresh `results/reports/thesis/*` from `results/results.jsonl`

If your Python environment is not default `python3`, override:

```bash
make pdf PYTHON=../.venv_gnn/bin/python
```
