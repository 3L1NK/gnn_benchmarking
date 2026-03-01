#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_normalize import load_config


TARGETS = [
    ("configs/runs/tuned_all/xgb_node2vec_corr_tuned_all.yaml", "xgb_node2vec_corr"),
    ("configs/runs/tuned_all/xgb_raw_tuned_all.yaml", "xgb_raw"),
    ("configs/runs/tuned_all/gat_corr_sector_granger_tuned_all.yaml", "gat_corr_sector_granger"),
]
TEST_YEARS = [2020, 2021, 2022, 2023, 2024]

GEN_DIR = REPO_ROOT / "configs" / "runs" / "rolling_cv_subset_generated"
DEFAULT_RESULTS_PATH = "results/ledger/ledger.jsonl"
DEFAULT_RESULTS_CSV = "results/ledger/ledger.csv"
DEFAULT_REPORT_OUT = "results/reports/thesis_vX"


def _run(cmd: list[str], *, env: dict | None = None) -> int:
    print("[rolling-cv]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return int(proc.returncode)


def _expected_run_tags() -> set[str]:
    return {f"{prefix}_rolling_cv_y{year}" for _, prefix in TARGETS for year in TEST_YEARS}


def _build_cfg(src_cfg: str, run_prefix: str, test_year: int, results_path: str) -> dict:
    cfg = load_config(src_cfg, REPO_ROOT)
    cfg = deepcopy(cfg)
    test_start = f"{int(test_year)}-01-01"
    test_end = f"{int(test_year)}-12-31"

    cfg["experiment_name"] = f"{run_prefix}_rolling_cv_y{int(test_year)}"
    cfg.setdefault("data", {})
    cfg["data"]["end_date"] = test_end

    cfg.setdefault("training", {})
    # Keep train_end deterministic at Dec 31 of test_year-1 via val_start == test_start.
    cfg["training"]["val_start"] = test_start
    cfg["training"]["test_start"] = test_start

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["results_path"] = results_path
    cfg["evaluation"]["out_dir"] = "results/runs/"
    cfg["evaluation"]["transaction_cost_bps"] = 0
    cfg["evaluation"]["primary_rebalance_freq"] = 5
    cfg["evaluation"]["backtest_policies"] = [5]

    cfg.setdefault("tuning", {})
    cfg["tuning"]["enabled"] = False

    cfg.setdefault("experiment", {})
    cfg["experiment"]["protocol_version"] = "v1_rolling_cv_subset"
    cfg["experiment"]["enforce_protocol"] = False
    return cfg


def _write_cfg(src_cfg: str, run_prefix: str, test_year: int, results_path: str) -> Path:
    cfg = _build_cfg(src_cfg, run_prefix, test_year, results_path)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_DIR / f"{run_prefix}_rolling_cv_y{int(test_year)}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def _refresh_csv_ledger(results_path: str, ledger_csv: str) -> pd.DataFrame:
    src = REPO_ROOT / results_path
    if not src.exists():
        raise FileNotFoundError(f"rolling CV ledger jsonl not found: {src}")
    df = pd.read_json(src, lines=True)
    out = REPO_ROOT / ledger_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _validate_ledger(df: pd.DataFrame) -> None:
    expected = _expected_run_tags()
    run_tags = set(df.get("run_tag", pd.Series(dtype=object)).astype(str).tolist())
    missing = sorted(expected - run_tags)
    if missing:
        raise ValueError(f"Missing rolling CV run tags in ledger: {missing}")
    present = sorted(expected & run_tags)
    if len(present) != len(expected):
        raise ValueError(f"Expected {len(expected)} rolling run tags, found {len(present)}")


def _validate_rolling_summary(report_out: str) -> None:
    summary_path = REPO_ROOT / report_out / "rolling_cv_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing rolling CV summary: {summary_path}")
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError("rolling_cv_summary.csv is empty")

    req_numeric = [
        "portfolio_sharpe_annualized",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
        "portfolio_turnover",
    ]
    for col in req_numeric:
        if col not in df.columns:
            raise ValueError(f"rolling_cv_summary.csv missing column: {col}")
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            raise ValueError(f"rolling_cv_summary.csv has NaN values in {col}")

    folds = df[df.get("summary_scope", "") == "fold"].copy()
    if folds.empty:
        raise ValueError("rolling_cv_summary.csv has no fold-level rows")
    years = sorted(pd.to_numeric(folds.get("fold_year"), errors="coerce").dropna().astype(int).unique().tolist())
    if years != TEST_YEARS:
        raise ValueError(f"Fold years mismatch: expected {TEST_YEARS}, got {years}")

    for year in TEST_YEARS:
        y_rows = folds[pd.to_numeric(folds["fold_year"], errors="coerce") == int(year)].copy()
        if y_rows.empty:
            raise ValueError(f"No fold rows for year {year}")
        configured = y_rows.get("configured_test_start", pd.Series(dtype=object)).astype(str).tolist()
        expected_prefix = f"{int(year)}-01-01"
        if not any(str(v).startswith(expected_prefix) for v in configured):
            raise ValueError(f"Configured test start mismatch for fold {year}: {configured[:3]}")
        obs_start = pd.to_datetime(y_rows.get("observed_test_start"), errors="coerce")
        obs_end = pd.to_datetime(y_rows.get("observed_test_end"), errors="coerce")
        if obs_start.isna().all() or obs_end.isna().all():
            raise ValueError(f"Observed test window missing for fold {year}")
        if not (obs_start.dropna().dt.year.eq(year).all() and obs_end.dropna().dt.year.eq(year).all()):
            raise ValueError(f"Observed test window year mismatch for fold {year}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rolling walk-forward CV subset (3 models x 5 yearly folds).")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--results-path", type=str, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--ledger-csv", type=str, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--report-out", type=str, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--fresh-ledger", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-report", action="store_true")
    args = parser.parse_args()

    if args.fresh_ledger:
        p = REPO_ROOT / args.results_path
        if p.exists():
            p.unlink()
        csv = REPO_ROOT / args.ledger_csv
        if csv.exists():
            csv.unlink()

    cfg_paths: list[Path] = []
    for src, prefix in TARGETS:
        for year in TEST_YEARS:
            cfg_paths.append(_write_cfg(src, prefix, year, args.results_path))
    print(f"[rolling-cv] generated {len(cfg_paths)} configs in {GEN_DIR}")

    if args.dry_run:
        return 0

    for cfg in cfg_paths:
        code = _run([args.python, "train.py", "--config", str(cfg)])
        if code != 0:
            return code

    code = _run([args.python, "scripts/deduplicate_results_ledger.py", "--results", args.results_path])
    if code != 0:
        return code

    ledger_df = _refresh_csv_ledger(args.results_path, args.ledger_csv)
    _validate_ledger(ledger_df)

    if args.with_report:
        report_cmd = [
            args.python,
            "scripts/generate_thesis_report.py",
            "--results",
            args.results_path,
            "--out",
            args.report_out,
            "--skip-prediction-audit",
        ]
        code = _run(report_cmd)
        if code != 0:
            return code

        env = os.environ.copy()
        env["THESIS_REPORT_DIR"] = str((REPO_ROOT / args.report_out).resolve())
        code = _run([args.python, "thesis/scripts/export_tables.py"], env=env)
        if code != 0:
            return code

        _validate_rolling_summary(args.report_out)

    print("[rolling-cv] completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
