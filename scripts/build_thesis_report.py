#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"[build-report] running {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command thesis artifact build (tuned-all source of truth).")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--run-matrix", action="store_true", help="Run tuned-all matrix before report generation.")
    parser.add_argument("--budget", type=str, default="medium", choices=["quick", "medium", "heavy"])
    parser.add_argument("--fresh-results", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-pdf", action="store_true", help="Compile thesis/main.tex at the end.")
    parser.add_argument("--results", type=str, default="results/results_tuned_all.jsonl")
    parser.add_argument("--report-out", type=str, default="results/reports/thesis_tuned_all")
    parser.add_argument("--expected-runs", type=int, default=45)
    parser.add_argument("--skip-prediction-audit", action="store_true")
    args = parser.parse_args()

    py = args.python
    results = args.results
    report_out = args.report_out

    if args.run_matrix:
        cmd = [py, "scripts/run_all_models_tuned_matrix.py", "--budget", args.budget]
        if args.fresh_results:
            cmd.append("--fresh-results")
        if args.rebuild_cache:
            cmd.append("--rebuild-cache")
        _run(cmd)

    _run([py, "scripts/deduplicate_results_ledger.py", "--results", results])

    report_cmd = [
        py,
        "scripts/generate_thesis_report.py",
        "--results",
        results,
        "--out",
        report_out,
        "--expected-runs",
        str(args.expected_runs),
    ]
    if args.skip_prediction_audit:
        report_cmd.append("--skip-prediction-audit")
    _run(report_cmd)

    env = os.environ.copy()
    env["THESIS_REPORT_DIR"] = str(REPO_ROOT / report_out)
    _run([py, "thesis/scripts/export_tables.py"], env=env)

    if args.build_pdf:
        _run(["make", "pdf", f"PYTHON={py}"], cwd=REPO_ROOT / "thesis")

    print(f"[build-report] done. report: {(REPO_ROOT / report_out).resolve()}")


if __name__ == "__main__":
    main()
