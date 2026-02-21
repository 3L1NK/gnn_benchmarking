#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RETUNE_CONFIGS = [
    "configs/runs/retune_medium/gcn_granger_only.yaml",
    "configs/runs/retune_medium/gcn_sector_only.yaml",
    "configs/runs/retune_medium/xgb_node2vec_corr.yaml",
    "configs/runs/retune_medium/lstm.yaml",
]


def _reset_outputs(repo_root: Path, report_out: str) -> None:
    results_file = repo_root / "results" / "results_retune.jsonl"
    if results_file.exists():
        results_file.unlink()
    report_dir = repo_root / report_out
    if report_dir.exists():
        for p in report_dir.glob("*"):
            if p.is_file():
                p.unlink()


def run_matrix(repo_root: Path, rebuild_cache: bool = False) -> int:
    for cfg in RETUNE_CONFIGS:
        cmd = [sys.executable, "train.py", "--config", cfg]
        if rebuild_cache:
            cmd.append("--rebuild-cache")
        print(f"[retune] running {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=repo_root)
        if proc.returncode != 0:
            print(f"[retune] failed at {cfg} with code {proc.returncode}")
            return proc.returncode
    print("[retune] completed targeted retune matrix")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted medium-budget retune matrix.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--fresh-results",
        action="store_true",
        help="Delete results/results_retune.jsonl and retune report files before running.",
    )
    parser.add_argument(
        "--with-report",
        action="store_true",
        help="Generate thesis-style report for retune results after successful matrix run.",
    )
    parser.add_argument("--report-out", type=str, default="results/reports/thesis_retune")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.fresh_results:
        _reset_outputs(repo_root, args.report_out)

    code = run_matrix(repo_root, rebuild_cache=args.rebuild_cache)
    if code == 0 and args.with_report:
        report_cmd = [
            sys.executable,
            "scripts/generate_thesis_report.py",
            "--results",
            "results/results_retune.jsonl",
            "--out",
            args.report_out,
        ]
        print(f"[retune] running {' '.join(report_cmd)}")
        proc = subprocess.run(report_cmd, cwd=repo_root)
        code = proc.returncode
    raise SystemExit(code)


if __name__ == "__main__":
    main()
