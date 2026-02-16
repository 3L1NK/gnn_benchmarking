#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CORE_MATRIX_CONFIGS = [
    "configs/runs/core/xgb_raw.yaml",
    "configs/runs/core/lstm.yaml",
    "configs/runs/core/xgb_node2vec_corr.yaml",
    "configs/runs/core/gcn_corr_only.yaml",
    "configs/runs/core/gcn_sector_only.yaml",
    "configs/runs/core/gcn_granger_only.yaml",
    "configs/runs/core/gcn_corr_sector_granger.yaml",
    "configs/runs/core/gat_corr_only.yaml",
    "configs/runs/core/gat_sector_only.yaml",
    "configs/runs/core/gat_granger_only.yaml",
    "configs/runs/core/gat_corr_sector_granger.yaml",
    "configs/runs/core/tgcn_static_corr_only.yaml",
    "configs/runs/core/tgat_static_corr_only.yaml",
]


def _reset_outputs(repo_root: Path) -> None:
    results_file = repo_root / "results" / "results.jsonl"
    if results_file.exists():
        results_file.unlink()
    report_dir = repo_root / "results" / "reports" / "thesis"
    if report_dir.exists():
        for p in report_dir.glob("*"):
            if p.is_file():
                p.unlink()


def run_matrix(repo_root: Path, rebuild_cache: bool = False) -> int:
    for cfg in CORE_MATRIX_CONFIGS:
        cmd = [sys.executable, "train.py", "--config", cfg]
        if rebuild_cache:
            cmd.append("--rebuild-cache")
        print(f"[matrix] running {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=repo_root)
        if proc.returncode != 0:
            print(f"[matrix] failed at {cfg} with code {proc.returncode}")
            return proc.returncode
    print("[matrix] completed core thesis matrix")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run the core thesis benchmark matrix.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--fresh-results", action="store_true", help="Delete results/results.jsonl and thesis report artifacts before running.")
    parser.add_argument("--with-report", action="store_true", help="Generate thesis report after successful matrix run.")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if args.fresh_results:
        _reset_outputs(repo_root)
    code = run_matrix(repo_root, rebuild_cache=args.rebuild_cache)
    if code == 0 and args.with_report:
        report_cmd = [
            sys.executable,
            "scripts/generate_thesis_report.py",
            "--results",
            "results/results.jsonl",
            "--out",
            "results/reports/thesis",
        ]
        print(f"[matrix] running {' '.join(report_cmd)}")
        proc = subprocess.run(report_cmd, cwd=repo_root)
        code = proc.returncode
    raise SystemExit(code)


if __name__ == "__main__":
    main()
