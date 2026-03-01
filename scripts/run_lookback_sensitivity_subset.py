#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_normalize import load_config


TARGETS = [
    ("configs/runs/core/xgb_node2vec_corr.yaml", "xgb_node2vec_corr"),
    ("configs/runs/core/xgb_raw.yaml", "xgb_raw"),
    ("configs/runs/core/gat_corr_sector_granger.yaml", "gat_corr_sector_granger"),
]
LOOKBACKS = [14, 30, 60]
RESULTS_PATH = "results/results_lookback_sensitivity.jsonl"
REPORT_OUT = "results/reports/lookback_sensitivity"
GEN_DIR = REPO_ROOT / "configs" / "runs" / "sensitivity_generated" / "lookback"


def _write_cfg(src_cfg: str, run_prefix: str, lookback: int) -> Path:
    cfg = load_config(src_cfg, REPO_ROOT)
    cfg = deepcopy(cfg)
    cfg["experiment_name"] = f"{run_prefix}_lb{int(lookback)}_sensitivity"
    cfg.setdefault("data", {})
    cfg["data"]["lookback_window"] = int(lookback)
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["results_path"] = RESULTS_PATH
    cfg["evaluation"]["primary_rebalance_freq"] = 5
    cfg["evaluation"]["backtest_policies"] = [5]

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEN_DIR / f"{run_prefix}_lb{int(lookback)}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def _run(cmd: list[str]) -> int:
    print("[lookback]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lookback sensitivity subset (14/30/60).")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-report", action="store_true")
    parser.add_argument("--fresh-results", action="store_true")
    args = parser.parse_args()

    if args.fresh_results:
        p = REPO_ROOT / RESULTS_PATH
        if p.exists():
            p.unlink()

    cfgs: list[Path] = []
    for src, prefix in TARGETS:
        for lb in LOOKBACKS:
            cfgs.append(_write_cfg(src, prefix, lb))

    print(f"[lookback] generated {len(cfgs)} configs in {GEN_DIR}")
    if args.dry_run:
        return 0

    for cfg in cfgs:
        code = _run([args.python, "train.py", "--config", str(cfg)])
        if code != 0:
            return code

    code = _run([args.python, "scripts/deduplicate_results_ledger.py", "--results", RESULTS_PATH])
    if code != 0:
        return code

    if args.with_report:
        # eval_runner enforces reb=1/5/21 outputs; 9 config runs -> 27 rows
        code = _run(
            [
                args.python,
                "scripts/generate_thesis_report.py",
                "--results",
                RESULTS_PATH,
                "--out",
                REPORT_OUT,
                "--expected-runs",
                "27",
                "--skip-prediction-audit",
            ]
        )
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

