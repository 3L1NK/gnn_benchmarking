#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.artifacts import resolve_output_dirs
from utils.config_normalize import load_config
from utils.prediction_audit import PredictionAuditIssue, audit_prediction_file


CORE_MATRIX_CONFIGS = [
    "configs/runs/core/xgb_raw.yaml",
    "configs/runs/core/mlp.yaml",
    "configs/runs/core/lstm.yaml",
    "configs/runs/core/xgb_node2vec_corr.yaml",
    "configs/runs/core/xgb_graphlasso_linear.yaml",
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


def _prediction_files_for_config(cfg: dict) -> List[Path]:
    out_dirs = resolve_output_dirs(cfg, model_type=cfg.get("model", {}).get("type"))
    roots = [out_dirs.canonical]
    if out_dirs.legacy is not None:
        roots.append(out_dirs.legacy)
    files: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.glob("*_predictions.csv"):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            files.append(rp)
    return sorted(files)


def _stale_issues_for_config(cfg_path: Path, cfg: dict, include_missing: bool) -> List[PredictionAuditIssue]:
    files = _prediction_files_for_config(cfg)
    issues: List[PredictionAuditIssue] = []
    if not files:
        if include_missing:
            issues.append(
                PredictionAuditIssue(
                    path=str(cfg_path),
                    issue="missing_prediction_artifact",
                    details="no *_predictions.csv found in canonical/legacy output dirs",
                )
            )
        return issues

    for p in files:
        file_issues = audit_prediction_file(p)
        # Only treat hard prediction integrity issues as stale.
        for item in file_issues:
            if item.issue in {"duplicate_pairs", "missing_columns", "invalid_dates", "read_error"}:
                issues.append(item)
    return issues


def _build_gnn_inference_override(config_path: Path) -> Path:
    payload = {
        "include": str(config_path),
        "training": {
            "skip_training_if_checkpoint": True,
        },
    }
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return Path(handle.name)


def _run_train(py_exe: str, cfg_path: Path, rebuild_cache: bool) -> int:
    cmd = [py_exe, "train.py", "--config", str(cfg_path)]
    if rebuild_cache:
        cmd.append("--rebuild-cache")
    print(f"[rerun] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(proc.returncode)


def _collect_targets(configs: Sequence[str]) -> List[Path]:
    return [Path(c).resolve() for c in configs]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun only stale core thesis configs whose prediction artifacts are invalid."
    )
    parser.add_argument("--config", action="append", default=[], help="Optional specific config path(s).")
    parser.add_argument("--dry-run", action="store_true", help="Only detect stale configs; do not rerun.")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Treat missing prediction artifact files as stale and rerun them too.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap on number of stale configs rerun.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to invoke train.py.")
    args = parser.parse_args()

    config_paths = _collect_targets(args.config or CORE_MATRIX_CONFIGS)
    loaded: Dict[Path, dict] = {}
    for c in config_paths:
        loaded[c] = load_config(c, REPO_ROOT)

    stale: List[Tuple[Path, List[PredictionAuditIssue]]] = []
    for c in config_paths:
        issues = _stale_issues_for_config(c, loaded[c], include_missing=bool(args.include_missing))
        if issues:
            stale.append((c, issues))

    print(f"[rerun] checked {len(config_paths)} config(s)")
    print(f"[rerun] stale config(s): {len(stale)}")
    for c, issues in stale:
        print(f" - {c}")
        for item in issues[:3]:
            print(f"   * {item.issue}: {item.details}")
        if len(issues) > 3:
            print(f"   * ... {len(issues) - 3} more issue(s)")

    if args.dry_run or not stale:
        return 0

    run_count = 0
    for c, _ in stale:
        if args.max_runs and run_count >= int(args.max_runs):
            break
        cfg = loaded[c]
        family = str(cfg.get("model", {}).get("family", "")).lower()

        temp_override = None
        run_cfg = c
        try:
            if family == "gnn":
                # For stale duplicate predictions caused by old mapping logic, we can
                # regenerate with fixed inference using existing best checkpoints.
                temp_override = _build_gnn_inference_override(c)
                run_cfg = temp_override

            code = _run_train(args.python, run_cfg, rebuild_cache=bool(args.rebuild_cache))
            if code != 0:
                print(f"[rerun] failed: {c} (exit={code})")
                return code
            run_count += 1
        finally:
            if temp_override is not None and temp_override.exists():
                temp_override.unlink(missing_ok=True)

    # Post-check rerun targets
    failed_post = 0
    for c, _ in stale[:run_count if args.max_runs else None]:
        issues = _stale_issues_for_config(c, loaded[c], include_missing=bool(args.include_missing))
        if issues:
            failed_post += 1
            print(f"[rerun] post-check failed for {c}")
            for item in issues[:5]:
                print(f"   * {item.issue}: {item.details}")
    if failed_post:
        print(f"[rerun] {failed_post} rerun config(s) still have stale prediction artifacts.")
        return 1

    print(f"[rerun] completed reruns: {run_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
