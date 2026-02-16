#!/usr/bin/env python3
"""
Run all ablation configs in a folder and record basic results to results/manifest.csv.

Usage:
  python3 scripts/run_ablation.py configs/runs/ablation/tgcn

This script resolves `include` chains (same logic as `train.py`), runs
`python3 train.py --config <cfg>` for each YAML, and then attempts to read the
produced equity curve to extract a final value. Results are appended to
`results/manifest.csv`.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import csv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_normalize import load_config

FOLDER_ALIASES = {
    "configs/ablation/tgcn": "configs/runs/ablation/tgcn",
    "configs/ablation/tgat": "configs/runs/ablation/tgat",
    "configs/models/tgcn/ablation": "configs/runs/ablation/tgcn",
    "configs/models/tgat/ablation": "configs/runs/ablation/tgat",
}


def find_equity_file(cfg, cfg_path):
    # Canonical path uses evaluation.out_dir directly; legacy GNN path also exists.
    out = Path(cfg.get("evaluation", {}).get("out_dir", "experiments/"))
    mtype = cfg.get("model", {}).get("type")
    if mtype is None:
        # try to infer from included base (shouldn't happen if includes resolved)
        mtype = Path(cfg_path).stem
    out_dir = out if isinstance(out, Path) else Path(out)
    canonical = out_dir / f"{mtype}_equity_curve.csv"
    if canonical.exists():
        return canonical, out_dir
    legacy_dir = out_dir / str(mtype)
    legacy = legacy_dir / f"{mtype}_equity_curve.csv"
    return legacy, legacy_dir


def extract_final_value(eq_file):
    if not eq_file.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(eq_file, index_col=0)
        # assume last row value column
        if df.shape[1] == 0:
            return None
        last = df.iloc[-1, 0]
        return float(last)
    except Exception:
        return None


def main(folder):
    folder = Path(FOLDER_ALIASES.get(str(Path(folder).as_posix()), str(folder)))
    if not folder.exists() or not folder.is_dir():
        print(f"Folder {folder} does not exist")
        sys.exit(1)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    manifest = results_dir / "manifest.csv"
    headers = ["config", "model", "out_dir", "final_value", "status"]
    if not manifest.exists():
        with manifest.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    for cfg_path in sorted(folder.glob("*.yaml")):
        print(f"Running config: {cfg_path}")
        proc = subprocess.run([sys.executable, "train.py", "--config", str(cfg_path)], cwd=REPO_ROOT)
        status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"

        try:
            cfg = load_config(cfg_path, REPO_ROOT)
        except Exception as e:
            print(f"Failed to load config for postprocessing: {e}")
            cfg = {}

        eq_file, model_dir = find_equity_file(cfg, cfg_path)
        final_value = extract_final_value(eq_file) if eq_file is not None else None

        with manifest.open("a") as f:
            writer = csv.writer(f)
            writer.writerow([str(cfg_path), cfg.get("model", {}).get("type", "unknown"), str(model_dir), final_value if final_value is not None else "", status])

        print(f"Finished {cfg_path}: status={status}, final_value={final_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all ablation configs in a folder.")
    parser.add_argument("folder", help="Ablation folder, e.g. configs/runs/ablation/tgcn")
    args = parser.parse_args()
    main(args.folder)
