#!/usr/bin/env python3
"""
Run all ablation configs in a folder and record basic results to results/manifest.csv.

Usage:
  python3 scripts/run_ablation.py configs/ablation/tgcn

This script resolves `include` chains (same logic as `train.py`), runs
`python3 train.py --config <cfg>` for each YAML, and then attempts to read the
produced equity curve to extract a final value. Results are appended to
`results/manifest.csv`.
"""
import subprocess
import sys
from pathlib import Path
import csv
import yaml
from copy import deepcopy


def _deep_update(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path):
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    include_path = cfg.pop("include", None)
    if include_path:
        include_path = Path(include_path)
        if not include_path.is_absolute():
            candidate = (config_path.parent / include_path).resolve()
            if candidate.exists():
                include_path = candidate
            else:
                include_path = (Path(__file__).resolve().parent.parent / include_path).resolve()
        if not include_path.exists():
            raise FileNotFoundError(f"Included config '{include_path}' does not exist")
        base = deepcopy(load_config(include_path))
        return _deep_update(base, cfg)

    return cfg


def find_equity_file(cfg, cfg_path):
    # Determine out_dir used by train.py: Path(config['evaluation']['out_dir']) / config['model']['type']
    out = Path(cfg.get("evaluation", {}).get("out_dir", "experiments/"))
    mtype = cfg.get("model", {}).get("type")
    if mtype is None:
        # try to infer from included base (shouldn't happen if includes resolved)
        mtype = Path(cfg_path).stem
    out_dir = out if isinstance(out, Path) else Path(out)
    model_dir = out_dir / str(mtype)
    eq_file = model_dir / f"{mtype}_equity_curve.csv"
    return eq_file, model_dir


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
    folder = Path(folder)
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
        proc = subprocess.run([sys.executable, "train.py", "--config", str(cfg_path)], cwd=Path(__file__).resolve().parent.parent)
        status = "ok" if proc.returncode == 0 else f"failed:{proc.returncode}"

        try:
            cfg = load_config(cfg_path)
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
    if len(sys.argv) < 2:
        print("Usage: scripts/run_ablation.py <configs/ablation/tgcn|tgat>  # static baselines")
        sys.exit(1)
    main(sys.argv[1])
