#!/usr/bin/env python3
"""
Sweep `emb_scale` for the XGB + Node2Vec pipeline and collect mean IC / hit-rate.

Usage (from repo root):
  source .venv_gnn/bin/activate
  PYTHONPATH=. python3 scripts/ablate_emb_scale.py --scales 0.1,0.2,0.5,1.0 --rebuild-cache

This script imports the project's `train_xgboost` entrypoint and runs the
training loop for each scale. It expects the environment to have the same
dependencies as training (pyarrow, xgboost, torch_geometric if Node2Vec is used).

Note: runs can be slow; run on your local `.venv_gnn`.
"""
import argparse
import csv
import os
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_normalize import load_config as load_normalized_config


def load_config(path):
    return load_normalized_config(path, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/runs/core/xgb_node2vec_corr.yaml")
    parser.add_argument("--scales", type=str, default="0.1,0.2,0.5,1.0")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    scales = [float(s) for s in args.scales.split(",")]
    cfg_base = load_config(args.config)

    results = []
    for s in scales:
        cfg = deepcopy(cfg_base)
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model"]["emb_scale"] = float(s)
        # ensure cache rebuild flag is set if requested
        cfg["cache"] = {"rebuild": args.rebuild_cache}

        print(f"Running emb_scale={s}")

        # Run training programmatically
        from trainers.train_xgboost import train_xgboost

        try:
            train_xgboost(cfg)
        except Exception as e:
            print(f"Run failed for emb_scale={s}: {e}")
            results.append({"emb_scale": s, "mean_ic": None, "mean_hit": None, "error": str(e)})
            continue

        out_dir = cfg.get("evaluation", {}).get("out_dir", "experiments/xgb_node2vec_corr")
        metrics_fp = os.path.join(out_dir, "xgb_node2vec_daily_metrics.csv")
        mean_ic = None
        mean_hit = None
        if os.path.exists(metrics_fp):
            import pandas as pd

            dfm = pd.read_csv(metrics_fp)
            if not dfm.empty:
                mean_ic = float(dfm["ic"].mean())
                mean_hit = float(dfm["hit"].mean())

        results.append({"emb_scale": s, "mean_ic": mean_ic, "mean_hit": mean_hit, "error": None})

    # write summary CSV to the base out_dir
    summary_out = cfg_base.get("evaluation", {}).get("out_dir", "experiments/xgb_node2vec_corr")
    os.makedirs(summary_out, exist_ok=True)
    summary_fp = os.path.join(summary_out, "emb_scale_ablation.csv")
    keys = ["emb_scale", "mean_ic", "mean_hit", "error"]
    with open(summary_fp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Ablation complete. Summary written to {summary_fp}")


if __name__ == "__main__":
    main()
