"""Inspect GNN snapshot caches and print diagnostics.

Usage:
  python3 scripts/inspect_snapshots.py --config configs/runs/core/gcn_corr_sector_granger.yaml

This script will:
- Load the YAML config
- Compute the same cache id used by `train_gnn` and try to load cached snapshots
- If cache missing, call `_build_snapshots_and_targets` to construct snapshots
- Print overall statistics and per-snapshot samples (nodes, edges, feature stats, target stats)
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import sys

# Ensure repository root is on sys.path so `utils` and `trainers` imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cache import cache_key, cache_path, cache_load
from utils.config_normalize import load_config as load_normalized_config


def compute_cache_id(cfg):
    graph_flags = cfg.get("graph", {})
    gr_cfg = cfg.get("granger", {})
    key = {
        "model": cfg.get("model", {}),
        "data": cfg.get("data", {}),
        "graph": {
            "use_corr": bool(graph_flags.get("use_corr", False)),
            "use_sector": bool(graph_flags.get("use_sector", False)),
            "use_granger": bool(graph_flags.get("use_granger", False)),
            "corr_top_k": int(graph_flags.get("corr_top_k", 10)),
            "corr_min_periods": int(graph_flags.get("corr_min_periods", 0)),
        },
        "granger": {
            "max_lag": int(gr_cfg.get("max_lag", 2)),
            "p_threshold": float(gr_cfg.get("p_threshold", 0.05)),
        },
    }
    return cache_key(key, dataset_version="gnn_snapshots", extra_files=[cfg["data"]["price_file"], "data/processed/universe.csv"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_normalized_config(args.config, REPO_ROOT)

    cache_id = compute_cache_id(cfg)
    cache_file = cache_path("gnn_snapshots", cache_id)
    print(f"Looking for cache: {cache_file}")

    cached = cache_load(cache_file)
    if cached is None:
        print("Cache not found. Building snapshots by importing trainer builder... this may take time.")
        # import builder function dynamically
        from trainers.train_gnn import _build_snapshots_and_targets
        snapshots, feat_cols, dates = _build_snapshots_and_targets(cfg)
        print(f"Built {len(snapshots)} snapshots; saving to cache")
        # save via cache_save to be consistent
        from utils.cache import cache_save
        cache_save(cache_file, {"snapshots": snapshots, "feat_cols": feat_cols, "dates": dates})
        cached = {"snapshots": snapshots, "feat_cols": feat_cols, "dates": dates}
    else:
        print("Loaded snapshots from cache")

    snapshots = cached["snapshots"]
    feat_cols = cached["feat_cols"]
    dates = cached.get("dates", [])

    print(f"Total snapshots: {len(snapshots)}")
    if not snapshots:
        print("No snapshots produced. Check date range, lookback window, and min_history in the config.")
        return

    node_counts = [int(g.x.shape[0]) for g in snapshots]
    edge_counts = [int(g.edge_index.shape[1]) for g in snapshots]
    valid_node_counts = []
    feats_zero_frac = []
    y_means = []
    y_stds = []

    for g in snapshots:
        valid_mask = g.valid_mask.numpy() if hasattr(g, "valid_mask") else np.ones(g.x.shape[0], dtype=bool)
        valid_node_counts.append(int(valid_mask.sum()))
        x = g.x.numpy()
        # fraction of features that are exactly zero across valid nodes
        if valid_mask.sum() > 0:
            zx = (np.abs(x[valid_mask]) < 1e-8)
            feats_zero_frac.append(zx.mean())
            y = g.y.numpy()
            y_means.append(float(np.nanmean(y[valid_mask])))
            y_stds.append(float(np.nanstd(y[valid_mask])))
        else:
            feats_zero_frac.append(1.0)
            y_means.append(float('nan'))
            y_stds.append(float('nan'))

    print(f"Mean nodes per snapshot: {np.mean(node_counts):.2f}")
    print(f"Mean valid nodes per snapshot: {np.mean(valid_node_counts):.2f}")
    print(f"Mean directed edges per snapshot: {np.mean(edge_counts):.2f}")
    print(f"Mean fraction of feature entries that are zero (per-snapshot, valid nodes): {np.mean(feats_zero_frac):.3f}")
    print(f"Mean target mean across snapshots: {np.nanmean(y_means):.6f}, mean target std: {np.nanmean(y_stds):.6f}")

    # print sample snapshot with more detail
    sample_idx = 0
    g = snapshots[sample_idx]
    print(f"\nSample snapshot {sample_idx} date={getattr(g, 'date', 'NA')} nodes={g.x.shape[0]} edges={g.edge_index.shape[1]}")
    valid_mask = g.valid_mask.numpy() if hasattr(g, "valid_mask") else np.ones(g.x.shape[0], dtype=bool)
    x = g.x.numpy()
    y = g.y.numpy()
    print(f"Valid nodes: {int(valid_mask.sum())}/{g.x.shape[0]}")
    print(f"Feature dims: {g.x.shape[1]} (showing per-feature mean/std over valid nodes)")
    means = np.mean(x[valid_mask], axis=0) if valid_mask.sum() > 0 else np.zeros(x.shape[1])
    stds = np.std(x[valid_mask], axis=0) if valid_mask.sum() > 0 else np.zeros(x.shape[1])
    for i, (m, s) in enumerate(zip(means, stds)):
        print(f"  feat_{i}: mean={m:.4f} std={s:.4f}")

    print(f"Target: mean={np.nanmean(y[valid_mask]):.6f} std={np.nanstd(y[valid_mask]):.6f}")

    ew = g.edge_weight.numpy() if hasattr(g, 'edge_weight') else np.array([])
    if ew.size:
        print(f"Edge weight: min={ew.min():.4f} max={ew.max():.4f} mean={ew.mean():.4f}")
    else:
        print("No edge weights present for this snapshot")

    # Check if features are constant across valid nodes (which would make model output near-constant)
    if valid_mask.sum() > 1:
        row_var = np.var(x[valid_mask], axis=0)
        n_zero_var = np.sum(row_var < 1e-8)
        print(f"Number of feature dims with near-zero variance across valid nodes: {n_zero_var}/{x.shape[1]}")


if __name__ == '__main__':
    main()
