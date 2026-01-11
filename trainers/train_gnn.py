# trainers/train_gnn.py

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import amp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.cuda.amp import GradScaler

from models.gnn_model import StaticGNN
try:
    from models.tgcn_model import StaticTGCN
except Exception:
    StaticTGCN = None
try:
    from models.tgat_model import StaticTGAT
except Exception:
    StaticTGAT = None
from utils.seeds import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.metrics import rank_ic, hit_rate, sharpe_ratio
from utils.backtest import backtest_long_only
from utils.baseline import get_global_buy_and_hold
from utils.plot import (
    plot_daily_ic,
    plot_ic_hist,
    plot_equity_curve,
    plot_equity_comparison,
)
from utils.cache import cache_load, cache_save, cache_key, cache_path
from utils.device import get_device, default_num_workers
from utils.sanity import check_tensor
from utils.targets import build_target
from utils.preprocessing import scale_features
from utils.splits import split_time


def _build_snapshots_and_targets(config):
    debug = bool(config.get("debug", {}).get("leakage", False))
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    corr_window = config["data"]["corr_window"]
    corr_thr = config["data"]["corr_threshold"]

    # New graph ablation flags (must fully control edge inclusion)
    graph_cfg = config.get("graph", {})
    use_corr = bool(graph_cfg.get("use_corr", False))
    use_sector = bool(graph_cfg.get("use_sector", False))
    use_granger = bool(graph_cfg.get("use_granger", False))
    make_undirected_cfg = graph_cfg.get("make_undirected")
    if make_undirected_cfg is None:
        make_undirected_cfg = graph_cfg.get("force_undirected")
    if make_undirected_cfg is None:
        make_undirected = not use_granger
    else:
        make_undirected = bool(make_undirected_cfg)

    # correlation params
    corr_top_k = int(graph_cfg.get("corr_top_k", 10))
    corr_min_periods = int(graph_cfg.get("corr_min_periods", max(5, corr_window // 2)))
    # per-edge-type degree budgets (mandatory)
    sector_top_k = int(graph_cfg.get("sector_top_k", 5))
    granger_top_k = int(graph_cfg.get("granger_top_k", 5))

    # per-edge-type scalar weights (mandatory)
    w_corr = float(graph_cfg.get("w_corr", 1.0))
    w_sector = float(graph_cfg.get("w_sector", 0.2))
    w_granger = float(graph_cfg.get("w_granger", 0.2))
    # global clamp for merged edge weights to avoid amplification
    max_edge_weight = float(graph_cfg.get("max_edge_weight", 1.0))

    # sector/industry weights (only used when use_sector is True)
    sector_weight = float(graph_cfg.get("sector_weight", 0.2))
    industry_weight = float(graph_cfg.get("industry_weight", 0.1))

    df = load_price_panel(price_file, start, end)
    df, feat_cols = add_technical_features(df)
    if "log_ret_1d" not in df.columns:
        raise ValueError("Expected column 'log_ret_1d' in feature dataframe")

    df, target_col = build_target(df, config, target_col="target")
    feature_cols = list(feat_cols) + ["log_ret_1d"]
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    df_raw = df.copy()
    train_mask, _, _, _ = split_time(df["date"], config, label="gnn", debug=debug)
    pre_cfg = config.get("preprocess", {})
    do_scale = bool(pre_cfg.get("scale_features", True))
    df, _ = scale_features(df, feat_cols, train_mask, label="gnn", debug=debug, scale=do_scale)

    universe_path = Path("data/processed/universe.csv")
    if not universe_path.exists():
        raise FileNotFoundError("Universe metadata is required at data/processed/universe.csv")
    universe_df = pd.read_csv(universe_path)

    universe_list = sorted(universe_df["ticker"].unique().tolist())
    sector_map = dict(zip(universe_df["ticker"], universe_df.get("sector", pd.Series(index=universe_df.index))))
    industry_map = dict(zip(universe_df["ticker"], universe_df.get("industry", pd.Series(index=universe_df.index))))

    ret_pivot = (
        df_raw.pivot(index="date", columns="ticker", values="log_ret_1d")
        .reindex(columns=universe_list)
        .sort_index()
    )

    dates = sorted(df["date"].unique())

    # Precompute Granger edges from train-only data when requested.
    granger_map_dir = {}
    if use_granger:
        from utils.graphs import granger_edges
        val_start = pd.to_datetime(config["training"]["val_start"]) if "training" in config else None
        if val_start is None:
            raise ValueError("training.val_start must be set to compute granger edges from train-only data")
        train_mask, _, _, _ = split_time(df["date"], config, label="gnn_granger", debug=debug)
        if debug and not df.loc[train_mask].empty:
            train_dates = pd.to_datetime(df.loc[train_mask, "date"])
            print(f"[gnn] granger window={train_dates.min().date()}..{train_dates.max().date()}")
        try:
            gr_edges = granger_edges(df_raw.loc[train_mask], max_lag=int(config.get("granger", {}).get("max_lag", 2)), p_threshold=float(config.get("granger", {}).get("p_threshold", 0.05)))
            for u, v, w in gr_edges:
                if u == v:
                    continue
                key = (u, v)
                granger_map_dir[key] = max(granger_map_dir.get(key, 0.0), float(abs(w)))
        except Exception:
            granger_map_dir = {}
    snapshots = []
    meta_dates = []

    for d in dates:
        d = pd.to_datetime(d)
        if d not in ret_pivot.index:
            continue
        idx = ret_pivot.index.get_loc(d)
        if isinstance(idx, slice):
            continue
        if idx < corr_window:
            continue

        window_ret = ret_pivot.iloc[idx - corr_window + 1 : idx + 1]
        window_start = window_ret.index.min()
        window_end = window_ret.index.max()
        if debug:
            print(f"[gnn] snapshot {d.date()} graph_window={window_start.date()}..{window_end.date()}")
        if window_end > d:
            raise ValueError(f"[gnn] Leakage detected: window_end {window_end} exceeds snapshot date {d}")
        universe_today = df[df["date"] == d].set_index("ticker")

        feat_for_date = {}
        target_ret_for_date = {}
        valid_mask = []

        for t in universe_list:
            if t in universe_today.index:
                row = universe_today.loc[t]
                feat_vec = row[feat_cols].values.astype(float)
                y_ret = float(row[target_col])
                valid_mask.append(True)
            else:
                feat_vec = np.zeros(len(feat_cols), dtype=float)
                y_ret = 0.0
                valid_mask.append(False)
            feat_for_date[t] = feat_vec
            target_ret_for_date[t] = y_ret

        tickers_list = list(universe_list)
        n_nodes = len(tickers_list)
        if n_nodes < 2:
            continue

        x = np.vstack([feat_for_date[t] for t in tickers_list])
        y_ret = np.array([target_ret_for_date[t] for t in tickers_list], dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y_ret).all():
            continue

        valid_mask_np = np.array(valid_mask, dtype=bool)
        if not np.any(np.abs(x[valid_mask_np]) > 1e-8):
            continue

        # Build per-type edge weight maps (unordered pairs) and normalize separately
        corr_map = {}
        sector_map_pairs = {}

        # Correlation edges (per-day)
        if use_corr:
            window_sub = window_ret[tickers_list]
            if (~valid_mask_np).any():
                window_sub.loc[:, [t for t, m in zip(tickers_list, valid_mask_np) if not m]] = np.nan
            corr_mat = window_sub.corr(min_periods=corr_min_periods).values
            if corr_mat.shape != (n_nodes, n_nodes):
                continue
            for i in range(n_nodes):
                row = corr_mat[i]
                if len(row) != n_nodes:
                    continue
                row = row.copy()
                row[i] = np.nan
                valid_idx = np.where(np.isfinite(row))[0]
                if corr_top_k > 0 and len(valid_idx) > corr_top_k:
                    top_idx = sorted(valid_idx, key=lambda j: abs(row[j]), reverse=True)[:corr_top_k]
                else:
                    top_idx = valid_idx
                for j in top_idx:
                    if not (valid_mask_np[i] and valid_mask_np[j]):
                        continue
                    w = row[j]
                    if not np.isfinite(w) or abs(w) < corr_thr:
                        continue
                    key = tuple(sorted((i, j)))
                    corr_map[key] = max(corr_map.get(key, 0.0), float(abs(w)))

        # Sector + Industry edges (controlled by use_sector)
        if use_sector and (sector_weight > 0 or industry_weight > 0):
            for i in range(n_nodes):
                ti = tickers_list[i]
                if not valid_mask_np[i]:
                    continue
                si = sector_map.get(ti)
                ii = industry_map.get(ti)
                for j in range(i + 1, n_nodes):
                    tj = tickers_list[j]
                    if not valid_mask_np[j]:
                        continue
                    sj = sector_map.get(tj)
                    ij = industry_map.get(tj)
                    # sector match
                    if sector_weight > 0 and si is not None and sj is not None and si == sj:
                        key = tuple(sorted((i, j)))
                        sector_map_pairs[key] = max(sector_map_pairs.get(key, 0.0), float(sector_weight))
                    # industry match
                    if industry_weight > 0 and ii is not None and ij is not None and ii == ij:
                        key = tuple(sorted((i, j)))
                        sector_map_pairs[key] = max(sector_map_pairs.get(key, 0.0), float(industry_weight))

        # Granger edges: use precomputed directed map (ticker-name keys). Map to indices for today's universe
        gr_map_idx = {}
        if use_granger and granger_map_dir:
            for (u, v), w in granger_map_dir.items():
                if u not in tickers_list or v not in tickers_list:
                    continue
                i = tickers_list.index(u)
                j = tickers_list.index(v)
                if not (valid_mask_np[i] and valid_mask_np[j]):
                    continue
                key = (i, j)
                gr_map_idx[key] = max(gr_map_idx.get(key, 0.0), float(w))

        # --- Apply per-edge-type trimming and scalar weighting BEFORE merging ---
        def trim_pair_map(pair_map, top_k, n_nodes):
            """Trim unordered-pair map so each node has at most `top_k` neighbors.
            Keep an edge if it is in the top_k of at least one endpoint.
            """
            if not pair_map or top_k <= 0:
                return {}
            # build adjacency lists
            adj = {i: [] for i in range(n_nodes)}
            for (a, b), w in pair_map.items():
                adj[a].append(((a, b), b, w))
                adj[b].append(((a, b), a, w))

            keep_keys = set()
            for node, nbrs in adj.items():
                if not nbrs:
                    continue
                # sort by weight descending
                nbrs_sorted = sorted(nbrs, key=lambda t: t[2], reverse=True)
                for item in nbrs_sorted[:top_k]:
                    keep_keys.add(item[0])

            return {k: v for k, v in pair_map.items() if k in keep_keys}

        def trim_directed_map(edge_map, top_k):
            """Trim directed edge map so each source has at most `top_k` outgoing edges."""
            if not edge_map or top_k <= 0:
                return {}
            adj = {}
            for (src, dst), w in edge_map.items():
                adj.setdefault(src, []).append(((src, dst), w))
            keep_keys = set()
            for _, edges in adj.items():
                edges_sorted = sorted(edges, key=lambda t: t[1], reverse=True)
                for (key, _) in edges_sorted[:top_k]:
                    keep_keys.add(key)
            return {k: v for k, v in edge_map.items() if k in keep_keys}

        # trim per-type maps
        corr_map = trim_pair_map(corr_map, corr_top_k if corr_top_k is not None else 0, n_nodes)
        sector_map_pairs = trim_pair_map(sector_map_pairs, sector_top_k if sector_top_k is not None else 0, n_nodes)
        gr_map_idx = trim_directed_map(gr_map_idx, granger_top_k if granger_top_k is not None else 0)

        # apply per-edge-type scalar weights
        # Corr: scale normalized corr weights by w_corr
        if corr_map:
            vmax = max(corr_map.values())
            if vmax > 0:
                for k in list(corr_map.keys()):
                    corr_map[k] = (corr_map[k] / vmax) * w_corr

        # Sector: set sector edges to uniform w_sector (as requested)
        if sector_map_pairs:
            for k in list(sector_map_pairs.keys()):
                sector_map_pairs[k] = float(w_sector)

        # Granger: scale normalized granger weights by w_granger
        if gr_map_idx:
            vmax = max(gr_map_idx.values())
            if vmax > 0:
                for k in list(gr_map_idx.keys()):
                    gr_map_idx[k] = (gr_map_idx[k] / vmax) * w_granger

        # Merge maps (no cross-type normalization)
        final_map = {}

        def add_edge(src, dst, weight):
            final_map[(src, dst)] = final_map.get((src, dst), 0.0) + weight

        for (i, j), v in corr_map.items():
            add_edge(i, j, v)
            add_edge(j, i, v)
        for (i, j), v in sector_map_pairs.items():
            add_edge(i, j, v)
            add_edge(j, i, v)
        for (i, j), v in gr_map_idx.items():
            add_edge(i, j, v)
            if make_undirected:
                add_edge(j, i, v)

        # Ensure self-loops are present and strong: weight = 1.0
        for i_node in range(n_nodes):
            final_map[(i_node, i_node)] = 1.0

        if not final_map:
            continue

        # Expand directed map into edge list
        src = []
        dst = []
        w_vals = []
        for (i, j), w in final_map.items():
            src.append(i)
            dst.append(j)
            if i == j:
                w_vals.append(1.0)
            else:
                w_clamped = float(min(w, max_edge_weight)) if np.isfinite(w) else 0.0
                w_vals.append(w_clamped)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(w_vals, dtype=torch.float32)
        mask_edge = torch.isfinite(edge_weight)
        if mask_edge.sum() == 0:
            continue
        edge_index = edge_index[:, mask_edge]
        edge_weight = edge_weight[mask_edge]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        g = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y_ret_tensor,
        )
        g.valid_mask = valid_mask_tensor
        g.tickers = tickers_list
        g.date = d
        g.window_start = window_start
        g.window_end = window_end
        g.is_directed = bool(use_granger and not make_undirected)
        # Log snapshot diagnostics: number of nodes and edges for defendability
        try:
            n_nodes = g.x.shape[0]
            n_edges = int(g.edge_index.shape[1]) if g.edge_index is not None else 0
            valid_nodes = int(valid_mask_np.sum())
            print(f"[gnn] snapshot {d.date()} nodes={n_nodes} valid_nodes={valid_nodes} edges={n_edges}")
        except Exception:
            pass

        snapshots.append(g)
        meta_dates.append(d)

    return snapshots, feat_cols, meta_dates


def _split_snapshots_by_date(snapshots, dates, config, *, label="gnn", debug=False):
    train_mask, val_mask, test_mask, _ = split_time(dates, config, label=label, debug=debug)

    train_list, val_list, test_list = [], [], []
    for g, is_train, is_val, is_test in zip(snapshots, train_mask, val_mask, test_mask):
        if is_train:
            train_list.append(g)
        elif is_val:
            val_list.append(g)
        elif is_test:
            test_list.append(g)

    return train_list, val_list, test_list


def _target_stats(snaps):
    """Compute mean/std of targets over valid nodes in training set for scaling."""
    ys = []
    for g in snaps:
        if not hasattr(g, "valid_mask"):
            continue
        mask = g.valid_mask
        y = g.y
        if mask is None or y is None:
            continue
        if mask.numel() != y.numel():
            continue
        mask = mask & torch.isfinite(y)
        if mask.sum() == 0:
            continue
        ys.append(y[mask])
    if not ys:
        return 0.0, 1.0
    all_y = torch.cat(ys)
    mean = float(all_y.mean().item())
    std = float(all_y.std().item())
    if std < 1e-6:
        std = 1.0
    return mean, std


def train_gnn(config):
    set_seed(42)
    device = get_device(config["training"]["device"])
    use_cuda = device.type == "cuda"
    print(f"[gnn] device={device}, cuda_available={torch.cuda.is_available()}")
    rebuild = config.get("cache", {}).get("rebuild", False)

    graph_flags = config.get("graph", {})
    gr_cfg = config.get("granger", {})
    make_undirected_cfg = graph_flags.get("make_undirected")
    if make_undirected_cfg is None:
        make_undirected_cfg = graph_flags.get("force_undirected")
    if make_undirected_cfg is None:
        make_undirected = not bool(graph_flags.get("use_granger", False))
    else:
        make_undirected = bool(make_undirected_cfg)
    graph_directed = bool(graph_flags.get("use_granger", False) and not make_undirected)
    cache_id = cache_key(
        {
            "model": config["model"],
            "data": config["data"],
            "preprocess": config.get("preprocess", {"scale_features": True, "scaler": "standard"}),
            "graph": {
                "use_corr": bool(graph_flags.get("use_corr", False)),
                "use_sector": bool(graph_flags.get("use_sector", False)),
                "use_granger": bool(graph_flags.get("use_granger", False)),
                "corr_top_k": int(graph_flags.get("corr_top_k", 10)),
                "corr_min_periods": int(graph_flags.get("corr_min_periods", 0)),
                "make_undirected": make_undirected,
            },
            "granger": {
                "max_lag": int(gr_cfg.get("max_lag", 2)),
                "p_threshold": float(gr_cfg.get("p_threshold", 0.05)),
            },
        },
        dataset_version="gnn_snapshots",
        extra_files=[config["data"]["price_file"], "data/processed/universe.csv"],
    )
    cache_file = cache_path("gnn_snapshots", cache_id)

    if not rebuild:
        cached = cache_load(cache_file)
        if cached is not None:
            snapshots, feat_cols, dates = cached["snapshots"], cached["feat_cols"], cached["dates"]
            print(f"[gnn] loaded snapshots from cache {cache_file}")
        else:
            snapshots, feat_cols, dates = _build_snapshots_and_targets(config)
            cache_save(cache_file, {"snapshots": snapshots, "feat_cols": feat_cols, "dates": dates, "graph_directed": graph_directed, "make_undirected": make_undirected})
            print(f"[gnn] saved snapshots to cache {cache_file}")
    else:
        snapshots, feat_cols, dates = _build_snapshots_and_targets(config)
        cache_save(cache_file, {"snapshots": snapshots, "feat_cols": feat_cols, "dates": dates, "graph_directed": graph_directed, "make_undirected": make_undirected})
        print(f"[gnn] saved snapshots to cache {cache_file}")

    # Mandatory logging for defensibility: edge types enabled and snapshot stats
    graph_cfg_print = config.get("graph", {})
    use_corr = bool(graph_cfg_print.get("use_corr", False))
    use_sector = bool(graph_cfg_print.get("use_sector", False))
    use_granger = bool(graph_cfg_print.get("use_granger", False))
    make_undirected_cfg = graph_cfg_print.get("make_undirected")
    if make_undirected_cfg is None:
        make_undirected_cfg = graph_cfg_print.get("force_undirected")
    if make_undirected_cfg is None:
        make_undirected = not use_granger
    else:
        make_undirected = bool(make_undirected_cfg)
    graph_directed = bool(use_granger and not make_undirected)
    # mean nodes/edges per snapshot
    try:
        node_counts = [g.x.shape[0] for g in snapshots]
        edge_counts = [int(g.edge_index.shape[1]) for g in snapshots]
        mean_nodes = float(np.mean(node_counts)) if node_counts else 0.0
        mean_edges = float(np.mean(edge_counts)) if edge_counts else 0.0
        print(f"[gnn] edge types enabled: use_corr={use_corr} use_sector={use_sector} use_granger={use_granger}")
        print(f"[gnn] graph directed={graph_directed} make_undirected={make_undirected}")
        print(f"[gnn] mean nodes per snapshot={mean_nodes:.1f}, mean directed edges per snapshot={mean_edges:.1f}")
        # Compute mean degree (undirected) and max degree per node across snapshots
        try:
            mean_degrees = []
            max_degrees = []
            for g in snapshots:
                if g.edge_index is None or g.edge_index.shape[1] == 0:
                    mean_degrees.append(0.0)
                    max_degrees.append(0)
                    continue
                ei = g.edge_index.cpu().numpy()
                pairs = set()
                for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                    pairs.add(tuple(sorted((int(a), int(b)))))
                # build adjacency counts (exclude self-loops)
                n = g.x.shape[0]
                deg = [0] * n
                for u, v in pairs:
                    if u == v:
                        continue
                    deg[u] += 1
                    deg[v] += 1
                mean_degrees.append(float(np.mean(deg)))
                max_degrees.append(int(np.max(deg) if deg else 0))
            overall_mean_degree = float(np.mean(mean_degrees)) if mean_degrees else 0.0
            overall_max_degree = int(np.max(max_degrees)) if max_degrees else 0
            print(f"[gnn] mean degree per node (averaged across snapshots)={overall_mean_degree:.2f}, max degree per node (across snapshots)={overall_max_degree}")
        except Exception:
            pass
        if use_granger:
            try:
                # compute granger edges count from train-only data for reporting
                from utils.data_loading import load_price_panel
                from utils.graphs import granger_edges
                start = config["data"]["start_date"]
                end = config["data"]["end_date"]
                df_full = load_price_panel(config["data"]["price_file"], start, end)
                df_full["date"] = pd.to_datetime(df_full["date"])
                val_start = pd.to_datetime(config["training"]["val_start"])
                train_mask = df_full["date"] < val_start
                gr_edges = granger_edges(df_full.loc[train_mask], max_lag=int(config.get("granger", {}).get("max_lag", 2)), p_threshold=float(config.get("granger", {}).get("p_threshold", 0.05)))
                gr_count = len(gr_edges) if gr_edges else 0
                print(f"[gnn] granger edges (precomputed count): {gr_count}")
            except Exception:
                print("[gnn] Warning: failed to compute granger edge count for logging")
    except Exception:
        pass

    # sanity checks
    for name, lst in [("snapshots", snapshots)]:
        for g in lst:
            issues = check_tensor("x", g.x) + check_tensor("y", g.y)
            if issues:
                raise ValueError(f"[gnn] Sanity failed: {'; '.join(issues)}")

    debug = bool(config.get("debug", {}).get("leakage", False))
    train_snaps, val_snaps, test_snaps = _split_snapshots_by_date(
        snapshots, dates, config, label="gnn", debug=debug
    )

    # Target scaling to avoid collapse to tiny constants
    tgt_mean, tgt_std = _target_stats(train_snaps)
    print(f"[gnn] target scaling mean={tgt_mean:.6f} std={tgt_std:.6f}")

    if not train_snaps:
        raise ValueError("No training snapshots available. Check the date range and lookback window.")
    if not val_snaps:
        raise ValueError("No validation snapshots available. Adjust 'training.val_start'/'training.test_start'.")

    bs = max(2, config["training"]["batch_size"])
    num_workers = default_num_workers()
    loader_kwargs = {
        "batch_size": bs,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0,
    }

    train_loader = GeoDataLoader(train_snaps, **loader_kwargs)
    val_loader = GeoDataLoader(val_snaps, **{**loader_kwargs, "shuffle": False})
    test_loader = GeoDataLoader(test_snaps, **{**loader_kwargs, "shuffle": False})

    requested_layers = int(config["model"].get("num_layers", 1))
    model_type = config["model"]["type"].lower()
    model_label = model_type
    if model_type == "tgcn":
        print("[gnn] Warning: 'tgcn' runs a static TGCN cell (no temporal state). Use type 'tgcn_static'.")
        model_label = "tgcn_static"
    elif model_type == "tgat":
        print("[gnn] Warning: 'tgat' runs a static GAT baseline (no temporal attention). Use type 'tgat_static'.")
        model_label = "tgat_static"

    num_layers = max(1, min(requested_layers, 3))
    if model_type == "gcn" and requested_layers > 2:
        print("[gnn] Warning: GCN depth >2 can over-smooth; clamping to at most 3 layers.")

    # Ensure dropout enabled and non-zero for GCN to reduce oversmoothing
    dropout_val = float(config["model"].get("dropout", 0.0))
    if model_type == "gcn" and dropout_val <= 0.0:
        print("[gnn] Warning: forcing non-zero dropout=0.2 for GCN to reduce oversmoothing")
        dropout_val = 0.2
    attn_dropout_val = float(config["model"].get("attn_dropout", dropout_val))

    # Allow TGCN/TGAT-style baselines in static (snapshot) mode.
    mtype = model_label
    if mtype == "tgcn_static":
        if StaticTGCN is None:
            raise RuntimeError("TGCN model support is unavailable (missing dependency).")
        model = StaticTGCN(
            input_dim=len(feat_cols),
            hidden_dim=config["model"]["hidden_dim"],
            dropout=dropout_val,
        ).to(device)
    elif mtype == "tgat_static":
        if StaticTGAT is None:
            raise RuntimeError("TGAT model support is unavailable (missing dependency).")
        # allow requesting more layers for TGAT-like model
        model = StaticTGAT(
            input_dim=len(feat_cols),
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=int(config["model"].get("num_layers", 2)),
            heads=int(config["model"].get("heads", 2)),
            dropout=dropout_val,
        ).to(device)
    else:
        model = StaticGNN(
            gnn_type=model_type,
            input_dim=len(feat_cols),
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=num_layers,
            dropout=dropout_val,
            attn_dropout=attn_dropout_val,
            heads=config["model"].get("heads", 1),
            use_residual=True,
        ).to(device)

    # If using GAT with weighted edges enabled, warn that GATConv ignores edge_weight
    try:
        if model_type in {"gat", "tgat_static"} and (use_corr or use_sector or use_granger):
            print("[gnn] Warning: model GAT ignores edge weights provided in `edge_weight`.\n" \
                  "If you rely on weighting, consider using GCN or implementing weighted attention.")
    except Exception:
        pass

    # Coerce hyperparams to numeric types to avoid YAML/string parsing issues
    lr_val = float(config["training"].get("lr", 1e-3))
    weight_decay_val = float(config["training"].get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_val,
        weight_decay=weight_decay_val,
    )
    loss_fn = torch.nn.MSELoss()
    scaler = GradScaler(enabled=use_cuda)

    best_val = float("inf")
    bad_epochs = 0
    patience = config["training"]["patience"]

    out_dir = Path(config["evaluation"]["out_dir"]) / config["model"]["type"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"best_{config['model']['type']}.pt"

    max_epochs = config["training"]["max_epochs"]
    tgt_mean_t = torch.tensor(tgt_mean, device=device)
    tgt_std_t = torch.tensor(tgt_std, device=device)

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with amp.autocast(device_type="cuda", enabled=use_cuda):
                logits = model(batch.x, batch.edge_index, batch.edge_weight)
                mask = batch.valid_mask & torch.isfinite(batch.y)
                if mask.sum() == 0:
                    continue
                target_scaled = (batch.y - tgt_mean_t) / tgt_std_t
                loss = loss_fn(logits[mask], target_scaled[mask])
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.detach())
        compute_time = time.time() - t0

        model.eval()
        val_losses = []
        with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_cuda):
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                logits = model(batch.x, batch.edge_index, batch.edge_weight)
                mask = batch.valid_mask & torch.isfinite(batch.y)
                if mask.sum() == 0:
                    continue
                target_scaled = (batch.y - tgt_mean_t) / tgt_std_t
                loss = loss_fn(logits[mask], target_scaled[mask])
                val_losses.append(loss.detach())

        mean_train = float(torch.stack(train_losses).mean().item()) if train_losses else float("nan")
        mean_val = float(torch.stack(val_losses).mean().item()) if val_losses else float("nan")
        print(f"[{config['model']['type'].upper()}] epoch {epoch} train {mean_train:.5f} val {mean_val:.5f} (compute {compute_time:.2f}s)")

        if np.isfinite(mean_val) and mean_val < best_val:
            best_val = mean_val
            bad_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    if not model_path.exists():
        raise RuntimeError(f"No checkpoint saved to '{model_path}'. Validation never produced a usable batch.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []
    with torch.no_grad(), amp.autocast(device_type="cuda", enabled=use_cuda):
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            logits = model(batch.x, batch.edge_index, batch.edge_weight)
            pred = (logits * tgt_std_t + tgt_mean_t).cpu().numpy()
            y_ret = batch.y.cpu().numpy()
            valid = batch.valid_mask.cpu().numpy()
            tickers_raw = batch.tickers
            tickers = []
            if isinstance(tickers_raw, list):
                for item in tickers_raw:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        tickers.extend(list(item))
                    else:
                        tickers.append(item)
            else:
                tickers = list(tickers_raw)
            tickers = [str(t) for t in tickers]
            d = batch.date
            if isinstance(d, (list, np.ndarray, pd.DatetimeIndex)):
                d = d[0]
            d = pd.to_datetime(d)

            for i, t in enumerate(tickers):
                if i >= len(valid):
                    break
                if not valid[i] or not np.isfinite(y_ret[i]):
                    continue
                rows.append({
                    "date": d,
                    "ticker": t,
                    "pred": float(pred[i]),
                    "realized_ret": float(y_ret[i]),
                })
            # log stats to diagnose collapse
            if len(pred) > 0:
                print(f"[{config['model']['type'].upper()}] test day {d.date()} pred mean {float(np.mean(pred)):.6f} std {float(np.std(pred)):.6f}")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / f"{config['model']['type']}_predictions.csv", index=False)

    daily_metrics = []
    for d, g in pred_df.groupby("date"):
        std_pred = g["pred"].std()
        if std_pred < 1e-8:
            print(f"[{config['model']['type'].upper()}] warning: near-constant predictions on {d.date()}, skipping IC")
            ic = np.nan
        else:
            ic = rank_ic(g["pred"], g["realized_ret"])
        hit = hit_rate(g["pred"], g["realized_ret"], top_k=config["evaluation"]["top_k"])
        daily_metrics.append({"date": d, "ic": ic, "hit": hit})
    daily_metrics = pd.DataFrame(daily_metrics).sort_values("date")

    ic_path = out_dir / f"{config['model']['type']}_ic_timeseries.png"
    ic_hist_path = out_dir / f"{config['model']['type']}_ic_histogram.png"
    if daily_metrics["ic"].notna().any():
        plot_daily_ic(daily_metrics, ic_path)
        plot_ic_hist(daily_metrics, ic_hist_path)
        print(f"[{config['model']['type'].upper()}] Saved IC plots: {ic_path}, {ic_hist_path}")
    else:
        print(f"[{config['model']['type'].upper()}] Skipping IC plots (all IC values NaN)")

    equity_curve, daily_ret, stats = backtest_long_only(
        pred_df,
        top_k=config["evaluation"]["top_k"],
        transaction_cost_bps=config["evaluation"]["transaction_cost_bps"],
        risk_free_rate=config["evaluation"]["risk_free_rate"],
        rebalance_freq=5,
    )
    equity_curve.to_csv(out_dir / f"{config['model']['type']}_equity_curve.csv", header=["value"])
    eq_path = out_dir / f"{config['model']['type']}_equity_curve.png"
    plot_equity_curve(
        equity_curve,
        f"{config['model']['type'].upper()} long only",
        eq_path,
    )
    print(f"[{config['model']['type'].upper()}] Saved equity curve to {eq_path}")

    # Enrich daily metrics with returns and drawdown for thesis-grade reporting
    eq_series = equity_curve.copy()
    eq_series.index = pd.to_datetime(eq_series.index)
    daily_ret_series = pd.Series(daily_ret, index=eq_series.index[: len(daily_ret)])
    dd = eq_series / eq_series.cummax() - 1.0
    daily_metrics = daily_metrics.merge(
        pd.DataFrame(
            {
                "date": daily_ret_series.index,
                "daily_return": daily_ret_series.values,
                "drawdown": dd.reindex(daily_ret_series.index).values,
            }
        ),
        on="date",
        how="left",
    )
    daily_metrics.to_csv(out_dir / f"{config['model']['type']}_daily_metrics.csv", index=False)

    # Global buy-and-hold baseline
    eq_bh_full, ret_bh_full, stats_bh = get_global_buy_and_hold(
        config,
        rebuild=config.get("cache", {}).get("rebuild", False),
        align_start_date=config["training"]["test_start"],
    )
    print("[baseline] global buy-and-hold stats", stats_bh)
    start_d, end_d = pred_df["date"].min(), pred_df["date"].max()
    eq_bh = eq_bh_full.loc[(eq_bh_full.index >= start_d) & (eq_bh_full.index <= end_d)]

    bh_path = out_dir / f"{config['model']['type']}_buy_and_hold_equity_curve.png"
    eq_bh.to_csv(out_dir / f"{config['model']['type']}_buy_and_hold_equity_curve.csv", header=["value"])
    plot_equity_curve(
        eq_bh,
        "Buy and Hold",
        bh_path,
    )
    print(f"[{config['model']['type'].upper()}] Saved buy-and-hold curve to {bh_path}")

    comp_path = out_dir / f"{config['model']['type']}_equity_comparison.png"
    plot_equity_comparison(
        model_curve=equity_curve,
        bh_curve=eq_bh,
        title=f"{config['model']['type'].upper()} vs Buy and Hold",
        out_path=comp_path,
    )
    print(f"[{config['model']['type'].upper()}] Saved equity comparison to {comp_path}")

    # Summary metrics (IC stats, volatility, max drawdown)
    run_tag = config.get("experiment_name", config["model"]["type"])
    ic_series = daily_metrics.set_index("date")["ic"].dropna()
    ic_mean = float(ic_series.mean()) if not ic_series.empty else float("nan")
    ic_std = float(ic_series.std()) if not ic_series.empty else float("nan")
    ic_tstat = float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_series.size > 1 and ic_std > 0 else float("nan")
    vol = float(np.std(daily_ret_series.values) * np.sqrt(252)) if not daily_ret_series.empty else float("nan")
    max_dd = float(dd.min()) if not dd.empty else float("nan")

    summary = {
        "run_tag": run_tag,
        "model_type": config["model"]["type"],
        "stats": stats,
        "buy_and_hold_stats": stats_bh,
        "ic_mean": ic_mean,
        "ic_tstat": ic_tstat,
        "volatility": vol,
        "max_drawdown": max_dd,
        "graph_directed": graph_directed,
        "graph_make_undirected": make_undirected,
    }
    summary_path = out_dir / f"{run_tag}_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Rolling metrics
    rolling_window = int(config.get("evaluation", {}).get("rolling_window", 63))
    rolling = pd.DataFrame({"date": daily_metrics["date"]}).copy()
    rolling["rolling_sharpe"] = daily_ret_series.rolling(rolling_window).apply(
        lambda x: sharpe_ratio(x, config["evaluation"]["risk_free_rate"]) if len(x) > 1 else np.nan,
        raw=False,
    ).values
    rolling["rolling_ic_mean"] = ic_series.reindex(daily_metrics["date"]).rolling(rolling_window).mean().values
    rolling.to_csv(out_dir / f"{run_tag}_rolling_metrics.csv", index=False)

    print(f"{config['model']['type'].upper()} backtest stats", stats)
    print("Buy-and-hold stats", stats_bh)
