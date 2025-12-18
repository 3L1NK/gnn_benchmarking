# trainers/train_gnn.py

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import amp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.cuda.amp import GradScaler

from models.gnn_model import StaticGNN
from utils.seeds import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.metrics import rank_ic, hit_rate
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


def _build_snapshots_and_targets(config):
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    horizon = config["data"]["target_horizon"]
    corr_window = config["data"]["corr_window"]
    corr_thr = config["data"]["corr_threshold"]
    graph_cfg = config.get("graph_edges", {})
    use_corr = graph_cfg.get("use_correlation", True)
    use_sector = graph_cfg.get("use_sector", True)
    use_industry = graph_cfg.get("use_industry", True)
    corr_top_k = int(graph_cfg.get("corr_top_k", 10))
    corr_min_periods = int(graph_cfg.get("corr_min_periods", max(5, corr_window // 2)))
    sector_weight = float(graph_cfg.get("sector_weight", 0.2))
    industry_weight = float(graph_cfg.get("industry_weight", 0.1))

    df = load_price_panel(price_file, start, end)
    df, feat_cols = add_technical_features(df)
    if "log_ret_1d" not in df.columns:
        raise ValueError("Expected column 'log_ret_1d' in feature dataframe")

    feature_cols = list(feat_cols) + ["log_ret_1d"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    universe_path = Path("data/processed/universe.csv")
    if not universe_path.exists():
        raise FileNotFoundError("Universe metadata is required at data/processed/universe.csv")
    universe_df = pd.read_csv(universe_path)

    universe_list = sorted(universe_df["ticker"].unique().tolist())
    sector_map = dict(zip(universe_df["ticker"], universe_df.get("sector", pd.Series(index=universe_df.index))))
    industry_map = dict(zip(universe_df["ticker"], universe_df.get("industry", pd.Series(index=universe_df.index))))

    df["ret_target"] = df.groupby("ticker")["log_ret_1d"].shift(-horizon)
    df = df.dropna(subset=["ret_target"]).reset_index(drop=True)

    ret_pivot = (
        df.pivot(index="date", columns="ticker", values="log_ret_1d")
        .reindex(columns=universe_list)
        .sort_index()
    )

    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
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
        universe_today = df[df["date"] == d].set_index("ticker")

        feat_for_date = {}
        target_ret_for_date = {}
        valid_mask = []

        for t in universe_list:
            if t in universe_today.index:
                row = universe_today.loc[t]
                feat_vec = row[feat_cols].values.astype(float)
                y_ret = float(row["ret_target"])
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

        # standardize per day using valid nodes only
        valid_mask_np = np.array(valid_mask, dtype=bool)
        x_std = x.copy()
        for col in range(x_std.shape[1]):
            vals = x_std[valid_mask_np, col]
            if vals.size == 0:
                x_std[:, col] = 0.0
                continue
            mean = vals.mean()
            std = vals.std()
            if std < 1e-8:
                x_std[:, col] = 0.0
            else:
                x_std[:, col] = (x_std[:, col] - mean) / std
        x = x_std
        if not np.any(np.abs(x[valid_mask_np]) > 1e-8):
            continue

        edge_dict = {}

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
                    w_abs = float(abs(w))
                    edge_dict[(i, j)] = max(edge_dict.get((i, j), 0.0), w_abs)
                    edge_dict[(j, i)] = max(edge_dict.get((j, i), 0.0), w_abs)

        if use_sector and sector_weight > 0:
            for i in range(n_nodes):
                ti = tickers_list[i]
                if not valid_mask_np[i]:
                    continue
                si = sector_map.get(ti)
                if si is None or (isinstance(si, float) and np.isnan(si)):
                    continue
                for j in range(i + 1, n_nodes):
                    tj = tickers_list[j]
                    if not valid_mask_np[j]:
                        continue
                    sj = sector_map.get(tj)
                    if sj is None or (isinstance(sj, float) and np.isnan(sj)):
                        continue
                    if si == sj:
                        edge_dict[(i, j)] = max(edge_dict.get((i, j), 0.0), sector_weight)
                        edge_dict[(j, i)] = max(edge_dict.get((j, i), 0.0), sector_weight)

        if use_industry and industry_weight > 0:
            for i in range(n_nodes):
                ti = tickers_list[i]
                if not valid_mask_np[i]:
                    continue
                ii = industry_map.get(ti)
                if ii is None or (isinstance(ii, float) and np.isnan(ii)):
                    continue
                for j in range(i + 1, n_nodes):
                    tj = tickers_list[j]
                    if not valid_mask_np[j]:
                        continue
                    ij = industry_map.get(tj)
                    if ij is None or (isinstance(ij, float) and np.isnan(ij)):
                        continue
                    if ii == ij:
                        edge_dict[(i, j)] = max(edge_dict.get((i, j), 0.0), industry_weight)
                        edge_dict[(j, i)] = max(edge_dict.get((j, i), 0.0), industry_weight)

        if not edge_dict:
            continue

        src, dst = zip(*edge_dict.keys())
        w_vals = list(edge_dict.values())
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

        snapshots.append(g)
        meta_dates.append(d)

    return snapshots, feat_cols, meta_dates


def _split_snapshots_by_date(snapshots, dates, val_start, test_start):
    val_start = pd.to_datetime(val_start)
    test_start = pd.to_datetime(test_start)

    train_list, val_list, test_list = [], [], []

    for g, d in zip(snapshots, dates):
        if d < val_start:
            train_list.append(g)
        elif d < test_start:
            val_list.append(g)
        else:
            test_list.append(g)

    return train_list, val_list, test_list


def train_gnn(config):
    set_seed(42)
    device = get_device(config["training"]["device"])
    use_cuda = device.type == "cuda"
    print(f"[gnn] device={device}, cuda_available={torch.cuda.is_available()}")
    rebuild = config.get("cache", {}).get("rebuild", False)

    cache_id = cache_key(
        {
            "model": config["model"],
            "data": config["data"],
            "graph_edges": config.get("graph_edges", {}),
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
            cache_save(cache_file, {"snapshots": snapshots, "feat_cols": feat_cols, "dates": dates})
            print(f"[gnn] saved snapshots to cache {cache_file}")
    else:
        snapshots, feat_cols, dates = _build_snapshots_and_targets(config)
        cache_save(cache_file, {"snapshots": snapshots, "feat_cols": feat_cols, "dates": dates})
        print(f"[gnn] saved snapshots to cache {cache_file}")

    # sanity checks
    for name, lst in [("snapshots", snapshots)]:
        for g in lst:
            issues = check_tensor("x", g.x) + check_tensor("y", g.y)
            if issues:
                raise ValueError(f"[gnn] Sanity failed: {'; '.join(issues)}")

    train_snaps, val_snaps, test_snaps = _split_snapshots_by_date(
        snapshots, dates, config["training"]["val_start"], config["training"]["test_start"]
    )

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

    model = StaticGNN(
        gnn_type=config["model"]["type"],
        input_dim=len(feat_cols),
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=min(config["model"]["num_layers"], 2),
        dropout=config["model"]["dropout"],
        heads=config["model"].get("heads", 1),
        use_residual=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
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
                loss = loss_fn(logits[mask], batch.y[mask])
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
                loss = loss_fn(logits[mask], batch.y[mask])
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
            pred = logits.cpu().numpy()
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
    daily_metrics.to_csv(out_dir / f"{config['model']['type']}_daily_metrics.csv", index=False)

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

    # Global buy-and-hold baseline
    eq_bh_full, ret_bh_full, stats_bh = get_global_buy_and_hold(
        config,
        rebuild=config.get("cache", {}).get("rebuild", False),
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

    print(f"{config['model']['type'].upper()} backtest stats", stats)
    print("Buy-and-hold stats", stats_bh)
