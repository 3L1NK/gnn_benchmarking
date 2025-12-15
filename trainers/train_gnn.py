# trainers/train_gnn.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

from models.gnn_model import StaticGNN
from utils.seeds import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.metrics import rank_ic, hit_rate
from utils.backtest import backtest_long_only, backtest_buy_and_hold
from utils.plot import (
    plot_daily_ic,
    plot_ic_hist,
    plot_equity_curve,
    plot_equity_comparison,
)


def _build_snapshots_and_targets(config):
    """
    Create one graph snapshot per trading day with node features, edges and targets.

    Target is regression: next log return (ret_target) stored as Data.y.

    Important: we enforce a fixed universe of tickers for every day so that
    all graph snapshots have the same number and ordering of nodes.
    This is required for temporal models like TGCN.
    """

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
    log_edge_stats = graph_cfg.get("log_edge_stats", False)

    # load panel and compute features
    df = load_price_panel(price_file, start, end)
    df, feat_cols = add_technical_features(df)

    # require log_ret_1d for targets and correlation
    if "log_ret_1d" not in df.columns:
        raise ValueError("Expected column 'log_ret_1d' in feature dataframe")

    # drop rows with any NaN in features or log_ret_1d
    feature_cols = list(feat_cols) + ["log_ret_1d"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # load universe metadata for sector and industry
    universe_path = Path("data/processed/universe.csv")
    if not universe_path.exists():
        raise FileNotFoundError("Universe metadata is required at data/processed/universe.csv")
    universe_df = pd.read_csv(universe_path)

    # fixed universe of tickers (same for all days)
    universe_list = sorted(universe_df["ticker"].unique().tolist())

    sector_map = dict(zip(universe_df["ticker"], universe_df.get("sector", pd.Series(index=universe_df.index))))
    industry_map = dict(zip(universe_df["ticker"], universe_df.get("industry", pd.Series(index=universe_df.index))))

    # regression style target: next horizon log return
    df["ret_target"] = df.groupby("ticker")["log_ret_1d"].shift(-horizon)

    # drop rows where we do not know next return
    df = df.dropna(subset=["ret_target"]).reset_index(drop=True)

    # pivot for correlation on returns, reindex to full universe
    ret_pivot = (
        df.pivot(index="date", columns="ticker", values="log_ret_1d")
        .reindex(columns=universe_list)
        .sort_index()
    )

    # work per day
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    snapshots = []
    meta_dates = []

    for d in dates:
        d = pd.to_datetime(d)

        if d not in ret_pivot.index:
            continue

        # need at least corr_window rows of returns
        idx = ret_pivot.index.get_loc(d)
        if isinstance(idx, slice):
            # should not happen for unique index
            continue
        if idx < corr_window:
            continue

        window_ret = ret_pivot.iloc[idx - corr_window + 1 : idx + 1]

        # universe data at this date
        universe_today = df[df["date"] == d].set_index("ticker")

        feat_for_date = {}
        target_ret_for_date = {}
        valid_mask = []

        # build features and targets for the full fixed universe
        for t in universe_list:
            if t in universe_today.index:
                row = universe_today.loc[t]
                feat_vec = row[feat_cols].values.astype(float)
                y_ret = float(row["ret_target"])
                valid_mask.append(True)
            else:
                # ticker not present on this date or dropped by NaNs earlier
                # use zeros so node exists but carries no signal
                feat_vec = np.zeros(len(feat_cols), dtype=float)
                y_ret = 0.0
                valid_mask.append(False)

            feat_for_date[t] = feat_vec
            target_ret_for_date[t] = y_ret

        # node count is fixed
        tickers_list = list(universe_list)
        n_nodes = len(tickers_list)
        if n_nodes < 2:
            continue

        # node feature matrix and realized returns
        x = np.vstack([feat_for_date[t] for t in tickers_list])
        y_ret = np.array([target_ret_for_date[t] for t in tickers_list], dtype=np.float32)

        if not np.isfinite(x).all() or not np.isfinite(y_ret).all():
            continue

        # ---------------------------------------
        # 1) Cross-sectional feature standardization (per day, valid nodes only)
        # ---------------------------------------
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

        # skip days where all valid nodes are zero after standardization
        if not np.any(np.abs(x[valid_mask_np]) > 1e-8):
            print(f"[graph] {d.date()} skipped (standardized features all zero for valid nodes)")
            continue

        edge_dict = {}  # (i, j) -> weight

        # 1) correlation based edges (top-k by |rho|, no zero-filling)
        if use_corr:
            window_sub = window_ret[tickers_list].copy()
            # ignore invalid nodes in correlation
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

        # 2) sector edges (reduced weight, structural prior)
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

        # 3) industry edges (reduced weight, structural prior)
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
            # no edges, skip this day
            continue

        # build edge_index and edge_weight tensors
        src, dst = zip(*edge_dict.keys())
        w_vals = list(edge_dict.values())

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(w_vals, dtype=torch.float32)

        # filter any non finite edge weights
        mask_edge = torch.isfinite(edge_weight)
        if mask_edge.sum() == 0:
            continue
        edge_index = edge_index[:, mask_edge]
        edge_weight = edge_weight[mask_edge]

        # diagnostics to catch edge explosion
        num_edges = edge_index.shape[1]
        deg = torch.bincount(edge_index[0], minlength=n_nodes)
        avg_deg = float(deg.float().mean().item()) if len(deg) > 0 else 0.0
        max_deg = int(deg.max().item()) if len(deg) > 0 else 0
        if log_edge_stats:
            print(f"[graph] {d.date()} nodes={n_nodes} edges={num_edges} avg_deg={avg_deg:.2f} max_deg={max_deg}")

        # build PyG Data object
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_ret_tensor = torch.tensor(y_ret, dtype=torch.float32)
        valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)

        graph = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y_ret_tensor,
        )
        graph.y_ret = y_ret_tensor
        graph.tickers = tickers_list
        graph.date = d
        graph.valid_mask = valid_mask_tensor

        snapshots.append(graph)
        meta_dates.append(d)

    return snapshots, feat_cols, meta_dates


def _split_snapshots_by_date(snapshots, dates, val_start, test_start):
    """
    Split daily graph snapshots into train, validation and test sets by date.

    Dates are compared to val_start and test_start.
    Training uses the earliest period, validation sits in the middle,
    and testing uses the most recent part of the sample.
    """
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


# 1. Training for static GCN and GAT
def _train_static_gnn(config):
    set_seed(42)

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    snapshots, feat_cols, dates = _build_snapshots_and_targets(config)
    train_snaps, val_snaps, test_snaps = _split_snapshots_by_date(
        snapshots, dates, config["training"]["val_start"], config["training"]["test_start"]
    )

    if not train_snaps:
        raise ValueError(
            "No training snapshots available. Check the date range and lookback window."
        )
    if not val_snaps:
        raise ValueError(
            "No validation snapshots available. Adjust 'training.val_start'/'training.test_start'."
        )

    train_loader = GeoDataLoader(
        train_snaps,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = GeoDataLoader(val_snaps, batch_size=1, shuffle=False)
    test_loader = GeoDataLoader(test_snaps, batch_size=1, shuffle=False)

    # Write outputs under evaluation.out_dir / model_name to avoid collisions
    out_dir = Path(config["evaluation"]["out_dir"]) / config["model"]["type"]
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = torch.nn.MSELoss()
    hidden_candidates = config["model"].get("hidden_dim_candidates", [32, 64])
    hidden_candidates = list(dict.fromkeys(hidden_candidates))
    num_layers = 1  # force shallow GCN to reduce over-smoothing

    patience_cfg = config["training"]["patience"]
    patience = max(patience_cfg * 2, patience_cfg + 20)
    min_epochs = config["training"].get("min_epochs", patience)
    max_epochs = config["training"]["max_epochs"]

    best_overall_val = float("inf")
    best_state = None
    best_hidden = None

    for hidden_dim in hidden_candidates:
        print(f"[{config['model']['type'].upper()}] training candidate hidden_dim={hidden_dim}")
        model = StaticGNN(
            gnn_type=config["model"]["type"],
            input_dim=len(feat_cols),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=config["model"]["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        best_val = float("inf")
        bad_epochs = 0
        best_state_candidate = None

        for epoch in range(max_epochs):
            model.train()
            train_losses = []

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.edge_weight)

                # simple mask: only finite labels
                mask = batch.valid_mask & torch.isfinite(batch.y)
                if mask.sum() == 0:
                    continue

                loss = loss_fn(logits[mask], batch.y[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["gradient_clip"],
                )
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = model(batch.x, batch.edge_index, batch.edge_weight)
                    mask = batch.valid_mask & torch.isfinite(batch.y)
                    if mask.sum() == 0:
                        continue
                    loss = loss_fn(logits[mask], batch.y[mask])
                    val_losses.append(loss.item())

            mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
            mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
            print(f"[{config['model']['type'].upper()}-{hidden_dim}] epoch {epoch} train {mean_train:.5f} val {mean_val:.5f}")

            if np.isfinite(mean_val) and mean_val < best_val:
                best_val = mean_val
                bad_epochs = 0
                best_state_candidate = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if (epoch + 1) >= min_epochs and bad_epochs >= patience:
                    print(f"[{config['model']['type'].upper()}-{hidden_dim}] Early stopping")
                    break

        if best_state_candidate is not None and best_val < best_overall_val:
            best_overall_val = best_val
            best_state = best_state_candidate
            best_hidden = hidden_dim

    if best_state is None:
        raise RuntimeError("No checkpoint saved; validation never produced a usable batch.")

    # rebuild best model
    model = StaticGNN(
        gnn_type=config["model"]["type"],
        input_dim=len(feat_cols),
        hidden_dim=best_hidden,
        num_layers=num_layers,
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(best_state)
    model.eval()

    # test predictions and backtest
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_weight)
            pred = logits.cpu().numpy()
            y_ret = batch.y.cpu().numpy()
            valid = batch.valid_mask.cpu().numpy()
            tickers_raw = batch.tickers

            if isinstance(tickers_raw, list) and len(tickers_raw) > 0:
                if isinstance(tickers_raw[0], str):
                    tickers = tickers_raw
                elif isinstance(tickers_raw[0], list):
                    tickers = tickers_raw[0]
                else:
                    tickers = [str(x) for x in tickers_raw]
            else:
                tickers = list(tickers_raw)

            # safely extract scalar date for grouping
            d = batch.date
            if isinstance(d, (list, np.ndarray, pd.DatetimeIndex)):
                d = d[0]
            d = pd.to_datetime(d)

            for i, t in enumerate(tickers):
                if not valid[i] or not np.isfinite(y_ret[i]):
                    continue
                rows.append({
                    "date": d,
                    "ticker": t,
                    "pred": float(pred[i]),
                    "realized_ret": float(y_ret[i]),
                })


    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / f"{config['model']['type']}_predictions.csv", index=False)

    # Diagnostic: check per-day prediction diversity
    pred_counts = pred_df.groupby("date")["pred"].nunique()
    low_var_days = pred_counts[pred_counts < 2]
    if not low_var_days.empty:
        print(f"[{config['model']['type'].upper()}] Warning: {len(low_var_days)} days with constant predictions")
        low_var_days.to_frame("unique_preds").reset_index().to_csv(
            out_dir / f"{config['model']['type']}_constant_pred_days.csv",
            index=False,
        )

    daily_metrics = []
    for d, g in pred_df.groupby("date"):
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

    print(
        f"{config['model']['type'].upper()} mean IC",
        daily_metrics["ic"].mean(),
        "mean hit",
        daily_metrics["hit"].mean(),
    )

    equity_curve, daily_ret, stats = backtest_long_only(
        pred_df,
        top_k=config["evaluation"]["top_k"],
        transaction_cost_bps=config["evaluation"]["transaction_cost_bps"],
        risk_free_rate=config["evaluation"]["risk_free_rate"],
    )
    eq_path = out_dir / f"{config['model']['type']}_equity_curve.png"
    equity_curve.to_csv(out_dir / f"{config['model']['type']}_equity_curve.csv", header=["value"])
    plot_equity_curve(
        equity_curve,
        f"{config['model']['type'].upper()} long only",
        eq_path,
    )
    print(f"[{config['model']['type'].upper()}] Saved equity curve to {eq_path}")

    # Buy-and-hold baseline for the same window
    bh_df = pred_df[["date", "ticker", "realized_ret"]].rename(columns={"realized_ret": "log_ret_1d"})
    eq_bh, ret_bh, stats_bh = backtest_buy_and_hold(
        bh_df,
        risk_free_rate=config["evaluation"]["risk_free_rate"],
    )
    bh_path = out_dir / f"{config['model']['type']}_buy_and_hold_equity_curve.png"
    eq_bh.to_csv(out_dir / f"{config['model']['type']}_buy_and_hold_equity_curve.csv", header=["value"])
    plot_equity_curve(
        eq_bh,
        "Buy and Hold",
        bh_path,
    )
    print(f"[{config['model']['type'].upper()}] Saved buy-and-hold curve to {bh_path}")

    # Combined comparison plot
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


# 2. Training for temporal TGCN
def _train_tgcn(config):
    """
    Temporal training using TGCN over a time ordered sequence of daily graphs.
    """

    from models.tgcn_model import TemporalGCNModel

    set_seed(42)

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    snapshots, feat_cols, dates = _build_snapshots_and_targets(config)

    # snapshots and dates are already aligned in time order
    train_snaps, val_snaps, test_snaps = _split_snapshots_by_date(
        snapshots, dates,
        config["training"]["val_start"],
        config["training"]["test_start"],
    )

    if not train_snaps:
        raise ValueError("No training snapshots for TGCN. Adjust date range.")
    if not val_snaps:
        raise ValueError("No validation snapshots for TGCN. Adjust val_start or test_start.")

    model = TemporalGCNModel(
        input_dim=len(feat_cols),
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    bad_epochs = 0
    patience = config["training"]["patience"]

    out_dir = Path(config["evaluation"]["out_dir"]) / config["model"]["type"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_tgcn.pt"

    # ---------------- TRAINING LOOP ----------------
    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_losses = []

        # reset hidden state at start of each epoch
        h = None

        for g in train_snaps:
            g = g.to(device)
            optimizer.zero_grad()

            logits, h = model(g.x, g.edge_index, g.edge_weight, h)
            h = h.detach()

            mask = g.valid_mask & torch.isfinite(g.y)
            if mask.sum() == 0:
                continue

            loss = loss_fn(logits[mask], g.y[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["gradient_clip"],
            )
            optimizer.step()

            train_losses.append(loss.item())

        # ---------------- VALIDATION ----------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            h_val = None
            for g in val_snaps:
                g = g.to(device)
                logits, h_val = model(g.x, g.edge_index, g.edge_weight, h_val)
                h_val = h_val.detach()
                mask = g.valid_mask & torch.isfinite(g.y)
                if mask.sum() == 0:
                    continue
                loss = loss_fn(logits[mask], g.y[mask])
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"[TGCN] epoch {epoch} train {mean_train:.5f} val {mean_val:.5f}")

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
        raise RuntimeError("TGCN saved no model. Validation may be empty.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---------------- TEST, PREDICTIONS, BACKTEST ----------------
    rows = []
    with torch.no_grad():
        h_test = None
        for g in test_snaps:
            g = g.to(device)

            logits, h_test = model(g.x, g.edge_index, g.edge_weight, h_test)
            h_test = h_test.detach()
            pred = logits.cpu().numpy()
            y_realized = g.y.cpu().numpy()
            valid = g.valid_mask.cpu().numpy()
            tickers = g.tickers
            d = pd.to_datetime(g.date)

            for i, t in enumerate(tickers):
                if not valid[i] or not np.isfinite(y_realized[i]):
                    continue
                rows.append(
                    {
                        "date": d,
                        "ticker": t,
                        "pred": float(pred[i]),
                        "realized_ret": float(y_realized[i]),
                    }
                )

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / "tgcn_predictions.csv", index=False)

    # IC and hit rate
    daily_metrics = []
    for d, g in pred_df.groupby("date"):
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred", "realized_ret"])

        if g["pred"].nunique() < 2 or g["realized_ret"].nunique() < 2:
            ic = np.nan
        else:
            ic = rank_ic(g["pred"], g["realized_ret"])

        hit = hit_rate(g["pred"], g["realized_ret"], top_k=config["evaluation"]["top_k"])
        daily_metrics.append({"date": d, "ic": ic, "hit": hit})

    daily_metrics = pd.DataFrame(daily_metrics).sort_values("date")
    mean_ic = daily_metrics["ic"].dropna().mean()
    mean_hit = daily_metrics["hit"].dropna().mean()

    daily_metrics.to_csv(out_dir / "tgcn_daily_metrics.csv", index=False)
    ic_path = out_dir / "tgcn_ic_timeseries.png"
    ic_hist_path = out_dir / "tgcn_ic_histogram.png"
    if daily_metrics["ic"].notna().any():
        plot_daily_ic(daily_metrics, ic_path)
        plot_ic_hist(daily_metrics, ic_hist_path)
        print(f"[TGCN] Saved IC plots: {ic_path}, {ic_hist_path}")
    else:
        print("[TGCN] Skipping IC plots (all IC values NaN)")
    print("TGCN mean IC", mean_ic)
    print("TGCN mean hit", mean_hit)

    curve, daily_ret, stats = backtest_long_only(
        pred_df,
        top_k=config["evaluation"]["top_k"],
        transaction_cost_bps=config["evaluation"]["transaction_cost_bps"],
        risk_free_rate=config["evaluation"]["risk_free_rate"],
    )
    curve.to_csv(out_dir / "tgcn_equity_curve.csv", header=["value"])
    plot_equity_curve(
        curve,
        "TGCN long only",
        out_dir / "tgcn_equity_curve.png",
    )
    print(f"[TGCN] Saved equity curve to {out_dir / 'tgcn_equity_curve.png'}")

    bh_df = pred_df[["date", "ticker", "realized_ret"]].rename(columns={"realized_ret": "log_ret_1d"})
    eq_bh, ret_bh, stats_bh = backtest_buy_and_hold(
        bh_df,
        risk_free_rate=config["evaluation"]["risk_free_rate"],
    )
    eq_bh.to_csv(out_dir / "tgcn_buy_and_hold_equity_curve.csv", header=["value"])
    plot_equity_curve(
        eq_bh,
        "Buy and Hold",
        out_dir / "tgcn_buy_and_hold_equity_curve.png",
    )
    print(f"[TGCN] Saved buy-and-hold curve to {out_dir / 'tgcn_buy_and_hold_equity_curve.png'}")

    plot_equity_comparison(
        model_curve=curve,
        bh_curve=eq_bh,
        title="TGCN vs Buy and Hold",
        out_path=out_dir / "tgcn_equity_comparison.png",
    )
    print(f"[TGCN] Saved equity comparison to {out_dir / 'tgcn_equity_comparison.png'}")

    print("TGCN backtest stats", stats)
    print("Buy-and-hold stats", stats_bh)


# public entry from train.py
def train_gnn(config):
    gnn_type = config["model"]["type"].lower()
    if gnn_type in {"gcn", "gat"}:
        _train_static_gnn(config)
    elif gnn_type == "tgcn":
        _train_tgcn(config)
    else:
        raise ValueError(f"Unknown gnn type {gnn_type}")
