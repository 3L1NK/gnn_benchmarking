# trainers/train_gnn.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

from models.gnn_model import StaticGNN
from models.tgat_model import TGATModel
from utils.seed import set_seed
from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.graphs import rolling_corr_edges, build_graph_snapshot
from utils.metrics import mse, rank_ic, hit_rate
from utils.backtest import backtest_long_short
from utils.plots import plot_equity_curve


def _build_snapshots_and_targets(config):
    """
    Build per day graph snapshots with node features and targets.

    Used for:
      - static GCN and GAT
      - temporal TGAT (we convert snapshots to temporal stream)
    """
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    horizon = config["data"]["target_horizon"]
    lookback = config["data"]["lookback_window"]
    corr_window = config["data"]["corr_window"]
    corr_thr = config["data"]["corr_threshold"]

    df = load_price_panel(price_file, start, end)
    df, feat_cols = add_technical_features(df)

    # target next day log return
    df["target"] = df.groupby("ticker")["log_ret_1d"].shift(-horizon)
    df = df.dropna(subset=["target"])

    dates = sorted(df["date"].unique())
    snapshots = []
    meta_dates = []

    for d in dates:
        window_start = d - pd.Timedelta(days=lookback * 2)
        hist = df[(df["date"] > window_start) & (df["date"] <= d)]
        if hist["date"].nunique() < lookback // 2:
            continue

        feat_for_date = {}
        target_for_date = {}
        universe_today = hist[hist["date"] == d]

        for _, row in universe_today.iterrows():
            t = row["ticker"]
            feat_for_date[t] = row[feat_cols].values.astype(float)
            target_for_date[t] = float(row["target"])

        if not feat_for_date:
            continue

        edges = rolling_corr_edges(df, d, corr_window, corr_thr)
        graph = build_graph_snapshot(
            d,
            universe_today[["ticker"]].drop_duplicates(),
            feat_for_date,
            edges,
        )

        y = []
        for t in graph.tickers:
            y.append(target_for_date.get(t, np.nan))
        graph.y = torch.tensor(np.asarray(y), dtype=torch.float32)

        snapshots.append(graph)
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

    train_loader = GeoDataLoader(
        train_snaps,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = GeoDataLoader(val_snaps, batch_size=1, shuffle=False)
    test_loader = GeoDataLoader(test_snaps, batch_size=1, shuffle=False)

    model = StaticGNN(
        gnn_type=config["model"]["type"],
        input_dim=len(feat_cols),
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
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

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"best_{config['model']['type']}.pt"

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_weight)
            mask = torch.isfinite(batch.y) * (batch.valid_mask > 0)
            if mask.sum() == 0:
                continue
            loss = loss_fn(pred[mask], batch.y[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_weight)
                mask = torch.isfinite(batch.y) * (batch.valid_mask > 0)
                if mask.sum() == 0:
                    continue
                loss = loss_fn(pred[mask], batch.y[mask])
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"[{config['model']['type'].upper()}] epoch {epoch} train {mean_train:.5f} val {mean_val:.5f}")

        if mean_val < best_val:
            best_val = mean_val
            bad_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # test predictions and backtest
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_weight)
            pred = pred.cpu().numpy()
            y = batch.y.cpu().numpy()
            tickers = batch.tickers
            date = batch.date

            for i, t in enumerate(tickers):
                if not np.isfinite(y[i]):
                    continue
                rows.append(
                    {
                        "date": pd.to_datetime(date),
                        "ticker": t,
                        "pred": float(pred[i]),
                        "realized_ret": float(y[i]),
                    }
                )

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / f"{config['model']['type']}_predictions.csv", index=False)

    daily_metrics = []
    for d, g in pred_df.groupby("date"):
        ic = rank_ic(g["pred"], g["realized_ret"])
        hit = hit_rate(g["pred"], g["realized_ret"], top_k=config["evaluation"]["top_k"])
        daily_metrics.append({"date": d, "ic": ic, "hit": hit})
    daily_metrics = pd.DataFrame(daily_metrics).sort_values("date")
    daily_metrics.to_csv(out_dir / f"{config['model']['type']}_daily_metrics.csv", index=False)

    print(
        f"{config['model']['type'].upper()} mean IC",
        daily_metrics["ic"].mean(),
        "mean hit",
        daily_metrics["hit"].mean(),
    )

    equity_curve, daily_ret, stats = backtest_long_short(
        pred_df,
        top_k=config["evaluation"]["top_k"],
        transaction_cost_bps=config["evaluation"]["transaction_cost_bps"],
        risk_free_rate=config["evaluation"]["risk_free_rate"],
    )
    equity_curve.to_csv(out_dir / f"{config['model']['type']}_equity_curve.csv", header=["value"])
    plot_equity_curve(
        equity_curve,
        f"{config['model']['type'].upper()} long short",
        out_dir / f"{config['model']['type']}_equity_curve.png",
    )
    print(f"{config['model']['type'].upper()} backtest stats", stats)


# 2. Training for temporal TGAT
def _train_tgat(config):
    """
    Simple temporal training.

    We still build daily snapshots, but TGAT sees time stamps
    and can attend over time.
    """
    set_seed(42)

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    snapshots, feat_cols, dates = _build_snapshots_and_targets(config)

    # assign integer time index per snapshot
    date_to_time = {d: i for i, d in enumerate(sorted(set(dates)))}
    for g, d in zip(snapshots, dates):
        t_idx = date_to_time[d]
        # per node time stamp tensor
        time_tensor = torch.full((g.x.size(0),), float(t_idx), dtype=torch.float32)
        g.time = time_tensor

    train_snaps, val_snaps, test_snaps = _split_snapshots_by_date(
        snapshots, dates, config["training"]["val_start"], config["training"]["test_start"]
    )

    train_loader = GeoDataLoader(
        train_snaps,
        batch_size=1,
        shuffle=True,
    )
    val_loader = GeoDataLoader(val_snaps, batch_size=1, shuffle=False)
    test_loader = GeoDataLoader(test_snaps, batch_size=1, shuffle=False)

    model = TGATModel(
        input_dim=len(feat_cols),
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
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

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_tgat.pt"

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.time)
            mask = torch.isfinite(batch.y) * (batch.valid_mask > 0)
            if mask.sum() == 0:
                continue
            loss = loss_fn(pred[mask], batch.y[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.time)
                mask = torch.isfinite(batch.y) * (batch.valid_mask > 0)
                if mask.sum() == 0:
                    continue
                loss = loss_fn(pred[mask], batch.y[mask])
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"[TGAT] epoch {epoch} train {mean_train:.5f} val {mean_val:.5f}")

        if mean_val < best_val:
            best_val = mean_val
            bad_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_weight, batch.time)
            pred = pred.cpu().numpy()
            y = batch.y.cpu().numpy()
            tickers = batch.tickers
            date = batch.date

            for i, t in enumerate(tickers):
                if not np.isfinite(y[i]):
                    continue
                rows.append(
                    {
                        "date": pd.to_datetime(date),
                        "ticker": t,
                        "pred": float(pred[i]),
                        "realized_ret": float(y[i]),
                    }
                )

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / "tgat_predictions.csv", index=False)

    daily_metrics = []
    for d, g in pred_df.groupby("date"):
        ic = rank_ic(g["pred"], g["realized_ret"])
        hit = hit_rate(g["pred"], g["realized_ret"], top_k=config["evaluation"]["top_k"])
        daily_metrics.append({"date": d, "ic": ic, "hit": hit})
    daily_metrics = pd.DataFrame(daily_metrics).sort_values("date")
    daily_metrics.to_csv(out_dir / "tgat_daily_metrics.csv", index=False)

    print(
        "TGAT mean IC",
        daily_metrics["ic"].mean(),
        "mean hit",
        daily_metrics["hit"].mean(),
    )

    equity_curve, daily_ret, stats = backtest_long_short(
        pred_df,
        top_k=config["evaluation"]["top_k"],
        transaction_cost_bps=config["evaluation"]["transaction_cost_bps"],
        risk_free_rate=config["evaluation"]["risk_free_rate"],
    )
    equity_curve.to_csv(out_dir / "tgat_equity_curve.csv", header=["value"])
    plot_equity_curve(
        equity_curve,
        "TGAT long short",
        out_dir / "tgat_equity_curve.png",
    )
    print("TGAT backtest stats", stats)


# public entry from train.py
def train_gnn(config):
    gnn_type = config["model"]["type"].lower()
    if gnn_type in {"gcn", "gat"}:
        _train_static_gnn(config)
    elif gnn_type == "tgat":
        _train_tgat(config)
    else:
        raise ValueError(f"Unknown gnn type {gnn_type}")
