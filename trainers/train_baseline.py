# trainers/train_baseline.py

from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.graphs import rolling_corr_edges, graphical_lasso_precision
from utils.metrics import mse, rank_ic, hit_rate
from utils.backtest import backtest_long_short
from utils.plot import plot_equity_curve
from utils.seeds import set_seed


def train_baseline(config):
    baseline_type = config["model"]["type"].lower()

    if baseline_type == "xgb_raw":
        train_xgb_raw(config)
    elif baseline_type == "xgb_node2vec":
        train_xgb_node2vec(config)
    elif baseline_type == "graphlasso_linear":
        train_graphlasso_linear(config)
    elif baseline_type == "graphlasso_xgb":
        train_graphlasso_xgb(config)
    else:
        raise ValueError(f"Unknown baseline type {baseline_type}")

def _build_feature_panel(config):
    """Shared helper. Returns df with features and target, list of feature columns."""
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    horizon = config["data"]["target_horizon"]

    df = load_price_panel(price_file, start, end)
    df, feat_cols = add_technical_features(df)

    # target next day log return
    df["target"] = df.groupby("ticker")["log_ret_1d"].shift(-horizon)
    df = df.dropna(subset=["target"])
    return df, feat_cols


def _time_masks(dates, val_start, test_start):
    dates = pd.to_datetime(dates)
    val_start = pd.to_datetime(val_start)
    test_start = pd.to_datetime(test_start)
    train_mask = dates < val_start
    val_mask = (dates >= val_start) & (dates < test_start)
    test_mask = dates >= test_start
    return train_mask, val_mask, test_mask


def train_xgb_raw(config):
    from xgboost import XGBRegressor

    set_seed(42)

    df, feat_cols = _build_feature_panel(config)
    dates = df["date"]
    train_mask, val_mask, test_mask = _time_masks(
        dates,
        config["training"]["val_start"],
        config["training"]["test_start"],
    )

    X = df[feat_cols].values.astype(float)
    y = df["target"].values.astype(float)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    preds_test = model.predict(X_test)

    df_test = df.loc[test_mask, ["date", "ticker"]].copy()
    df_test["pred"] = preds_test
    df_test["realized_ret"] = y_test

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(out_dir / "xgb_raw_predictions.csv", index=False)

    _evaluate_and_backtest(
        df_test,
        out_dir,
        name="xgb_raw",
        config=config,
    )

def train_xgb_node2vec(config):
    from torch_geometric.nn import Node2Vec
    from xgboost import XGBRegressor
    import networkx as nx

    set_seed(42)

    df, feat_cols = _build_feature_panel(config)

    # build one global correlation graph on train period only
    val_start = pd.to_datetime(config["training"]["val_start"])
    train_df = df[df["date"] < val_start]

    last_train_date = train_df["date"].max()
    corr_window = config["data"]["lookback_window"]
    corr_thr = config["data"].get("corr_threshold", 0.4)

    edges = rolling_corr_edges(
        train_df,
        last_train_date,
        corr_window,
        corr_thr,
    )

    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=abs(w))

    # Node2Vec on this graph
    n2v = Node2Vec(
        n_components=16,
        walklen=40,
        epochs=15,
        return_weight=1.0,
        neighbor_weight=1.0,
        threads=4,
    )
    n2v.fit(G)
    emb = {node: n2v.predict(node) for node in G.nodes()}

    rows = []
    for _, row in df.iterrows():
        t = row["ticker"]
        if t not in emb:
            continue
        feat = row[feat_cols].values.astype(float)
        vec = emb[t]
        x = np.concatenate([feat, vec])
        rows.append(
            {
                "date": row["date"],
                "ticker": t,
                "target": row["target"],
                "features": x,
            }
        )

    tab = pd.DataFrame(rows)
    X = np.stack(tab["features"].values)
    y = tab["target"].values.astype(float)

    train_mask, val_mask, test_mask = _time_masks(
        tab["date"],
        config["training"]["val_start"],
        config["training"]["test_start"],
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    preds_test = model.predict(X_test)

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tab_test = tab.loc[test_mask, ["date", "ticker"]].copy()
    tab_test["pred"] = preds_test
    tab_test["realized_ret"] = y_test
    tab_test.to_csv(out_dir / "xgb_node2vec_predictions.csv", index=False)

    _evaluate_and_backtest(
        tab_test,
        out_dir,
        name="xgb_node2vec",
        config=config,
    )

def _add_graph_smooth_features(df, feat_cols, adj_dict, alpha=0.5):
    """
    For each ticker and date:
      new_feat = alpha * own + (1 - alpha) * neighbor weighted average

    Returns df with extra columns for smoothed features.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    smoothed_cols = []

    for col in feat_cols:
        new_col = col + "_smooth"
        smoothed_cols.append(new_col)
        df[new_col] = df[col]

    # pre group by date for speed
    by_date = dict(tuple(df.groupby("date", sort=False)))

    out_frames = []
    for date, g in by_date.items():
        g = g.copy()
        ticker_to_idx = {t: i for i, t in enumerate(g["ticker"])}
        for idx, row in g.iterrows():
            t = row["ticker"]
            if t not in adj_dict:
                continue
            neighbors = adj_dict[t]
            if not neighbors:
                continue

            weights = []
            neigh_feat = {c: [] for c in feat_cols}

            for nb, w in neighbors:
                if nb not in ticker_to_idx:
                    continue
                nb_row = g.iloc[ticker_to_idx[nb]]
                weights.append(abs(w))
                for c in feat_cols:
                    neigh_feat[c].append(nb_row[c])

            if not weights:
                continue
            w_arr = np.array(weights)
            w_arr = w_arr / (w_arr.sum() + 1e-8)

            for c in feat_cols:
                neigh_vals = np.array(neigh_feat[c])
                neigh_mean = (w_arr * neigh_vals).sum()
                new_val = alpha * row[c] + (1 - alpha) * neigh_mean
                g.at[idx, c + "_smooth"] = new_val

        out_frames.append(g)

    df_out = pd.concat(out_frames, ignore_index=True)
    return df_out, smoothed_cols


def train_graphlasso_linear(config):
    from sklearn.linear_model import Ridge

    set_seed(42)

    df, feat_cols = _build_feature_panel(config)

    # fit graphical lasso on train period
    val_start = config["training"]["val_start"]
    cols, prec, edges, adj = graphical_lasso_precision(
        df,
        start_date=config["data"]["start_date"],
        end_date=val_start,
        alpha=config.get("graphlasso_alpha", 0.01),
    )

    df_smooth, smooth_cols = _add_graph_smooth_features(df, feat_cols, adj, alpha=0.5)

    X = df_smooth[feat_cols + smooth_cols].values.astype(float)
    y = df_smooth["target"].values.astype(float)

    train_mask, val_mask, test_mask = _time_masks(
        df_smooth["date"],
        config["training"]["val_start"],
        config["training"]["test_start"],
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test = df_smooth.loc[test_mask, ["date", "ticker"]].copy()
    df_test["pred"] = preds_test
    df_test["realized_ret"] = y_test
    df_test.to_csv(out_dir / "graphlasso_linear_predictions.csv", index=False)

    _evaluate_and_backtest(
        df_test,
        out_dir,
        name="graphlasso_linear",
        config=config,
    )

def train_graphlasso_xgb(config):
    from xgboost import XGBRegressor

    set_seed(42)

    df, feat_cols = _build_feature_panel(config)

    val_start = config["training"]["val_start"]
    cols, prec, edges, adj = graphical_lasso_precision(
        df,
        start_date=config["data"]["start_date"],
        end_date=val_start,
        alpha=config.get("graphlasso_alpha", 0.01),
    )

    df_smooth, smooth_cols = _add_graph_smooth_features(df, feat_cols, adj, alpha=0.5)

    X = df_smooth[feat_cols + smooth_cols].values.astype(float)
    y = df_smooth["target"].values.astype(float)

    train_mask, val_mask, test_mask = _time_masks(
        df_smooth["date"],
        config["training"]["val_start"],
        config["training"]["test_start"],
    )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    preds_test = model.predict(X_test)

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test = df_smooth.loc[test_mask, ["date", "ticker"]].copy()
    df_test["pred"] = preds_test
    df_test["realized_ret"] = y_test
    df_test.to_csv(out_dir / "graphlasso_xgb_predictions.csv", index=False)

    _evaluate_and_backtest(
        df_test,
        out_dir,
        name="graphlasso_xgb",
        config=config,
    )

def _evaluate_and_backtest(pred_df, out_dir: Path, name: str, config):
    # day level IC and hit rate
    daily_metrics = []
    for d, g in pred_df.groupby("date"):
        ic = rank_ic(g["pred"], g["realized_ret"])
        hit = hit_rate(g["pred"], g["realized_ret"], top_k=config["evaluation"]["top_k"])
        daily_metrics.append({"date": d, "ic": ic, "hit": hit})

    daily_metrics = pd.DataFrame(daily_metrics).sort_values("date")
    daily_metrics.to_csv(out_dir / f"{name}_daily_metrics.csv", index=False)

    print(
        f"[{name}] mean IC",
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
    equity_curve.to_csv(out_dir / f"{name}_equity_curve.csv", header=["value"])
    plot_equity_curve(
        equity_curve,
        f"{name} long short",
        out_dir / f"{name}_equity_curve.png",
    )
    print(f"[{name}] backtest stats", stats)
