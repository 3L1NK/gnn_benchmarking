# trainers/train_xgboost.py

from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.graphs import rolling_corr_edges, graphical_lasso_precision, granger_edges
from utils.metrics import mse, rank_ic, hit_rate
from utils.backtest import backtest_long_short
from utils.plot import plot_equity_curve
from utils.seeds import set_seed


def train_xgboost(config):
    """Dispatch XGBoost-based models."""
    xgboost_type = config["model"]["type"].lower()

    if xgboost_type == "xgb_raw":
        train_xgb_raw(config)
    elif xgboost_type == "xgb_node2vec":
        train_xgb_node2vec(config)
    elif xgboost_type == "graphlasso_linear":
        train_graphlasso_linear(config)
    elif xgboost_type == "graphlasso_xgb":
        train_graphlasso_xgb(config)
    elif xgboost_type == "granger_xgb":
        train_granger_xgb(config)
    else:
        raise ValueError(f"Unknown xgboost type {xgboost_type}")

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
    """
    Plain XGBoost model:
    Predict next day returns using only per asset technical features.
    """

    from xgboost import XGBRegressor

    set_seed(42)

    # -----------------------------
    # 1. Load data and features
    # -----------------------------
    df, feat_cols = _build_feature_panel(config)
    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------
    # 2. Time split
    # -----------------------------
    train_mask, val_mask, test_mask = _time_masks(
        df["date"],
        config["training"]["val_start"],
        config["training"]["test_start"],
    )

    X = df[feat_cols].values.astype(float)
    y = df["target"].values.astype(float)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # -----------------------------
    # 3. XGBoost model
    # -----------------------------
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    preds_test = model.predict(X_test)

    # -----------------------------
    # 4. Build prediction dataframe
    # -----------------------------
    df_test = df.loc[test_mask, ["date", "ticker"]].copy()
    df_test["pred"] = preds_test
    df_test["realized_ret"] = y_test

    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(out_dir / "xgb_raw_predictions.csv", index=False)

    # -----------------------------
    # 5. Evaluate and backtest
    # -----------------------------
    _evaluate_and_backtest(
        df_test,
        out_dir,
        name="xgb_raw",
        config=config,
    )

def train_xgb_node2vec(config):
    """
    XGBoost model with Node2Vec embeddings using PyTorch Geometric.
    This avoids nodevectors and avoids old networkx, so it is stable with numpy 2.
    """

    import torch
    import networkx as nx
    import numpy as np
    from xgboost import XGBRegressor
    from torch_geometric.nn import Node2Vec

    set_seed(42)

    # -------------------------------------------------
    # 1. Load feature panel
    # -------------------------------------------------
    df, feat_cols = _build_feature_panel(config)
    df["date"] = pd.to_datetime(df["date"])

    val_start = pd.to_datetime(config["training"]["val_start"])
    train_df = df[df["date"] < val_start]

    last_train_date = train_df["date"].max()
    corr_window = config["data"]["lookback_window"]
    corr_thr = config["data"].get("corr_threshold", 0.4)

    # -------------------------------------------------
    # 2. Build correlation graph (train period only)
    # -------------------------------------------------
    edges = rolling_corr_edges(
        train_df,
        last_train_date,
        corr_window,
        corr_thr,
    )

    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=abs(w))

    tickers = list(G.nodes())
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    # PyG edge_index
    if len(G.edges()) == 0:
        raise ValueError("No edges in correlation graph. Increase corr_threshold or check data.")

    edge_index = torch.tensor(
        [[ticker_to_idx[u] for u, v in G.edges()],
         [ticker_to_idx[v] for u, v in G.edges()]],
        dtype=torch.long
    )

    # -------------------------------------------------
    # 3. Train Node2Vec with PyTorch Geometric
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_dim = config["model"].get("embedding_dim", 16)
    n2v = Node2Vec(
        edge_index=edge_index,
        embedding_dim=emb_dim,
        walk_length=40,
        context_size=20,
        walks_per_node=10,
        num_negative_samples=1,
        sparse=True,
    ).to(device)

    loader = n2v.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(n2v.parameters(), lr=0.01)

    for epoch in range(30):  # 30 epochs is enough
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

    # final embedding matrix
    emb_matrix = n2v().detach().cpu().numpy()

    # -------------------------------------------------
    # 4. Build full dataset (features + embeddings)
    # -------------------------------------------------
    rows = []
    for _, row in df.iterrows():
        t = row["ticker"]
        if t not in ticker_to_idx:
            continue  # ticker missing from graph

        f = row[feat_cols].values.astype(float)
        vec = emb_matrix[ticker_to_idx[t]]
        full_feat = np.concatenate([f, vec])

        rows.append({
            "date": row["date"],
            "ticker": t,
            "target": row["target"],
            "features": full_feat,
        })

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

    # -------------------------------------------------
    # 5. XGBoost on features + embeddings
    # -------------------------------------------------
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    preds_test = model.predict(X_test)

    # -------------------------------------------------
    # 6. Save predictions + backtest
    # -------------------------------------------------
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
    """
    Graphical Lasso + XGBoost model.

    Steps:
      1) Fit Graphical Lasso on log returns only.
      2) Build adjacency graph.
      3) Smooth technical features using neighbors.
      4) Train XGBoost on [raw + smoothed] features.
      5) Evaluate IC, hit, long-short.
    """

    from xgboost import XGBRegressor
    set_seed(42)

    # -----------------------------
    # 1. Load original price panel + features
    # -----------------------------
    df, feat_cols = _build_feature_panel(config)

    # -----------------------------
    # 2. Prepare return-only DataFrame for GraphLasso
    # -----------------------------
    returns_df = df[["date", "ticker", "log_ret_1d"]].copy()

    # -----------------------------
    # 3. Fit Graphical Lasso using TRAIN period only
    # -----------------------------
    val_start = config["training"]["val_start"]
    cols, prec, edges, adj = graphical_lasso_precision(
        returns_df,
        start_date=config["data"]["start_date"],
        end_date=val_start,
        alpha=config.get("graphlasso_alpha", 0.01),
    )

    # -----------------------------
    # 4. Smooth features using adjacency dictionary
    # -----------------------------
    df_smooth, smooth_cols = _add_graph_smooth_features(df, feat_cols, adj, alpha=0.5)

    # -----------------------------
    # 5. Build feature matrix for XGB
    # -----------------------------
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

    # -----------------------------
    # 6. Train XGBoost
    # -----------------------------
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    preds_test = model.predict(X_test)

    # -----------------------------
    # 7. Save results
    # -----------------------------
    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test = df_smooth.loc[test_mask, ["date", "ticker"]].copy()
    df_test["pred"] = preds_test
    df_test["realized_ret"] = y_test
    df_test.to_csv(out_dir / "graphlasso_xgb_predictions.csv", index=False)

    # -----------------------------
    # 8. Evaluate + backtest
    # -----------------------------
    _evaluate_and_backtest(
        df_test,
        out_dir,
        name="graphlasso_xgb",
        config=config,
    )
    
def train_granger_xgb(config):
    from xgboost import XGBRegressor
    set_seed(42)

    # load features
    df, feat_cols = _build_feature_panel(config)

    # build returns panel
    returns_df = df[["date", "ticker", "log_ret_1d"]].copy()

    # get granger edges using only train period
    val_start = config["training"]["val_start"]
    mask = returns_df["date"] < pd.to_datetime(val_start)
    edges = granger_edges(returns_df.loc[mask], max_lag=2, p_threshold=0.05)

    # build adjacency dict
    adj = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, [])  # ensure existence

    # smooth features
    df_smooth, smooth_cols = _add_graph_smooth_features(df, feat_cols, adj, alpha=0.5)

    X = df_smooth[feat_cols + smooth_cols].values
    y = df_smooth["target"].values

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
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    preds_test = model.predict(X_test)

    # save results
    out_dir = Path(config["evaluation"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test = df_smooth.loc[test_mask, ["date", "ticker"]].copy()
    df_test["pred"] = preds_test
    df_test["realized_ret"] = y_test
    df_test.to_csv(out_dir / "granger_xgb_predictions.csv", index=False)

    _evaluate_and_backtest(
        df_test,
        out_dir,
        name="granger_xgb",
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
