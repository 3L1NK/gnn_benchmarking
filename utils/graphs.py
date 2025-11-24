import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso
from torch_geometric.utils import from_networkx

def industry_edges(universe_df: pd.DataFrame):
    edges = []
    for industry, group in universe_df.groupby("industry"):
        tickers = list(group["ticker"])
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                edges.append((tickers[i], tickers[j], 1.0))
    return edges

def sector_edges(universe_df: pd.DataFrame):
    # universe_df: columns ticker, sector
    edges = []
    for sector, group in universe_df.groupby("sector"):
        tickers = list(group["ticker"])
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                edges.append((tickers[i], tickers[j], 1.0))
    return edges


def rolling_corr_edges(panel_df, date, window, threshold):
    """panel_df has columns date, ticker, log_ret_1d"""
    end_date = date
    start_date = end_date - pd.Timedelta(days=window * 2)
    hist = panel_df[(panel_df["date"] > start_date) & (panel_df["date"] <= end_date)]

    if hist.empty:
        return []

    pivot = hist.pivot(index="date", columns="ticker", values="log_ret_1d").dropna(axis=1, how="all")
    corr = pivot.corr()

    edges = []
    for i, ti in enumerate(corr.columns):
        for j, tj in enumerate(corr.columns):
            if j <= i:
                continue
            c = corr.iloc[i, j]
            if np.abs(c) >= threshold:
                edges.append((ti, tj, float(c)))
    return edges

def graphical_lasso_precision(
    panel_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    alpha: float = 0.01,
):
    """
    Fit Graphical Lasso on log returns between start_date and end_date.

    Returns:
      tickers: list of tickers in the model
      precision: precision matrix as numpy array
      edges: list of (ticker_i, ticker_j, weight_ij)
      adj_dict: dict ticker -> list of (neighbor, weight)
    """
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    df = df.loc[mask]

    pivot = df.pivot(index="date", columns="ticker", values="log_ret_1d").dropna(axis=1, how="any")
    if pivot.shape[1] < 3:
        raise ValueError("Not enough tickers with full history for Graphical Lasso")

    model = GraphicalLasso(alpha=alpha, max_iter=200)
    model.fit(pivot.values)
    prec = model.precision_
    cols = pivot.columns.tolist()

    edges = []
    adj = {t: [] for t in cols}

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            w = -prec[i, j]
            if np.abs(w) > 1e-4:
                ti, tj = cols[i], cols[j]
                edges.append((ti, tj, float(w)))
                adj[ti].append((tj, float(w)))
                adj[tj].append((ti, float(w)))

    return cols, prec, edges, adj

def graphical_lasso_edges(panel_df, date, window, alpha=0.01):
    end_date = date
    start_date = end_date - pd.Timedelta(days=window * 2)
    hist = panel_df[(panel_df["date"] > start_date) & (panel_df["date"] <= end_date)]

    if hist.empty:
        return []

    pivot = hist.pivot(index="date", columns="ticker", values="log_ret_1d").dropna(axis=1, how="any")
    if pivot.shape[1] < 3:
        return []

    model = GraphicalLasso(alpha=alpha, max_iter=100)
    model.fit(pivot.values)
    prec = model.precision_

    cols = pivot.columns
    edges = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            w = -prec[i, j]  # negative off diagonal indicates conditional dependence
            if np.abs(w) > 1e-4:
                edges.append((cols[i], cols[j], float(w)))
    return edges


def build_graph_snapshot(date, universe, features_for_date, edge_list):
    """Return torch_geometric Data object."""
    import torch
    from torch_geometric.data import Data

    tickers = universe["ticker"].tolist()
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    feat_mat = []
    valid_mask = []

    for t in tickers:
        row = features_for_date.get(t)
        if row is None or np.any(np.isnan(row)):
            feat_mat.append(np.zeros(len(next(iter(features_for_date.values())))))
            valid_mask.append(0.0)
        else:
            feat_mat.append(row)
            valid_mask.append(1.0)

    x = torch.tensor(np.asarray(feat_mat), dtype=torch.float32)
    valid_mask = torch.tensor(valid_mask, dtype=torch.float32)

    edges_idx = []
    edge_weight = []
    for u, v, w in edge_list:
        if u not in ticker_to_idx or v not in ticker_to_idx:
            continue
        i, j = ticker_to_idx[u], ticker_to_idx[v]
        edges_idx.append([i, j])
        edges_idx.append([j, i])
        edge_weight.append(w)
        edge_weight.append(w)

    if len(edges_idx) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight_tensor = torch.empty((0,), dtype=torch.float32)
    else:
        edge_index = torch.tensor(np.array(edges_idx).T, dtype=torch.long)
        edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight_tensor)
    data.valid_mask = valid_mask
    data.tickers = tickers
    data.date = date
    return data
