import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso
from torch_geometric.utils import from_networkx

from statsmodels.tsa.stattools import grangercausalitytests

def granger_edges(panel_df, max_lag=2, p_threshold=0.05):
    """
    Build directed Granger causality edges.
    panel_df must have columns: date, ticker, log_ret_1d.
    """
    df = panel_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # pivot to matrix shape [dates x tickers]
    pivot = df.pivot(index="date", columns="ticker", values="log_ret_1d").dropna()

    tickers = pivot.columns.tolist()
    raw = []  # collect (u, v, best_p)

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            if i == j:
                continue

            x = pivot[tickers[i]].values
            y = pivot[tickers[j]].values

            # A causes B means: past A helps predict B
            data = np.column_stack([y, x])

            try:
                result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            except Exception:
                continue

            # find the best (smallest) p-value across tested lags
            best_p = 1.0
            for lag in range(1, max_lag + 1):
                try:
                    p_val = result[lag][0]["ssr_ftest"][1]
                except Exception:
                    continue
                if p_val < best_p:
                    best_p = p_val

            if best_p < p_threshold:
                raw.append((tickers[i], tickers[j], float(best_p)))

    # Convert p-values to weights using -log(p) then normalize to [0,1]
    if not raw:
        return []

    eps = 1e-12
    pvals = np.array([r[2] for r in raw], dtype=float)
    # transform: w = -log(p); larger when p small
    w = -np.log(np.clip(pvals, eps, 1.0))
    if np.all(w == 0):
        # numerical fallback
        w = 1.0 - pvals
    # normalize to [0,1]
    w = w / (w.max() + 1e-12)

    edges = []
    for (u, v, _), weight in zip(raw, w.tolist()):
        edges.append((u, v, float(weight)))

    return edges


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
    # Use the last `window` trading rows up to `date` (inclusive) instead of
    # approximating with calendar days. This better matches trading-day windows.
    end_date = pd.to_datetime(date)

    pivot_all = (
        panel_df.assign(date=pd.to_datetime(panel_df["date"]))
        .pivot(index="date", columns="ticker", values="log_ret_1d")
        .sort_index()
    )

    if end_date not in pivot_all.index:
        # if the exact date isn't present, select the last index <= end_date
        idxs = pivot_all.index[pivot_all.index <= end_date]
        if len(idxs) == 0:
            return []
        end_pos = idxs.max()
    else:
        end_pos = end_date

    hist = pivot_all.loc[:end_pos].tail(window)

    if hist.empty:
        return []

    pivot = hist.dropna(axis=1, how="all")
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
    returns_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    alpha: float = 0.01,
):
    """
    Fit Graphical Lasso on log returns only.
    returns_df must have columns: date, ticker, log_ret_1d
    """

    df = returns_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # make end date exclusive to avoid leaking the first validation date into training
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] < pd.to_datetime(end_date))
    df = df.loc[mask]

    # pivot into matrix [dates × tickers]
    pivot = df.pivot(index="date", columns="ticker", values="log_ret_1d")

    # drop rows with NaN (not columns)
    # this preserves max number of tickers
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.shape[1] < 3:
        raise ValueError("Not enough full-history tickers for Graphical Lasso")

    # fit Graphical Lasso on covariance of returns
    model = GraphicalLasso(alpha=alpha, max_iter=200)
    model.fit(pivot.values)

    prec = model.precision_
    cols = pivot.columns.tolist()

    edges = []
    adj = {t: [] for t in cols}

    # build undirected weighted graph from precision matrix
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
    # use end_date as exclusive upper bound to avoid including the first validation day
    hist = panel_df[(panel_df["date"] > start_date) & (panel_df["date"] < end_date)]

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
