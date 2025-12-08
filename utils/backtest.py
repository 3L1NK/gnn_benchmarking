# utils/backtest.py

import pandas as pd
import numpy as np
from .metrics import sharpe_ratio, sortino_ratio


def backtest_long_short(pred_df, top_k=20, transaction_cost_bps=5, risk_free_rate=0.0):
    """
    Market neutral long short portfolio.

    Input:
        pred_df: DataFrame with columns [date, ticker, pred, realized_ret]
        top_k: number of long and short positions per day
        transaction_cost_bps: cost in basis points per unit of turnover
        risk_free_rate: daily risk free rate used in Sharpe and Sortino

    Logic:
        - each day sort by pred
        - go long top_k, short bottom_k
        - equal weights on long and short side
        - daily rebalancing with transaction costs based on turnover
    """
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "pred"], ascending=[True, False])

    dates = df["date"].drop_duplicates().sort_values().tolist()

    equity = 1.0
    equity_curve = []
    daily_returns = []
    daily_turnover = []

    prev_weights = {}

    for d in dates:
        day_df = df[df["date"] == d]
        if day_df.empty:
            continue

        day_df = day_df.sort_values("pred", ascending=False)

        long = day_df.head(top_k)
        short = day_df.tail(top_k)

        # equal weighting on both legs
        w_long = 0.5 / max(len(long), 1)
        w_short = -0.5 / max(len(short), 1)

        new_weights = {}
        for t in long["ticker"]:
            new_weights[t] = w_long
        for t in short["ticker"]:
            new_weights[t] = w_short

        # turnover
        turnover = 0.0
        tickers = set(new_weights.keys()) | set(prev_weights.keys())
        for t in tickers:
            old_w = prev_weights.get(t, 0.0)
            new_w = new_weights.get(t, 0.0)
            turnover += abs(new_w - old_w)
        daily_turnover.append(turnover)

        transaction_cost = turnover * (transaction_cost_bps / 10000.0)

        todays_ret = day_df.set_index("ticker")["realized_ret"].to_dict()

        r = 0.0
        for t, w in new_weights.items():
            r += w * todays_ret.get(t, 0.0)

        r_net = r - transaction_cost

        equity *= (1 + r_net)
        equity_curve.append((d, equity))
        daily_returns.append(r_net)

        prev_weights = new_weights

    if not equity_curve:
        raise ValueError("No equity points computed in backtest_long_short")

    eq_series = pd.Series(
        [v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
    )

    stats = {
        "final_value": float(equity),
        "sharpe": sharpe_ratio(daily_returns, risk_free_rate),
        "sortino": sortino_ratio(daily_returns, risk_free_rate),
        "avg_turnover": float(np.mean(daily_turnover)) if daily_turnover else 0.0,
    }

    return eq_series, daily_returns, stats


def backtest_long_only(pred_df, top_k=20, transaction_cost_bps=5, risk_free_rate=0.0):
    """
    Long only top_k portfolio.

    Input:
        pred_df: DataFrame with columns [date, ticker, pred, realized_ret]
        top_k: number of long positions per day
        transaction_cost_bps: cost in basis points per unit of turnover
        risk_free_rate: daily risk free rate used in Sharpe and Sortino

    Logic:
        - each day sort by pred
        - go long top_k assets only
        - equal weights, fully invested, no shorts
        - daily rebalancing with transaction costs based on turnover
    """
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "pred"], ascending=[True, False])

    dates = df["date"].drop_duplicates().sort_values().tolist()

    equity = 1.0
    equity_curve = []
    daily_returns = []
    daily_turnover = []

    prev_weights = {}

    for d in dates:
        day_df = df[df["date"] == d]
        if day_df.empty:
            continue

        day_df = day_df.sort_values("pred", ascending=False)
        long = day_df.head(top_k)

        # equal weights on long only side
        w_long = 1.0 / max(len(long), 1)

        new_weights = {t: w_long for t in long["ticker"]}

        # turnover
        turnover = 0.0
        tickers = set(new_weights.keys()) | set(prev_weights.keys())
        for t in tickers:
            old_w = prev_weights.get(t, 0.0)
            new_w = new_weights.get(t, 0.0)
            turnover += abs(new_w - old_w)
        daily_turnover.append(turnover)

        transaction_cost = turnover * (transaction_cost_bps / 10000.0)

        todays_ret = day_df.set_index("ticker")["realized_ret"].to_dict()

        r = 0.0
        for t, w in new_weights.items():
            r += w * todays_ret.get(t, 0.0)

        r_net = r - transaction_cost

        equity *= (1 + r_net)
        equity_curve.append((d, equity))
        daily_returns.append(r_net)

        prev_weights = new_weights

    if not equity_curve:
        raise ValueError("No equity points computed in backtest_long_only")

    eq_series = pd.Series(
        [v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
    )

    stats = {
        "final_value": float(equity),
        "sharpe": sharpe_ratio(daily_returns, risk_free_rate),
        "sortino": sortino_ratio(daily_returns, risk_free_rate),
        "avg_turnover": float(np.mean(daily_turnover)) if daily_turnover else 0.0,
    }

    return eq_series, daily_returns, stats


def backtest_buy_and_hold(price_panel, risk_free_rate=0.0, return_col="log_ret_1d"):
    """
    Equal weight buy and hold portfolio.

    Input:
        price_panel: DataFrame with columns [date, ticker, <return_col>]
                     already filtered to the test period
        risk_free_rate: daily risk free rate for Sharpe and Sortino
        return_col: name of the column containing daily returns

    Logic:
        - at first test date, invest equally in all available tickers
        - hold weights constant, no rebalancing, no transaction costs
    """
    df = price_panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # pivot returns to matrix [dates, tickers]
    ret_wide = df.pivot(index="date", columns="ticker", values=return_col).sort_index()
    ret_wide = ret_wide.fillna(0.0)

    ret_mat = ret_wide.values  # shape [T, N]
    n_assets = ret_mat.shape[1]
    if n_assets == 0:
        raise ValueError("No assets in price_panel for buy and hold backtest")

    w = np.ones(n_assets) / n_assets

    port_ret = ret_mat.dot(w)
    equity = (1 + port_ret).cumprod()

    eq_series = pd.Series(equity, index=ret_wide.index)

    stats = {
        "final_value": float(eq_series.iloc[-1]),
        "sharpe": sharpe_ratio(port_ret, risk_free_rate),
        "sortino": sortino_ratio(port_ret, risk_free_rate),
        "avg_turnover": 0.0,  # no trading after initial allocation
    }

    return eq_series, port_ret, stats
