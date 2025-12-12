import pandas as pd
import numpy as np
from .metrics import sharpe_ratio, sortino_ratio

def backtest_buy_and_hold(price_panel, risk_free_rate=0.0):
    """
    price_panel has columns: date, ticker, log_ret_1d (or some daily return).
    Equal weight in all tickers at start, never rebalance.
    """

    df = price_panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # pivot returns to matrix [dates, tickers]
    ret_mat = df.pivot(index="date", columns="ticker", values="log_ret_1d").sort_index()
    ret_mat = ret_mat.fillna(0.0).values  # shape [T, N]

    n_assets = ret_mat.shape[1]
    w = np.ones(n_assets) / n_assets

    port_ret = ret_mat.dot(w)  # shape [T]
    equity = (1 + port_ret).cumprod()

    eq_series = pd.Series(equity, index=sorted(df["date"].unique()))

    stats = {
        "final_value": float(eq_series.iloc[-1]),
        "sharpe": sharpe_ratio(port_ret, risk_free_rate),
        "sortino": sortino_ratio(port_ret, risk_free_rate),
    }

    return eq_series, port_ret, stats


def backtest_long_only(pred_df, top_k=20, transaction_cost_bps=5, risk_free_rate=0.0):
    """
    pred_df columns: date, ticker, pred, realized_ret
    Long only top_k portfolio, equal weight, daily rebalancing.
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

        # equal weight only on long side
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

    eq_series = pd.Series(
        [v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
    )

    stats = {
        "final_value": equity,
        "sharpe": sharpe_ratio(daily_returns, risk_free_rate),
        "sortino": sortino_ratio(daily_returns, risk_free_rate),
        "avg_turnover": float(np.mean(daily_turnover)) if daily_turnover else 0.0,
    }

    return eq_series, daily_returns, stats


def backtest_long_short(pred_df, top_k=20, transaction_cost_bps=5, risk_free_rate=0.0):
    """
    pred_df columns: date, ticker, pred, realized_ret

    Implements:
        - equal weighted long short portfolio
        - daily rebalancing
        - turnover based transaction costs
        - net returns
    """

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "pred"], ascending=[True, False])

    dates = df["date"].drop_duplicates().sort_values().tolist()

    equity = 1.0
    equity_curve = []
    daily_returns = []

    prev_weights = {}

    for d in dates:
        day_df = df[df["date"] == d]
        if day_df.empty:
            continue

        day_df = day_df.sort_values("pred", ascending=False)

        long = day_df.head(top_k)
        short = day_df.tail(top_k)

        # equal weighting
        w_long = 0.5 / max(len(long), 1)
        w_short = -0.5 / max(len(short), 1)

        new_weights = {}

        for t in long["ticker"]:
            new_weights[t] = w_long
        for t in short["ticker"]:
            new_weights[t] = w_short

        # compute turnover cost
        turnover = 0.0
        tickers = set(new_weights.keys()) | set(prev_weights.keys())

        for t in tickers:
            old_w = prev_weights.get(t, 0.0)
            new_w = new_weights.get(t, 0.0)
            turnover += abs(new_w - old_w)

        transaction_cost = turnover * (transaction_cost_bps / 10000.0)

        # compute portfolio return
        todays_ret = day_df.set_index("ticker")["realized_ret"].to_dict()

        r = 0.0
        for t, w in new_weights.items():
            r += w * todays_ret.get(t, 0.0)

        # net return after cost
        r_net = r - transaction_cost

        equity *= (1 + r_net)
        equity_curve.append((d, equity))
        daily_returns.append(r_net)

        prev_weights = new_weights

    eq_series = pd.Series(
        [v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
    )

    stats = {
        "final_value": equity,
        "sharpe": sharpe_ratio(daily_returns, risk_free_rate),
        "sortino": sortino_ratio(daily_returns, risk_free_rate),
        "avg_turnover": turnover / len(dates),
    }

    return eq_series, daily_returns, stats
