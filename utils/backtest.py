import pandas as pd
import numpy as np
from .metrics import sharpe_ratio, sortino_ratio

def backtest_buy_and_hold(price_panel, risk_free_rate=0.0):
    """
    price_panel has columns: date, ticker, log_ret_1d (or some daily return).
    Equal weight in all tickers at start, never rebalance (weights drift).
    """

    df = price_panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # pivot returns to matrix [dates, tickers]
    ret_df = df.pivot(index="date", columns="ticker", values="log_ret_1d").sort_index()

    # Fix universe at start date (no entry for tickers without a start return).
    start_date = ret_df.index[0]
    start_mask = ret_df.loc[start_date].notna()
    ret_df = ret_df.loc[:, start_mask]

    # Fill remaining missing values as 0.0 (treat gaps as flat returns)
    ret_df = ret_df.fillna(0.0)

    # The input column is log returns (log_ret_1d). Convert to simple returns
    # before compounding: simple_ret = exp(log_ret) - 1
    simple_ret = np.expm1(ret_df.values)  # shape [T, N]

    # Buy-and-hold: equal dollars at t0, no rebalance.
    # Portfolio equity is mean of cumulative wealth paths.
    cum_wealth = (1.0 + simple_ret).cumprod(axis=0)  # shape [T, N]
    equity = cum_wealth.mean(axis=1)
    port_ret = pd.Series(equity, index=ret_df.index).pct_change().fillna(0.0).values

    eq_series = pd.Series(equity, index=ret_df.index)

    stats = {
        "final_value": float(eq_series.iloc[-1]),
        "sharpe": sharpe_ratio(port_ret, risk_free_rate),
        "sortino": sortino_ratio(port_ret, risk_free_rate),
    }

    return eq_series, port_ret, stats


def backtest_long_only(pred_df, top_k=20, transaction_cost_bps=5, risk_free_rate=0.0, rebalance_freq=5):
    """
    pred_df columns: date, ticker, pred, realized_ret
    Long only top_k portfolio, equal weight, rebalancing every `rebalance_freq` days (default 5).
    Positions are held between rebalances; transaction costs apply on rebalance days.
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

    for i, d in enumerate(dates):
        day_df = df[df["date"] == d]
        if day_df.empty:
            continue

        # realized_ret in our pipeline is log returns (log_ret_1d). Convert to
        # simple returns before compounding: simple = exp(log) - 1
        raw_ret = day_df.set_index("ticker")["realized_ret"].to_dict()
        todays_ret = {}
        for k, v in raw_ret.items():
            try:
                if pd.isna(v):
                    todays_ret[k] = 0.0
                else:
                    todays_ret[k] = float(np.expm1(v))
            except Exception:
                # fallback: if value not numeric, treat as zero
                try:
                    todays_ret[k] = float(v)
                except Exception:
                    todays_ret[k] = 0.0
        transaction_cost = 0.0

        # Rebalance on schedule; otherwise hold previous weights
        if (i % rebalance_freq) == 0 or not prev_weights:
            day_df = day_df.sort_values("pred", ascending=False)
            long = day_df.head(top_k)

            w_long = 1.0 / max(len(long), 1)
            new_weights = {t: w_long for t in long["ticker"]}

            turnover = 0.0
            tickers = set(new_weights.keys()) | set(prev_weights.keys())
            for t in tickers:
                old_w = prev_weights.get(t, 0.0)
                new_w = new_weights.get(t, 0.0)
                turnover += abs(new_w - old_w)
            daily_turnover.append(turnover)

            transaction_cost = turnover * (transaction_cost_bps / 10000.0)
            prev_weights = new_weights
        else:
            # hold weights, no turnover
            if daily_turnover:
                daily_turnover.append(0.0)
            else:
                daily_turnover.append(0.0)

        r = sum(prev_weights.get(t, 0.0) * todays_ret.get(t, 0.0) for t in prev_weights)
        r_net = r - transaction_cost

        equity *= (1 + r_net)
        equity_curve.append((d, equity))
        daily_returns.append(r_net)

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
        # convert log realized returns to simple returns for portfolio math
        raw_ret = day_df.set_index("ticker")["realized_ret"].to_dict()
        todays_ret = {}
        for k, v in raw_ret.items():
            try:
                if pd.isna(v):
                    todays_ret[k] = 0.0
                else:
                    todays_ret[k] = float(np.expm1(v))
            except Exception:
                try:
                    todays_ret[k] = float(v)
                except Exception:
                    todays_ret[k] = 0.0

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
