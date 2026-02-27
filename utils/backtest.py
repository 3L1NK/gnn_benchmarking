import pandas as pd
import numpy as np
from .metrics import portfolio_metrics, validate_portfolio_series

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
    port_ret = pd.Series(equity, index=ret_df.index).pct_change().dropna().values

    eq_series = pd.Series(equity, index=ret_df.index)

    validate_portfolio_series(eq_series, port_ret)
    stats = portfolio_metrics(eq_series, port_ret, risk_free_rate)

    return eq_series, port_ret, stats


def backtest_long_only(pred_df, top_k=20, transaction_cost_bps=0, risk_free_rate=0.0, rebalance_freq=1):
    """
    pred_df columns: date, ticker, pred, realized_ret
    Long only top_k portfolio, equal weight, rebalancing every `rebalance_freq` days (default 1).
    Positions are held between rebalances; transaction costs apply on rebalance days.
    """
    if rebalance_freq < 1:
        raise ValueError(f"rebalance_freq must be >= 1, got {rebalance_freq}")

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

    validate_portfolio_series(eq_series, daily_returns)
    stats = portfolio_metrics(eq_series, daily_returns, risk_free_rate)
    stats["avg_turnover"] = float(np.mean(daily_turnover)) if daily_turnover else 0.0

    return eq_series, daily_returns, stats


def backtest_long_short(
    pred_df,
    top_k=20,
    bottom_k=None,
    transaction_cost_bps=0,
    risk_free_rate=0.0,
    rebalance_freq=1,
    long_leg_gross=0.5,
    short_leg_gross=0.5,
):
    """
    pred_df columns: date, ticker, pred, realized_ret

    Implements:
        - equal weighted long-short portfolio
        - configurable rebalance frequency
        - turnover-based transaction costs
        - net returns
    """
    if rebalance_freq < 1:
        raise ValueError(f"rebalance_freq must be >= 1, got {rebalance_freq}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if bottom_k is None:
        bottom_k = top_k
    if bottom_k < 1:
        raise ValueError(f"bottom_k must be >= 1, got {bottom_k}")
    if long_leg_gross < 0 or short_leg_gross < 0:
        raise ValueError("long_leg_gross and short_leg_gross must be non-negative.")

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "pred"], ascending=[True, False])

    dates = df["date"].drop_duplicates().sort_values().tolist()

    equity = 1.0
    equity_curve = []
    daily_returns = []

    prev_close_weights = {}
    turnover_series = []

    for i, d in enumerate(dates):
        day_df = df[df["date"] == d]
        if day_df.empty:
            continue

        transaction_cost = 0.0
        rebalance_now = (i % rebalance_freq) == 0 or not prev_close_weights
        if rebalance_now:
            day_df = day_df.sort_values("pred", ascending=False)
            long = day_df.head(top_k)
            short = day_df.tail(bottom_k)

            # equal weighting then scale legs to fixed gross exposures.
            w_long = float(long_leg_gross) / max(len(long), 1)
            w_short = -float(short_leg_gross) / max(len(short), 1)

            open_weights = {}
            for t in long["ticker"]:
                open_weights[t] = open_weights.get(t, 0.0) + w_long
            for t in short["ticker"]:
                open_weights[t] = open_weights.get(t, 0.0) + w_short
            long_sum = float(sum(w for w in open_weights.values() if w > 0.0))
            short_sum = float(sum(w for w in open_weights.values() if w < 0.0))
            if not np.isclose(long_sum, float(long_leg_gross), rtol=1e-12, atol=1e-12):
                raise ValueError(
                    f"Long-leg neutrality check failed: long_sum={long_sum:.12f}, expected={float(long_leg_gross):.12f}"
                )
            if not np.isclose(short_sum, -float(short_leg_gross), rtol=1e-12, atol=1e-12):
                raise ValueError(
                    f"Short-leg neutrality check failed: short_sum={short_sum:.12f}, expected={-float(short_leg_gross):.12f}"
                )

            turnover = 0.0
            tickers = set(open_weights.keys()) | set(prev_close_weights.keys())
            for t in tickers:
                old_w = float(prev_close_weights.get(t, 0.0))
                new_w = float(open_weights.get(t, 0.0))
                turnover += abs(new_w - old_w)
            transaction_cost = float(turnover * (transaction_cost_bps / 10000.0))
            turnover_series.append(float(turnover))
        else:
            open_weights = prev_close_weights
            turnover_series.append(0.0)

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
        for t, w in open_weights.items():
            r += w * todays_ret.get(t, 0.0)

        # net return after cost
        r_net = r - transaction_cost

        equity *= (1 + r_net)
        equity_curve.append((d, equity))
        daily_returns.append(r_net)

        # Drift weights to close-of-day weights for turnover computation.
        gross_weights = {t: float(w * (1.0 + todays_ret.get(t, 0.0))) for t, w in open_weights.items()}
        gross_exposure = sum(abs(v) for v in gross_weights.values())
        if gross_exposure <= 0.0 or not np.isfinite(gross_exposure):
            prev_close_weights = {}
        else:
            prev_close_weights = {t: float(v / gross_exposure) for t, v in gross_weights.items()}

    eq_series = pd.Series(
        [v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
    )

    validate_portfolio_series(eq_series, daily_returns)
    stats = portfolio_metrics(eq_series, daily_returns, risk_free_rate)
    stats["avg_turnover"] = float(np.mean(turnover_series)) if turnover_series else 0.0

    return eq_series, daily_returns, stats
