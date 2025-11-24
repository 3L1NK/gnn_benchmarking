import pandas as pd
import numpy as np
from .metrics import sharpe_ratio, sortino_ratio


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
