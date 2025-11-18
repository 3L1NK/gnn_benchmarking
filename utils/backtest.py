import pandas as pd
import numpy as np
from .metrics import sharpe_ratio, sortino_ratio


def backtest_long_short(pred_df, top_k=20, transaction_cost_bps=5, risk_free_rate=0.0):
    """
    pred_df columns: date, ticker, pred, realized_ret
    """
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "pred"], ascending=[True, False])

    portfolio_values = []
    portfolio_ret = []

    dates = df["date"].drop_duplicates().sort_values().tolist()
    value = 1.0

    for d in dates:
        day = df[df["date"] == d]
        if day.empty:
            continue

        day = day.sort_values("pred", ascending=False)

        long = day.head(top_k)
        short = day.tail(top_k)

        w_long = 0.5 / max(len(long), 1)
        w_short = -0.5 / max(len(short), 1)

        gross_turnover = 1.0  # approximate
        tc = transaction_cost_bps / 10000.0 * gross_turnover

        ret_long = (long["realized_ret"] * w_long).sum()
        ret_short = (short["realized_ret"] * w_short).sum()
        day_ret = ret_long + ret_short - tc

        value *= (1.0 + day_ret)
        portfolio_values.append(value)
        portfolio_ret.append(day_ret)

    stats = {
        "final_value": value,
        "sharpe": sharpe_ratio(portfolio_ret, risk_free_rate),
        "sortino": sortino_ratio(portfolio_ret, risk_free_rate),
    }
    series = pd.Series(portfolio_values, index=dates[: len(portfolio_values)])
    return series, portfolio_ret, stats
