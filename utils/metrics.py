import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
_STD_EPS = 1e-12


def rank_ic(pred, target):
    df = pd.DataFrame({"pred": pred, "target": target}).dropna()
    # Require some variation; Spearman warns on constant inputs.
    if len(df) < 3 or df["pred"].nunique() < 2 or df["target"].nunique() < 2:
        return np.nan
    return df["pred"].rank().corr(df["target"].rank(), method="spearman")


def hit_rate(pred, target, top_k=20):
    df = pd.DataFrame({"pred": pred, "target": target}).dropna()
    if df.empty:
        return np.nan
    df = df.sort_values("pred", ascending=False)
    top = df.head(top_k)
    return float((top["target"] > 0).mean())


def mse(pred, target):
    diff = np.array(pred) - np.array(target)
    return float(np.mean(diff ** 2))


def sharpe_ratio(returns, risk_free_rate=0.0):
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    excess = r - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / (excess.std() + 1e-8))


def sortino_ratio(returns, risk_free_rate=0.0):
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    excess = r - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    denom = downside.std()
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(excess.mean() / denom)


def sharpe_ratio_annualized(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS_PER_YEAR):
    daily = sharpe_ratio(returns, risk_free_rate=risk_free_rate)
    return float(daily * np.sqrt(periods_per_year))


def sortino_ratio_annualized(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS_PER_YEAR):
    daily = sortino_ratio(returns, risk_free_rate=risk_free_rate)
    return float(daily * np.sqrt(periods_per_year))


def annualized_sharpe_from_returns(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS_PER_YEAR):
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    if not np.isfinite(r).all():
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    denom = float(excess.std())
    if not np.isfinite(denom) or denom <= _STD_EPS:
        return float("nan")
    return float((float(excess.mean()) / denom) * np.sqrt(periods_per_year))


def annualized_return(equity_curve, periods_per_year=252, n_periods=None):
    eq = pd.Series(equity_curve).dropna()
    if len(eq) < 2:
        return 0.0
    if n_periods is None:
        n_periods = max(len(eq) - 1, 0)
    if n_periods <= 0:
        return 0.0
    eq_start = float(eq.iloc[0])
    eq_end = float(eq.iloc[-1])
    if eq_start <= 0 or eq_end <= 0:
        return 0.0
    return float((eq_end / eq_start) ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(returns, periods_per_year=252):
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    return float(r.std() * np.sqrt(periods_per_year))


def max_drawdown(equity_curve):
    eq = pd.Series(equity_curve).dropna()
    if eq.empty:
        return 0.0
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min())


def validate_portfolio_series(equity_curve, daily_returns):
    eq = pd.Series(equity_curve).dropna()
    r = pd.Series(daily_returns).dropna()

    if eq.empty:
        raise ValueError("Portfolio validation failed: equity curve is empty.")
    if not np.isfinite(eq).all():
        raise ValueError("Portfolio validation failed: equity curve has non-finite values.")
    if isinstance(eq.index, pd.DatetimeIndex):
        if not eq.index.is_monotonic_increasing:
            raise ValueError("Portfolio validation failed: equity curve index is not monotonic increasing.")
        if eq.index.has_duplicates:
            raise ValueError("Portfolio validation failed: equity curve index contains duplicate dates.")

    if r.empty:
        raise ValueError("Portfolio validation failed: daily return series is empty.")
    if not np.isfinite(r).all():
        raise ValueError("Portfolio validation failed: daily return series has non-finite values.")
    ret_std = float(r.std())
    if not np.isfinite(ret_std) or ret_std <= _STD_EPS or int(r.nunique()) < 2:
        raise ValueError("Portfolio validation failed: daily return series is constant or near-constant.")

    if len(eq) > 1 and len(r) not in {len(eq), len(eq) - 1}:
        raise ValueError(
            f"Portfolio validation failed: returns length {len(r)} misaligned with equity length {len(eq)}."
        )
    return eq, r


def portfolio_metrics(equity_curve, daily_returns, risk_free_rate=0.0, periods_per_year=252):
    eq, r = validate_portfolio_series(equity_curve, daily_returns)
    n_periods = len(r) if len(r) > 0 else max(len(eq) - 1, 0)
    sharpe_daily = sharpe_ratio(r, risk_free_rate)
    sortino_daily = sortino_ratio(r, risk_free_rate)
    sharpe_annualized = annualized_sharpe_from_returns(r, risk_free_rate, periods_per_year=periods_per_year)
    if not np.isfinite(sharpe_annualized):
        sharpe_annualized = float(sharpe_daily * np.sqrt(periods_per_year))
    return {
        "final_value": float(eq.iloc[-1]) if not eq.empty else 0.0,
        "cumulative_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0,
        "annualized_return": annualized_return(eq, periods_per_year=periods_per_year, n_periods=n_periods),
        "annualized_volatility": annualized_volatility(r, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe_daily,
        "sortino": sortino_daily,
        "sharpe_annualized": float(sharpe_annualized),
        "sortino_annualized": float(sortino_daily * np.sqrt(periods_per_year)),
    }
