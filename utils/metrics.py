import numpy as np
import pandas as pd


def rank_ic(pred, target):
    df = pd.DataFrame({"pred": pred, "target": target}).dropna()
    if len(df) < 3:
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
    excess = r - risk_free_rate / 252.0
    return float(excess.mean() / (excess.std() + 1e-8))


def sortino_ratio(returns, risk_free_rate=0.0):
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    excess = r - risk_free_rate / 252.0
    downside = excess[excess < 0]
    denom = downside.std()
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(excess.mean() / denom)
