import numpy as np
import pandas as pd


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - 100 / (1 + rs)


def macd(price, fast=12, slow=26, signal=9):
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_technical_features(df, id_col="ticker", date_col="date", price_col="close", vol_col="volume"):
    df = df.sort_values([id_col, date_col]).copy()

    feature_cols = []

    def _per_asset(g):
        p = g[price_col]
        v = g[vol_col]

        g["ret_1d"] = p.pct_change()
        g["ret_5d"] = p.pct_change(5)
        g["ret_20d"] = p.pct_change(20)
        g["log_ret_1d"] = np.log(p / p.shift(1))
        g["vol_20d"] = g["log_ret_1d"].rolling(20).std()

        g["ema_10"] = p.ewm(span=10, adjust=False).mean()
        g["ema_20"] = p.ewm(span=20, adjust=False).mean()
        g["mom_10"] = p / p.shift(10) - 1.0

        # RSI and MACD
        g["rsi_14"] = rsi(p, window=14)
        macd_line, sig, hist = macd(p)
        g["macd_line"] = macd_line
        g["macd_signal"] = sig
        g["macd_hist"] = hist

        # volume features
        if v.notna().any():
            g["vol_z_20"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-8)
        else:
            g["vol_z_20"] = np.nan

        return g

    df = df.groupby(id_col, group_keys=False).apply(_per_asset)

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "log_ret_1d",
        "vol_20d",
        "ema_10",
        "ema_20",
        "mom_10",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "vol_z_20",
    ]

    return df, feature_cols
