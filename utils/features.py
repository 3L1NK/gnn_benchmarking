import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(series):
    """
    Perform the Augmented Dickey Fuller test on a time series.

    Parameters
    ----------
    series : pandas.Series
        The time series to test. Missing values are removed.

    Returns
    -------
    float or None
        The p value of the ADF test. Lower values (below 0.05) indicate stationarity.
        Returns None if the series is too short.
    """
    series = series.dropna()
    if len(series) < 20:
        return None
    stat, pvalue, *_ = adfuller(series, autolag="AIC")
    return pvalue

def kpss_test(series):
    """
    Perform the KPSS test on a time series.

    Parameters
    ----------
    series : pandas.Series
        The time series to test. Missing values are removed.

    Returns
    -------
    float or None
        The p value of the KPSS test. Higher values (above 0.05) indicate stationarity.
        Returns None if the series is too short.
    """
    series = series.dropna()
    if len(series) < 20:
        return None
    stat, pvalue, *_ = kpss(series, regression="c", nlags="auto")
    return pvalue


def rsi_stationary(returns, window=14):
    """
    Compute a stationary version of the RSI indicator using returns instead of prices.

    Parameters
    ----------
    returns : pandas.Series
        Daily return values used to compute RSI.
    window : int, optional
        Rolling window size for averaging gains and losses.

    Returns
    -------
    pandas.Series
        The RSI values based on returns. This version avoids non stationary price inputs.
    """
    delta = returns
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    return rsi



def macd_stationary(returns, fast=12, slow=26, signal=9):
    """
    Compute a stationary MACD indicator using returns instead of prices.

    Parameters
    ----------
    returns : pandas.Series
        Daily returns used to build MACD.
    fast : int, optional
        Span for the fast EMA.
    slow : int, optional
        Span for the slow EMA.
    signal : int, optional
        Span for the signal line EMA.

    Returns
    -------
    tuple of pandas.Series
        macd_line : EMA fast minus EMA slow.
        signal_line : Smoothed version of the MACD line.
        hist : Difference between MACD and signal line.
    """
    ema_fast = returns.ewm(span=fast, adjust=False).mean()
    ema_slow = returns.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist



def add_technical_features(df, id_col="ticker", date_col="date", price_col="close", vol_col="volume"):
    """
    Add stationary technical features for each asset in a panel dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least ticker, date, close price and volume.
    id_col : str, optional
        Column name that identifies each asset.
    date_col : str, optional
        Column name for the date field. The data must be sortable by this column.
    price_col : str, optional
        Column name for the close price.
    vol_col : str, optional
        Column name for trade volume.

    Returns
    -------
    df : pandas.DataFrame
        The input dataframe with added stationary features.
    feature_cols : list of str
        Names of all generated feature columns. These can be used for ML models.
    """
    df = df.sort_values([id_col, date_col]).copy()

    def _per_asset(g):
        p = g[price_col]
        v = g[vol_col]

        # returns
        g["ret_1d"] = p.pct_change()
        g["ret_5d"] = p.pct_change(5)
        g["ret_20d"] = p.pct_change(20)
        g["log_ret_1d"] = np.log(p / p.shift(1))

        # volatility of returns
        g["vol_20d"] = g["log_ret_1d"].rolling(20).std()

        # momentum (already stationary)
        g["mom_10"] = p.pct_change(10)

        # stationary RSI
        g["rsi_14"] = rsi_stationary(g["ret_1d"], window=14)

        # stationary MACD on returns
        macd_line, sig, hist = macd_stationary(g["ret_1d"])
        g["macd_line"] = macd_line
        g["macd_signal"] = sig
        g["macd_hist"] = hist

        # volume z score
        g["vol_z_20"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-8)

        return g

    df = df.groupby(id_col, group_keys=False).apply(_per_asset)

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "log_ret_1d",
        "vol_20d",
        "mom_10",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "vol_z_20",
    ]

    return df, feature_cols


def test_all_features(df, feature_cols, id_col="ticker"):
    """
    Apply ADF and KPSS stationarity tests to all features and all tickers.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the computed features.
    feature_cols : list of str
        Names of feature columns to test.
    id_col : str, optional
        Column name identifying the asset.

    Returns
    -------
    pandas.DataFrame
        A table with one row per ticker and feature showing ADF and KPSS p values.
        This helps identify features that are not stationary.
    """

    results = []
    for ticker, g in df.groupby(id_col):
        for col in feature_cols:
            adf_p = adf_test(g[col])
            kpss_p = kpss_test(g[col])
            results.append([ticker, col, adf_p, kpss_p])
    return pd.DataFrame(results, columns=["ticker", "feature", "adf_p", "kpss_p"])
