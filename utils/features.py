import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series):
    """
    Run the Augmented Dickey Fuller stationarity test.

    Parameters
    ----------
    series : pandas.Series
        Time series to test. Missing values are removed.

    Returns
    -------
    float or None
        P value of the ADF test. A value below 0.05 suggests stationarity.
        Returns None if the series is too short to test.
    """
    series = series.dropna()
    if len(series) < 20:
        return None
    stat, pvalue, *_ = adfuller(series, autolag="AIC")
    return pvalue


def kpss_test(series):
    """
    Run the KPSS stationarity test.

    Parameters
    ----------
    series : pandas.Series
        Time series to test. Missing values are removed.

    Returns
    -------
    float or None
        P value of the KPSS test. A value above 0.05 suggests stationarity.
        Returns None if the series is too short to test.
    """
    series = series.dropna()
    if len(series) < 20:
        return None
    stat, pvalue, *_ = kpss(series, regression="c", nlags="auto")
    return pvalue


def rsi_stationary(returns, window=14):
    """
    Compute RSI using returns to improve stationarity.

    Parameters
    ----------
    returns : pandas.Series
        Daily returns used to compute RSI.
    window : int, optional
        Rolling window length for averaging gains and losses.

    Returns
    -------
    pandas.Series
        Stationary RSI values between 0 and 100.
    """
    delta = returns
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def macd_stationary(returns, fast=12, slow=26, signal=9):
    """
    Compute MACD, signal line and histogram using returns instead of prices.

    Parameters
    ----------
    returns : pandas.Series
        Daily returns used as input for EMAs.
    fast : int, optional
        Span for the fast EMA.
    slow : int, optional
        Span for the slow EMA.
    signal : int, optional
        Span for the signal EMA.

    Returns
    -------
    tuple of pandas.Series
        macd_line : Difference between fast and slow EMA of returns.
        signal_line : Smoothed version of the MACD line.
        hist : MACD histogram which is macd_line minus signal_line.
    """
    ema_fast = returns.ewm(span=fast, adjust=False).mean()
    ema_slow = returns.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_technical_features(
    df,
    id_col="ticker",
    date_col="date",
    price_col="close",
    vol_col="volume"
):
    """
    Compute stationary technical features for each asset in a panel dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with at least ticker, date, close price and volume.
    id_col : str, optional
        Column name identifying each asset.
    date_col : str, optional
        Column name for the timestamp.
    price_col : str, optional
        Column name with close prices.
    vol_col : str, optional
        Column name with trading volume.

    Returns
    -------
    pandas.DataFrame
        Dataframe with added features. Rows with NaN created by rolling windows are removed.
    list of str
        List of names of the feature columns added to the dataframe.
    """
    df = df.sort_values([id_col, date_col]).copy()

    def _per_asset(g):
        p = g[price_col]
        v = g[vol_col]

        # returns
        g["ret_1d"] = p.pct_change()
        g["ret_5d"] = p.pct_change(5)
        g["ret_20d"] = p.pct_change(20)
        g["log_ret_1d"] = np.log(p / p.shift(1)).replace([np.inf, -np.inf], np.nan)

        # momentum
        g["mom_3d"] = p.pct_change(3)
        g["mom_10"] = p.pct_change(10)
        g["mom_21d"] = p.pct_change(21)

        # volatility
        g["vol_5d"] = g["log_ret_1d"].rolling(5).std()
        g["vol_20d"] = g["log_ret_1d"].rolling(20).std()
        g["vol_60d"] = g["log_ret_1d"].rolling(60).std()

        # drawdown
        rolling_max = p.rolling(20).max()
        g["drawdown_20d"] = (p / rolling_max) - 1

        # volume features
        g["volume_pct_change"] = v.pct_change()
        g["vol_z_20"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-8)
        g["vol_z_5"] = (v - v.rolling(5).mean()) / (v.rolling(5).std() + 1e-8)

        # stationary RSI
        g["rsi_14"] = rsi_stationary(g["ret_1d"], window=14)

        # stationary MACD
        macd_line, sig, hist = macd_stationary(g["ret_1d"])
        g["macd_line"] = macd_line
        g["macd_signal"] = sig
        g["macd_hist"] = hist

        return g

    df = df.groupby(id_col, group_keys=False).apply(_per_asset)

    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "ret_1d", "ret_5d", "ret_20d", "log_ret_1d",
        "mom_3d", "mom_10", "mom_21d",
        "vol_5d", "vol_20d", "vol_60d",
        "drawdown_20d",
        "volume_pct_change", "vol_z_5", "vol_z_20",
        "rsi_14", "macd_line", "macd_signal", "macd_hist"
    ]

    return df, feature_cols


def test_all_features(df, feature_cols, id_col="ticker"):
    """
    Run ADF and KPSS stationarity tests for each feature and each asset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that contains the computed feature columns.
    feature_cols : list of str
        Names of feature columns to test.
    id_col : str, optional
        Column name identifying each asset.

    Returns
    -------
    pandas.DataFrame
        Table with ticker, feature name, ADF p value and KPSS p value.
        Helps identify which features may need transformation.
    """
    results = []
    for ticker, g in df.groupby(id_col):
        for col in feature_cols:
            adf_p = adf_test(g[col])
            kpss_p = kpss_test(g[col])
            results.append([ticker, col, adf_p, kpss_p])

    return pd.DataFrame(
        results,
        columns=["ticker", "feature", "adf_p", "kpss_p"]
    )
