from pathlib import Path
import pandas as pd

from utils.cache import cache_key, cache_path, cache_load, cache_save
from utils.data_loading import load_price_panel
from utils.backtest import backtest_buy_and_hold


def get_global_buy_and_hold(config, rebuild=False, align_start_date=None):
    """
    Compute or load a single global buy-and-hold baseline from raw price data.
    This is a model-independent sanity check: equal weight, no rebalance,
    computed before any model-specific preprocessing.
    """
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    risk_free = config["evaluation"].get("risk_free_rate", 0.0)
    universe_file = config["data"].get("universe_file")

    key = cache_key(
        {
            "baseline": "global_buy_and_hold",
            "price_file": price_file,
            "start": start,
            "end": end,
            "risk_free": risk_free,
        },
        dataset_version="global_bh",
        extra_files=[price_file] + ([universe_file] if universe_file else []),
    )
    path = cache_path("global_buy_and_hold", key)

    if not rebuild:
        cached = cache_load(path)
        if cached is not None:
            print(f"[baseline] loaded global buy-and-hold from cache {path}")
            eq_bh = cached["eq"]
            ret_bh = pd.Series(cached["ret"], index=eq_bh.index)
            stats_bh = cached["stats"]
            return _align_baseline(eq_bh, ret_bh, stats_bh, align_start_date, risk_free)

    df = load_price_panel(price_file, start, end)
    # only raw columns needed: date, ticker, log_ret_1d
    price_panel = df[["date", "ticker", "log_ret_1d"]].dropna(subset=["log_ret_1d"])
    eq_bh, ret_bh, stats_bh = backtest_buy_and_hold(
        price_panel,
        risk_free_rate=risk_free,
    )

    cache_save(path, {"eq": eq_bh, "ret": ret_bh, "stats": stats_bh})
    print(f"[baseline] saved global buy-and-hold to cache {path}")
    ret_bh_series = pd.Series(ret_bh, index=eq_bh.index)
    return _align_baseline(eq_bh, ret_bh_series, stats_bh, align_start_date, risk_free)


def _align_baseline(eq_bh, ret_bh_series, stats_bh, align_start_date, risk_free):
    """
    Optionally align baseline to a start date and renormalize equity to 1.0.
    """
    if align_start_date is None:
        return eq_bh, ret_bh_series.values, stats_bh

    mask = eq_bh.index >= pd.to_datetime(align_start_date)
    eq_slice = eq_bh.loc[mask]
    ret_slice = ret_bh_series.loc[mask]
    if eq_slice.empty:
        return eq_bh, ret_bh_series.values, stats_bh
    eq_norm = eq_slice / eq_slice.iloc[0]
    from utils.metrics import sharpe_ratio, sortino_ratio
    stats_slice = {
        "final_value": float(eq_norm.iloc[-1]),
        "sharpe": sharpe_ratio(ret_slice, risk_free),
        "sortino": sortino_ratio(ret_slice, risk_free),
    }
    return eq_norm, ret_slice.values, stats_slice
