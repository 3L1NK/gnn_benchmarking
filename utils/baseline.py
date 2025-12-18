from pathlib import Path
import pandas as pd

from utils.cache import cache_key, cache_path, cache_load, cache_save
from utils.data_loading import load_price_panel
from utils.backtest import backtest_buy_and_hold


def get_global_buy_and_hold(config, rebuild=False):
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
            return cached["eq"], cached["ret"], cached["stats"]

    df = load_price_panel(price_file, start, end)
    # only raw columns needed: date, ticker, log_ret_1d
    price_panel = df[["date", "ticker", "log_ret_1d"]].dropna(subset=["log_ret_1d"])
    eq_bh, ret_bh, stats_bh = backtest_buy_and_hold(
        price_panel,
        risk_free_rate=risk_free,
    )

    cache_save(path, {"eq": eq_bh, "ret": ret_bh, "stats": stats_bh})
    print(f"[baseline] saved global buy-and-hold to cache {path}")
    return eq_bh, ret_bh, stats_bh
