from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

from utils.cache import cache_key, cache_path, cache_load, cache_save, cache_dir
from utils.data_loading import load_price_panel
from utils.metrics import portfolio_metrics


BASELINE_VERSION = "price_bh_v2_eqw_v2"
BASELINE_ROOT = Path("results/runs/baselines")
GLOBAL_BASELINE_CSV = BASELINE_ROOT / "buy_and_hold_global.csv"
GLOBAL_EQUAL_WEIGHT_CSV = BASELINE_ROOT / "equal_weight_global.csv"


def clear_buy_and_hold_cache():
    """
    Remove cached global buy-and-hold files so the baseline is recomputed.
    """
    for p in cache_dir().glob("global_buy_and_hold_*"):
        p.unlink(missing_ok=True)
    for p in cache_dir().glob("global_equal_weight_*"):
        p.unlink(missing_ok=True)


def _select_price_column(df: pd.DataFrame) -> str:
    candidates = ["adj_close", "close", "price", "last", "px_last"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("No price column found; expected one of: adj_close, close, price, last, px_last.")


def _clean_price_panel(
    df: pd.DataFrame,
    price_col: str,
    start_date=None,
    end_date=None,
    universe=None,
    max_ffill_gap: int = 5,
) -> pd.DataFrame:
    """
    Return a cleaned price matrix with index=date and columns=ticker.
    - Drop tickers without a price on the global start date.
    - Forward-fill prices with a limit; drop tickers with remaining gaps.
    - Align calendar by dropping any residual NaN rows.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    if start_date is not None:
        data = data.loc[data["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        data = data.loc[data["date"] <= pd.to_datetime(end_date)]
    if universe is not None:
        data = data.loc[data["ticker"].isin(universe)]

    data = data.dropna(subset=[price_col])
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data = data.dropna(subset=[price_col])

    if data.empty:
        raise ValueError("No price data available after filtering.")

    # Use pivot_table to make duplicate (date, ticker) deterministic.
    pivot = (
        data.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last")
        .sort_index()
    )

    # Fix universe at first available date
    global_start = pivot.index[0]
    start_mask = pivot.loc[global_start].notna()
    pivot = pivot.loc[:, start_mask]

    # Forward fill prices per ticker with a max gap, then drop tickers with remaining gaps.
    pivot = pivot.ffill(limit=max_ffill_gap)
    pivot = pivot.dropna(axis=1, how="any")

    # Align calendar by intersection across remaining tickers.
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.empty or pivot.shape[1] == 0:
        raise ValueError("Price panel empty after cleaning; check data coverage and filters.")

    return pivot


def compute_buy_and_hold_fixed_shares(
    prices: pd.DataFrame,
    price_col: str,
    start_date=None,
    end_date=None,
    universe=None,
    initial_value: float = 1.0,
    max_ffill_gap: int = 5,
    risk_free_rate: float = 0.0,
):
    """
    Compute fixed-shares buy-and-hold from prices:
    - Equal-weight at t0 across fixed universe
    - Shares stay constant
    - Portfolio value = sum(shares_i * price_i,t)
    """
    price_matrix = _clean_price_panel(
        prices,
        price_col=price_col,
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        max_ffill_gap=max_ffill_gap,
    )

    prices0 = price_matrix.iloc[0]
    n_assets = len(prices0)
    shares = (initial_value / n_assets) / prices0

    equity = (price_matrix * shares).sum(axis=1)
    eq_series = pd.Series(equity.values, index=price_matrix.index, name="buy_and_hold")

    daily_ret = eq_series.pct_change().dropna().values
    stats = portfolio_metrics(eq_series, daily_ret, risk_free_rate)

    return eq_series, daily_ret, stats


def slice_and_rebase_equity_curve(eq_series: pd.Series, start_date=None, end_date=None, rebase=True):
    s = eq_series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if start_date is not None:
        s = s.loc[s.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        s = s.loc[s.index <= pd.to_datetime(end_date)]
    if rebase and not s.empty:
        s = s / s.iloc[0]
    return s


def _window_key(start_date, end_date) -> str:
    s = pd.to_datetime(start_date).date() if start_date is not None else "na"
    e = pd.to_datetime(end_date).date() if end_date is not None else "na"
    return f"{s}__{e}"


def _window_baseline_paths(start_date, end_date, *, rebalance_freq: Optional[int] = None):
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    d = BASELINE_ROOT / f"window_{_window_key(start_date, end_date)}"
    d.mkdir(parents=True, exist_ok=True)
    if rebalance_freq is None:
        curve = d / "buy_and_hold.csv"
        stats = d / "buy_and_hold_stats.json"
    else:
        curve = d / f"equal_weight_reb{int(rebalance_freq)}.csv"
        stats = d / f"equal_weight_reb{int(rebalance_freq)}_stats.json"
    return curve, stats


def _load_cached_window_baseline(curve_path: Path, stats_path: Path):
    if not curve_path.exists() or not stats_path.exists():
        return None
    try:
        curve_df = pd.read_csv(curve_path)
        if "date" not in curve_df.columns:
            return None
        value_cols = [c for c in curve_df.columns if c != "date"]
        if not value_cols:
            return None
        col = value_cols[0]
        series = pd.Series(
            pd.to_numeric(curve_df[col], errors="coerce").values,
            index=pd.to_datetime(curve_df["date"], errors="coerce"),
            name=col,
        ).dropna()
        stats = pd.read_json(stats_path, typ="series").to_dict()
        daily_ret = series.pct_change().dropna().values
        return series, daily_ret, stats
    except Exception:
        return None


def _save_cached_window_baseline(eq_series: pd.Series, stats: dict, curve_path: Path, stats_path: Path, value_col: str):
    write_baseline_curve_csv(eq_series, curve_path, value_column=value_col)
    pd.Series(stats).to_json(stats_path)


def write_buy_and_hold_csv(eq_series: pd.Series, out_path: Path):
    write_baseline_curve_csv(eq_series, out_path, value_column="buy_and_hold")


def write_baseline_curve_csv(eq_series: pd.Series, out_path: Path, value_column: str):
    s = eq_series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    df = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), value_column: s.values})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def get_buy_and_hold_for_window(config, start_date, end_date, rebuild=False, eq_full=None):
    risk_free = config["evaluation"].get("risk_free_rate", 0.0)
    curve_path, stats_path = _window_baseline_paths(start_date, end_date, rebalance_freq=None)
    if not rebuild:
        cached = _load_cached_window_baseline(curve_path, stats_path)
        if cached is not None:
            return cached
    if eq_full is None:
        eq_full, _, _ = get_global_buy_and_hold(config, rebuild=rebuild)
    eq_window = slice_and_rebase_equity_curve(eq_full, start_date, end_date, rebase=True)
    daily_ret = eq_window.pct_change().dropna().values
    stats = portfolio_metrics(eq_window, daily_ret, risk_free)
    _save_cached_window_baseline(eq_window, stats, curve_path, stats_path, value_col="buy_and_hold")
    return eq_window, daily_ret, stats


def compute_equal_weight_rebalanced(
    prices: pd.DataFrame,
    price_col: str,
    start_date=None,
    end_date=None,
    universe=None,
    initial_value: float = 1.0,
    max_ffill_gap: int = 5,
    risk_free_rate: float = 0.0,
    rebalance_freq: int = 1,
    return_diagnostics: bool = False,
):
    if rebalance_freq < 1:
        raise ValueError(f"rebalance_freq must be >= 1, got {rebalance_freq}")

    price_matrix = _clean_price_panel(
        prices,
        price_col=price_col,
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        max_ffill_gap=max_ffill_gap,
    )
    ret = price_matrix.pct_change().fillna(0.0)
    n_assets = ret.shape[1]
    if n_assets == 0:
        raise ValueError("No assets available for equal-weight baseline.")

    eq = float(initial_value)
    eq_vals = []
    daily_ret = []
    turnover_series = []
    prev_close_w = np.zeros(n_assets, dtype=float)
    rebalance_dates = []
    close_weights = []
    for i, (_, row) in enumerate(ret.iterrows()):
        row_ret = row.values.astype(float, copy=False)

        rebalance_now = (i % rebalance_freq == 0) or (i == 0)
        if rebalance_now:
            target_w = np.full(n_assets, 1.0 / n_assets, dtype=float)
            turnover = float(np.abs(target_w - prev_close_w).sum())
            open_w = target_w
            rebalance_dates.append(ret.index[i])
        else:
            turnover = 0.0
            open_w = prev_close_w

        port_ret = float(np.dot(open_w, row_ret))
        eq *= (1.0 + port_ret)
        eq_vals.append(eq)
        daily_ret.append(port_ret)
        turnover_series.append(turnover)

        gross = open_w * (1.0 + row_ret)
        denom = float(gross.sum())
        if denom <= 0.0 or not np.isfinite(denom):
            prev_close_w = np.full(n_assets, 1.0 / n_assets, dtype=float)
        else:
            prev_close_w = gross / denom
        close_weights.append(prev_close_w.copy())

    eq_series = pd.Series(eq_vals, index=ret.index, name="equal_weight")
    stats = portfolio_metrics(eq_series, daily_ret, risk_free_rate)
    stats["avg_turnover"] = float(np.mean(turnover_series)) if turnover_series else 0.0
    if return_diagnostics:
        diagnostics = {
            "rebalance_dates": pd.DatetimeIndex(rebalance_dates),
            "rebalance_count": int(len(rebalance_dates)),
            "turnover_series": pd.Series(turnover_series, index=ret.index, name="turnover"),
            "turnover_mean": float(np.mean(turnover_series)) if turnover_series else 0.0,
            "turnover_median": float(np.median(turnover_series)) if turnover_series else 0.0,
            "turnover_max": float(np.max(turnover_series)) if turnover_series else 0.0,
            "close_weights": pd.DataFrame(close_weights, index=ret.index, columns=ret.columns),
        }
        return eq_series, np.asarray(daily_ret, dtype=float), stats, diagnostics
    return eq_series, np.asarray(daily_ret, dtype=float), stats


def get_global_equal_weight(config, rebalance_freq=1, rebuild=False, align_start_date=None):
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    risk_free = config["evaluation"].get("risk_free_rate", 0.0)
    universe_file = config["data"].get("universe_file")
    max_ffill_gap = config.get("evaluation", {}).get("bh_max_ffill_gap", 5)

    key = cache_key(
        {
            "baseline": "global_equal_weight",
            "price_file": price_file,
            "start": start,
            "end": end,
            "risk_free": risk_free,
            "baseline_version": BASELINE_VERSION,
            "max_ffill_gap": max_ffill_gap,
            "rebalance_freq": int(rebalance_freq),
        },
        dataset_version=BASELINE_VERSION,
        extra_files=[price_file] + ([universe_file] if universe_file else []),
    )
    path = cache_path("global_equal_weight", key)
    if not rebuild:
        cached = cache_load(path)
        if cached is not None:
            print(f"[baseline] loaded global equal-weight from cache {path}")
            eq = cached["eq"]
            ret = np.asarray(cached["ret"], dtype=float)
            ret_series = pd.Series(ret, index=eq.index, name="eqw_ret")
            stats = cached["stats"]
            _maybe_write_global_equal_weight_csv(eq)
            return _align_baseline(eq, ret_series, stats, align_start_date, risk_free)

    df = load_price_panel(price_file, start, end)
    price_col = _select_price_column(df)
    universe = None
    if universe_file:
        u = pd.read_csv(universe_file)
        if "ticker" in u.columns:
            universe = u["ticker"].dropna().unique().tolist()

    eq, ret, stats = compute_equal_weight_rebalanced(
        df,
        price_col=price_col,
        start_date=start,
        end_date=end,
        universe=universe,
        initial_value=1.0,
        max_ffill_gap=max_ffill_gap,
        risk_free_rate=risk_free,
        rebalance_freq=int(rebalance_freq),
    )
    stats = {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in stats.items()}
    cache_save(path, {"eq": eq, "ret": ret, "stats": stats})
    print(f"[baseline] saved global equal-weight to cache {path}")
    _maybe_write_global_equal_weight_csv(eq)
    return _align_baseline(eq, pd.Series(ret, index=eq.index, name="eqw_ret"), stats, align_start_date, risk_free)


def get_equal_weight_for_window(config, start_date, end_date, rebalance_freq=1, rebuild=False, eq_full=None):
    risk_free = config["evaluation"].get("risk_free_rate", 0.0)
    curve_path, stats_path = _window_baseline_paths(start_date, end_date, rebalance_freq=int(rebalance_freq))
    if not rebuild:
        cached = _load_cached_window_baseline(curve_path, stats_path)
        if cached is not None:
            return cached
    if eq_full is None:
        eq_full, _, _ = get_global_equal_weight(
            config,
            rebalance_freq=rebalance_freq,
            rebuild=rebuild,
        )
    eq_window = slice_and_rebase_equity_curve(eq_full, start_date, end_date, rebase=True)
    daily_ret = eq_window.pct_change().dropna().values
    stats = portfolio_metrics(eq_window, daily_ret, risk_free)
    _save_cached_window_baseline(eq_window, stats, curve_path, stats_path, value_col="equal_weight")
    return eq_window, daily_ret, stats


def get_global_buy_and_hold(config, rebuild=False, align_start_date=None):
    """
    Compute or load a single global buy-and-hold baseline from raw price data.
    Canonical definition: fixed-shares buy-and-hold from prices (not returns).
    """
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    risk_free = config["evaluation"].get("risk_free_rate", 0.0)
    universe_file = config["data"].get("universe_file")
    max_ffill_gap = config.get("evaluation", {}).get("bh_max_ffill_gap", 5)

    key = cache_key(
        {
            "baseline": "global_buy_and_hold",
            "price_file": price_file,
            "start": start,
            "end": end,
            "risk_free": risk_free,
            "baseline_version": BASELINE_VERSION,
            "max_ffill_gap": max_ffill_gap,
        },
        dataset_version=BASELINE_VERSION,
        extra_files=[price_file] + ([universe_file] if universe_file else []),
    )
    path = cache_path("global_buy_and_hold", key)

    if not rebuild:
        cached = cache_load(path)
        if cached is not None:
            print(f"[baseline] loaded global buy-and-hold from cache {path}")
            eq_bh = cached["eq"]
            ret_bh = np.asarray(cached["ret"], dtype=float)
            ret_bh_series = pd.Series(ret_bh, index=eq_bh.index[1:], name="bh_ret")
            stats_bh = cached["stats"]
            _maybe_write_global_csv(eq_bh)
            return _align_baseline(eq_bh, ret_bh_series, stats_bh, align_start_date, risk_free)

    df = load_price_panel(price_file, start, end)
    price_col = _select_price_column(df)

    universe = None
    if universe_file:
        u = pd.read_csv(universe_file)
        if "ticker" in u.columns:
            universe = u["ticker"].dropna().unique().tolist()

    eq_bh, ret_bh, stats_bh = compute_buy_and_hold_fixed_shares(
        df,
        price_col=price_col,
        start_date=start,
        end_date=end,
        universe=universe,
        initial_value=1.0,
        max_ffill_gap=max_ffill_gap,
        risk_free_rate=risk_free,
    )

    stats_bh = {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in stats_bh.items()}
    cache_save(path, {"eq": eq_bh, "ret": ret_bh, "stats": stats_bh})
    print(f"[baseline] saved global buy-and-hold to cache {path}")
    ret_bh_series = pd.Series(ret_bh, index=eq_bh.index[1:], name="bh_ret")
    _maybe_write_global_csv(eq_bh)
    return _align_baseline(eq_bh, ret_bh_series, stats_bh, align_start_date, risk_free)


def _align_baseline(eq_bh, ret_bh_series, stats_bh, align_start_date, risk_free):
    """
    Optionally align baseline to a start date and renormalize equity to 1.0.
    """
    if align_start_date is None:
        return eq_bh, ret_bh_series.values, stats_bh

    eq_slice = slice_and_rebase_equity_curve(eq_bh, align_start_date, None, rebase=True)
    if eq_slice.empty:
        return eq_bh, ret_bh_series.values, stats_bh
    ret_slice = eq_slice.pct_change().dropna().values
    stats_slice = portfolio_metrics(eq_slice, ret_slice, risk_free)
    return eq_slice, ret_slice, stats_slice


def _maybe_write_global_csv(eq_bh: pd.Series):
    try:
        write_buy_and_hold_csv(eq_bh, GLOBAL_BASELINE_CSV)
    except Exception as exc:
        print(f"[baseline] warning: failed to write global baseline CSV: {exc}")


def _maybe_write_global_equal_weight_csv(eq_series: pd.Series):
    try:
        write_baseline_curve_csv(eq_series, GLOBAL_EQUAL_WEIGHT_CSV, value_column="equal_weight")
    except Exception as exc:
        print(f"[baseline] warning: failed to write global equal-weight CSV: {exc}")
