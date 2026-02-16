
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_PANEL_COLUMNS = ("date", "ticker")


def _require_columns(df: pd.DataFrame, price_file: str) -> None:
    missing = [c for c in REQUIRED_PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Price panel '{price_file}' is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _assert_unique_pairs(df: pd.DataFrame, price_file: str) -> None:
    dup_mask = df.duplicated(["date", "ticker"], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count == 0:
        return
    sample = df.loc[dup_mask, ["date", "ticker"]].head(10).to_string(index=False)
    raise ValueError(
        f"Price panel '{price_file}' contains duplicate (date,ticker) rows: {dup_count}. "
        f"Sample duplicates:\n{sample}"
    )


def load_price_panel(
    price_file: str,
    start_date: str,
    end_date: str,
    *,
    require_unique: bool = True,
) -> pd.DataFrame:
    path = Path(price_file)
    if not path.exists():
        raise FileNotFoundError(f"Price panel file does not exist: {path}")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    if end_ts < start_ts:
        raise ValueError(f"Invalid date range: end_date ({end_date}) is before start_date ({start_date}).")

    df = pd.read_parquet(path)
    _require_columns(df, str(path))

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad = int(out["date"].isna().sum())
        raise ValueError(f"Price panel '{path}' contains {bad} rows with invalid date values.")

    out["ticker"] = out["ticker"].astype(str)
    mask = (out["date"] >= start_ts) & (out["date"] <= end_ts)
    out = out.loc[mask].sort_values(["date", "ticker"]).reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No rows left in '{path}' for date range {start_date}..{end_date}.")
    if require_unique:
        _assert_unique_pairs(out, str(path))
    return out
