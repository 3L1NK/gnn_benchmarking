import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


def download_prices(tickers, start, end, out_path: Path):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # yfinance multiindex (Date, field, ticker) to tidy
    close = data["Close"]
    vol = data.get("Volume", pd.DataFrame())

    close = close.stack().rename("close").reset_index()
    close.columns = ["date", "ticker", "close"]

    if not vol.empty:
        vol = vol.stack().rename("volume").reset_index()
        vol.columns = ["date", "ticker", "volume"]
        df = close.merge(vol, on=["date", "ticker"], how="left")
    else:
        df = close
        df["volume"] = np.nan

    df = df.sort_values(["date", "ticker"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df


def load_price_panel(price_file: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_parquet(price_file)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    df = df.loc[mask].sort_values(["date", "ticker"]).reset_index(drop=True)
    return df
