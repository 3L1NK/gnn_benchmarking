
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


def load_price_panel(price_file: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_parquet(price_file)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    df = df.loc[mask].sort_values(["date", "ticker"]).reset_index(drop=True)
    return df