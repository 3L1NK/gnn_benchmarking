import yfinance as yf
import pandas as pd
from pathlib import Path
import time

tickers = [
    "AAPL","MSFT","GOOGL","NVDA","IBM","ORCL",
    "JPM","BAC","GS","C","WFC",
    "WMT","COST","MCD","NKE","HD","AMZN",
    "META","DIS",
    "PFE","UNH","JNJ",
    "XOM","CVX",
    "CAT","RTX",
    "LIN","NEE","PLD",
    "IEF","TLT","GLD",
]


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def download_chunked(tickers, start, end, out_path):
    all_data = []

    for group in chunk_list(tickers, 5):   # download 5 tickers at a time
        print("Downloading:", group)
        try:
            data = yf.download(
                group,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            all_data.append(data)
        except Exception as e:
            print("Error for group:", group, e)
        time.sleep(2)   # pause to avoid Yahoo limits

    # merge all MultiIndex dataframes
    if not all_data:
        raise ValueError("No data downloaded")

    df = pd.concat(all_data, axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print("Saved merged raw data:", out_path)
    return df


if __name__ == "__main__":
    download_chunked(
        tickers,
        start="2000-01-01",
        end="2024-12-31",
        out_path="data/raw/raw_yfinance.parquet"
    )
