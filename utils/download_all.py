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
    """
    Purpose: Split a long list into smaller sublists.

    Parameters:
    • lst the list to split
    • size number of items in each sublist

    Return: a generator. Each iteration yields a chunk of the original list.

    Example:
    chunk_list(["A","B","C","D"], 2) yields ["A","B"] then ["C","D"].
    """
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def download_chunked(tickers, start, end, out_path):
    """Purpose: download data in chunks, merge them, save to disk.

    Parameters:
    • tickers: List of ticker symbols to download.
    • start: Start date for historical data. Example: "2000-01-01".
    • end: End date for historical data. Example: "2024-12-31".
    • out_path: Path for the final Parquet file. Example: "data/raw/raw_yfinance.parquet".
    
    Return: Merged DataFrame of all downloaded data.
    """
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

# download all tickers and save to Parquet for the last 25 years
if __name__ == "__main__":
    download_chunked(
        tickers,
        start="2000-01-01",
        end="2024-12-31",
        out_path="data/raw/raw_yfinance.parquet"
    )
