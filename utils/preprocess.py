import pandas as pd
import numpy as np
from pathlib import Path

from utils.features import add_technical_features


def load_raw_file():
    """
    Load whichever raw file exists.
    The script checks in this order:
    1. raw_tidy.parquet (already tidy)
    2. raw_yfinance_full.parquet (Yahoo MultiIndex)
    """

    tidy_path = Path("data/raw/raw_tidy.parquet")
    full_path = Path("data/raw/raw_yfinance_full.parquet")

    if tidy_path.exists():
        print("Loading tidy raw file:", tidy_path)
        df = pd.read_parquet(tidy_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    elif full_path.exists():
        print("Loading MultiIndex raw file:", full_path)
        df_raw = pd.read_parquet(full_path)

        # Yahoo format: columns like ('Close', 'AAPL'), ('Volume','MSFT')
        # Convert to tidy
        tidy_list = []

        for col_type in ["Close", "Volume"]:
            cols = [c for c in df_raw.columns if c[0] == col_type]
            if len(cols) == 0:
                continue

            sub = df_raw[cols].copy()
            sub.columns = [c[1] for c in cols]  # ticker names
            sub = sub.stack().reset_index()
            sub.columns = ["date", "ticker", col_type.lower()]
            tidy_list.append(sub)

        # merge Close + Volume
        df = tidy_list[0]
        for add in tidy_list[1:]:
            df = df.merge(add, on=["date", "ticker"], how="left")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        return df

    else:
        raise FileNotFoundError("No raw file found in data/raw/")


def main():
    print("Step 1. Loading raw data")
    df = load_raw_file()

    print("Rows:", len(df))
    print(df.head())

    print("Step 2. Adding technical features")
    df, feat_cols = add_technical_features(df)

    print("Feature columns:", feat_cols)

    # ensure sorting
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    out_path = Path("data/processed/prices.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Step 3. Saving processed dataset to:", out_path)
    df.to_parquet(out_path)

    print("Done. Final processed dataset saved.")


if __name__ == "__main__":
    main()
