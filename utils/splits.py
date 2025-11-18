import pandas as pd


def time_split_by_date(df, val_start, test_start):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train = df[df["date"] < pd.to_datetime(val_start)]
    val = df[(df["date"] >= pd.to_datetime(val_start)) & (df["date"] < pd.to_datetime(test_start))]
    test = df[df["date"] >= pd.to_datetime(test_start)]

    return train, val, test
