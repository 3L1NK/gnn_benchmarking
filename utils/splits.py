import pandas as pd


def split_time(dates, config, *, label="split", debug=False):
    dates = pd.to_datetime(pd.Series(dates))
    val_start = pd.to_datetime(config["training"]["val_start"])
    test_start = pd.to_datetime(config["training"]["test_start"])

    train_mask = dates < val_start
    val_mask = (dates >= val_start) & (dates < test_start)
    test_mask = dates >= test_start

    if debug:
        def _rng(mask):
            if mask.sum() == 0:
                return "empty"
            sub = dates[mask]
            return f"{sub.min().date()}..{sub.max().date()}"

        print(
            f"[{label}] split val_start={val_start.date()} test_start={test_start.date()} "
            f"train_range={_rng(train_mask)} val_range={_rng(val_mask)} test_range={_rng(test_mask)}"
        )

    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy(), {
        "val_start": val_start,
        "test_start": test_start,
    }


def time_split_by_date(df, val_start, test_start):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train = df[df["date"] < pd.to_datetime(val_start)]
    val = df[(df["date"] >= pd.to_datetime(val_start)) & (df["date"] < pd.to_datetime(test_start))]
    test = df[df["date"] >= pd.to_datetime(test_start)]

    return train, val, test
