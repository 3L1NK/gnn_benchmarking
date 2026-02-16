import pandas as pd

from utils.baseline import slice_and_rebase_equity_curve


def test_buy_and_hold_slice_same_window_is_identical():
    full = pd.Series(
        [1.0, 1.1, 1.05, 1.2, 1.25],
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
        name="buy_and_hold",
    )

    a = slice_and_rebase_equity_curve(full, start_date="2020-01-02", end_date="2020-01-07", rebase=True)
    b = slice_and_rebase_equity_curve(full, start_date="2020-01-02", end_date="2020-01-07", rebase=True)
    pd.testing.assert_series_equal(a, b)
