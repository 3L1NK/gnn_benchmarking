import pandas as pd

from utils.baseline import (
    compute_buy_and_hold_fixed_shares,
    slice_and_rebase_equity_curve,
    write_buy_and_hold_csv,
)


def _toy_prices():
    return pd.DataFrame(
        [
            {"date": "2020-01-02", "ticker": "A", "close": 100.0},
            {"date": "2020-01-02", "ticker": "B", "close": 200.0},
            {"date": "2020-01-03", "ticker": "A", "close": 110.0},
            {"date": "2020-01-03", "ticker": "B", "close": 210.0},
            {"date": "2020-01-06", "ticker": "A", "close": 105.0},
            {"date": "2020-01-06", "ticker": "B", "close": 205.0},
        ]
    )


def test_buy_and_hold_deterministic():
    prices = _toy_prices()
    eq1, ret1, _ = compute_buy_and_hold_fixed_shares(prices, price_col="close")
    eq2, ret2, _ = compute_buy_and_hold_fixed_shares(prices, price_col="close")
    pd.testing.assert_series_equal(eq1, eq2)
    assert (ret1 == ret2).all()


def test_buy_and_hold_gap_drops_ticker():
    prices = pd.DataFrame(
        [
            {"date": "2020-01-02", "ticker": "A", "close": 100.0},
            {"date": "2020-01-03", "ticker": "A", "close": 101.0},
            {"date": "2020-01-06", "ticker": "A", "close": 102.0},
            {"date": "2020-01-07", "ticker": "A", "close": 103.0},
            {"date": "2020-01-02", "ticker": "B", "close": 200.0},
            {"date": "2020-01-03", "ticker": "B", "close": 201.0},
        ]
    )
    eq, _, _ = compute_buy_and_hold_fixed_shares(prices, price_col="close", max_ffill_gap=1)
    # With B dropped, equity should track A only and start at 1.0
    expected = pd.Series([1.0, 1.01, 1.02, 1.03], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]), name="buy_and_hold")
    pd.testing.assert_series_equal(eq, expected)


def test_slice_and_rebase():
    s = pd.Series(
        [2.0, 2.2, 2.4],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        name="buy_and_hold",
    )
    out = slice_and_rebase_equity_curve(s, "2020-01-03", "2020-01-06", rebase=True)
    assert out.iloc[0] == 1.0
    assert out.index.is_monotonic_increasing
    assert out.index.is_unique


def test_csv_writer_identical(tmp_path):
    s = pd.Series(
        [1.0, 1.1, 1.2],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        name="buy_and_hold",
    )
    p1 = tmp_path / "bh1.csv"
    p2 = tmp_path / "bh2.csv"
    write_buy_and_hold_csv(s, p1)
    write_buy_and_hold_csv(s, p2)
    assert p1.read_text() == p2.read_text()
