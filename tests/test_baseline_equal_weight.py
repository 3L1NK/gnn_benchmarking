import pandas as pd

from utils.baseline import compute_equal_weight_rebalanced, write_baseline_curve_csv


def _toy_prices():
    return pd.DataFrame(
        [
            {"date": "2020-01-01", "ticker": "A", "close": 100.0},
            {"date": "2020-01-01", "ticker": "B", "close": 200.0},
            {"date": "2020-01-02", "ticker": "A", "close": 101.0},
            {"date": "2020-01-02", "ticker": "B", "close": 199.0},
            {"date": "2020-01-03", "ticker": "A", "close": 102.0},
            {"date": "2020-01-03", "ticker": "B", "close": 198.0},
        ]
    )


def test_equal_weight_baseline_runs():
    prices = _toy_prices()
    eq, ret, stats = compute_equal_weight_rebalanced(prices, price_col="close", rebalance_freq=1)
    assert len(eq) == 3
    assert len(ret) == 3
    assert "annualized_return" in stats


def test_write_baseline_curve_csv(tmp_path):
    s = pd.Series([1.0, 1.1], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    out = tmp_path / "eqw.csv"
    write_baseline_curve_csv(s, out, value_column="equal_weight")
    text = out.read_text()
    assert "equal_weight" in text
