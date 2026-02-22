import pandas as pd

from utils.backtest import backtest_long_only


def _toy_predictions():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "ticker": ["A", "B", "A", "B", "A", "B"],
            "pred": [0.8, 0.2, 0.1, 0.9, 0.9, 0.1],
            "realized_ret": [0.01, -0.01, -0.03, 0.02, -0.01, 0.03],
        }
    )


def test_backtest_rebalance_freq_changes_path():
    pred_df = _toy_predictions()
    eq_daily, ret_daily, _ = backtest_long_only(pred_df, top_k=1, rebalance_freq=1)
    eq_weekly, ret_weekly, _ = backtest_long_only(pred_df, top_k=1, rebalance_freq=5)
    assert len(eq_daily) == len(eq_weekly)
    assert len(ret_daily) == len(ret_weekly)
    # Same input but different rebalance frequencies should generally differ.
    assert float(eq_daily.iloc[-1]) != float(eq_weekly.iloc[-1])


def test_backtest_reproducible_for_fixed_input():
    pred_df = _toy_predictions()
    eq_1, ret_1, stats_1 = backtest_long_only(pred_df, top_k=1, rebalance_freq=1)
    eq_2, ret_2, stats_2 = backtest_long_only(pred_df, top_k=1, rebalance_freq=1)
    pd.testing.assert_series_equal(eq_1, eq_2)
    assert list(ret_1) == list(ret_2)
    assert stats_1["final_value"] == stats_2["final_value"]
