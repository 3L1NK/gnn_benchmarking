import pandas as pd
import pytest

from utils.data_loading import load_price_panel


def test_load_price_panel_requires_date_and_ticker(tmp_path):
    path = tmp_path / "prices.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "close": [100.0, 101.0],
        }
    ).to_parquet(path)

    with pytest.raises(ValueError, match="missing required columns"):
        load_price_panel(str(path), "2020-01-01", "2020-01-02")


def test_load_price_panel_rejects_duplicate_date_ticker_pairs(tmp_path):
    path = tmp_path / "prices.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["A", "A"],
            "close": [100.0, 100.1],
        }
    ).to_parquet(path)

    with pytest.raises(ValueError, match="duplicate"):
        load_price_panel(str(path), "2020-01-01", "2020-01-02")


def test_load_price_panel_filters_range_and_sorts_keys(tmp_path):
    path = tmp_path / "prices.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02", "2020-01-01"]),
            "ticker": ["B", "7", "A", "A"],
            "close": [101.0, 99.0, 100.0, 98.0],
        }
    ).to_parquet(path)

    out = load_price_panel(str(path), "2020-01-02", "2020-01-03")
    assert out["date"].min() == pd.Timestamp("2020-01-02")
    assert out["date"].max() == pd.Timestamp("2020-01-03")
    assert out[["date", "ticker"]].to_dict("records") == [
        {"date": pd.Timestamp("2020-01-02"), "ticker": "7"},
        {"date": pd.Timestamp("2020-01-02"), "ticker": "A"},
        {"date": pd.Timestamp("2020-01-03"), "ticker": "B"},
    ]
