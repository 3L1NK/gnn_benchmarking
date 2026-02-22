import pandas as pd
import pytest

from utils.predictions import sanitize_predictions, prediction_row_stats


def test_sanitize_predictions_unique_ok():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "ticker": ["A", "B", "A"],
            "pred": [0.1, 0.2, 0.3],
            "realized_ret": [0.01, -0.01, 0.02],
        }
    )
    out = sanitize_predictions(df, strict_unique=True)
    assert len(out) == 3
    stats = prediction_row_stats(out)
    assert stats["prediction_rows"] == 3
    assert stats["prediction_unique_pairs"] == 3


def test_sanitize_predictions_raises_on_duplicates():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "ticker": ["A", "A"],
            "pred": [0.1, 0.2],
            "realized_ret": [0.01, 0.02],
        }
    )
    with pytest.raises(ValueError):
        sanitize_predictions(df, strict_unique=True)
