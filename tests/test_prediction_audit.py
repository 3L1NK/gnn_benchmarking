from pathlib import Path

import pandas as pd
import pytest

from utils.prediction_audit import (
    assert_no_prediction_artifact_issues,
    audit_prediction_artifacts,
)


def test_prediction_audit_passes_clean_files(tmp_path):
    pred = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "ticker": ["A", "B", "A"],
            "pred": [0.1, 0.2, 0.3],
            "realized_ret": [0.01, 0.02, 0.03],
        }
    )
    p = tmp_path / "x_predictions.csv"
    pred.to_csv(p, index=False)

    files, issues = audit_prediction_artifacts([tmp_path])
    assert files == [p.resolve()]
    assert issues == []
    assert_no_prediction_artifact_issues([tmp_path])


def test_prediction_audit_detects_duplicate_pairs(tmp_path):
    pred = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "ticker": ["A", "A"],
            "pred": [0.1, 0.2],
            "realized_ret": [0.01, 0.02],
        }
    )
    p = tmp_path / "bad_predictions.csv"
    pred.to_csv(p, index=False)

    _, issues = audit_prediction_artifacts([tmp_path])
    assert any(i.issue == "duplicate_pairs" for i in issues)
    with pytest.raises(ValueError, match="Prediction artifact audit failed"):
        assert_no_prediction_artifact_issues([tmp_path])


def test_prediction_audit_detects_missing_columns(tmp_path):
    p = Path(tmp_path) / "bad_predictions.csv"
    pd.DataFrame({"date": ["2020-01-01"], "ticker": ["A"]}).to_csv(p, index=False)

    _, issues = audit_prediction_artifacts([tmp_path])
    assert any(i.issue == "missing_columns" for i in issues)
