from __future__ import annotations

from typing import Dict

import pandas as pd


REQUIRED_PRED_COLUMNS = ("date", "ticker", "pred", "realized_ret")


def validate_prediction_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction frame missing required columns: {missing}")


def sanitize_predictions(pred_df: pd.DataFrame, *, strict_unique: bool = True) -> pd.DataFrame:
    validate_prediction_columns(pred_df)
    out = pred_df.copy()
    out = out.loc[:, list(REQUIRED_PRED_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str)
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    out["realized_ret"] = pd.to_numeric(out["realized_ret"], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "pred", "realized_ret"]).reset_index(drop=True)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    dup_mask = out.duplicated(["date", "ticker"], keep=False)
    dup_count = int(dup_mask.sum())
    if strict_unique and dup_count > 0:
        sample = out.loc[dup_mask, ["date", "ticker"]].head(5)
        raise ValueError(
            f"Duplicate (date,ticker) predictions detected: {dup_count}. "
            f"Sample:\n{sample.to_string(index=False)}"
        )
    return out


def prediction_row_stats(pred_df: pd.DataFrame) -> Dict[str, int]:
    rows = int(len(pred_df))
    unique_pairs = int(pred_df.drop_duplicates(["date", "ticker"]).shape[0])
    return {
        "prediction_rows": rows,
        "prediction_unique_pairs": unique_pairs,
    }
