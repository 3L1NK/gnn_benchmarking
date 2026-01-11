from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureScaler:
    feature_cols: List[str]
    mean: np.ndarray
    std: np.ndarray


def fit_feature_scaler(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    train_mask: np.ndarray,
    *,
    label: str = "scaler",
    debug: bool = False,
) -> FeatureScaler:
    cols = list(feature_cols)
    train_df = df.loc[train_mask, cols]
    arr = train_df.to_numpy(dtype=float, copy=True)

    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std == 0, 1.0, std)

    if debug and "date" in df.columns:
        dates = pd.to_datetime(df.loc[train_mask, "date"])
        if not dates.empty:
            print(
                f"[{label}] scaler fit range={dates.min().date()}..{dates.max().date()} "
                f"rows={len(dates)}"
            )

    return FeatureScaler(feature_cols=cols, mean=mean, std=std)


def _impute_features(df: pd.DataFrame, scaler: FeatureScaler) -> np.ndarray:
    arr = df[scaler.feature_cols].to_numpy(dtype=float, copy=True)
    if np.isnan(arr).any():
        inds = np.isnan(arr)
        arr[inds] = np.take(scaler.mean, np.where(inds)[1])
    return arr


def apply_feature_scaler(df: pd.DataFrame, scaler: FeatureScaler) -> np.ndarray:
    arr = _impute_features(df, scaler)
    return (arr - scaler.mean) / (scaler.std + 1e-8)


def scale_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    train_mask: np.ndarray,
    *,
    label: str = "scaler",
    debug: bool = False,
    scale: bool = True,
) -> Tuple[pd.DataFrame, FeatureScaler]:
    scaler = fit_feature_scaler(df, feature_cols, train_mask, label=label, debug=debug)
    scaled = df.copy()
    if scale:
        scaled[scaler.feature_cols] = apply_feature_scaler(df, scaler)
    else:
        scaled[scaler.feature_cols] = _impute_features(df, scaler)
    return scaled, scaler
