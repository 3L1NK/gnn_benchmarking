from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from .predictions import REQUIRED_PRED_COLUMNS


@dataclass(frozen=True)
class PredictionAuditIssue:
    path: str
    issue: str
    details: str


def _resolve_prediction_files(roots: Sequence[Path], pattern: str = "*_predictions.csv") -> List[Path]:
    files: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(resolved)
    return sorted(files)


def audit_prediction_file(path: Path) -> List[PredictionAuditIssue]:
    issues: List[PredictionAuditIssue] = []
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [
            PredictionAuditIssue(
                path=str(path),
                issue="read_error",
                details=f"failed to read CSV: {exc}",
            )
        ]

    missing = [c for c in REQUIRED_PRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(
            PredictionAuditIssue(
                path=str(path),
                issue="missing_columns",
                details=f"missing required columns: {missing}",
            )
        )
        return issues

    work = df.loc[:, list(REQUIRED_PRED_COLUMNS)].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    invalid_dates = int(work["date"].isna().sum())
    if invalid_dates > 0:
        issues.append(
            PredictionAuditIssue(
                path=str(path),
                issue="invalid_dates",
                details=f"{invalid_dates} rows have invalid date values",
            )
        )

    work["ticker"] = work["ticker"].astype(str)
    dup_mask = work.duplicated(["date", "ticker"], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        sample = (
            work.loc[dup_mask, ["date", "ticker"]]
            .drop_duplicates(["date", "ticker"])
            .head(5)
            .to_string(index=False)
        )
        issues.append(
            PredictionAuditIssue(
                path=str(path),
                issue="duplicate_pairs",
                details=f"{dup_count} duplicate rows by (date,ticker). Sample:\n{sample}",
            )
        )
    return issues


def audit_prediction_artifacts(
    roots: Sequence[str | Path],
    pattern: str = "*_predictions.csv",
) -> tuple[List[Path], List[PredictionAuditIssue]]:
    root_paths = [Path(p) for p in roots]
    files = _resolve_prediction_files(root_paths, pattern=pattern)
    issues: List[PredictionAuditIssue] = []
    for path in files:
        issues.extend(audit_prediction_file(path))
    return files, issues


def assert_no_prediction_artifact_issues(
    roots: Sequence[str | Path],
    pattern: str = "*_predictions.csv",
) -> List[Path]:
    files, issues = audit_prediction_artifacts(roots, pattern=pattern)
    if issues:
        preview = "\n".join(
            [f"- {i.path}: {i.issue} ({i.details})" for i in issues[:10]]
        )
        raise ValueError(
            f"Prediction artifact audit failed with {len(issues)} issue(s) across {len(files)} file(s):\n{preview}"
        )
    return files
