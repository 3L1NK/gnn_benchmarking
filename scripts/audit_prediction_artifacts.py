#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.prediction_audit import audit_prediction_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit saved prediction artifacts for schema and duplicate (date,ticker) pairs."
    )
    parser.add_argument(
        "--root",
        action="append",
        default=["experiments"],
        help="Root directory to scan recursively. Can be repeated.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_predictions.csv",
        help="Glob pattern for prediction files (default: *_predictions.csv).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional path to write issue rows as CSV.",
    )
    args = parser.parse_args()

    files, issues = audit_prediction_artifacts(args.root, pattern=args.pattern)
    print(f"[audit] scanned files: {len(files)}")
    print(f"[audit] issues found: {len(issues)}")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        issue_df = pd.DataFrame([i.__dict__ for i in issues])
        issue_df.to_csv(out_path, index=False)
        print(f"[audit] wrote issue CSV: {out_path}")

    if issues:
        print("[audit] sample issues:")
        for item in issues[:15]:
            print(f" - {item.path} :: {item.issue} :: {item.details}")
        return 1

    print("[audit] all prediction artifacts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
