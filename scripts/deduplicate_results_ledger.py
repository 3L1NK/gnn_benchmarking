#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.results import (
    RESULT_FIELDS,
    run_key_from_fields,
    split_id_from_fields,
)


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"results ledger not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _normalize_row(row: dict) -> dict:
    out = {k: row.get(k, "") for k in RESULT_FIELDS}

    split_id = str(out.get("split_id", "") or "")
    if not split_id:
        split_id = split_id_from_fields(
            out.get("protocol_version", ""),
            out.get("split_train_end", ""),
            out.get("split_val_start", ""),
            out.get("split_test_start", ""),
            out.get("target_policy_hash", ""),
        )
    out["split_id"] = split_id

    config_hash = str(out.get("config_hash", "") or "")
    if not config_hash:
        # Old ledgers do not carry config payload; keep explicit legacy marker.
        config_hash = "legacy_config"
    out["config_hash"] = config_hash

    run_key = str(out.get("run_key", "") or "")
    if not run_key:
        run_key = run_key_from_fields(
            out.get("model_family", ""),
            out.get("model_name", ""),
            out.get("edge_type", ""),
            out.get("seed", ""),
            out.get("target_type", ""),
            out.get("target_horizon", ""),
            out.get("rebalance_freq", ""),
            out.get("split_id", ""),
            out.get("config_hash", ""),
        )
    out["run_key"] = run_key
    return out


def _dedup_latest(rows: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    for row in rows:
        norm = _normalize_row(row)
        key = str(norm.get("run_key", "") or "")
        if not key:
            key = str(norm.get("experiment_id", "") or "")
        if key not in by_key:
            by_key[key] = norm
            continue
        prev = by_key[key]
        prev_ts = pd.to_datetime(prev.get("timestamp", ""), errors="coerce")
        cur_ts = pd.to_datetime(norm.get("timestamp", ""), errors="coerce")
        if pd.isna(prev_ts) or (not pd.isna(cur_ts) and cur_ts >= prev_ts):
            by_key[key] = norm

    def _sort_key(r: dict) -> tuple[int, pd.Timestamp]:
        ts = pd.to_datetime(r.get("timestamp", ""), errors="coerce")
        if pd.isna(ts):
            return (1, pd.Timestamp.min)
        return (0, ts)

    return sorted(by_key.values(), key=_sort_key)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            ordered = {k: row.get(k, "") for k in RESULT_FIELDS}
            f.write(json.dumps(ordered) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate results JSONL by stable run_key (latest timestamp wins).")
    parser.add_argument("--results", type=str, default="results/results_tuned_all.jsonl")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_path = Path(args.out) if args.out else results_path

    rows = _load_rows(results_path)
    deduped = _dedup_latest(rows)
    _write_rows(out_path, deduped)

    print(f"[dedup] input rows: {len(rows)}")
    print(f"[dedup] output rows: {len(deduped)}")
    print(f"[dedup] wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
