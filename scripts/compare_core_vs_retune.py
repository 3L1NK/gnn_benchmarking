#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    df = pd.read_json(path, lines=True)
    if df.empty:
        raise ValueError(f"results file is empty: {path}")
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["rebalance_freq"] = pd.to_numeric(df.get("rebalance_freq"), errors="coerce").fillna(1).astype(int)
    df["portfolio_sharpe"] = pd.to_numeric(df.get("portfolio_sharpe"), errors="coerce")
    if "portfolio_sharpe_annualized" not in df.columns:
        df["portfolio_sharpe_annualized"] = df["portfolio_sharpe"] * np.sqrt(252.0)
    df["portfolio_sharpe_annualized"] = pd.to_numeric(df["portfolio_sharpe_annualized"], errors="coerce")
    df["portfolio_max_drawdown"] = pd.to_numeric(df.get("portfolio_max_drawdown"), errors="coerce")
    df["portfolio_annualized_return"] = pd.to_numeric(df.get("portfolio_annualized_return"), errors="coerce")
    df["portfolio_final_value"] = pd.to_numeric(df.get("portfolio_final_value"), errors="coerce")
    df["run_tag"] = df.get("run_tag", "").astype(str)
    return df


def _latest_per_run(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_tag", "rebalance_freq", "target_policy_hash"]
    for c in keys:
        if c not in df.columns:
            df[c] = pd.NA
    return df.sort_values("timestamp").groupby(keys, as_index=False, dropna=False).tail(1).reset_index(drop=True)


def _normalize_retune_tag(tag: str) -> str:
    out = str(tag)
    for suffix in ["_retune_medium", "_retune"]:
        if out.endswith(suffix):
            return out[: -len(suffix)]
    return out


def compare(core_path: Path, retune_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    core = _latest_per_run(_load_results(core_path))
    retune = _latest_per_run(_load_results(retune_path))

    retune["base_run_tag"] = retune["run_tag"].map(_normalize_retune_tag)
    core["base_run_tag"] = core["run_tag"].astype(str)

    keep_cols = [
        "base_run_tag",
        "run_tag",
        "rebalance_freq",
        "portfolio_sharpe_annualized",
        "portfolio_max_drawdown",
        "portfolio_annualized_return",
        "portfolio_final_value",
    ]
    merged = retune[keep_cols].merge(
        core[keep_cols],
        on=["base_run_tag", "rebalance_freq"],
        how="left",
        suffixes=("_retune", "_core"),
    )

    merged["delta_sharpe_annualized"] = (
        merged["portfolio_sharpe_annualized_retune"] - merged["portfolio_sharpe_annualized_core"]
    )
    merged["delta_max_drawdown"] = (
        merged["portfolio_max_drawdown_retune"] - merged["portfolio_max_drawdown_core"]
    )
    merged["delta_annualized_return"] = (
        merged["portfolio_annualized_return_retune"] - merged["portfolio_annualized_return_core"]
    )
    merged["delta_final_value"] = merged["portfolio_final_value_retune"] - merged["portfolio_final_value_core"]

    winners = merged[
        (merged["portfolio_sharpe_annualized_retune"] > merged["portfolio_sharpe_annualized_core"])
        & (merged["portfolio_max_drawdown_retune"] >= (merged["portfolio_max_drawdown_core"] - 0.02))
    ].copy()

    merged.to_csv(out_dir / "retune_delta_summary.csv", index=False)
    winners.to_csv(out_dir / "retune_winners.csv", index=False)

    print(f"[compare] wrote {out_dir / 'retune_delta_summary.csv'} rows={len(merged)}")
    if winners.empty:
        print("[compare] no retune winners under rule: improve sharpe_annualized and max_drawdown not worse by > 0.02")
    else:
        print(f"[compare] wrote {out_dir / 'retune_winners.csv'} rows={len(winners)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare core benchmark results vs targeted retune results.")
    parser.add_argument("--core", type=str, default="results/results.jsonl")
    parser.add_argument("--retune", type=str, default="results/results_retune.jsonl")
    parser.add_argument("--out", type=str, default="results/reports/retune_comparison")
    args = parser.parse_args()
    compare(Path(args.core), Path(args.retune), Path(args.out))


if __name__ == "__main__":
    main()
