#!/usr/bin/env python3
"""
Print simple leaderboards from results.jsonl.

Usage:
  python3 scripts/leaderboard.py --path results/results.jsonl --top 10
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="results/results.jsonl")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_json(path, lines=True)

    if df.empty:
        print("No results found.")
        return

    cols = ["experiment_id", "model_name", "edge_type", "rebalance_freq", "portfolio_sharpe", "prediction_rank_ic"]
    for col in cols:
        if col not in df.columns:
            df[col] = None

    top = args.top

    print(f"Top {top} by Sharpe")
    print(df.sort_values("portfolio_sharpe", ascending=False)[cols].head(top).to_string(index=False))
    print("")
    print(f"Top {top} by Rank IC")
    print(df.sort_values("prediction_rank_ic", ascending=False)[cols].head(top).to_string(index=False))


if __name__ == "__main__":
    main()
