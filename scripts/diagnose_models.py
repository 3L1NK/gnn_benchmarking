"""
Quick diagnostics to compare LSTM vs XGB+Node2Vec data pipeline.

Run from repo root: `python scripts/diagnose_models.py`
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config_normalize import load_config as load_normalized_config
from utils.data_loading import load_price_panel
# avoid importing add_technical_features (depends on statsmodels) in this lightweight script
# implement a lightweight rolling_corr_edges here to avoid importing statsmodels-dependent utils
def rolling_corr_edges(panel_df, date, window, threshold):
    import pandas as pd
    import numpy as np

    end_date = pd.to_datetime(date)
    start_date = end_date - pd.Timedelta(days=window * 2)
    hist = panel_df[(panel_df["date"] > start_date) & (panel_df["date"] <= end_date)]

    if hist.empty:
        return []

    pivot = hist.pivot(index="date", columns="ticker", values="log_ret_1d").dropna(axis=1, how="all")
    corr = pivot.corr()

    edges = []
    for i, ti in enumerate(corr.columns):
        for j, tj in enumerate(corr.columns):
            if j <= i:
                continue
            c = corr.iloc[i, j]
            if np.abs(c) >= threshold:
                edges.append((ti, tj, float(c)))
    return edges


def load_cfg(path):
    return load_normalized_config(path, REPO_ROOT)


def count_sequences(df, lookback, horizon, id_col="ticker"):
    cnt = 0
    per_t = {}
    for t, g in df.groupby(id_col):
        n = len(g)
        if n <= lookback:
            per_t[t] = 0
            continue
        # number of valid positions the trainer builds
        npos = max(0, (n - horizon) - lookback)
        per_t[t] = int(npos)
        cnt += npos
    return cnt, per_t


def main():
    repo = REPO_ROOT
    cfg_xgb = load_cfg(repo / "configs" / "runs" / "core" / "xgb_node2vec_corr.yaml")
    cfg_lstm = load_cfg(repo / "configs" / "runs" / "core" / "lstm.yaml")

    print("Config files loaded:")
    print(" - xgb_node2vec:", cfg_xgb.get("experiment_name", "<none>"))
    print(" - lstm:", cfg_lstm.get("experiment_name", "<none>"))

    price_file_x = cfg_xgb["data"]["price_file"]
    price_file_l = cfg_lstm["data"]["price_file"]

    print("Loading price panels (this uses the repo's loader)...")
    df_x = load_price_panel(price_file_x, cfg_xgb["data"]["start_date"], cfg_xgb["data"]["end_date"])
    df_l = load_price_panel(price_file_l, cfg_lstm["data"]["start_date"], cfg_lstm["data"]["end_date"])

    print("Price panels shapes:")
    print(" - xgb df:", df_x.shape)
    print(" - lstm df:", df_l.shape)

    # If technical features are already present in the parquet, use them.
    possible_feat_cols = [
        "ret_1d", "ret_5d", "ret_20d", "log_ret_1d",
        "mom_3d", "mom_10", "mom_21d",
        "vol_5d", "vol_20d", "vol_60d",
        "drawdown_20d",
        "volume_pct_change", "vol_z_5", "vol_z_20",
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
    ]
    feat_cols_l = [c for c in possible_feat_cols if c in df_l.columns]
    if feat_cols_l:
        print(f" - LSTM features columns found in parquet: {len(feat_cols_l)} -> {feat_cols_l}")
        df_l_feats = df_l.dropna(subset=feat_cols_l + ["log_ret_1d"]) if "log_ret_1d" in df_l.columns else df_l.dropna()
    else:
        print(" - LSTM features missing in parquet; skipping feature computation in this lightweight check.")
        df_l_feats = df_l.dropna(subset=["log_ret_1d"]) if "log_ret_1d" in df_l.columns else df_l

    print("For XGB pipeline, checking default feature list intersection with data columns")
    default_feat_cols = [
        "ret_1d", "ret_5d", "ret_20d", "log_ret_1d",
        "mom_3d", "mom_10", "mom_21d",
        "vol_5d", "vol_20d", "vol_60d",
        "drawdown_20d",
        "volume_pct_change", "vol_z_5", "vol_z_20",
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
    ]
    feat_cols_x = [c for c in default_feat_cols if c in df_x.columns]
    print(f" - XGB available default features: {len(feat_cols_x)} -> {feat_cols_x}")

    # Sequence counts for LSTM
    lookback = cfg_lstm["data"]["lookback_window"]
    horizon = cfg_lstm["data"]["target_horizon"]
    seq_cnt, per_t = count_sequences(df_l_feats, lookback, horizon)
    print(f"LSTM: total sequences built (approx): {seq_cnt}")

    # Node2Vec graph edges for XGB
    val_start = pd.to_datetime(cfg_xgb["training"]["val_start"]) if "val_start" in cfg_xgb["training"] else pd.to_datetime(cfg_xgb["training"]["test_start"])  
    train_df = df_x[df_x["date"] < val_start]
    last_train_date = train_df["date"].max()
    corr_window = cfg_xgb["data"].get("lookback_window", 60)
    corr_thr = cfg_xgb["data"].get("corr_threshold", 0.4)

    edges = rolling_corr_edges(train_df, last_train_date, corr_window, corr_thr)
    print(f"XGB Node2Vec: rolling_corr_edges found {len(edges)} edges (threshold {corr_thr})")

    # sample of edges
    if edges:
        print(" - sample edges:", edges[:5])

    # check universe metadata
    uni_path = repo / "data" / "processed" / "universe.csv"
    if uni_path.exists():
        uni = pd.read_csv(uni_path)
        print("Universe metadata found, tickers:", uni.shape[0])
        print(" - sectors present:", 'sector' in uni.columns, "industry present:", 'industry' in uni.columns)
    else:
        print("Universe metadata missing at data/processed/universe.csv")

    print("Diagnostics complete. Suggested next steps: check that embeddings cover test tickers, and whether embeddings were scaled too weakly (0.2 factor).")


if __name__ == "__main__":
    main()
