# trainers/train_xgboost.py
from pathlib import Path
import json

import numpy as np
import pandas as pd

from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.graphs import rolling_corr_edges, graphical_lasso_precision, granger_edges
from utils.metrics import rank_ic, hit_rate, sharpe_ratio
from utils.backtest import backtest_long_short, backtest_buy_and_hold, backtest_long_only
from utils.plot import (
    plot_equity_curve,
    plot_daily_ic,
    plot_ic_hist,
    plot_equity_comparison,
)
from utils.seeds import set_seed
from utils.cache import cache_load, cache_save, cache_key, cache_path
from utils.baseline import get_global_buy_and_hold
from utils.targets import build_target
from xgboost import XGBRegressor
from itertools import product
from sklearn.metrics import root_mean_squared_error


def _build_feature_panel(config):
    """Shared helper. Returns df with features and target, list of feature columns.

    Features are assumed to be precomputed in the parquet produced by preprocessing.
    """
    default_feat_cols = [
        "ret_1d", "ret_5d", "ret_20d", "log_ret_1d",
        "mom_3d", "mom_10", "mom_21d",
        "vol_5d", "vol_20d", "vol_60d",
        "drawdown_20d",
        "volume_pct_change", "vol_z_5", "vol_z_20",
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
    ]
    price_file = config["data"]["price_file"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    df = load_price_panel(price_file, start, end)
    feat_cols = [c for c in default_feat_cols if c in df.columns]

    # Fallback: if the parquet is raw prices without features, compute them here.
    if not feat_cols:
        df, feat_cols = add_technical_features(df)
        feat_cols = [c for c in default_feat_cols if c in df.columns]

    df, _ = build_target(df, config, target_col="target")

    return df, feat_cols

def _time_masks(dates, val_start, test_start):
    """
    Docstring for _time_masks
    
    it transforms dates into pd.Timestamp and creates boolean masks for train, validation, and test sets based on the provided start dates.
    
    :param dates: dates from the price.parquet DataFrame
    :param val_start: when the validation period starts
    :param test_start: when the test period starts
    :return: train_mask, val_mask, test_mask
    """
    dates = pd.to_datetime(dates)
    val_start = pd.to_datetime(val_start)
    test_start = pd.to_datetime(test_start)
    train_mask = dates < val_start
    val_mask = (dates >= val_start) & (dates < test_start)
    test_mask = dates >= test_start
    return train_mask, val_mask, test_mask

def _add_graph_smooth_features(df, feat_cols, adj_dict, alpha=0.5):
    """
    for granger or graphical lasso adjacency
    For each ticker and date:
      new_feat = alpha * own + (1 - alpha) * neighbor weighted average

    Returns df with extra columns for smoothed features.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    smoothed_cols = []

    for col in feat_cols:
        new_col = col + "_smooth"
        smoothed_cols.append(new_col)
        df[new_col] = df[col]

    by_date = dict(tuple(df.groupby("date", sort=False)))

    out_frames = []
    for date, g in by_date.items():
        g = g.copy()
        ticker_to_idx = {t: i for i, t in enumerate(g["ticker"])}
        for idx, row in g.iterrows():
            t = row["ticker"]
            if t not in adj_dict:
                continue
            neighbors = adj_dict[t]
            if not neighbors:
                continue

            weights = []
            neigh_feat = {c: [] for c in feat_cols}

            for nb, w in neighbors:
                if nb not in ticker_to_idx:
                    continue
                nb_row = g.iloc[ticker_to_idx[nb]]
                weights.append(abs(w))
                for c in feat_cols:
                    neigh_feat[c].append(nb_row[c])

            if not weights:
                continue
            w_arr = np.array(weights)
            w_arr = w_arr / (w_arr.sum() + 1e-8)

            for c in feat_cols:
                neigh_vals = np.array(neigh_feat[c])
                neigh_mean = (w_arr * neigh_vals).sum()
                new_val = alpha * row[c] + (1 - alpha) * neigh_mean
                g.at[idx, c + "_smooth"] = new_val

        out_frames.append(g)

    df_out = pd.concat(out_frames, ignore_index=True)
    return df_out, smoothed_cols

class XGBoostTrainer:
    def __init__(self, config):
        self.config = config
        self.df, self.feat_cols = _build_feature_panel(config)
        self.df["date"] = pd.to_datetime(self.df["date"])

        self.out_dir = Path(config["evaluation"]["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rebuild_cache = config.get("cache", {}).get("rebuild", False)

        self._registry = {
            "xgb_raw": self.train_xgb_raw,
            "xgb_node2vec": self.train_xgb_node2vec,
            "graphlasso_linear": self.train_graphlasso_linear,
            "graphlasso_xgb": self.train_graphlasso_xgb,
            "granger_xgb": self.train_granger_xgb,
        }

    def _groups_from_dates(self, dates, mask):
        """Return a list of group sizes per date in chronological order for rows where mask is True."""
        ds = pd.to_datetime(pd.Series(dates))
        sub = ds[mask]
        if sub.empty:
            return []
        # groupby on values gives groups in sorted order
        grp = sub.groupby(sub).size().tolist()
        return grp

    def run(self):
        key = self.config["model"]["type"].lower()

        # global buy-and-hold baseline (cached, model independent)
        eq_bh_full, ret_bh_full, stats_bh = get_global_buy_and_hold(
            self.config,
            rebuild=self.rebuild_cache,
            align_start_date=self.config["training"]["test_start"],
        )
        self.bh_curve = eq_bh_full
        self.bh_ret = ret_bh_full
        self.bh_stats = stats_bh
        print("[baseline] global buy-and-hold stats (aligned to test window)", stats_bh)

        try:
            result = self._registry[key]()   # train the chosen XGB variant
        except KeyError:
            raise ValueError(f"Unknown xgboost type {key}")

        return result

    def _evaluate_and_backtest(self, pred_df, name: str):
        daily_metrics = []
        for d, g in pred_df.groupby("date"):
            ic = rank_ic(g["pred"], g["realized_ret"])
            hit = hit_rate(g["pred"], g["realized_ret"], top_k=self.config["evaluation"]["top_k"])
            daily_metrics.append({"date": d, "ic": ic, "hit": hit})

        daily_metrics = pd.DataFrame(daily_metrics).sort_values("date")
        
        # IC time series plot
        plot_daily_ic(
            daily_metrics,
            self.out_dir / f"{name}_ic_timeseries.png"
        )

        # IC histogram
        plot_ic_hist(
            daily_metrics,
            self.out_dir / f"{name}_ic_histogram.png"
        )

        print(
            f"[{name}] mean IC",
            daily_metrics["ic"].mean(),
            "mean hit",
            daily_metrics["hit"].mean(),
        )
        if name == "xgb_node2vec":
            print("[xgb_node2vec] Note: static embeddings provided no temporal signal; no improvement observed vs raw XGB is expected.")

        equity_curve, daily_ret, stats = backtest_long_only(
            pred_df,
            top_k=self.config["evaluation"]["top_k"],
            transaction_cost_bps=self.config["evaluation"]["transaction_cost_bps"],
            risk_free_rate=self.config["evaluation"]["risk_free_rate"],
        )
        equity_curve = equity_curve.copy()
        equity_curve.index = pd.to_datetime(equity_curve.index)
        daily_ret_series = pd.Series(daily_ret, index=equity_curve.index[: len(daily_ret)])
        dd = equity_curve / equity_curve.cummax() - 1.0
        daily_metrics = daily_metrics.merge(
            pd.DataFrame(
                {
                    "date": daily_ret_series.index,
                    "daily_return": daily_ret_series.values,
                    "drawdown": dd.reindex(daily_ret_series.index).values,
                }
            ),
            on="date",
            how="left",
        )
        daily_metrics.to_csv(self.out_dir / f"{name}_daily_metrics.csv", index=False)

        equity_curve.to_csv(self.out_dir / f"{name}_equity_curve.csv", header=["value"])
        plot_equity_curve(
            equity_curve,
            f"{name} long only",
            self.out_dir / f"{name}_equity_curve.png",
        )
        print(f"[{name}] backtest stats", stats)
        
        # Align baseline to prediction window for plotting
        start_d, end_d = pred_df["date"].min(), pred_df["date"].max()
        bh_slice = self.bh_curve.loc[(self.bh_curve.index >= start_d) & (self.bh_curve.index <= end_d)]

        plot_equity_comparison(
            model_curve=equity_curve,
            bh_curve=bh_slice,
            title=f"{name}: Model vs Buy & Hold",
            out_path=self.out_dir / f"{name}_vs_buy_and_hold.png",
        )

        # Summary + rolling metrics for thesis reporting
        run_tag = self.config.get("experiment_name", name)
        ic_series = daily_metrics.set_index("date")["ic"].dropna()
        ic_mean = float(ic_series.mean()) if not ic_series.empty else float("nan")
        ic_std = float(ic_series.std()) if not ic_series.empty else float("nan")
        ic_tstat = float(ic_mean / (ic_std / np.sqrt(len(ic_series)))) if ic_series.size > 1 and ic_std > 0 else float("nan")
        vol = float(np.std(daily_ret_series.values) * np.sqrt(252)) if not daily_ret_series.empty else float("nan")
        max_dd = float(dd.min()) if not dd.empty else float("nan")

        summary = {
            "run_tag": run_tag,
            "model_type": name,
            "stats": stats,
            "buy_and_hold_stats": self.bh_stats,
            "ic_mean": ic_mean,
            "ic_tstat": ic_tstat,
            "volatility": vol,
            "max_drawdown": max_dd,
        }
        summary_path = self.out_dir / f"{name}_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        rolling_window = int(self.config.get("evaluation", {}).get("rolling_window", 63))
        rolling = pd.DataFrame({"date": daily_metrics["date"]}).copy()
        rolling["rolling_sharpe"] = daily_ret_series.rolling(rolling_window).apply(
            lambda x: sharpe_ratio(x, self.config["evaluation"]["risk_free_rate"]) if len(x) > 1 else np.nan,
            raw=False,
        ).values
        rolling["rolling_ic_mean"] = ic_series.reindex(daily_metrics["date"]).rolling(rolling_window).mean().values
        rolling.to_csv(self.out_dir / f"{name}_rolling_metrics.csv", index=False)
    
    
    def train_xgb_raw(self):
        set_seed(42)

        # -----------------------
        # 1. Split data
        # -----------------------
        train_mask, val_mask, test_mask = _time_masks(
            self.df["date"],
            self.config["training"]["val_start"],
            self.config["training"]["test_start"],
        )

        # Optional cross-sectional (per-date) z-score normalization for raw features.
        cs_cfg = self.config.get("xgb", {})
        cs_zscore = bool(cs_cfg.get("cross_sectional_zscore", False))

        raw_mat = self.df[self.feat_cols].to_numpy(dtype=float, copy=True)
        dates_arr = self.df["date"].values

        if cs_zscore:
            norm_raw = np.zeros_like(raw_mat, dtype=float)
            uniq_dates, inv_idx = np.unique(dates_arr, return_inverse=True)
            for i, d in enumerate(uniq_dates):
                idxs = np.where(inv_idx == i)[0]
                block = raw_mat[idxs].astype(float, copy=True)
                col_mean = np.nanmean(block, axis=0)
                col_std = np.nanstd(block, axis=0)
                col_std[col_std == 0] = 1.0
                inds_nan = np.isnan(block)
                if inds_nan.any():
                    block[inds_nan] = np.take(col_mean, np.where(inds_nan)[1])
                z = (block - col_mean) / (col_std + 1e-8)
                norm_raw[idxs] = z
            X = norm_raw
        else:
            # minimal imputation to avoid NaNs: fill with global column mean
            col_mean = np.nanmean(raw_mat, axis=0)
            inds_nan = np.isnan(raw_mat)
            if inds_nan.any():
                raw_mat[inds_nan] = np.take(col_mean, np.where(inds_nan)[1])
            X = raw_mat

        y = self.df["target"].values.astype(float)

        cache_id = cache_key(
            {
                "model": "xgb_raw",
                "params": self.config["model"],
                "data": self.config["data"],
                "training": self.config["training"],
                "cross_sectional_zscore": cs_zscore,
            },
            dataset_version="xgb_raw",
            extra_files=[self.config["data"]["price_file"]],
        )
        cache_file = cache_path("xgb_raw", cache_id)
        cached = None if self.rebuild_cache else cache_load(cache_file)
        if cached is not None:
            X_train, y_train = cached["X_train"], cached["y_train"]
            X_val, y_val = cached["X_val"], cached["y_val"]
            X_test, y_test = cached["X_test"], cached["y_test"]
            print(f"[xgb_raw] loaded splits from cache {cache_file}")
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_val,   y_val   = X[val_mask],   y[val_mask]
            X_test,  y_test  = X[test_mask],  y[test_mask]
            cache_save(cache_file, {
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test,
            })
            print(f"[xgb_raw] saved splits to cache {cache_file}")

        # Log daily coverage in test set for defendability checks
        try:
            test_dates = self.df.loc[test_mask, "date"]
            counts = test_dates.value_counts().sort_index()
            if not counts.empty:
                print(f"[xgb_raw] test tickers per day: mean={counts.mean():.1f}, min={counts.min()}, max={counts.max()}")
        except Exception:
            pass

        # -----------------------
        # 2. Read config
        # -----------------------
        model_cfg = self.config["model"]
        tuning_cfg = self.config.get("tuning", {})
        use_tuning = tuning_cfg.get("enabled", False)
        print("XGB tuning enabled:", use_tuning)
        fixed_params = model_cfg.get("params", {})
        param_grid = tuning_cfg.get("param_grid", {})

        # support ranking objectives via XGBRanker when configured
        obj = model_cfg.get("objective", "reg:squarederror")
        is_rank = isinstance(obj, str) and obj.startswith("rank")

        # -----------------------
        # 3. Model builder
        # -----------------------
        def make_model(params):
            if is_rank:
                from xgboost import XGBRanker

                return XGBRanker(
                    objective=obj,
                    tree_method="hist",
                    random_state=42,
                    **params,
                )
            else:
                return XGBRegressor(
                    objective=obj,
                    tree_method="hist",
                    random_state=42,
                    **params,
                )

        # -----------------------
        # 4. Hyperparameter logic
        # -----------------------
        if use_tuning:
            if not param_grid:
                print("No param_grid provided in config.tuning.param_grid; using fixed params only")
                param_grid = {}

            print("Starting XGB hyperparameter search from YAML param_grid")

            best_params = None
            best_rmse = float("inf")

            if not param_grid:
                candidates = [fixed_params]
            else:
                candidates = []
                keys = list(param_grid.keys())
                for values in product(*param_grid.values()):
                    overrides = dict(zip(keys, values))
                    candidates.append({**fixed_params, **overrides})

            for params in candidates:
                model = make_model(params)
                if is_rank:
                    train_group = self._groups_from_dates(self.df["date"].values, train_mask)
                    val_group = self._groups_from_dates(self.df["date"].values, val_mask)
                    model.fit(X_train, y_train, group=train_group, eval_set=[(X_val, y_val)], eval_group=[val_group], verbose=False)
                else:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                preds = model.predict(X_val)
                rmse = root_mean_squared_error(y_val, preds)

                print("Params", params, "RMSE", rmse)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params

            print("Best parameters:", best_params)

        else:
            print("Using fixed XGB parameters from YAML")
            best_params = fixed_params

        # -----------------------
        # 5. Train final model
        # -----------------------
        X_train_full = np.concatenate([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        model = make_model(best_params)
        if is_rank:
            train_group = self._groups_from_dates(self.df["date"].values, train_mask)
            val_group = self._groups_from_dates(self.df["date"].values, val_mask)
            train_group_full = list(train_group) + list(val_group)
            model.fit(
                X_train_full,
                y_train_full,
                group=train_group_full,
                eval_set=[(X_val, y_val)],
                eval_group=[val_group],
                verbose=50,
            )
        else:
            model.fit(
                X_train_full,
                y_train_full,
                eval_set=[(X_val, y_val)],
                verbose=50,
            )

        preds_test = model.predict(X_test)

        # -----------------------
        # 6. Output + backtest
        # -----------------------
        df_test = self.df.loc[test_mask, ["date", "ticker"]].copy()
        df_test["pred"] = preds_test
        df_test["realized_ret"] = y_test
        df_test.to_csv(self.out_dir / "xgb_raw_predictions.csv", index=False)

        self._evaluate_and_backtest(df_test, name="xgb_raw")



    def train_xgb_node2vec(self):
        """
        XGBoost model with Node2Vec embeddings using PyTorch Geometric.
        This avoids nodevectors and avoids old networkx, so it is stable with numpy 2.
        Note: embeddings are static structural features; they add no temporal signal.
        Expect no material improvement vs raw XGB; serves as a structural baseline.
        """
        import torch
        import networkx as nx
        from xgboost import XGBRegressor
        from torch_geometric.nn import Node2Vec

        set_seed(42)

        val_start = pd.to_datetime(self.config["training"]["val_start"])
        train_df = self.df[self.df["date"] < val_start]

        # universe metadata for sector / industry priors
        universe_path = Path("data/processed/universe.csv")
        if not universe_path.exists():
            raise FileNotFoundError("Universe metadata is required at data/processed/universe.csv")
        universe_df = pd.read_csv(universe_path)
        sector_map = dict(zip(universe_df["ticker"], universe_df.get("sector", pd.Series(index=universe_df.index))))
        industry_map = dict(zip(universe_df["ticker"], universe_df.get("industry", pd.Series(index=universe_df.index))))
        sector_weight = 0.2
        industry_weight = 0.1

        last_train_date = train_df["date"].max()
        corr_window = self.config["data"]["lookback_window"]
        corr_thr = self.config["data"].get("corr_threshold", 0.4)

        edges = rolling_corr_edges(
            train_df,
            last_train_date,
            corr_window,
            corr_thr,
        )

        graph_mode = self.config["model"].get("graph_mode", "correlation")  # "correlation" or "combined"

        # build weighted static graph combining correlation + optional sector/industry
        tickers_train = sorted(train_df["ticker"].unique())
        weight_map = {}

        def add_edge(u, v, w):
            if u == v:
                return
            key = tuple(sorted((u, v)))
            weight_map[key] = weight_map.get(key, 0.0) + float(w)

        # correlation edges
        for u, v, w in edges:
            add_edge(u, v, abs(w))

        if graph_mode == "combined":
            # sector edges (weak prior)
            sector_groups = {}
            for t in tickers_train:
                s = sector_map.get(t)
                if s is None or (isinstance(s, float) and np.isnan(s)):
                    continue
                sector_groups.setdefault(s, []).append(t)
            for _, lst in sector_groups.items():
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        add_edge(lst[i], lst[j], sector_weight)

            # industry edges (even weaker prior)
            industry_groups = {}
            for t in tickers_train:
                ind = industry_map.get(t)
                if ind is None or (isinstance(ind, float) and np.isnan(ind)):
                    continue
                industry_groups.setdefault(ind, []).append(t)
            for _, lst in industry_groups.items():
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        add_edge(lst[i], lst[j], industry_weight)

        if not weight_map:
            raise ValueError("No edges in combined graph. Check data or thresholds.")

        # normalize weights to [0,1]
        max_w = max(weight_map.values())
        norm_weights = {k: (v / max_w) for k, v in weight_map.items()}

        G = nx.Graph()
        G.add_nodes_from(tickers_train)
        for (u, v), w in norm_weights.items():
            G.add_edge(u, v, weight=w)

        edges_list = list(G.edges())
        if len(edges_list) == 0:
            raise ValueError("No edges in combined graph after normalization.")

        tickers = list(G.nodes())
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        # Node2Vec in PyG does not use edge weights directly, so we approximate
        # weights via edge multiplicity in the walk graph. Stronger edges appear
        # more times, increasing their sampling probability.
        edge_pairs = []
        for (u, v) in edges_list:
            w = G[u][v].get("weight", 1.0)
            mult = max(1, int(round(w * 10)))  # up to 10 duplicates since w <= 1
            edge_pairs.extend([(u, v)] * mult)

        edge_index = torch.tensor(
            [[ticker_to_idx[u] for u, v in edge_pairs],
             [ticker_to_idx[v] for u, v in edge_pairs]],
            dtype=torch.long,
        )

        cache_id = cache_key(
            {
                "model": "xgb_node2vec",
                "graph_mode": graph_mode,
                "emb_dim": self.config["model"].get("embedding_dim", 8),
                "data": self.config["data"],
                "training": self.config["training"],
            },
            dataset_version="xgb_node2vec",
            extra_files=[self.config["data"]["price_file"], universe_path],
        )
        emb_cache = cache_path("xgb_node2vec_emb", cache_id)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cached_emb = None if self.rebuild_cache else cache_load(emb_cache)
        if cached_emb is not None:
            emb_matrix = cached_emb["emb_matrix"]
            ticker_to_idx = cached_emb["ticker_to_idx"]
            print(f"[xgb_node2vec] loaded embeddings from cache {emb_cache}")
        else:
            emb_dim = self.config["model"].get("embedding_dim", 8)  # smaller to keep embeddings as weak priors
            n2v = Node2Vec(
                edge_index=edge_index,
                embedding_dim=emb_dim,
                walk_length=40,
                context_size=20,
                walks_per_node=10,
                num_negative_samples=1,
                sparse=True,
            ).to(device)

            loader = n2v.loader(batch_size=128, shuffle=True)
            optimizer = torch.optim.SparseAdam(n2v.parameters(), lr=0.01)

            for _ in range(30):
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
                    loss.backward()
                    optimizer.step()

            emb_matrix = n2v().detach().cpu().numpy()
            cache_save(emb_cache, {"emb_matrix": emb_matrix, "ticker_to_idx": ticker_to_idx})
            print(f"[xgb_node2vec] saved embeddings to cache {emb_cache}")

        # Standardize embeddings separately and scale them by a configurable factor
        emb_scale = self.config["model"].get("emb_scale", 0.2)
        emb_mean = emb_matrix.mean(axis=0, keepdims=True)
        emb_std = emb_matrix.std(axis=0, keepdims=True) + 1e-8
        emb_matrix = emb_scale * (emb_matrix - emb_mean) / emb_std

        # Build rows with raw features and embeddings separately so we can
        # apply per-date cross-sectional normalization to raw features.
        rows = []
        for _, row in self.df.iterrows():
            t = row["ticker"]
            # gather raw features (keep NaNs for now so we can handle them per-date)
            f = row.reindex(self.feat_cols).to_numpy(dtype=float, copy=False)
            if t in ticker_to_idx:
                vec = emb_matrix[ticker_to_idx[t]]
            else:
                # impute missing embeddings with zero vector and warn later
                vec = np.zeros(emb_matrix.shape[1], dtype=float)

            rows.append({
                "date": row["date"],
                "ticker": t,
                "target": row["target"],
                "raw_feat": f,
                "emb": vec,
            })

        tab = pd.DataFrame(rows)

        # Per-date cross-sectional z-score normalization for raw features (optional)
        cs_cfg = self.config.get("xgb", {})
        cs_zscore = bool(cs_cfg.get("cross_sectional_zscore", False))

        raw_mat = np.stack(tab["raw_feat"].values)
        dates = tab["date"].values
        if cs_zscore:
            norm_raw = np.zeros_like(raw_mat, dtype=float)
            uniq_dates, inv_idx = np.unique(dates, return_inverse=True)
            for i, d in enumerate(uniq_dates):
                idxs = np.where(inv_idx == i)[0]
                block = raw_mat[idxs].astype(float, copy=True)
                col_mean = np.nanmean(block, axis=0)
                col_std = np.nanstd(block, axis=0)
                col_std[col_std == 0] = 1.0
                inds_nan = np.isnan(block)
                if inds_nan.any():
                    block[inds_nan] = np.take(col_mean, np.where(inds_nan)[1])
                z = (block - col_mean) / (col_std + 1e-8)
                norm_raw[idxs] = z
        else:
            # minimal imputation to avoid NaNs when not z-scoring: fill with global col mean
            norm_raw = raw_mat.astype(float, copy=True)
            col_mean = np.nanmean(norm_raw, axis=0)
            inds_nan = np.isnan(norm_raw)
            if inds_nan.any():
                norm_raw[inds_nan] = np.take(col_mean, np.where(inds_nan)[1])

        # combine normalized raw features and embeddings
        emb_stack = np.stack(tab["emb"].values)

        # Debug / diagnostics: report embedding coverage and relative scale
        # Embeddings have already been standardized and scaled by `emb_scale` above.
        try:
            mean_emb_l2 = float(np.linalg.norm(emb_stack, axis=1).mean())
        except Exception:
            mean_emb_l2 = float(np.nan)

        # mean absolute entry in normalized raw features (should be ~O(1) after z-scoring)
        mean_abs_raw = float(np.nanmean(np.abs(norm_raw))) if norm_raw.size else float(np.nan)

        print(f"[xgb_node2vec] emb_scale={emb_scale}, emb_matrix.shape={emb_matrix.shape}, emb_stack.shape={emb_stack.shape}")
        print(f"[xgb_node2vec] mean emb L2 (post-scale)={mean_emb_l2:.6f}, mean abs raw feature={mean_abs_raw:.6f}")
        if not np.isnan(mean_abs_raw) and mean_abs_raw > 0:
            print(f"[xgb_node2vec] emb/raw magnitude ratio={mean_emb_l2/mean_abs_raw:.6f}")

        X = np.hstack([norm_raw, emb_stack])
        y = tab["target"].values.astype(float)

        # warn if some tickers had missing embeddings
        missing_emb_tickers = [t for t in tab["ticker"].unique() if t not in ticker_to_idx]
        if missing_emb_tickers:
            print(f"[xgb_node2vec] Warning: {len(missing_emb_tickers)} tickers missing embeddings; imputing zeros.")

        # report missing coverage per split (train/val/test) and percent missing in test
        all_tickers = sorted(self.df["ticker"].unique())
        missing_global = [t for t in all_tickers if t not in ticker_to_idx]
        if missing_global:
            print(f"[xgb_node2vec] Warning: {len(missing_global)} universe tickers have no embedding mapping (ticker_to_idx size={len(ticker_to_idx)}, universe size={len(all_tickers)})")
            print(f"[xgb_node2vec] Sample missing tickers: {missing_global[:10]}")

        # time masks are needed even when loading cached splits
        train_mask, val_mask, test_mask = _time_masks(
            tab["date"],
            self.config["training"]["val_start"],
            self.config["training"]["test_start"],
        )

        try:
            test_counts = tab.loc[test_mask].groupby("date").size()
            if not test_counts.empty:
                print(f"[xgb_node2vec] test tickers per day: mean={test_counts.mean():.1f}, min={test_counts.min()}, max={test_counts.max()}")
            # percent missing embeddings in test rows
            test_tab = tab.loc[test_mask]
            if len(test_tab) > 0:
                missing_in_test = (~test_tab["ticker"].isin(ticker_to_idx)).sum()
                pct_missing = 100.0 * missing_in_test / len(test_tab)
                print(f"[xgb_node2vec] percent missing embeddings in test rows: {pct_missing:.2f}% ({missing_in_test}/{len(test_tab)})")
        except Exception:
            pass

        # time masks are needed even when loading cached splits
        train_mask, val_mask, test_mask = _time_masks(
            tab["date"],
            self.config["training"]["val_start"],
            self.config["training"]["test_start"],
        )

        cache_id_feats = cache_key(
            {
                "model": "xgb_node2vec_features",
                "graph_mode": graph_mode,
                "data": self.config["data"],
                "training": self.config["training"],
                "cross_sectional_zscore": cs_zscore,
                "emb_dim": self.config["model"].get("embedding_dim", 8),
            },
            dataset_version="xgb_node2vec_features",
            extra_files=[self.config["data"]["price_file"], universe_path],
        )
        feat_cache = cache_path("xgb_node2vec_feats", cache_id_feats)
        cached_feats = None if self.rebuild_cache else cache_load(feat_cache)

        if cached_feats is not None:
            X_train, y_train = cached_feats["X_train"], cached_feats["y_train"]
            X_val, y_val = cached_feats["X_val"], cached_feats["y_val"]
            X_test, y_test = cached_feats["X_test"], cached_feats["y_test"]
            print(f"[xgb_node2vec] loaded feature splits from cache {feat_cache}")
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            cache_save(feat_cache, {
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test,
            })
            print(f"[xgb_node2vec] saved feature splits to cache {feat_cache}")

        # Optional tuning similar to xgb_raw
        model_cfg = self.config["model"]
        tuning_cfg = self.config.get("tuning", {})
        use_tuning = tuning_cfg.get("enabled", False)
        fixed_params = model_cfg.get("params", {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        })
        param_grid = tuning_cfg.get("param_grid", {})

        obj = model_cfg.get("objective", "reg:squarederror")
        is_rank = isinstance(obj, str) and obj.startswith("rank")

        def make_model(params):
            if is_rank:
                from xgboost import XGBRanker

                return XGBRanker(
                    objective=obj,
                    tree_method="hist",
                    random_state=42,
                    **params,
                )
            else:
                return XGBRegressor(
                    objective=obj,
                    tree_method="hist",
                    random_state=42,
                    **params,
                )

        if use_tuning and param_grid:
            print("XGB Node2Vec tuning enabled")
            best_params = None
            best_rmse = float("inf")
            keys = list(param_grid.keys())
            candidates = []
            from itertools import product
            for values in product(*param_grid.values()):
                overrides = dict(zip(keys, values))
                candidates.append({**fixed_params, **overrides})
            for params in candidates:
                mdl = make_model(params)
                if is_rank:
                    train_group = self._groups_from_dates(tab["date"].values, train_mask)
                    val_group = self._groups_from_dates(tab["date"].values, val_mask)
                    mdl.fit(X_train, y_train, group=train_group, eval_set=[(X_val, y_val)], eval_group=[val_group], verbose=False)
                else:
                    mdl.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                preds = mdl.predict(X_val)
                rmse = root_mean_squared_error(y_val, preds)
                print("Params", params, "RMSE", rmse)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
            print("Best XGB Node2Vec params:", best_params)
        else:
            best_params = fixed_params

        model = make_model(best_params)
        if is_rank:
            train_group = self._groups_from_dates(tab["date"].values, train_mask)
            val_group = self._groups_from_dates(tab["date"].values, val_mask)
            train_group_full = list(train_group) + list(val_group)
            X_train_full = np.concatenate([X_train, X_val])
            y_train_full = np.concatenate([y_train, y_val])
            model.fit(X_train_full, y_train_full, group=train_group_full, eval_set=[(X_val, y_val)], eval_group=[val_group], verbose=50)
            preds_test = model.predict(X_test)
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
            preds_test = model.predict(X_test)

        # Feature importance diagnostics: show whether embeddings contribute.
        try:
            # Build feature names: raw features then emb_0..emb_{d-1}
            raw_names = self.feat_cols
            emb_dim = emb_stack.shape[1]
            emb_names = [f"emb_{i}" for i in range(emb_dim)]
            feature_names = raw_names + emb_names

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                # fallback for some xgb wrappers
                booster = getattr(model, "get_booster", lambda: None)()
                if booster is not None:
                    fmap = booster.get_score(importance_type="weight")
                    importances = np.array([fmap.get(fn, 0.0) for fn in feature_names], dtype=float)
                else:
                    importances = None

            if importances is not None:
                fi = list(zip(feature_names, importances))
                fi_sorted = sorted(fi, key=lambda x: x[1], reverse=True)
                print("[xgb_node2vec] Top feature importances:")
                for n, v in fi_sorted[:20]:
                    print(f"  {n}: {v:.6f}")
                # save to CSV
                try:
                    import pandas as _pd
                    _pd.DataFrame(fi_sorted, columns=["feature", "importance"]).to_csv(self.out_dir / "xgb_node2vec_feature_importances.csv", index=False)
                except Exception:
                    pass
        except Exception:
            pass

        tab_test = tab.loc[test_mask, ["date", "ticker"]].copy()
        tab_test["pred"] = preds_test
        tab_test["realized_ret"] = y_test
        tab_test.to_csv(self.out_dir / "xgb_node2vec_predictions.csv", index=False)

        self._evaluate_and_backtest(
            tab_test,
            name="xgb_node2vec",
        )

    def train_graphlasso_linear(self):
        from sklearn.linear_model import Ridge

        set_seed(42)

        val_start = self.config["training"]["val_start"]
        # `graphical_lasso_precision` treats end_date as exclusive, so pass the
        # validation start directly and let the function exclude it.
        _, _, _, adj = graphical_lasso_precision(
            self.df,
            start_date=self.config["data"]["start_date"],
            end_date=val_start,
            alpha=self.config.get("graphlasso_alpha", 0.01),
        )

        df_smooth, smooth_cols = _add_graph_smooth_features(self.df, self.feat_cols, adj, alpha=0.5)

        X = df_smooth[self.feat_cols + smooth_cols].values.astype(float)
        y = df_smooth["target"].values.astype(float)

        train_mask, val_mask, test_mask = _time_masks(
            df_smooth["date"],
            self.config["training"]["val_start"],
            self.config["training"]["test_start"],
        )

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        preds_test = model.predict(X_test)

        df_test = df_smooth.loc[test_mask, ["date", "ticker"]].copy()
        df_test["pred"] = preds_test
        df_test["realized_ret"] = y_test
        df_test.to_csv(self.out_dir / "graphlasso_linear_predictions.csv", index=False)

        self._evaluate_and_backtest(
            df_test,
            name="graphlasso_linear",
        )

    def train_graphlasso_xgb(self):
        """
        Graphical Lasso + XGBoost model.

        Steps:
          1) Fit Graphical Lasso on log returns only.
          2) Build adjacency graph.
          3) Smooth technical features using neighbors.
          4) Train XGBoost on [raw + smoothed] features.
          5) Evaluate IC, hit, long-short.
        """
        from xgboost import XGBRegressor
        set_seed(42)

        returns_df = self.df[["date", "ticker", "log_ret_1d"]].copy()

        val_start = self.config["training"]["val_start"]
        # `graphical_lasso_precision` treats end_date as exclusive, so pass the
        # validation start directly and let the function exclude it.
        _, _, _, adj = graphical_lasso_precision(
            returns_df,
            start_date=self.config["data"]["start_date"],
            end_date=val_start,
            alpha=self.config.get("graphlasso_alpha", 0.01),
        )

        df_smooth, smooth_cols = _add_graph_smooth_features(self.df, self.feat_cols, adj, alpha=0.5)

        X = df_smooth[self.feat_cols + smooth_cols].values.astype(float)
        y = df_smooth["target"].values.astype(float)

        train_mask, val_mask, test_mask = _time_masks(
            df_smooth["date"],
            self.config["training"]["val_start"],
            self.config["training"]["test_start"],
        )

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        preds_test = model.predict(X_test)

        df_test = df_smooth.loc[test_mask, ["date", "ticker"]].copy()
        df_test["pred"] = preds_test
        df_test["realized_ret"] = y_test
        df_test.to_csv(self.out_dir / "graphlasso_xgb_predictions.csv", index=False)

        self._evaluate_and_backtest(
            df_test,
            name="graphlasso_xgb",
        )

    def train_granger_xgb(self):
        from xgboost import XGBRegressor
        set_seed(42)

        returns_df = self.df[["date", "ticker", "log_ret_1d"]].copy()

        val_start = self.config["training"]["val_start"]
        mask = returns_df["date"] < pd.to_datetime(val_start)

        # allow configuring Granger params from YAML
        granger_cfg = self.config.get("granger", {})
        max_lag = int(granger_cfg.get("max_lag", 2))
        p_threshold = float(granger_cfg.get("p_threshold", 0.05))

        edges = granger_edges(returns_df.loc[mask], max_lag=max_lag, p_threshold=p_threshold)

        # log top-k edges for quick sanity check
        if edges:
            top_k = min(10, len(edges))
            sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_k]
            print(f"[granger_xgb] Top {top_k} Granger edges (u,v,weight):")
            for u, v, w in sorted_edges:
                print(f"  {u} -> {v}: {w:.4f}")

        # Build adjacency for smoothing. granger_edges now returns weighted directed edges.
        # We'll symmetrize and row-normalize weights so smoothing is stable.
        adj = {}
        for u, v, w in edges:
            if u == v:
                continue
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, []).append((u, w))

        # If edges exist, row-normalize neighbor weights to sum to 1 per node.
        if adj:
            for node, nbrs in list(adj.items()):
                # sum weights, avoid zero
                total = sum(abs(w) for _, w in nbrs) + 1e-12
                adj[node] = [(nb, float(w) / total) for nb, w in nbrs]
        else:
            # fallback: use sector/industry priors when Granger finds no edges
            print("[granger_xgb] Warning: no Granger edges found; falling back to sector/industry priors.")
            universe_path = Path("data/processed/universe.csv")
            if universe_path.exists():
                universe_df = pd.read_csv(universe_path)
                sector_map = dict(zip(universe_df["ticker"], universe_df.get("sector", pd.Series(index=universe_df.index))))
                industry_map = dict(zip(universe_df["ticker"], universe_df.get("industry", pd.Series(index=universe_df.index))))
                # build weak priors
                universe_list = universe_df["ticker"].unique().tolist()
                for i, ti in enumerate(universe_list):
                    for j, tj in enumerate(universe_list):
                        if i == j:
                            continue
                        # sector adjacency
                        si = sector_map.get(ti)
                        sj = sector_map.get(tj)
                        if si is not None and sj is not None and si == sj:
                            adj.setdefault(ti, []).append((tj, 0.2))
                        # industry adjacency
                        ii = industry_map.get(ti)
                        ij = industry_map.get(tj)
                        if ii is not None and ij is not None and ii == ij:
                            adj.setdefault(ti, []).append((tj, 0.1))
            # normalize fallback
            for node, nbrs in list(adj.items()):
                total = sum(abs(w) for _, w in nbrs) + 1e-12
                adj[node] = [(nb, float(w) / total) for nb, w in nbrs]

        df_smooth, smooth_cols = _add_graph_smooth_features(self.df, self.feat_cols, adj, alpha=self.config.get("graph_smooth_alpha", 0.5))

        X = df_smooth[self.feat_cols + smooth_cols].values
        y = df_smooth["target"].values

        train_mask, val_mask, test_mask = _time_masks(
            df_smooth["date"],
            self.config["training"]["val_start"],
            self.config["training"]["test_start"],
        )

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        preds_test = model.predict(X_test)

        df_test = df_smooth.loc[test_mask, ["date", "ticker"]].copy()
        df_test["pred"] = preds_test
        df_test["realized_ret"] = y_test
        df_test.to_csv(self.out_dir / "granger_xgb_predictions.csv", index=False)

        self._evaluate_and_backtest(
            df_test,
            name="granger_xgb",
        )

    


def train_xgboost(config):
    """Entry point used by train.py."""
    trainer = XGBoostTrainer(config)
    trainer.run()
