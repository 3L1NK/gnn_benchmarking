# trainers/train_xgboost.py
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loading import load_price_panel
from utils.features import add_technical_features
from utils.graphs import rolling_corr_edges, graphical_lasso_precision, granger_edges
from utils.metrics import rank_ic, hit_rate
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
    horizon = config["data"]["target_horizon"]

    df = load_price_panel(price_file, start, end)
    feat_cols = [c for c in default_feat_cols if c in df.columns]

    # Fallback: if the parquet is raw prices without features, compute them here.
    if not feat_cols:
        df, feat_cols = add_technical_features(df)
        feat_cols = [c for c in default_feat_cols if c in df.columns]

    df["target"] = df.groupby("ticker")["log_ret_1d"].shift(-horizon)
    df = df.dropna(subset=["target"])

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

    def run(self):
        key = self.config["model"]["type"].lower()

        # global buy-and-hold baseline (cached, model independent)
        eq_bh_full, ret_bh_full, stats_bh = get_global_buy_and_hold(
            self.config,
            rebuild=self.rebuild_cache,
        )
        # align to test window for fair comparison
        test_start = pd.to_datetime(self.config["training"]["test_start"])
        mask_bh = eq_bh_full.index >= test_start
        self.bh_curve = eq_bh_full
        self.bh_ret = ret_bh_full
        from utils.metrics import sharpe_ratio, sortino_ratio
        stats_bh_window = {
            "final_value": float(eq_bh_full.loc[mask_bh].iloc[-1]) if mask_bh.any() else float("nan"),
            "sharpe": sharpe_ratio(ret_bh_full[mask_bh], self.config["evaluation"]["risk_free_rate"]),
            "sortino": sortino_ratio(ret_bh_full[mask_bh], self.config["evaluation"]["risk_free_rate"]),
        }
        self.bh_stats = stats_bh_window
        print("[baseline] global buy-and-hold stats (test window)", stats_bh_window)

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
        daily_metrics.to_csv(self.out_dir / f"{name}_daily_metrics.csv", index=False)
        
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

        X = self.df[self.feat_cols].values.astype(float)
        y = self.df["target"].values.astype(float)

        cache_id = cache_key(
            {
                "model": "xgb_raw",
                "params": self.config["model"],
                "data": self.config["data"],
                "training": self.config["training"],
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

        # -----------------------
        # 2. Read config
        # -----------------------
        model_cfg = self.config["model"]
        tuning_cfg = self.config.get("tuning", {})
        use_tuning = tuning_cfg.get("enabled", False)
        print("XGB tuning enabled:", use_tuning)
        fixed_params = model_cfg.get("params", {})
        param_grid = tuning_cfg.get("param_grid", {})

        # -----------------------
        # 3. Model builder
        # -----------------------
        def make_model(params):
            return XGBRegressor(
                objective="reg:squarederror",
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

        # Standardize embeddings separately and scale down so they act as weak structural priors.
        emb_mean = emb_matrix.mean(axis=0, keepdims=True)
        emb_std = emb_matrix.std(axis=0, keepdims=True) + 1e-8
        emb_matrix = 0.2 * (emb_matrix - emb_mean) / emb_std

        rows = []
        for _, row in self.df.iterrows():
            t = row["ticker"]
            if t not in ticker_to_idx:
                continue

            available_cols = [c for c in self.feat_cols if c in row.index]
            if not available_cols:
                continue
            f = row[available_cols].values.astype(float)
            vec = emb_matrix[ticker_to_idx[t]]
            full_feat = np.concatenate([f, vec])

            rows.append({
                "date": row["date"],
                "ticker": t,
                "target": row["target"],
                "features": full_feat,
            })

        tab = pd.DataFrame(rows)

        X = np.stack(tab["features"].values)
        y = tab["target"].values.astype(float)

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
        edges = granger_edges(returns_df.loc[mask], max_lag=2, p_threshold=0.05)

        adj = {}
        for u, v, w in edges:
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, [])

        df_smooth, smooth_cols = _add_graph_smooth_features(self.df, self.feat_cols, adj, alpha=0.5)

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
