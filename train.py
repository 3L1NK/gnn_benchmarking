import argparse
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

from trainers.train_gnn import train_gnn
from trainers.train_lstm import train_lstm
from trainers.train_xgboost import train_xgboost


def main(config):
    model_type = config["model"]["family"]

    if model_type == "gnn":
        train_gnn(config)
    elif model_type == "lstm":
        train_lstm(config)
    elif model_type == "xgboost":
        train_xgboost(config)
    else:
        raise ValueError(f"Unknown model family {model_type}")


if __name__ == "__main__":
    def _deep_update(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = _deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def load_config(config_path):
        config_path = Path(config_path)
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}

        include_path = cfg.pop("include", None)
        if include_path:
            include_path = Path(include_path)
            if not include_path.is_absolute():
                candidate = (config_path.parent / include_path).resolve()
                if candidate.exists():
                    include_path = candidate
                else:
                    include_path = (PROJECT_ROOT / include_path).resolve()
            if not include_path.exists():
                raise FileNotFoundError(f"Included config '{include_path}' does not exist")
            base = deepcopy(load_config(include_path))
            return _deep_update(base, cfg)

        # Backwards compatibility: translate old `graph_edges` section into
        # the new `graph` section (explicit separation of model vs graph config).
        # This allows older YAMLs to keep working while encouraging the new layout.
        if "graph" not in cfg and "graph_edges" in cfg:
            ge = cfg.get("graph_edges", {}) or {}
            graph = {}
            # booleans: prefer explicit names if present
            graph["use_corr"] = bool(ge.get("use_correlation", ge.get("use_corr", False)))
            graph["use_sector"] = bool(ge.get("use_sector", False))
            # If a legacy `use_industry` or `use_sector` was used, keep sector toggle true
            if ge.get("use_industry", False):
                graph["use_sector"] = True
            # If a granger config existed, map to use_granger
            graph["use_granger"] = bool(cfg.get("granger", {}).get("enabled", False))

            # correlation params mapping
            if "corr_top_k" in ge:
                graph["corr_top_k"] = int(ge.get("corr_top_k", 10))
            if "corr_min_periods" in ge:
                graph["corr_min_periods"] = int(ge.get("corr_min_periods", 0))

            # sector / industry weights
            if "sector_weight" in ge:
                graph["sector_weight"] = float(ge.get("sector_weight", 0.2))
            if "industry_weight" in ge:
                graph["industry_weight"] = float(ge.get("industry_weight", 0.1))

            cfg["graph"] = graph

        return cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rebuild-cache", action="store_true", help="Force recomputation of cached artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    config["cache"] = {"rebuild": args.rebuild_cache}

    main(config)
