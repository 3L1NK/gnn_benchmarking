import argparse
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

#from trainers.train_gnn import train_gnn
#from trainers.train_lstm import train_lstm
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
        return cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)
