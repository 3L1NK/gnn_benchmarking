import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from models.registry import run_model
from utils.config_normalize import load_config
from utils.protocol import assert_canonical_protocol


def main(config):
    run_model(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rebuild-cache", action="store_true", help="Force recomputation of cached artifacts")
    args = parser.parse_args()

    config = load_config(args.config, PROJECT_ROOT)
    cache_cfg = dict(config.get("cache", {}))
    cache_cfg["rebuild"] = bool(args.rebuild_cache)
    config["cache"] = cache_cfg
    assert_canonical_protocol(config)

    main(config)
