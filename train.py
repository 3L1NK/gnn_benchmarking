import argparse
import yaml

from trainers.train_gnn import train_gnn
from trainers.train_lstm import train_lstm
from trainers.train_baseline import train_baseline


def main(config):
    model_type = config["model"]["family"]

    if model_type == "gnn":
        train_gnn(config)
    elif model_type == "lstm":
        train_lstm(config)
    elif model_type == "baseline":
        train_baseline(config)
    else:
        raise ValueError(f"Unknown model family {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
