from __future__ import annotations

from typing import Callable, Dict

from trainers.train_gnn import train_gnn
from trainers.train_lstm import train_lstm
from trainers.train_xgboost import train_xgboost


_FAMILY_REGISTRY: Dict[str, Callable[[dict], None]] = {
    "gnn": train_gnn,
    "lstm": train_lstm,
    "xgboost": train_xgboost,
}


def get_family_runner(model_family: str) -> Callable[[dict], None]:
    key = str(model_family).lower()
    if key not in _FAMILY_REGISTRY:
        raise ValueError(f"Unknown model family '{model_family}'. Available: {sorted(_FAMILY_REGISTRY)}")
    return _FAMILY_REGISTRY[key]


def run_model(config: dict) -> None:
    family = config.get("model", {}).get("family", "")
    runner = get_family_runner(family)
    runner(config)
