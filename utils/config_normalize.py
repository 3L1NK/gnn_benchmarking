from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config_aliases import resolve_config_alias

DEFAULT_PROTOCOL_VERSION = "v1_thesis_core"
DEFAULT_BACKTEST_POLICIES = [1, 5, 21]
DEFAULT_PRIMARY_REBALANCE_FREQ = 1


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_existing_or_alias(path: Path, project_root: Path) -> Optional[Path]:
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        alias = resolve_config_alias(rel)
    except Exception:
        alias = resolve_config_alias(path)
    if alias is not None:
        alias_path = (project_root / alias).resolve()
        if alias_path.exists():
            return alias_path
    if path.exists():
        return path.resolve()
    return None


def _resolve_config_path(config_path: str | Path, project_root: Path) -> Path:
    config_path = Path(config_path)

    if config_path.is_absolute():
        resolved = _resolve_existing_or_alias(config_path, project_root)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"Config '{config_path}' does not exist")

    direct = _resolve_existing_or_alias(config_path, project_root)
    if direct is not None:
        return direct

    project_candidate = _resolve_existing_or_alias((project_root / config_path), project_root)
    if project_candidate is not None:
        return project_candidate

    alias = resolve_config_alias(config_path)
    if alias is not None:
        alias_path = (project_root / alias).resolve()
        if alias_path.exists():
            return alias_path

    raise FileNotFoundError(f"Config '{config_path}' does not exist")


def _resolve_include_path(config_path: Path, include_path: str | Path, project_root: Path) -> Path:
    include_path = Path(include_path)
    if include_path.is_absolute():
        resolved = _resolve_existing_or_alias(include_path, project_root)
        return resolved if resolved is not None else include_path

    candidate = _resolve_existing_or_alias((config_path.parent / include_path), project_root)
    if candidate is not None:
        return candidate

    project_candidate = _resolve_existing_or_alias((project_root / include_path), project_root)
    if project_candidate is not None:
        return project_candidate

    alias = resolve_config_alias(include_path)
    if alias is not None:
        alias_path = (project_root / alias).resolve()
        if alias_path.exists():
            return alias_path

    return (project_root / include_path).resolve()


def _translate_legacy_graph_edges(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "graph" in cfg or "graph_edges" not in cfg:
        return cfg

    ge = cfg.get("graph_edges", {}) or {}
    graph: Dict[str, Any] = {}
    graph["use_corr"] = bool(ge.get("use_correlation", ge.get("use_corr", False)))
    graph["use_sector"] = bool(ge.get("use_sector", False))
    if ge.get("use_industry", False):
        graph["use_sector"] = True
    graph["use_granger"] = bool(ge.get("use_granger", cfg.get("granger", {}).get("enabled", False)))

    if "corr_top_k" in ge:
        graph["corr_top_k"] = int(ge.get("corr_top_k", 10))
    if "corr_min_periods" in ge:
        graph["corr_min_periods"] = int(ge.get("corr_min_periods", 0))

    if "sector_weight" in ge:
        graph["sector_weight"] = float(ge.get("sector_weight", 0.2))
    if "industry_weight" in ge:
        graph["industry_weight"] = float(ge.get("industry_weight", 0.1))

    if "w_corr" in ge:
        graph["w_corr"] = float(ge.get("w_corr", 1.0))
    if "w_sector" in ge:
        graph["w_sector"] = float(ge.get("w_sector", 0.2))
    if "w_granger" in ge:
        graph["w_granger"] = float(ge.get("w_granger", 0.2))
    if "max_edge_weight" in ge:
        graph["max_edge_weight"] = float(ge.get("max_edge_weight", 1.0))
    if "make_undirected" in ge:
        graph["make_undirected"] = bool(ge.get("make_undirected"))
    if "sector_top_k" in ge:
        graph["sector_top_k"] = int(ge.get("sector_top_k", 5))
    if "granger_top_k" in ge:
        graph["granger_top_k"] = int(ge.get("granger_top_k", 5))

    cfg["graph"] = graph
    return cfg


def _normalize_backtest_policies(cfg: Dict[str, Any]) -> Dict[str, Any]:
    eval_cfg = cfg.setdefault("evaluation", {})
    policies_raw = eval_cfg.get("backtest_policies", DEFAULT_BACKTEST_POLICIES)
    if isinstance(policies_raw, (int, float)):
        policies = [int(policies_raw)]
    else:
        policies = [int(v) for v in list(policies_raw)]
    policies = [p for p in policies if p >= 1]
    if not policies:
        policies = list(DEFAULT_BACKTEST_POLICIES)
    seen = set()
    normalized = []
    for p in policies:
        if p not in seen:
            normalized.append(p)
            seen.add(p)
    eval_cfg["backtest_policies"] = normalized

    primary = int(eval_cfg.get("primary_rebalance_freq", DEFAULT_PRIMARY_REBALANCE_FREQ))
    if primary < 1:
        primary = DEFAULT_PRIMARY_REBALANCE_FREQ
    if primary not in normalized:
        primary = normalized[0]
    eval_cfg["primary_rebalance_freq"] = primary
    return cfg


def _normalize_granger_aliases(cfg: Dict[str, Any]) -> Dict[str, Any]:
    graph_cfg = cfg.setdefault("graph", {})
    gr_cfg = cfg.setdefault("granger", {})

    # Graph-level aliases used in historical run configs.
    if "granger_topk" in graph_cfg and "granger_top_k" not in graph_cfg:
        graph_cfg["granger_top_k"] = int(graph_cfg.get("granger_topk", 5))
    if "granger_top_k" in graph_cfg:
        graph_cfg["granger_top_k"] = int(graph_cfg.get("granger_top_k", 5))

    # Legacy graph fields for Granger test configuration -> canonical granger.*.
    if "granger_maxlag" in graph_cfg and "max_lag" not in gr_cfg:
        gr_cfg["max_lag"] = int(graph_cfg.get("granger_maxlag", 2))
    if "granger_alpha" in graph_cfg and "p_threshold" not in gr_cfg:
        gr_cfg["p_threshold"] = float(graph_cfg.get("granger_alpha", 0.05))

    if "max_lag" in gr_cfg:
        gr_cfg["max_lag"] = int(gr_cfg.get("max_lag", 2))
    if "p_threshold" in gr_cfg:
        gr_cfg["p_threshold"] = float(gr_cfg.get("p_threshold", 0.05))
    return cfg


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(cfg or {})
    cfg = _translate_legacy_graph_edges(cfg)
    cfg = _normalize_granger_aliases(cfg)

    data_cfg = cfg.setdefault("data", {})
    data_cfg.setdefault("target_horizon", 1)
    data_cfg.setdefault("target_type", "regression")
    data_cfg.setdefault("target_name", "log_return")

    training_cfg = cfg.setdefault("training", {})
    training_cfg.setdefault("val_start", "2016-01-01")
    training_cfg.setdefault("test_start", "2020-01-01")

    eval_cfg = cfg.setdefault("evaluation", {})
    eval_cfg.setdefault("transaction_cost_bps", 0)
    eval_cfg.setdefault("out_dir", "results/runs/")
    eval_cfg.setdefault("results_path", "results/results.jsonl")
    cfg = _normalize_backtest_policies(cfg)

    exp_cfg = cfg.setdefault("experiment", {})
    exp_cfg.setdefault("protocol_version", DEFAULT_PROTOCOL_VERSION)
    # Keep compatibility for non-thesis legacy configs unless explicitly enabled.
    exp_cfg.setdefault("enforce_protocol", False)
    return cfg


def load_config(config_path: str | Path, project_root: str | Path) -> Dict[str, Any]:
    project_root = Path(project_root)
    config_path = _resolve_config_path(config_path, project_root)

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    include_path = cfg.pop("include", None)
    if include_path:
        include_list = include_path if isinstance(include_path, list) else [include_path]
        base: Dict[str, Any] = {}
        for inc in include_list:
            resolved = _resolve_include_path(config_path, inc, project_root)
            if not resolved.exists():
                raise FileNotFoundError(f"Included config '{resolved}' does not exist")
            base = _deep_update(base, deepcopy(load_config(resolved, project_root)))
        cfg = _deep_update(base, cfg)

    return normalize_config(cfg)
