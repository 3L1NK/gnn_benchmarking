from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import re

import yaml

from .protocol import protocol_from_config
from .results import split_id_from_fields


@dataclass(frozen=True)
class OutputDirs:
    canonical: Path
    legacy: Optional[Path]
    run_id: str
    root: Path


def _slug(value: object) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = re.sub(r"-+", "-", raw).strip("-")
    return raw or "na"


def _infer_edge_type(config: dict) -> str:
    model_type = str(config.get("model", {}).get("type", "")).lower()
    graph_cfg = config.get("graph", {}) or {}
    if model_type == "xgb_node2vec":
        return "node2vec_correlation"
    if "graphlasso" in model_type:
        return "graphlasso"
    if model_type == "granger_xgb":
        return "granger"
    parts = []
    if bool(graph_cfg.get("use_corr", False)):
        parts.append("corr")
    if bool(graph_cfg.get("use_sector", False)):
        parts.append("sector")
    if bool(graph_cfg.get("use_granger", False)):
        parts.append("granger")
    return "+".join(parts) if parts else "none"


def build_stable_run_id(config: dict, *, model_type: str | None = None) -> str:
    family = str(config.get("model", {}).get("family", "")).lower()
    model_type = model_type or str(config.get("model", {}).get("type", "model"))
    data_cfg = config.get("data", {}) or {}
    tune_cfg = config.get("tuning", {}) or {}
    seed = int(config.get("seed", 42))
    edge_type = _infer_edge_type(config)
    corr_window = data_cfg.get("corr_window", "na")
    lookback = data_cfg.get("lookback_window", "na")

    protocol = protocol_from_config(config, baseline_version="")
    split_id = split_id_from_fields(
        protocol.protocol_version,
        protocol.split_train_end,
        protocol.split_val_start,
        protocol.split_test_start,
        protocol.target_policy_hash,
    )
    tuning_enabled = bool(tune_cfg.get("enabled", False))
    tuning_profile = str(tune_cfg.get("profile", "")).strip()
    tuning_tag = "tuned" if tuning_enabled else "fixed"
    if tuning_profile:
        tuning_tag = f"{tuning_tag}-{tuning_profile}"

    return (
        f"m-{_slug(model_type)}"
        f"__f-{_slug(family)}"
        f"__e-{_slug(edge_type)}"
        f"__cw-{_slug(corr_window)}"
        f"__lb-{_slug(lookback)}"
        f"__sp-{_slug(split_id)}"
        f"__sd-{seed}"
        f"__tg-{_slug(tuning_tag)}"
    )


def _write_run_metadata(run_root: Path, config: dict, run_id: str) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    cfg_path = run_root / "resolved_config.yaml"
    manifest_path = run_root / "run_manifest.json"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest_payload = {"run_id": run_id, "run_root": str(run_root)}
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")


def resolve_output_dirs(config: dict, *, model_type: str | None = None) -> OutputDirs:
    # Freeze all new writes to a single root to avoid duplicate artifact trees.
    run_id = build_stable_run_id(config, model_type=model_type)
    root = Path("results") / "runs" / run_id
    _write_run_metadata(root, config, run_id=run_id)

    canonical = root
    legacy: Optional[Path] = None

    canonical.mkdir(parents=True, exist_ok=True)
    if legacy is not None:
        legacy.mkdir(parents=True, exist_ok=True)
    return OutputDirs(canonical=canonical, legacy=legacy, run_id=run_id, root=root)
