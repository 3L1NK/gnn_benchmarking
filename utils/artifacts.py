from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OutputDirs:
    canonical: Path
    legacy: Optional[Path]


def resolve_output_dirs(config: dict, *, model_type: str | None = None) -> OutputDirs:
    eval_out = Path(config.get("evaluation", {}).get("out_dir", "experiments/"))
    family = str(config.get("model", {}).get("family", "")).lower()
    model_type = model_type or str(config.get("model", {}).get("type", "model"))

    canonical = eval_out
    legacy: Optional[Path] = None
    if family == "gnn":
        candidate = eval_out / model_type
        if candidate != canonical:
            legacy = candidate

    canonical.mkdir(parents=True, exist_ok=True)
    if legacy is not None:
        legacy.mkdir(parents=True, exist_ok=True)
    return OutputDirs(canonical=canonical, legacy=legacy)
