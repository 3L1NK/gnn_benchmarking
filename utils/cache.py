from pathlib import Path
import hashlib
import json
import os
import time
import pickle
import numpy as np
import torch
from joblib import dump, load


def cache_dir():
    path = Path("data/processed/cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_key(config, dataset_version="default", extra_files=None):
    """
    Build a stable key from config, dataset version, and input file mtimes.
    extra_files: list of Paths to include (e.g., price/universe files).
    """
    extra_files = extra_files or []
    meta = {
        "dataset_version": dataset_version,
        "config": config,
        "files": {},
    }
    for p in extra_files:
        p = Path(p)
        try:
            stat = p.stat()
            meta["files"][str(p)] = {"mtime": stat.st_mtime, "size": stat.st_size}
        except FileNotFoundError:
            meta["files"][str(p)] = {"missing": True}
    raw = json.dumps(meta, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def cache_path(name, key):
    return cache_dir() / f"{name}_{key}.pt"


def cache_load(path):
    path = Path(path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        # We store arbitrary Python objects (not just weights), so allow full pickle.
        return torch.load(path, map_location="cpu", weights_only=False)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".joblib":
        return load(path)
    # fallback
    with path.open("rb") as f:
        return pickle.load(f)


def cache_save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        torch.save(obj, path)
    elif suffix == ".npy":
        np.save(path, obj)
    elif suffix == ".joblib":
        dump(obj, path)
    else:
        with path.open("wb") as f:
            pickle.dump(obj, f)
