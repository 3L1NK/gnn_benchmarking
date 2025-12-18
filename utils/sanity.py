import torch
import numpy as np


def check_tensor(name, t):
    if t is None:
        return []
    issues = []
    if torch.isnan(t).any() or torch.isinf(t).any():
        issues.append(f"{name}: contains NaN or Inf")
    if t.numel() > 0 and float(torch.var(t.float())) < 1e-10:
        issues.append(f"{name}: near-zero variance")
    if t.ndim >= 2 and (t.abs().sum(dim=-1) == 0).any():
        issues.append(f"{name}: contains all-zero feature rows")
    return issues


def check_numpy(name, arr):
    arr = np.asarray(arr)
    issues = []
    if np.isnan(arr).any() or np.isinf(arr).any():
        issues.append(f"{name}: contains NaN or Inf")
    if arr.size > 0 and np.var(arr.astype("float64")) < 1e-10:
        issues.append(f"{name}: near-zero variance")
    return issues
