import torch
import os


def get_device(prefer="cuda"):
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer)
    return torch.device("cpu")


def default_num_workers():
    return min(8, os.cpu_count() or 0)
