"""Reproducibility and device helpers.

Seeding a GNN is slightly more involved than seeding an MLP: dropout on
attention coefficients, the random split in :func:`gnn_lab.data.cora.load_cora_linqs`,
and CUDA's non-deterministic scatter kernels all contribute. ``set_seed`` covers
the first two; for bitwise-reproducible scatter reductions you additionally need
``torch.use_deterministic_algorithms(True)``, which is slower and is therefore
left opt-in.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import torch

__all__ = ["set_seed", "get_device", "count_parameters", "Timer", "early_stopping"]


def set_seed(seed: int = 42, deterministic_cuda: bool = False) -> None:
    """Seed python, numpy (if present) and torch, CPU and CUDA."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a device. ``prefer`` in {"auto", "cpu", "cuda"}."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


class Timer:
    """Context manager + attribute for elapsed wall-clock seconds."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        import time
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time
        self.elapsed = time.perf_counter() - self._t
        return False


def early_stopping(patience: int = 50, mode: str = "max"):
    """Return ``(should_stop, best)`` updater for a monitored metric.

    Used by the graph-level trainer, where 188 MUTAG graphs overfit in a handful
    of epochs. Node classification on Cora is instead run for a fixed number of
    epochs, matching the published protocol -- so the two trainers differ
    deliberately.
    """
    best = -float("inf") if mode == "max" else float("inf")
    bad = 0

    def step(value: float):
        nonlocal best, bad
        improved = value > best if mode == "max" else value < best
        if improved:
            best, bad = value, 0
        else:
            bad += 1
        return bad >= patience, best

    return step
