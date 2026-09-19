"""
Step 16.x — PPO experiment seeding.

One canonical seed entry point for Python, NumPy and PyTorch.
Environment/scenario/QPSO components should receive the same experiment seed
through their own explicit APIs rather than generating private seeds.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic_torch: bool = True,
) -> int:
    """
    Seed all process-level RNGs used by PPO.

    Returns the normalized integer seed so callers can pass it onward to
    environment/scenario/QPSO owners.
    """
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer, not bool")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        # cuBLAS reproducibility on supported CUDA versions.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    return seed


def make_component_seed(
    experiment_seed: int,
    component: str,
) -> int:
    """
    Derive a stable component seed without Python's randomized hash().

    This is useful when independent RNG streams are required for scenario,
    environment, QPSO, evaluation, etc.
    """
    if not isinstance(component, str) or not component.strip():
        raise ValueError("component must be a non-empty string")

    # Stable 32-bit FNV-1a derivation.
    h = 2166136261
    for byte in component.strip().encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF

    seed = int(experiment_seed)
    if seed < 0:
        raise ValueError("experiment_seed must be non-negative")

    return (seed ^ h) & 0xFFFFFFFF
