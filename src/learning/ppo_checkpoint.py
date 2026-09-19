"""
Step 16.x — PPO reproducible checkpoint manager.

Captures the state needed to resume PPO without silently changing the
optimization trajectory:

- actor-critic parameters
- optimizer state
- trainer minibatch RNG
- Python random state
- NumPy global RNG state
- PyTorch CPU RNG state
- CUDA RNG states when CUDA is available
- experiment metadata/configuration supplied by the caller

The checkpoint manager does not know about SUMO, QPSO, routing, or forecasting.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    """Capture process-level RNG state."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore process-level RNG state captured by capture_rng_state()."""
    if not isinstance(state, dict):
        raise TypeError("rng state must be a dictionary")

    required = {"python", "numpy", "torch_cpu"}
    missing = required - state.keys()
    if missing:
        raise ValueError(
            f"rng state missing required entries: {sorted(missing)}"
        )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class PPOCheckpointManager:
    """Save/load complete PPO experiment state."""

    def save(
        self,
        path: str | Path,
        *,
        model: torch.nn.Module,
        trainer,
        update_index: int,
        env_steps: int,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if update_index < 0:
            raise ValueError("update_index must be non-negative")
        if env_steps < 0:
            raise ValueError("env_steps must be non-negative")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": copy.deepcopy(model.state_dict()),
            "trainer": trainer.training_state_dict(),
            "rng": capture_rng_state(),
            "update_index": int(update_index),
            "env_steps": int(env_steps),
            "metadata": dict(metadata or {}),
        }

        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(target)
        return target

    def load(
        self,
        path: str | Path,
        *,
        model: torch.nn.Module,
        trainer,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)

        payload = torch.load(
            source,
            map_location=map_location,
            weights_only=False,
        )

        if not isinstance(payload, dict):
            raise ValueError("invalid PPO checkpoint payload")

        version = payload.get("checkpoint_version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version: {version}"
            )

        for key in ("model", "trainer", "rng", "update_index", "env_steps"):
            if key not in payload:
                raise ValueError(
                    f"checkpoint missing required field: {key}"
                )

        model.load_state_dict(payload["model"])
        trainer.load_training_state_dict(payload["trainer"])

        if restore_rng:
            restore_rng_state(payload["rng"])

        return {
            "update_index": int(payload["update_index"]),
            "env_steps": int(payload["env_steps"]),
            "metadata": dict(payload.get("metadata", {})),
        }
