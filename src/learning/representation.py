"""
Step 16.12 — Model-independent PPO representation contract.

This module defines the boundary between the traffic-state representation
pipeline and PPO.

The PPO controller must NOT depend on:
    - a particular GNN architecture
    - a particular Transformer architecture
    - a particular hidden dimension
    - a particular forecasting checkpoint

Instead, an upstream model/adapter produces a fixed representation contract.

Architecture
------------

Raw causal state
    |
    +--> GNN / spatial encoder
    |
    +--> temporal / forecast encoder
    |
    +--> event / service / decision summaries
    |
    v
RepresentationProvider
    |
    v
RepresentationOutput
    |
    +--> PPO actor
    +--> PPO critic

A later GNN/Transformer can change internally as long as its adapter continues
to satisfy this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


# ============================================================================
# Representation specification
# ============================================================================


@dataclass(frozen=True)
class RepresentationSpec:
    """
    Versioned dimensional contract consumed by PPO.

    These dimensions are interface dimensions, NOT GNN/Transformer hidden
    dimensions.

    An upstream model may have any internal architecture. Its adapter is
    responsible for mapping that model's outputs into these dimensions.
    """

    version: str = "ppo-representation-v1"

    spatial_dim: int = 128
    temporal_dim: int = 128
    forecast_dim: int = 12
    uncertainty_dim: int = 6
    context_dim: int = 58

    def __post_init__(self) -> None:
        for name, value in (
            ("spatial_dim", self.spatial_dim),
            ("temporal_dim", self.temporal_dim),
            ("forecast_dim", self.forecast_dim),
            ("uncertainty_dim", self.uncertainty_dim),
            ("context_dim", self.context_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

        if not self.version.strip():
            raise ValueError("version must not be empty")

    @property
    def total_dim(self) -> int:
        """Total PPO input dimension represented by this contract."""

        return (
            self.spatial_dim
            + self.temporal_dim
            + self.forecast_dim
            + self.uncertainty_dim
            + self.context_dim
        )


# ============================================================================
# Representation output
# ============================================================================


@dataclass(frozen=True)
class RepresentationOutput:
    """
    One causal fixed-size representation for a PPO decision.

    All arrays are one-dimensional and already normalized by the upstream
    representation pipeline.

    No future simulator information is permitted here.
    """

    spec: RepresentationSpec

    spatial: np.ndarray
    temporal: np.ndarray
    forecast: np.ndarray
    uncertainty: np.ndarray
    context: np.ndarray

    observation_time_s: float
    graph_version: str
    representation_version: str

    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        expected = {
            "spatial": self.spec.spatial_dim,
            "temporal": self.spec.temporal_dim,
            "forecast": self.spec.forecast_dim,
            "uncertainty": self.spec.uncertainty_dim,
            "context": self.spec.context_dim,
        }

        for name, expected_dim in expected.items():
            value = getattr(self, name)

            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"{name} must be a numpy.ndarray"
                )

            if value.shape != (expected_dim,):
                raise ValueError(
                    f"{name} must have shape {(expected_dim,)}, "
                    f"got {value.shape}"
                )

            if value.dtype.kind not in "fc":
                raise TypeError(
                    f"{name} must contain floating-point values"
                )

            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{name} contains NaN or infinite values"
                )

            value.setflags(write=False)

        if self.observation_time_s < 0:
            raise ValueError(
                "observation_time_s must be non-negative"
            )

        if not self.graph_version.strip():
            raise ValueError(
                "graph_version must not be empty"
            )

        if not self.representation_version.strip():
            raise ValueError(
                "representation_version must not be empty"
            )

        if self.representation_version != self.spec.version:
            raise ValueError(
                "representation_version must match spec.version"
            )

    @property
    def flat(self) -> np.ndarray:
        """
        Return the complete PPO representation.

        Ordering is part of the versioned contract.
        """

        result = np.concatenate(
            (
                self.spatial,
                self.temporal,
                self.forecast,
                self.uncertainty,
                self.context,
            )
        ).astype(np.float32, copy=False)

        result.setflags(write=False)
        return result


# ============================================================================
# Provider protocol
# ============================================================================


class RepresentationProvider(Protocol):
    """
    Interface implemented by the model/state adapter used by PPO.

    PPO depends only on this interface.

    The implementation may internally use:
        - GNN + Transformer
        - temporal-only forecasting
        - persistence forecasting
        - another learned encoder
        - a deterministic development representation

    PPO must not inspect the implementation internals.
    """

    @property
    def spec(self) -> RepresentationSpec:
        """Return the fixed representation contract."""
        ...

    def encode(
        self,
        state: Any,
    ) -> RepresentationOutput:
        """
        Convert one causal environment state into a PPO representation.

        `state` must contain only information visible at the current
        observation time.
        """
        ...


# ============================================================================
# Validation helper
# ============================================================================


def validate_representation_provider(
    provider: RepresentationProvider,
) -> RepresentationSpec:
    """
    Validate the provider's public contract before PPO starts.

    This intentionally checks only the interface, not the provider's internal
    architecture.
    """

    spec = provider.spec

    if not isinstance(spec, RepresentationSpec):
        raise TypeError(
            "provider.spec must be a RepresentationSpec"
        )

    probe = getattr(provider, "encode", None)

    if not callable(probe):
        raise TypeError(
            "provider must expose a callable encode(state) method"
        )

    return spec