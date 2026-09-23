"""Small, versioned forecast contract between a forecaster and PPO.

The GNN--Transformer may change internally, but PPO always receives twelve
forecast summary values and six uncertainty values.  A provider must use only
traffic known at the supplied decision time; future labels are never accepted
by this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PPOForecastSignal:
    """Causal traffic summary consumed by one PPO decision."""

    forecast_features: np.ndarray
    uncertainty_features: np.ndarray
    forecaster_version: str

    def __post_init__(self) -> None:
        forecast = np.asarray(self.forecast_features, dtype=np.float32)
        uncertainty = np.asarray(self.uncertainty_features, dtype=np.float32)
        if forecast.shape != (12,):
            raise ValueError("forecast_features must have shape (12,)")
        if uncertainty.shape != (6,):
            raise ValueError("uncertainty_features must have shape (6,)")
        if not np.all(np.isfinite(forecast)):
            raise ValueError("forecast_features must be finite")
        if not np.all(np.isfinite(uncertainty)):
            raise ValueError("uncertainty_features must be finite")
        if not self.forecaster_version.strip():
            raise ValueError("forecaster_version must be non-empty")
        forecast.setflags(write=False)
        uncertainty.setflags(write=False)
        object.__setattr__(self, "forecast_features", forecast)
        object.__setattr__(self, "uncertainty_features", uncertainty)

    @classmethod
    def zero_baseline(cls) -> "PPOForecastSignal":
        """Explicit no-forecast baseline for pipeline-only PPO smoke tests."""
        return cls(
            forecast_features=np.zeros(12, dtype=np.float32),
            uncertainty_features=np.zeros(6, dtype=np.float32),
            forecaster_version="zero-baseline-v1",
        )


def coerce_ppo_forecast_signal(value: PPOForecastSignal | Sequence[float]) -> PPOForecastSignal:
    """Accept the new signal or the former 12-value provider contract.

    Sequence support preserves existing callers.  It means forecast-only
    providers still have unknown uncertainty, represented by six zeroes.
    """
    if isinstance(value, PPOForecastSignal):
        return value
    return PPOForecastSignal(
        forecast_features=np.asarray(tuple(float(item) for item in value), dtype=np.float32),
        uncertainty_features=np.zeros(6, dtype=np.float32),
        forecaster_version="legacy-forecast-provider-v1",
    )
