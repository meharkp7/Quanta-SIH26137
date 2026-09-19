"""Step 14 uncertainty calibration and forecast serving primitives.

The calibration is split from the neural model on purpose.  A held-out
validation residual distribution is the source of the interval radius, so
the routing process never gets access to future simulator labels.  The
resulting policy is serialisable and can be attached to a causal ``Forecast``
record at issue time.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Iterable, Sequence

import numpy as np

from src.contracts.forecast import Forecast


@dataclass(frozen=True)
class IntervalMetrics:
    nominal_coverage: float
    actual_coverage: float
    mean_width: float
    count: int


@dataclass(frozen=True)
class ResidualCalibration:
    """Per-horizon split-conformal absolute-residual radii."""

    radii: tuple[float, ...]
    nominal_coverage: float = 0.80
    calibration_count: tuple[int, ...] = ()
    policy_version: str = "residual-conformal-v1"

    def __post_init__(self) -> None:
        if not self.radii:
            raise ValueError("at least one horizon radius is required")
        if not 0.0 < self.nominal_coverage < 1.0:
            raise ValueError("nominal_coverage must be between zero and one")
        if any(not isfinite(float(x)) or float(x) < 0.0 for x in self.radii):
            raise ValueError("radii must be finite and non-negative")
        if self.calibration_count and any(int(x) < 0 for x in self.calibration_count):
            raise ValueError("calibration_count must be non-negative")
        if not self.policy_version.strip():
            raise ValueError("policy_version must not be empty")
        if self.calibration_count and len(self.calibration_count) != len(self.radii):
            raise ValueError("calibration_count must match radii")

    def to_dict(self) -> dict:
        return {
            "radii": list(self.radii),
            "nominal_coverage": self.nominal_coverage,
            "calibration_count": list(self.calibration_count),
            "policy_version": self.policy_version,
        }


def _finite_residuals(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, horizon: int) -> np.ndarray:
    values = np.abs(prediction[..., horizon] - target[..., horizon])
    valid = mask[..., horizon] & np.isfinite(values)
    return values[valid].astype(float, copy=False)


def fit_residual_calibration(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    nominal_coverage: float = 0.80,
) -> ResidualCalibration:
    """Fit finite-sample split-conformal radii per forecast horizon."""
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if prediction.shape != target.shape or prediction.shape != valid_mask.shape:
        raise ValueError("prediction, target, and valid_mask must have equal shapes")
    if prediction.ndim != 3:
        raise ValueError("arrays must have shape [batch, edge, horizon]")
    if prediction.shape[-1] == 0:
        raise ValueError("at least one forecast horizon is required")
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be between zero and one")
    radii: list[float] = []
    counts: list[int] = []
    for h in range(prediction.shape[-1]):
        residuals = np.sort(_finite_residuals(prediction, target, valid_mask, h))
        counts.append(int(residuals.size))
        if residuals.size == 0:
            radii.append(0.0)
            continue
        # Conservative finite-sample conformal order statistic.
        rank = int(ceil((residuals.size + 1) * nominal_coverage)) - 1
        radii.append(float(residuals[min(max(rank, 0), residuals.size - 1)]))
    return ResidualCalibration(tuple(radii), nominal_coverage, tuple(counts))


def interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    nominal_coverage: float = 0.80,
) -> dict[str, float | int]:
    """Measure held-out interval coverage and width."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(target)
    if not (lower.shape == upper.shape == target.shape == mask.shape):
        raise ValueError("interval arrays must have equal shapes")
    finite_bounds = np.isfinite(lower) & np.isfinite(upper)
    if np.any(mask & ~finite_bounds):
        raise ValueError("valid intervals must have finite bounds")
    if np.any(mask & (lower > upper)):
        raise ValueError("interval lower bounds must not exceed upper bounds")
    count = int(mask.sum())
    if count == 0:
        return {"nominal_coverage": nominal_coverage, "actual_coverage": 0.0, "mean_width": 0.0, "count": 0}
    covered = mask & (target >= lower) & (target <= upper)
    width = np.where(mask, upper - lower, np.nan)
    return {
        "nominal_coverage": float(nominal_coverage),
        "actual_coverage": float(covered.sum() / count),
        "mean_width": float(np.nanmean(width)),
        "count": count,
    }


def attach_intervals(forecast: Forecast, calibration: ResidualCalibration) -> Forecast:
    """Return a forecast with q0.1/q0.5/q0.9-style 80% intervals."""
    if len(forecast.target_times_s) != len(calibration.radii):
        raise ValueError("calibration horizon count does not match forecast")
    median = tuple(tuple(value for value in row) for row in forecast.prediction)
    lower = tuple(
        tuple(None if value is None else float(value) - calibration.radii[h] for h, value in enumerate(row))
        for row in forecast.prediction
    )
    upper = tuple(
        tuple(None if value is None else float(value) + calibration.radii[h] for h, value in enumerate(row))
        for row in forecast.prediction
    )
    return forecast.model_copy(update={
        "lower_prediction": lower,
        "median_prediction": median,
        "upper_prediction": upper,
        "model_version": f"{forecast.model_version}+{calibration.policy_version}",
    })


def calibrated_arrays(
    prediction: np.ndarray,
    calibration: ResidualCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build median/lower/upper arrays for evaluation and reporting."""
    prediction = np.asarray(prediction, dtype=float)
    radii = np.asarray(calibration.radii, dtype=float)
    if prediction.ndim != 3 or prediction.shape[-1] != radii.size:
        raise ValueError("prediction shape does not match calibration")
    return prediction, prediction - radii.reshape(1, 1, -1), prediction + radii.reshape(1, 1, -1)

