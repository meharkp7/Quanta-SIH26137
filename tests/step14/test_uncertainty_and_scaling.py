from __future__ import annotations

import numpy as np
import pytest

from src.learning.scaling import FeatureScaler
from src.learning.uncertainty import (
    ResidualCalibration,
    interval_metrics,
    fit_residual_calibration,
)


def test_scaler_rejects_feature_axis_mismatch():
    scaler = FeatureScaler((0.0, 1.0), (1.0, 2.0), ("a", "b"))
    with pytest.raises(ValueError, match="feature axis"):
        scaler.transform(np.zeros((2, 3), dtype=np.float32))


def test_scaler_requires_unique_feature_names():
    with pytest.raises(ValueError, match="unique"):
        FeatureScaler.fit(np.ones((1, 1, 1, 2), dtype=np.float32), ("a", "a"))


def test_scaler_roundtrip_preserves_finite_values():
    values = np.array([[[[1.0, 4.0], [3.0, 8.0]]]], dtype=np.float32)
    scaler = FeatureScaler.fit(values, ("speed", "occupancy"))
    transformed = scaler.transform(values)
    restored = scaler.inverse_transform(transformed)
    np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)


def test_uncertainty_calibration_rejects_zero_horizon():
    with pytest.raises(ValueError, match="at least one"):
        fit_residual_calibration(
            np.empty((1, 1, 0)),
            np.empty((1, 1, 0)),
            np.empty((1, 1, 0), dtype=bool),
        )


def test_interval_metrics_rejects_nonfinite_valid_bounds():
    lower = np.array([[[np.nan]]])
    upper = np.array([[[1.0]]])
    target = np.array([[[1.0]]])
    mask = np.array([[[True]]])
    with pytest.raises(ValueError, match="finite bounds"):
        interval_metrics(lower, upper, target, mask)


def test_interval_metrics_rejects_inverted_interval():
    lower = np.array([[[2.0]]])
    upper = np.array([[[1.0]]])
    target = np.array([[[1.0]]])
    mask = np.array([[[True]]])
    with pytest.raises(ValueError, match="lower bounds"):
        interval_metrics(lower, upper, target, mask)


def test_residual_calibration_rejects_empty_radii():
    with pytest.raises(ValueError, match="at least one"):
        ResidualCalibration(())
