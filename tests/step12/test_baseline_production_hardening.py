from __future__ import annotations

import numpy as np
import pytest

from src.learning.baselines import PersistenceForecaster, TemporalOnlyForecaster, masked_speed_metrics
from src.learning.loader import ForecastBatch
from src.learning.inspect import inspect_window, write_inspect_example


def _batch() -> ForecastBatch:
    features = np.zeros((1, 3, 3, 6), dtype=np.float32)
    mask = np.zeros_like(features, dtype=bool)
    raw = np.asarray([[[0.5, 0.8, 0.0], [0.6, np.nan, 0.9], [0.7, 0.7, np.nan]]], dtype=np.float32)
    mask[:, :, :, 0] = np.isfinite(raw)
    speed_targets = np.full((1, 3, 2), 0.6, dtype=np.float32)
    speed_mask = np.ones_like(speed_targets, dtype=bool)
    traversal_targets = np.zeros_like(speed_targets)
    traversal_mask = np.zeros_like(speed_targets, dtype=bool)
    available = np.full((1, 3, 2, 2), 120.0, dtype=np.float32)
    return ForecastBatch(
        features=features,
        raw_speed_ratio=raw,
        feature_mask=mask,
        speed_targets=speed_targets,
        speed_target_mask=speed_mask,
        traversal_targets=traversal_targets,
        traversal_target_mask=traversal_mask,
        issue_times_s=(120,),
        episode_ids=("ep-1",),
        edge_ids=(),
        edge_padding_mask=np.asarray([[False, False, True]]),
        edge_ids_by_sample=np.asarray([["e1", "e2", ""]]),
        history_times_s=np.asarray([[0, 60, 120]]),
        target_times_s=np.asarray([[420, 720]]),
        label_available_at_s=available,
        scenario_ids=("s1",),
        graph_versions=("g1",),
        splits=("train",),
    )


def test_persistence_uses_latest_finite_observation_and_respects_padding():
    prediction = PersistenceForecaster().predict(_batch())
    np.testing.assert_allclose(prediction[0, 0], [0.7, 0.7])
    np.testing.assert_allclose(prediction[0, 1], [0.7, 0.7])
    assert np.isnan(prediction[0, 2]).all()


def test_temporal_baseline_excludes_padded_edges_and_validates_fit():
    batch = _batch()
    model = TemporalOnlyForecaster(ridge=1e-2).fit(batch)
    prediction = model.predict(batch)
    assert prediction.shape == (1, 3, 2)
    assert np.isnan(prediction[0, 2]).all()
    with pytest.raises(ValueError, match="ridge"):
        TemporalOnlyForecaster(ridge=-1.0)


def test_masked_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="equal shapes"):
        masked_speed_metrics(np.zeros((1, 2)), np.ones((1, 2)), np.zeros((1, 3)))
