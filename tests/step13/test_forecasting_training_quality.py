import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.learning.baselines import PersistenceForecaster
from src.learning.gnn_transformer import (
    ForecastLossConfig,
    forecast_loss,
)
from src.learning.loader import ForecastBatch
from src.learning.train_forecaster import _label_coverage


def _batch():
    features = np.zeros((1, 2, 3, 6), dtype=np.float32)
    feature_mask = np.ones_like(features, dtype=bool)
    padding = np.array([[False, False, True]], dtype=bool)
    speed_targets = np.ones((1, 3, 3), dtype=np.float32)
    speed_mask = np.zeros((1, 3, 3), dtype=bool)
    speed_mask[0, :2, 0] = True
    traversal_targets = np.ones((1, 3, 3), dtype=np.float32) * 60.0
    traversal_mask = speed_mask.copy()
    return ForecastBatch(
        features=features,
        raw_speed_ratio=features[..., 0],
        feature_mask=feature_mask,
        speed_targets=speed_targets,
        speed_target_mask=speed_mask,
        traversal_targets=traversal_targets,
        traversal_target_mask=traversal_mask,
        issue_times_s=(60,),
        episode_ids=("ep",),
        edge_ids=(),
        edge_padding_mask=padding,
        edge_ids_by_sample=np.array([["a", "b", ""]]),
        history_times_s=np.zeros((1, 2), dtype=int),
        target_times_s=np.zeros((1, 3), dtype=int),
        label_available_at_s=np.ones((1, 3, 3, 2), dtype=float),
        scenario_ids=("s",),
        graph_versions=("g",),
        splits=("train",),
    )


def test_loss_config_rejects_invalid_policy():
    with pytest.raises(ValueError):
        ForecastLossConfig(speed_weight=0.0, traversal_weight=0.0)
    with pytest.raises(ValueError):
        ForecastLossConfig(traversal_delta=0.0)


def test_forecast_loss_uses_log_space_for_second_scale_targets():
    batch = _batch()
    outputs = {
        "speed_ratio": torch.full((1, 3, 3), 0.5, requires_grad=True),
        "traversal_time_s": torch.full((1, 3, 3), 30.0, requires_grad=True),
    }
    losses = forecast_loss(outputs, batch, loss_config=ForecastLossConfig())
    assert torch.isfinite(losses["total"])
    assert losses["traversal"].detach().item() < 1.0
    losses["total"].backward()
    assert outputs["speed_ratio"].grad is not None
    assert outputs["traversal_time_s"].grad is not None


def test_label_coverage_uses_non_padding_cells_as_denominator():
    batch = _batch()
    # 2 valid labels out of 2 real edges = 100% for horizon 0.
    assert _label_coverage(batch, batch.speed_target_mask[:, :, 0]) == pytest.approx(1.0)
    # Only 2 real edges exist; padded edge must never dilute coverage.
    assert _label_coverage(batch, np.zeros((1, 3), dtype=bool)) == pytest.approx(0.0)


def test_persistence_uses_latest_finite_observation_not_only_last_timestep():
    batch = _batch()
    batch.raw_speed_ratio[:, 0, 0] = 0.7
    batch.raw_speed_ratio[:, 1, 0] = np.nan
    batch.feature_mask[:, 1, 0] = False
    prediction = PersistenceForecaster().predict(batch)
    np.testing.assert_allclose(prediction[0, 0], [0.7, 0.7, 0.7])
    assert np.isnan(prediction[0, 2]).all()
