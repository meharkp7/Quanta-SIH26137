import numpy as np
import torch

from src.learning.gnn_transformer import forecast_loss
from src.learning.train_forecaster import _supervision_report
from src.learning.windows import ForecastWindow


def _window(speed_mask, traversal_mask, edges=("e1", "e2")):
    e = len(edges)
    speed_mask = np.asarray(speed_mask, dtype=bool).reshape(e, 3)
    traversal_mask = np.asarray(traversal_mask, dtype=bool).reshape(e, 3)
    return ForecastWindow(
        episode_id="ep", split="train", issue_time_s=720,
        history_times_s=tuple(range(60, 780, 60)),
        target_times_s=(1020, 1320, 1620), edge_ids=tuple(edges),
        features=np.zeros((12, e, 6), dtype=np.float32),
        feature_mask=np.ones((12, e, 6), dtype=bool),
        speed_targets=np.where(speed_mask, 0.5, np.nan).astype(np.float32),
        speed_target_mask=speed_mask,
        traversal_targets=np.where(traversal_mask, 60.0, np.nan).astype(np.float32),
        traversal_target_mask=traversal_mask,
        label_available_at_s=np.where(
            np.stack([speed_mask, traversal_mask], axis=-1),
            np.asarray((1020, 1320, 1620))[None, :, None],
            np.nan,
        ).astype(np.float64),
    )


def test_supervision_report_uses_edge_instances_not_valid_labels():
    w = _window([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0])
    class Dataset:
        windows = [w]
    report = _supervision_report(Dataset())
    assert report["edge_instances"] == 2
    assert report["horizons"]["5"]["speed_valid_count"] == 1
    assert report["horizons"]["5"]["speed_coverage"] == 0.5


def test_forecast_loss_normalizes_traversal_to_minute_scale():
    outputs = {
        "speed_ratio": torch.tensor([[[0.6]]]),
        "traversal_time_s": torch.tensor([[[120.0]]]),
    }
    # Shape [B,E,H], with one horizon.
    class Batch:
        speed_targets = np.array([[[0.5]]], dtype=np.float32)
        speed_target_mask = np.array([[[True]]])
        traversal_targets = np.array([[[60.0]]], dtype=np.float32)
        traversal_target_mask = np.array([[[True]]])

    losses = forecast_loss(outputs, Batch(), traversal_scale_s=60.0)
    assert torch.isclose(losses["speed"], torch.tensor(0.005))
    assert torch.isclose(losses["traversal"], torch.tensor(0.5))
    assert torch.isclose(losses["total"], torch.tensor(0.505))


def test_forecast_loss_rejects_invalid_scale_and_zero_weights():
    class Batch:
        speed_targets = np.array([[[0.5]]], dtype=np.float32)
        speed_target_mask = np.array([[[True]]])
        traversal_targets = np.array([[[60.0]]], dtype=np.float32)
        traversal_target_mask = np.array([[[True]]])

    outputs = {
        "speed_ratio": torch.tensor([[[0.6]]]),
        "traversal_time_s": torch.tensor([[[60.0]]]),
    }
    try:
        forecast_loss(outputs, Batch(), traversal_scale_s=0)
    except ValueError as exc:
        assert "traversal_scale_s" in str(exc)
    else:
        raise AssertionError("invalid traversal scale was accepted")

    try:
        forecast_loss(outputs, Batch(), speed_weight=0, traversal_weight=0)
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("zero loss weights were accepted")
