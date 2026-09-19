import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.learning.gnn_transformer import CausalGNNTransformer, GraphBatch, forecast_loss
from src.learning.loader import ForecastBatch


def batch():
    features = np.zeros((2, 12, 3, 6), dtype=np.float32)
    return ForecastBatch(
        features=features, raw_speed_ratio=features[..., 0],
        feature_mask=np.ones_like(features, dtype=bool),
        speed_targets=np.ones((2, 3, 3), dtype=np.float32),
        speed_target_mask=np.ones((2, 3, 3), dtype=bool),
        traversal_targets=np.ones((2, 3, 3), dtype=np.float32) * 10,
        traversal_target_mask=np.ones((2, 3, 3), dtype=bool),
        issue_times_s=(660, 660), episode_ids=("a", "b"), edge_ids=(),
        edge_padding_mask=np.array([[False, False, True], [False, False, False]]),
        edge_ids_by_sample=np.array([["a", "b", ""], ["a", "b", "c"]]),
        history_times_s=np.zeros((2, 12), dtype=int), target_times_s=np.zeros((2, 3), dtype=int),
        label_available_at_s=np.ones((2, 3, 3, 2)), scenario_ids=("a", "b"),
        graph_versions=("ga", "gb"), splits=("train", "train"),
    )


def test_model_preserves_per_road_outputs_and_masks_padding():
    b = batch()
    model = CausalGNNTransformer(width=32)
    graph = GraphBatch(torch.eye(3).repeat(2, 1, 1), torch.ones(2, 3, 6))
    outputs = model(torch.as_tensor(b.features), graph, torch.as_tensor(b.edge_padding_mask))
    assert outputs["speed_ratio"].shape == (2, 3, 3)
    assert outputs["traversal_time_s"].shape == (2, 3, 3)
    assert torch.all(outputs["road_embeddings"][0, 2] == 0)


def test_masked_loss_ignores_nan_and_unavailable_targets():
    b = batch()
    b.speed_targets[0, 0, 0] = np.nan
    b.speed_target_mask[0, 0, 0] = False
    model = CausalGNNTransformer(width=32)
    graph = GraphBatch(torch.eye(3).repeat(2, 1, 1), torch.ones(2, 3, 6))
    outputs = model(torch.as_tensor(b.features), graph)
    losses = forecast_loss(outputs, b)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_transformer_rejects_non_divisible_attention_width():
    with pytest.raises(ValueError, match="divisible"):
        CausalGNNTransformer(width=30, heads=4)


def test_forecast_loss_uses_log_space_for_traversal_by_default():
    import numpy as np
    import torch
    from src.learning.gnn_transformer import forecast_loss

    class Batch:
        speed_targets = np.zeros((1, 1, 1), dtype=np.float32)
        speed_target_mask = np.zeros((1, 1, 1), dtype=bool)
        traversal_targets = np.array([[[100.0]]], dtype=np.float32)
        traversal_target_mask = np.ones((1, 1, 1), dtype=bool)

    outputs = {"speed_ratio": torch.zeros(1, 1, 1),
               "traversal_time_s": torch.tensor([[[50.0]]], requires_grad=True)}
    losses = forecast_loss(outputs, Batch())
    expected = torch.nn.functional.huber_loss(
        torch.log1p(torch.tensor([50.0])),
        torch.log1p(torch.tensor([100.0])),
    )
    assert torch.allclose(losses["traversal"], expected)
    losses["total"].backward()
    assert outputs["traversal_time_s"].grad is not None


def test_forecast_loss_rejects_zero_task_weights():
    import numpy as np
    import torch
    from src.learning.gnn_transformer import forecast_loss

    class Batch:
        speed_targets = np.zeros((1, 1, 1), dtype=np.float32)
        speed_target_mask = np.zeros((1, 1, 1), dtype=bool)
        traversal_targets = np.zeros((1, 1, 1), dtype=np.float32)
        traversal_target_mask = np.zeros((1, 1, 1), dtype=bool)

    outputs = {"speed_ratio": torch.zeros(1, 1, 1), "traversal_time_s": torch.ones(1, 1, 1)}
    import pytest
    with pytest.raises(ValueError, match="at least one loss weight"):
        forecast_loss(outputs, Batch(), speed_weight=0.0, traversal_weight=0.0)
