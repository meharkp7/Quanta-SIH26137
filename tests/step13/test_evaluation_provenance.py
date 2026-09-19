import numpy as np
import pytest

from src.learning.evaluation import (
    baseline_delta,
    evaluate_prediction_set,
    padding_aware_label_coverage,
    validate_manifest_splits,
)


def _manifest():
    return {
        "train": ["e1", "e2"],
        "validation": ["e3"],
        "test": ["e4"],
        "split_episode_counts": {"train": 2, "validation": 1, "test": 1},
        "map_records": {
            "0": {"split": "train", "map_fingerprint": "map-a"},
            "1": {"split": "validation", "map_fingerprint": "map-b"},
            "2": {"split": "test", "map_fingerprint": "map-c"},
        },
    }


def test_manifest_requires_episode_and_map_disjointness():
    integrity = validate_manifest_splits(_manifest())
    assert integrity.map_disjoint is True
    assert integrity.episode_counts == {"train": 2, "validation": 1, "test": 1}

    broken = _manifest()
    broken["test"] = ["e2"]
    broken["split_episode_counts"]["test"] = 1
    with pytest.raises(ValueError, match="episode leakage"):
        validate_manifest_splits(broken)


def test_padding_aware_coverage_excludes_padded_edges():
    mask = np.array([[[True, False], [True, True], [True, True]]])
    padding = np.array([[False, True, False]])
    assert padding_aware_label_coverage(mask, padding) == pytest.approx(0.75)


def test_complete_split_evaluation_reports_horizon_metrics_and_coverage():
    targets = np.ones((1, 3, 2), dtype=np.float32)
    predictions = targets.copy()
    predictions[0, 0, 0] = 1.2
    masks = np.ones_like(targets, dtype=bool)
    masks[0, 2, 1] = False
    padding = np.array([[False, True, False]])
    result = evaluate_prediction_set(targets, masks, predictions, padding)
    assert result["label_coverage"] == pytest.approx(0.75)
    assert result["horizons"]["5"]["count"] == 2
    assert result["horizons"]["5"]["mae"] == pytest.approx(0.1)


def test_baseline_delta_is_explicit_and_preserves_missing_metrics():
    model = {"mae": 0.2, "rmse": 0.3, "relative_mae": 0.4}
    baseline = {"mae": 0.25, "rmse": 0.35, "relative_mae": 0.45}
    delta = baseline_delta(model, baseline)
    assert delta["mae"] == pytest.approx(-0.05)
    assert delta["rmse"] == pytest.approx(-0.05)
    assert delta["relative_mae"] == pytest.approx(-0.05)

    assert baseline_delta({"mae": None}, {"mae": 0.1})["mae"] is None


def test_legacy_episode_only_manifest_remains_supported():
    manifest = {
        "train": ["e1", "e2"],
        "validation": ["e3"],
        "test": ["e4"],
        "split_episode_counts": {"train": 2, "validation": 1, "test": 1},
        "rule": "split by episode before overlappingwindows",
    }
    integrity = validate_manifest_splits(manifest)
    assert integrity.map_disjoint is False
    assert integrity.map_counts == {"train": 0, "validation": 0, "test": 0}
