"""Production evaluation helpers for causal forecasting experiments.

This module keeps evaluation policy separate from model architecture.  It
validates split provenance, computes padding-aware label coverage, and reports
model-vs-baseline deltas without fitting on validation/test data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np



SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitIntegrity:
    """Immutable summary of the validated corpus split contract."""

    episode_counts: Mapping[str, int]
    map_counts: Mapping[str, int]
    map_disjoint: bool


def validate_manifest_splits(manifest: Mapping[str, object]) -> SplitIntegrity:
    """Validate episode/map isolation before any model evaluation.

    The check intentionally operates only on manifest provenance.  It does not
    infer split membership from filesystem ordering or episode filenames.
    """
    episode_ids_by_split: dict[str, list[str]] = {}
    for split in SPLITS:
        raw = manifest.get(split)
        if not isinstance(raw, list):
            raise ValueError(f"manifest.{split} must be a list of episode IDs")
        ids = [str(value) for value in raw]
        if any(not value for value in ids):
            raise ValueError(f"manifest.{split} contains an empty episode ID")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate episode IDs inside {split} split")
        episode_ids_by_split[split] = ids

    seen: set[str] = set()
    for split in SPLITS:
        overlap = seen.intersection(episode_ids_by_split[split])
        if overlap:
            raise ValueError(
                f"episode leakage across splits: {sorted(overlap)[:5]}"
            )
        seen.update(episode_ids_by_split[split])

    expected_counts = manifest.get("split_episode_counts")
    if isinstance(expected_counts, Mapping):
        for split in SPLITS:
            expected = int(expected_counts.get(split, -1))
            actual = len(episode_ids_by_split[split])
            if expected != actual:
                raise ValueError(
                    f"{split} episode count mismatch: actual={actual}, expected={expected}"
                )

    raw_maps = manifest.get("map_records")
    # Legacy Step-12 pilot manifests predate map-level provenance.  They still
    # have a valid episode-disjoint split contract, so preserve compatibility
    # and report that map disjointness could not be established. Production
    # OSM manifests provide map_records and are validated strictly below.
    if raw_maps is None:
        return SplitIntegrity(
            episode_counts={split: len(episode_ids_by_split[split]) for split in SPLITS},
            map_counts={split: 0 for split in SPLITS},
            map_disjoint=False,
        )
    if not isinstance(raw_maps, Mapping):
        raise ValueError("manifest.map_records must be an object when provided")

    map_sets: dict[str, set[str]] = {split: set() for split in SPLITS}
    for raw_record in raw_maps.values():
        if not isinstance(raw_record, Mapping):
            raise ValueError("map_records contains a non-object record")
        split = str(raw_record.get("split", ""))
        fingerprint = str(raw_record.get("map_fingerprint", ""))
        if split not in map_sets:
            raise ValueError(f"invalid map split {split!r}")
        if not fingerprint:
            raise ValueError("map record is missing map_fingerprint")
        if fingerprint in map_sets[split]:
            raise ValueError(f"duplicate map fingerprint inside {split}: {fingerprint}")
        map_sets[split].add(fingerprint)

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            overlap = map_sets[left].intersection(map_sets[right])
            if overlap:
                raise ValueError(
                    f"base-map leakage between {left} and {right}: {sorted(overlap)}"
                )

    return SplitIntegrity(
        episode_counts={split: len(episode_ids_by_split[split]) for split in SPLITS},
        map_counts={split: len(map_sets[split]) for split in SPLITS},
        map_disjoint=True,
    )


def padding_aware_label_coverage(
    mask: np.ndarray,
    edge_padding_mask: np.ndarray,
) -> float:
    """Return valid labels / eligible edge-horizon cells, excluding padding."""
    labels = np.asarray(mask, dtype=bool)
    padding = np.asarray(edge_padding_mask, dtype=bool)
    if labels.ndim != 3:
        raise ValueError("mask must have shape [B, E, H]")
    if padding.shape != labels.shape[:2]:
        raise ValueError("edge_padding_mask must have shape [B, E]")
    eligible = np.broadcast_to(~padding[:, :, None], labels.shape)
    denominator = int(eligible.sum())
    return float((labels & eligible).sum() / denominator) if denominator else 0.0


def evaluate_prediction_set(
    targets: np.ndarray,
    masks: np.ndarray,
    predictions: np.ndarray,
    edge_padding_mask: np.ndarray,
    *,
    horizon_minutes: Sequence[int] | None = None,
) -> dict[str, object]:
    """Evaluate a complete split with explicit horizon and coverage metrics."""
    from src.learning.baselines import masked_speed_metrics
    y = np.asarray(targets, dtype=np.float32)
    m = np.asarray(masks, dtype=bool)
    pred = np.asarray(predictions, dtype=np.float32)
    if y.shape != m.shape or y.shape != pred.shape:
        raise ValueError("targets, masks, and predictions must have equal shapes")
    if y.ndim != 3:
        raise ValueError("forecast arrays must have shape [B, E, H]")
    coverage = padding_aware_label_coverage(m, edge_padding_mask)
    horizons = list(horizon_minutes or ((np.arange(y.shape[-1]) + 1) * 5).tolist())
    if len(horizons) != y.shape[-1]:
        raise ValueError("horizon_minutes length must match forecast horizon count")
    per_horizon: dict[str, dict[str, object]] = {}
    padding = np.asarray(edge_padding_mask, dtype=bool)
    if padding.shape != y.shape[:2]:
        raise ValueError("edge_padding_mask must have shape [B, E]")
    eligible = ~padding
    for index, horizon in enumerate(horizons):
        horizon_mask = m[:, :, index] & eligible
        metric = masked_speed_metrics(y[:, :, index], horizon_mask, pred[:, :, index])
        metric["label_coverage"] = padding_aware_label_coverage(
            m[:, :, index : index + 1], padding
        )
        per_horizon[str(int(horizon))] = metric
    return {"label_coverage": coverage, "horizons": per_horizon}


def baseline_delta(
    model_metrics: Mapping[str, object],
    baseline_metrics: Mapping[str, object],
) -> dict[str, float | None]:
    """Compare MAE/RMSE to a baseline without hiding missing-label cases."""
    result: dict[str, float | None] = {}
    for metric_name in ("mae", "rmse", "relative_mae"):
        model_value = model_metrics.get(metric_name)
        baseline_value = baseline_metrics.get(metric_name)
        if model_value is None or baseline_value is None:
            result[metric_name] = None
            continue
        result[metric_name] = float(model_value) - float(baseline_value)
    return result
