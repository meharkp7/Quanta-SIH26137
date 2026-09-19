"""Causal speed-forecast baselines used as production reference models."""
from __future__ import annotations

import numpy as np

from src.learning.loader import ForecastBatch


def _validate_speed_arrays(
    targets: np.ndarray,
    mask: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    pred = np.asarray(predictions, dtype=np.float32)
    if y.shape != m.shape or y.shape != pred.shape:
        raise ValueError("targets, mask, and predictions must have equal shapes")
    return y, m, pred


def masked_speed_metrics(
    targets: np.ndarray,
    mask: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute MAE/RMSE only on finite, explicitly available labels."""
    y, m, pred = _validate_speed_arrays(targets, mask, predictions)
    valid = m & np.isfinite(y) & np.isfinite(pred)
    if not np.any(valid):
        return {"count": 0, "mae": None, "rmse": None}
    err = pred[valid].astype(np.float64) - y[valid].astype(np.float64)
    absolute = np.abs(err)
    scale = np.maximum(np.abs(y[valid].astype(np.float64)), 1e-6)
    return {
        "count": int(valid.sum()),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "relative_mae": float(np.mean(absolute / scale)),
        "mean_error": float(np.mean(err)),
    }


def _require_batch(batch: ForecastBatch) -> None:
    if not isinstance(batch, ForecastBatch):
        raise TypeError("batch must be a ForecastBatch")
    if batch.features.ndim != 4:
        raise ValueError("batch.features must have shape [B, L, E, F]")
    if batch.raw_speed_ratio.shape != batch.features[..., :1].squeeze(-1).shape:
        raise ValueError("raw_speed_ratio must have shape [B, L, E]")
    if batch.feature_mask.shape != batch.features.shape:
        raise ValueError("feature_mask must match features shape")
    if batch.speed_targets.ndim != 3:
        raise ValueError("speed_targets must have shape [B, E, H]")
    if batch.speed_target_mask.shape != batch.speed_targets.shape:
        raise ValueError("speed_target_mask must match speed_targets")
    if batch.edge_padding_mask.shape != batch.speed_targets.shape[:2]:
        raise ValueError("edge_padding_mask must have shape [B, E]")


class PersistenceForecaster:
    """Repeat the most recent finite causal speed ratio per road."""

    name = "persistence"

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        _require_batch(batch)
        raw = np.asarray(batch.raw_speed_ratio, dtype=np.float32)
        observed = (
            batch.feature_mask[..., 0]
            & np.isfinite(raw)
            & ~batch.edge_padding_mask[:, None, :]
        )
        b, l, e = raw.shape
        prediction = np.full(
            (b, e, batch.speed_targets.shape[-1]),
            np.nan,
            dtype=np.float32,
        )

        # Reverse scan avoids Python-side dependence on a particular history
        # length and correctly handles intermittent sensor gaps.
        for t in range(l - 1, -1, -1):
            available = observed[:, t, :] & ~np.isfinite(prediction[:, :, 0])
            if np.any(available):
                prediction[available] = raw[:, t, :][available, None]

        prediction[batch.edge_padding_mask] = np.nan
        return prediction


class TemporalOnlyForecaster:
    """Causal ridge baseline over flattened per-road history, without graph data."""

    name = "temporal_only"

    def __init__(self, ridge: float = 1e-2) -> None:
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        self.ridge = float(ridge)
        self.weights: np.ndarray | None = None
        self.feature_dim: int | None = None
        self.horizon_count: int | None = None

    def fit(self, batch: ForecastBatch) -> "TemporalOnlyForecaster":
        _require_batch(batch)
        design, targets, masks = self._xy(batch, require_targets=True)
        if design.shape[0] == 0:
            raise ValueError("cannot fit temporal baseline on an empty batch")
        valid_rows = np.any(masks & np.isfinite(targets), axis=1)
        if not np.any(valid_rows):
            raise ValueError("cannot fit temporal baseline: no valid speed labels")

        dim = design.shape[1]
        horizons = targets.shape[1]
        weights = np.zeros((dim, horizons), dtype=np.float32)
        for horizon in range(horizons):
            valid = masks[:, horizon] & np.isfinite(targets[:, horizon])
            if not np.any(valid):
                continue
            x = design[valid].astype(np.float64)
            y = targets[valid, horizon].astype(np.float64)
            system = x.T @ x + self.ridge * np.eye(dim, dtype=np.float64)
            weights[:, horizon] = np.linalg.solve(system, x.T @ y).astype(np.float32)

        self.weights = weights
        self.feature_dim = dim
        self.horizon_count = horizons
        return self

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        _require_batch(batch)
        if self.weights is None:
            raise RuntimeError("TemporalOnlyForecaster.fit must be called first")
        design, _, _ = self._xy(batch, require_targets=False)
        if design.shape[1] != self.feature_dim:
            raise ValueError("batch feature/history shape does not match fitted baseline")
        raw = design @ self.weights
        b, _l, e, _f = batch.features.shape
        prediction = np.full(
            (b, e, self.horizon_count),
            np.nan,
            dtype=np.float32,
        )
        cursor = 0
        for batch_index in range(b):
            for edge_index in range(e):
                if batch.edge_padding_mask[batch_index, edge_index]:
                    continue
                prediction[batch_index, edge_index] = raw[cursor]
                cursor += 1
        return prediction

    def _xy(
        self,
        batch: ForecastBatch,
        *,
        require_targets: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.asarray(batch.features, dtype=np.float32)
        b, l, e, _f = features.shape
        horizons = batch.speed_targets.shape[-1]
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for batch_index in range(b):
            for edge_index in range(e):
                if batch.edge_padding_mask[batch_index, edge_index]:
                    continue
                history = np.nan_to_num(
                    features[batch_index, :, edge_index, :],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).reshape(-1)
                rows.append(np.concatenate((history, np.ones(1, dtype=np.float32))))
                targets.append(batch.speed_targets[batch_index, edge_index])
                masks.append(batch.speed_target_mask[batch_index, edge_index])

        if not rows:
            dim = l * features.shape[-1] + 1
            empty_targets = np.empty((0, horizons), dtype=np.float32)
            empty_masks = np.empty((0, horizons), dtype=bool)
            return np.empty((0, dim), dtype=np.float32), empty_targets, empty_masks

        design = np.stack(rows).astype(np.float32)
        target_array = np.stack(targets).astype(np.float32)
        mask_array = np.stack(masks).astype(bool)
        if not require_targets:
            target_array = np.full((design.shape[0], horizons), np.nan, dtype=np.float32)
        return design, target_array, mask_array
