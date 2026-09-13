"""Persistence and temporal-only forecast baselines for Step 12.

Persistence copies the last *raw* observed speed ratio.  The temporal-only
baseline is a per-edge ridge model over the causal history and never receives
road adjacency or future labels.  Both baselines expose masked metrics so
missing target labels never become artificial zeros.
"""

from __future__ import annotations

import numpy as np

from src.learning.loader import ForecastBatch


def masked_speed_metrics(
    targets: np.ndarray,
    mask: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int | None]:
    """Compute MAE/RMSE only where the target label is actually available."""
    y = np.asarray(targets, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    pred = np.asarray(predictions, dtype=np.float32)
    valid = m & np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return {"count": 0, "mae": None, "rmse": None}
    err = pred[valid] - y[valid]
    return {
        "count": int(valid.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


class PersistenceForecaster:
    """Repeat the last observed speed ratio at every requested horizon."""

    name = "persistence"

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        # raw_speed_ratio is retained by ForecastBatch specifically so scaling
        # cannot silently turn persistence into a z-score prediction.
        last = batch.raw_speed_ratio[:, -1, :]
        last_valid = batch.feature_mask[:, -1, :, 0]
        prediction = np.repeat(last[:, :, None], batch.speed_targets.shape[-1], axis=2)
        prediction = np.where(last_valid[:, :, None], prediction, np.nan)
        return prediction.astype(np.float32)


class TemporalOnlyForecaster:
    """Per-edge ridge regression on flattened history; no graph structure."""

    name = "temporal_only"

    def __init__(self, ridge: float = 1e-2) -> None:
        self.ridge = float(ridge)
        self.weights: np.ndarray | None = None

    def fit(self, batch: ForecastBatch) -> "TemporalOnlyForecaster":
        design, targets, masks = self._xy(batch)
        dim = design.shape[1]
        horizons = targets.shape[1]
        weights = np.zeros((dim, horizons), dtype=np.float32)
        for horizon in range(horizons):
            valid = masks[:, horizon] & np.isfinite(targets[:, horizon])
            if not np.any(valid):
                continue
            x = design[valid]
            y = targets[valid, horizon]
            gram = x.T @ x + self.ridge * np.eye(dim, dtype=np.float32)
            weights[:, horizon] = np.linalg.solve(gram, x.T @ y)
        self.weights = weights
        return self

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("TemporalOnlyForecaster.fit must be called first")
        b, _l, e, _f = batch.features.shape
        horizons = batch.speed_targets.shape[-1]
        design, _, _ = self._xy(batch, require_targets=False)
        raw = design @ self.weights
        prediction = np.full((b, e, horizons), np.nan, dtype=np.float32)
        cursor = 0
        for batch_index in range(b):
            for edge_index in range(e):
                prediction[batch_index, edge_index] = raw[cursor]
                cursor += 1
        return prediction

    def _xy(
        self,
        batch: ForecastBatch,
        *,
        require_targets: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.nan_to_num(batch.features, nan=0.0)
        b, l, e, _f = features.shape
        horizons = batch.speed_targets.shape[-1]
        rows = []
        targets = []
        masks = []
        for batch_index in range(b):
            for edge_index in range(e):
                history = features[batch_index, :, edge_index, :].reshape(-1)
                rows.append(np.concatenate([history, np.ones((1,), dtype=np.float32)]))
                targets.append(batch.speed_targets[batch_index, edge_index])
                masks.append(batch.speed_target_mask[batch_index, edge_index])
        design = np.stack(rows, axis=0).astype(np.float32)
        target_array = np.stack(targets, axis=0).astype(np.float32)
        mask_array = np.stack(masks, axis=0).astype(bool)
        if not require_targets:
            target_array = np.full((design.shape[0], horizons), np.nan, dtype=np.float32)
        return design, target_array, mask_array
