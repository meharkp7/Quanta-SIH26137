"""Persistence and temporal-only forecast baselines.

Neither model sees the road graph. Persistence copies the last valid speed
ratio. The temporal-only model is a ridge regressor on the per-edge history.
"""

from __future__ import annotations

import numpy as np

from src.learning.loader import ForecastBatch


class PersistenceForecaster:
    """Repeat the last observed speed ratio at every horizon."""

    name = "persistence"

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        features = batch.features
        mask = batch.feature_mask
        last = features[:, -1, :, 0]
        last_valid = mask[:, -1, :, 0]
        prediction = np.repeat(last[:, :, None], batch.speed_targets.shape[-1], axis=2)
        prediction = np.where(last_valid[:, :, None], prediction, np.nan)
        return prediction.astype(np.float32)


class TemporalOnlyForecaster:
    """Per-horizon ridge regression on flattened per-edge histories. No GNN."""

    name = "temporal_only"

    def __init__(self, ridge: float = 1e-2) -> None:
        self.ridge = float(ridge)
        self.weights: np.ndarray | None = None

    def fit(self, batch: ForecastBatch) -> "TemporalOnlyForecaster":
        design, targets = self._xy(batch)
        dim = design.shape[1]
        gram = design.T @ design + self.ridge * np.eye(dim, dtype=np.float32)
        self.weights = np.linalg.solve(gram, design.T @ targets)
        return self

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("TemporalOnlyForecaster.fit must be called first")
        b, _l, e, _f = batch.features.shape
        horizons = batch.speed_targets.shape[-1]
        design, _ = self._xy(batch, require_targets=False)
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
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.nan_to_num(batch.features, nan=0.0)
        b, l, e, f = features.shape
        horizons = batch.speed_targets.shape[-1]
        rows = []
        targets = []
        for batch_index in range(b):
            for edge_index in range(e):
                history = features[batch_index, :, edge_index, :].reshape(-1)
                rows.append(np.concatenate([history, np.ones((1,), dtype=np.float32)]))
                if require_targets:
                    y = batch.speed_targets[batch_index, edge_index]
                    y = np.where(batch.speed_target_mask[batch_index, edge_index], y, 0.0)
                    targets.append(y)
        design = np.stack(rows, axis=0).astype(np.float32)
        if not require_targets:
            return design, np.zeros((design.shape[0], horizons), dtype=np.float32)
        return design, np.stack(targets, axis=0).astype(np.float32)
