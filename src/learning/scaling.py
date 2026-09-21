"""Train-only feature scaling for forecast windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    feature_names: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError("feature axis does not match scaler feature_names")
        return (array - mean) / scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError("feature axis does not match scaler feature_names")
        return array * scale + mean

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mean=tuple(data["mean"]),
            scale=tuple(data["scale"]),
            feature_names=tuple(data["feature_names"]),
        )

    @classmethod
    def fit_windows(cls, windows, feature_names: tuple[str, ...]) -> "FeatureScaler":
        """Fit on training windows without materializing a dense batch.

        Streaming Welford-equivalent accumulation (float64 sum/sumsq/count
        per channel over finite, feature-masked values). Identical semantics
        to ``fit`` on ``where(mask, features, nan)`` but O(1) memory, so
        thousand-episode corpora cannot OOM the loader.
        """
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")
        n_features = len(feature_names)
        total = np.zeros(n_features, dtype=np.float64)
        total_sq = np.zeros(n_features, dtype=np.float64)
        count = np.zeros(n_features, dtype=np.float64)
        seen = False
        for window in windows:
            values = np.asarray(window.features, dtype=np.float64)
            mask = np.asarray(window.feature_mask, dtype=bool)
            if values.ndim != 3 or values.shape[-1] != n_features:
                raise ValueError("window features must have shape [L, E, F]")
            if mask.shape != values.shape:
                raise ValueError("window feature_mask must match features shape")
            seen = True
            for index in range(n_features):
                column = values[..., index]
                valid = mask[..., index] & np.isfinite(column)
                selected = column[valid]
                if selected.size:
                    total[index] += selected.sum()
                    total_sq[index] += np.square(selected).sum()
                    count[index] += selected.size
        if not seen:
            raise ValueError("fit_windows requires at least one window")
        mean = []
        scale = []
        for index, name in enumerate(feature_names):
            if count[index] == 0:
                mean.append(0.0)
                scale.append(1.0)
                continue
            mu = total[index] / count[index]
            var = total_sq[index] / count[index] - mu * mu
            sigma = float(np.sqrt(max(var, 0.0)))
            if not np.isfinite(mu) or not np.isfinite(sigma):
                raise ValueError(
                    f"feature {name!r} produced non-finite scaler statistics"
                )
            mean.append(float(mu))
            scale.append(sigma if sigma > 1e-6 else 1.0)
        return cls(mean=tuple(mean), scale=tuple(scale), feature_names=feature_names)

    @classmethod
    def fit(cls, features: np.ndarray, feature_names: tuple[str, ...]) -> "FeatureScaler":
        """Fit on training windows only. Masked / non-finite values are ignored."""

        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError("features must have shape [B, L, E, F]")
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")
        if array.shape[-1] != len(feature_names):
            raise ValueError("feature axis does not match feature_names")
        mean = []
        scale = []
        for index in range(array.shape[-1]):
            column = array[..., index]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                mean.append(0.0)
                scale.append(1.0)
                continue
            mu = float(finite.mean())
            sigma = float(finite.std())
            if not np.isfinite(mu) or not np.isfinite(sigma):
                raise ValueError(f"feature {feature_names[index]!r} produced non-finite scaler statistics")
            mean.append(mu)
            scale.append(sigma if sigma > 1e-6 else 1.0)
        return cls(mean=tuple(mean), scale=tuple(scale), feature_names=feature_names)
