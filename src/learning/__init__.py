"""Step 12 forecasting loader and baselines.

Windows are [B, L, E, F] with L=12 minutes and targets at 5/10/15 minutes.
Scaling is fit on the training split only.
"""

from src.learning.baselines import PersistenceForecaster, TemporalOnlyForecaster
from src.learning.loader import ForecastBatch, WindowDataset, load_pilot_windows
from src.learning.scaling import FeatureScaler

__all__ = [
    "FeatureScaler",
    "ForecastBatch",
    "PersistenceForecaster",
    "TemporalOnlyForecaster",
    "WindowDataset",
    "load_pilot_windows",
]
