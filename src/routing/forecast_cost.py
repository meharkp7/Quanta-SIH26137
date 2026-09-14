"""Causal forecast serving and forecast-aware road costs (Step 14)."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Mapping, Iterable

from src.contracts.forecast import Forecast
from src.contracts.scenario import RoadEdge, Scenario
from src.routing.cost_view import CostView


@dataclass(frozen=True)
class ForecastCostPolicy:
    """Validation-tuned policy for using forecast values in routing."""

    uncertainty_weight: float = 0.35
    stale_half_life_s: float = 900.0
    max_forecast_reliance: float = 1.0
    min_speed_ratio: float = 0.10
    extension: str = "current"

    def __post_init__(self) -> None:
        if self.uncertainty_weight < 0 or not isfinite(self.uncertainty_weight):
            raise ValueError("uncertainty_weight must be finite and non-negative")
        if self.stale_half_life_s <= 0 or not isfinite(self.stale_half_life_s):
            raise ValueError("stale_half_life_s must be positive")
        if not 0.0 <= self.max_forecast_reliance <= 1.0:
            raise ValueError("max_forecast_reliance must be in [0, 1]")
        if not 0.0 < self.min_speed_ratio <= 1.0:
            raise ValueError("min_speed_ratio must be in (0, 1]")
        if self.extension not in {"current", "last"}:
            raise ValueError("extension must be current or last")


class ForecastAwareTravelTime:
    """Time-dependent edge provider with explicit closure and uncertainty rules.

    Forecasts contain speed ratios.  For a candidate edge departure, the
    target-time rows are linearly interpolated.  Outside the issued horizon,
    the declared current/last extension is used.  The lower speed bound is
    used as a risk surcharge, then stale/uncertain forecasts are blended back
    toward the current observation.
    """

    def __init__(
        self,
        forecast: Forecast,
        edges: Iterable[RoadEdge],
        *,
        current_speed_mps: Mapping[str, float] | None = None,
        known_closed_edge_ids: Iterable[str] = (),
        policy: ForecastCostPolicy | None = None,
    ) -> None:
        self.forecast = forecast
        self.edges = {str(edge.edge_id): edge for edge in edges}
        self.current_speed_mps = {str(k): float(v) for k, v in (current_speed_mps or {}).items()}
        self.known_closed_edge_ids = frozenset(str(x) for x in known_closed_edge_ids)
        self.policy = policy or ForecastCostPolicy()
        unit = str(forecast.target_unit).lower()
        if unit not in {"speed_ratio", "ratio", "m/s"}:
            raise ValueError("forecast target_unit must be speed_ratio, ratio, or m/s")

        def normalise(edge_id: str, row: tuple[float | None, ...]) -> tuple[float | None, ...]:
            scale = 1.0
            if unit == "m/s":
                edge = self.edges.get(str(edge_id))
                if edge is None:
                    return row
                scale = 1.0 / float(edge.speed_limit_mps)
            return tuple(None if value is None else float(value) * scale for value in row)

        self._row = {str(edge_id): normalise(str(edge_id), row) for edge_id, row in zip(forecast.edge_ids, forecast.prediction)}
        self._lower = {str(edge_id): normalise(str(edge_id), row) for edge_id, row in zip(forecast.edge_ids, forecast.lower_prediction or forecast.prediction)}
        self._upper = {str(edge_id): normalise(str(edge_id), row) for edge_id, row in zip(forecast.edge_ids, forecast.upper_prediction or forecast.prediction)}

    @property
    def closed_edge_ids(self) -> frozenset[str]:
        return self.known_closed_edge_ids

    def _value_at(self, row: tuple[float | None, ...], edge_id: str, departure_time_s: float, fallback: float) -> float:
        times = tuple(float(t) for t in self.forecast.target_times_s)
        values = tuple(None if v is None else float(v) for v in row)
        valid = [(t, v) for t, v in zip(times, values) if v is not None and isfinite(v)]
        if not valid:
            return fallback
        if departure_time_s <= valid[0][0]:
            return valid[0][1]
        if departure_time_s > valid[-1][0]:
            return fallback if self.policy.extension == "current" else valid[-1][1]
        for (t0, v0), (t1, v1) in zip(valid, valid[1:]):
            if t0 <= departure_time_s <= t1:
                alpha = (departure_time_s - t0) / max(1e-9, t1 - t0)
                return v0 + alpha * (v1 - v0)
        return fallback

    def _reliance(self, edge_id: str, departure_time_s: float) -> float:
        age = max(0.0, float(departure_time_s) - float(self.forecast.issued_at_s))
        temporal = exp(-age / self.policy.stale_half_life_s)
        lo = self._value_at(self._lower.get(edge_id, ()), edge_id, departure_time_s, 1.0)
        hi = self._value_at(self._upper.get(edge_id, ()), edge_id, departure_time_s, lo)
        spread = max(0.0, hi - lo)
        return min(self.policy.max_forecast_reliance, temporal / (1.0 + self.policy.uncertainty_weight * spread))

    def __call__(self, edge: RoadEdge, departure_time_s: float) -> float:
        edge_id = str(edge.edge_id)
        if edge_id in self.known_closed_edge_ids:
            # CostView excludes these edges from the graph.  This guard also
            # makes direct provider calls fail closed instead of estimating a
            # traversal through a known closure.
            raise ValueError(f"known closure overrides forecast for edge {edge_id!r}")
        free_flow = float(edge.length_m) / float(edge.speed_limit_mps)
        current = self.current_speed_mps.get(edge_id, float(edge.speed_limit_mps))
        current_ratio = min(1.0, max(self.policy.min_speed_ratio, current / float(edge.speed_limit_mps)))
        predicted = self._value_at(self._row.get(edge_id, ()), edge_id, departure_time_s, current_ratio)
        lower = self._value_at(self._lower.get(edge_id, ()), edge_id, departure_time_s, predicted)
        spread = max(0.0, predicted - lower)
        conservative = max(self.policy.min_speed_ratio, lower)
        reliance = self._reliance(edge_id, departure_time_s)
        ratio = (reliance * conservative) + ((1.0 - reliance) * current_ratio)
        return free_flow / max(self.policy.min_speed_ratio, ratio)


def forecast_cost_view(
    scenario: Scenario,
    forecast: Forecast,
    *,
    current_speed_mps: Mapping[str, float] | None = None,
    known_closed_edge_ids: Iterable[str] = (),
    policy: ForecastCostPolicy | None = None,
) -> CostView:
    """Create the immutable versioned CostView consumed by RouteEvaluator/QPSO."""
    provider = ForecastAwareTravelTime(
        forecast,
        scenario.edges,
        current_speed_mps=current_speed_mps,
        known_closed_edge_ids=known_closed_edge_ids,
        policy=policy,
    )
    closed = set(provider.closed_edge_ids)
    return CostView(
        scenario.edges,
        graph_version=str(scenario.graph_version),
        cost_version="forecast-aware-v1",
        forecast_version=str(forecast.forecast_version),
        closed_edge_ids=closed,
        travel_time_provider=provider,
    )
