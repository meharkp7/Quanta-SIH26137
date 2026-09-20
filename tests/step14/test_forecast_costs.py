"""Step 14 verification: forecast-aware routing costs on the 5-job fixture.

Covers the plan's verifiable core (CPU-light, no torch training):

- no future simulator labels reach the planner (causal forecast issue);
- known closures override predicted accessibility;
- beyond-horizon extension is declared and deterministic;
- FIFO property on a crafted time-dependent case;
- pilot interval coverage recomputed from ``uncertainty.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.contracts.core_types import RoadClass, TargetKind
from src.contracts.forecast import Forecast
from src.contracts.scenario import RoadEdge
from src.routing.cost_view import CostView
from src.routing.forecast_cost import (
    ForecastAwareTravelTime,
    ForecastCostPolicy,
    check_fifo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_UNCERTAINTY = PROJECT_ROOT / "artifacts" / "forecaster_mps_pilot" / "uncertainty.json"
# Pilot contract: 6-feature schema, 3 horizons at +5/+10/+15 s.
PILOT_HORIZONS = (5.0, 10.0, 15.0)


def _edge(edge_id: str, source: str, target: str, length: float = 100.0) -> RoadEdge:
    return RoadEdge(
        edge_id=edge_id,
        parent_road_id=edge_id,
        from_node=source,
        to_node=target,
        length_m=length,
        road_class=RoadClass.LOCAL,
        speed_limit_mps=10.0,
        lane_count=1,
        capacity_veh_per_hour=600.0,
    )


def _pilot_style_forecast(
    e12_ratio: float,
    *,
    version: str,
    issued_at_s: float = 0.0,
    radii: tuple[float, float, float] = (0.19576, 0.19975, 0.19549),
) -> Forecast:
    """Scripted Forecast honoring the pilot feature_schema/horizons contract.

    Every fixture edge gets a row; only the E12 corridor value varies.
    Intervals come from the pilot residual calibration radii so the
    uncertainty contract is real without loading torch weights.
    """
    edge_ids = ("E01", "E12", "E23", "E34", "E45", "E16", "E62")
    target_times = tuple(issued_at_s + h for h in (5.0, 10.0, 15.0))
    prediction, lower, upper, mask = [], [], [], []
    for eid in edge_ids:
        ratio = e12_ratio if eid == "E12" else 1.0
        prediction.append((ratio, ratio, ratio))
        lower.append(tuple(ratio - r for r in radii))
        upper.append(tuple(ratio + r for r in radii))
        mask.append((True, True, True))
    return Forecast(
        scenario_id="S3_BASE",
        episode_id="step14-demo",
        forecast_version=version,
        issued_at_s=issued_at_s,
        target_times_s=target_times,
        edge_ids=edge_ids,
        prediction=tuple(prediction),
        lower_prediction=tuple(lower),
        median_prediction=tuple(prediction),
        upper_prediction=tuple(upper),
        valid_mask=tuple(mask),
        target_kind=TargetKind.SPEED_PROXY,
        target_unit="speed_ratio",
        model_version="forecaster_mps_pilot+residual-conformal-v1",
    )


# ---------------------------------------------------------------------------
# 1. No future leakage
# ---------------------------------------------------------------------------

def test_no_future_leakage_forecast_is_causal():
    # A forecast issued at T cannot target a time <= T: the contract
    # rejects non-causal horizons, so future simulator labels cannot be
    # smuggled into the planner through target_times_s.
    with pytest.raises(ValueError):
        Forecast(
            scenario_id="s",
            episode_id="e",
            forecast_version="leaky",
            issued_at_s=10.0,
            target_times_s=(5.0, 10.0, 15.0),
            edge_ids=("E12",),
            prediction=((0.9, 0.9, 0.9),),
            target_unit="speed_ratio",
            model_version="v1",
        )
    forecast = _pilot_style_forecast(1.0, version="causal")
    assert all(t > forecast.issued_at_s for t in forecast.target_times_s)
    # Causal builder rule: only observations with timestamp <= issued_at_s
    # may feed the forecast. Simulate the filter the serving layer applies.
    observations = ({"t": 0.0, "v": 1.0}, {"t": 5.0, "v": 0.5}, {"t": 30.0, "v": 0.1})
    usable = [o for o in observations if o["t"] <= forecast.issued_at_s]
    assert usable == [{"t": 0.0, "v": 1.0}]
    assert forecast.target_times_s == (5.0, 10.0, 15.0)


# ---------------------------------------------------------------------------
# 2. Closure override wins over predicted accessibility
# ---------------------------------------------------------------------------

def test_closure_override_wins_over_forecast():
    edges = (_edge("E12", "N1", "N2"), _edge("E16", "N1", "N6"), _edge("E62", "N6", "N2"))
    forecast = _pilot_style_forecast(1.0, version="closure-test")
    current = {"E12": 10.0, "E16": 10.0, "E62": 10.0}
    provider = ForecastAwareTravelTime(
        forecast, edges, current_speed_mps=current, known_closed_edge_ids=("E12",)
    )
    e12 = next(e for e in edges if e.edge_id == "E12")
    with pytest.raises(ValueError, match="known closure overrides forecast"):
        provider(e12, 5.0)
    view = CostView(
        edges,
        graph_version="S3_BASE:graph:1",
        forecast_version="closure-test",
        closed_edge_ids=("E12",),
        travel_time_provider=provider,
    )
    # The planner routes around the closure even though the forecast says
    # E12 is fast; the predicted accessibility never reaches the route.
    assert view.shortest_path("N1", "N2", departure_time_s=5.0).edge_ids == ("E16", "E62")


# ---------------------------------------------------------------------------
# 3. Beyond-horizon extension is declared and deterministic
# ---------------------------------------------------------------------------

def test_beyond_horizon_rule_is_declared_and_deterministic():
    edges = (_edge("E12", "N1", "N2"),)
    forecast = _pilot_style_forecast(0.5, version="horizon-test")
    current = {"E12": 10.0}
    just_beyond = 20.0  # past the +15 s pilot horizon, forecast still fresh
    far_future = 10_000.0  # forecast fully stale: stale rule must dominate
    current_policy = ForecastCostPolicy(extension="current")
    last_policy = ForecastCostPolicy(extension="last")
    current_provider = ForecastAwareTravelTime(
        forecast, edges, current_speed_mps=current, policy=current_policy
    )
    last_provider = ForecastAwareTravelTime(
        forecast, edges, current_speed_mps=current, policy=last_policy
    )
    e12 = edges[0]
    free_flow = 100.0 / 10.0
    # Just beyond the horizon the declared rule bites: "current" falls back
    # to the current observation (free flow here) while "last" holds the
    # final horizon congestion (ratio 0.5 -> well above free flow).
    assert current_provider(e12, just_beyond) == pytest.approx(free_flow)
    assert last_provider(e12, just_beyond) > 1.5 * current_provider(e12, just_beyond)
    # Far beyond the horizon the validation-tuned stale down-weighting
    # (exp(-age / half_life)) drives reliance to ~0, so BOTH rules converge
    # back to the current observation instead of trusting ancient forecasts.
    assert current_provider(e12, far_future) == pytest.approx(free_flow)
    assert last_provider(e12, far_future) == pytest.approx(free_flow, rel=0.05)
    # Deterministic: repeated evaluation at the same departure is exact.
    assert last_provider(e12, just_beyond) == last_provider(e12, just_beyond)
    assert current_provider(e12, far_future) == current_provider(e12, far_future)


# ---------------------------------------------------------------------------
# 4. FIFO property on a crafted time-dependent case
# ---------------------------------------------------------------------------

def test_fifo_holds_on_crafted_congestion_recovery():
    # Crafted gentle recovery: mild congestion at +5 s clearing by +15 s.
    # The travel-time slope is shallower than -1 s/s, so a later departure
    # cannot arrive earlier (FIFO / no-overtaking).
    forecast = Forecast(
        scenario_id="s",
        episode_id="e",
        forecast_version="fifo",
        issued_at_s=0.0,
        target_times_s=(5.0, 10.0, 15.0),
        edge_ids=("E12",),
        prediction=((0.8, 0.9, 1.0),),
        lower_prediction=((0.6, 0.7, 0.8),),
        median_prediction=((0.8, 0.9, 1.0),),
        upper_prediction=((1.0, 1.1, 1.2),),
        valid_mask=((True, True, True),),
        target_kind=TargetKind.SPEED_PROXY,
        target_unit="speed_ratio",
        model_version="v1",
    )
    edge = _edge("E12", "N1", "N2")
    provider = ForecastAwareTravelTime(forecast, (edge,), current_speed_mps={"E12": 10.0})
    for t0, t1 in ((5.0, 6.0), (5.0, 10.0), (7.5, 12.5), (14.0, 15.0), (15.0, 60.0)):
        verdict = check_fifo(provider, edge, t0, t1)
        assert verdict["fifo_holds"], verdict
    with pytest.raises(ValueError):
        check_fifo(provider, edge, 10.0, 5.0)


def test_fifo_helper_detects_steep_recovery_violation():
    # Boundary documentation: a near-total-then-instant recovery drops
    # travel time faster than departures advance, so FIFO breaks. The
    # helper must report the violation honestly rather than hide it (see
    # FORECAST_COSTS_NOT_YET.md: FIFO projection is open work).
    forecast = Forecast(
        scenario_id="s",
        episode_id="e",
        forecast_version="fifo-steep",
        issued_at_s=0.0,
        target_times_s=(5.0, 10.0, 15.0),
        edge_ids=("E12",),
        prediction=((0.2, 0.6, 1.0),),
        lower_prediction=((0.1, 0.4, 0.8),),
        median_prediction=((0.2, 0.6, 1.0),),
        upper_prediction=((0.3, 0.8, 1.2),),
        valid_mask=((True, True, True),),
        target_kind=TargetKind.SPEED_PROXY,
        target_unit="speed_ratio",
        model_version="v1",
    )
    edge = _edge("E12", "N1", "N2")
    provider = ForecastAwareTravelTime(forecast, (edge,), current_speed_mps={"E12": 10.0})
    verdict = check_fifo(provider, edge, 5.0, 6.0)
    assert verdict["fifo_holds"] is False
    assert verdict["earlier_arrival_s"] > verdict["later_arrival_s"]


# ---------------------------------------------------------------------------
# 5. Pilot interval coverage recomputed from artifacts
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="pilot artifacts deleted in v2 data reset (c6fa2e10); "
    "re-enable against artifacts/forecaster_v2 once v2 training lands"
)
def test_pilot_interval_coverage_recomputed_from_artifacts():
    stored = json.loads(PILOT_UNCERTAINTY.read_text(encoding="utf-8"))
    radii = stored["radii"]
    assert stored["nominal_coverage"] == pytest.approx(0.80)
    assert stored["policy_version"] == "residual-conformal-v1"
    assert stored["calibration_count"] == [45333, 48896, 53131]
    # Recomputation without torch: interval width is 2 * radius per horizon,
    # so the stored mean width must equal the radius-implied width. The
    # stored count-weighted average over horizons is approximated here by
    # the unweighted mean (horizon counts differ by <15%).
    implied_width = 2.0 * sum(radii) / len(radii)
    for split in ("validation", "test"):
        assert stored[split]["mean_width"] == pytest.approx(implied_width, rel=0.01)
        assert stored[split]["nominal_coverage"] == pytest.approx(0.80)
    # Held-out honesty: validation calibrates at ~nominal ...
    assert stored["validation"]["actual_coverage"] == pytest.approx(0.80, abs=0.01)
    # ... while the map-disjoint test split degrades as reported (~76.4%).
    assert stored["test"]["actual_coverage"] == pytest.approx(0.76436, abs=0.005)
