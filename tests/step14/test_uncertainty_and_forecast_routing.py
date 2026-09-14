from __future__ import annotations

import numpy as np
import pytest

from src.contracts.forecast import Forecast
from src.contracts.scenario import RoadEdge
from src.contracts.core_types import RoadClass
from src.learning.uncertainty import attach_intervals, fit_residual_calibration, interval_metrics
from src.routing.cost_view import CostView
from src.routing.forecast_cost import ForecastCostPolicy, ForecastAwareTravelTime


def _edge(edge_id: str, source: str, target: str, length: float = 100.0) -> RoadEdge:
    return RoadEdge(
        edge_id=edge_id, parent_road_id=edge_id, from_node=source, to_node=target,
        length_m=length, road_class=RoadClass.LOCAL, speed_limit_mps=10.0,
        lane_count=1, capacity_veh_per_hour=100.0,
    )


def _forecast(values: tuple[float, ...], *, version: str = "f1") -> Forecast:
    return Forecast(
        scenario_id="s", episode_id="e", forecast_version=version,
        issued_at_s=0.0, target_times_s=(300.0, 600.0),
        edge_ids=("a", "b"),
        prediction=(values, values), target_unit="speed_ratio", model_version="v1",
        valid_mask=((True, True), (True, True)),
    )


def test_residual_intervals_have_measured_held_out_coverage():
    prediction = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    target = np.array([[[0.8, 0.9], [1.2, 1.1]], [[1.1, 0.9], [0.9, 1.2]]])
    mask = np.ones_like(target, dtype=bool)
    calibration = fit_residual_calibration(prediction, target, mask)
    _, lower, upper = (prediction, prediction - np.asarray(calibration.radii), prediction + np.asarray(calibration.radii))
    report = interval_metrics(lower, upper, target, mask)
    assert report["count"] == 8
    assert 0.0 <= report["actual_coverage"] <= 1.0
    assert report["mean_width"] > 0.0


def test_forecast_intervals_are_versioned_and_shape_safe():
    forecast = _forecast((0.8, 0.7))
    calibrated = attach_intervals(forecast, fit_residual_calibration(
        np.ones((2, 2, 2)), np.full((2, 2, 2), 0.8), np.ones((2, 2, 2), dtype=bool)
    ))
    assert calibrated.lower_prediction is not None
    assert calibrated.upper_prediction is not None
    assert calibrated.model_version.endswith("+residual-conformal-v1")
    with pytest.raises(ValueError):
        attach_intervals(forecast, fit_residual_calibration(
            np.ones((1, 1, 1)), np.ones((1, 1, 1)), np.ones((1, 1, 1), dtype=bool)
        ))


def test_forecast_changes_edge_cost_and_route_and_closure_wins():
    edges = (
        _edge("a", "s", "x"), _edge("b", "x", "t"),
        _edge("c", "s", "y", 120.0), _edge("d", "y", "t", 120.0),
    )
    fast_a = _forecast((1.0, 1.0), version="fast")
    slow_a = _forecast((0.25, 0.25), version="slow")
    # The forecast rows are deliberately edge-aligned only for a/b; c/d use
    # the current-speed extension and remain at free flow.
    current = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0}
    fast_provider = ForecastAwareTravelTime(fast_a, edges, current_speed_mps=current)
    slow_provider = ForecastAwareTravelTime(slow_a, edges, current_speed_mps=current)
    fast = CostView(edges, forecast_version="fast", travel_time_provider=fast_provider)
    slow = CostView(edges, forecast_version="slow", travel_time_provider=slow_provider)
    fast_path = fast.shortest_path("s", "t", departure_time_s=300.0)
    slow_path = slow.shortest_path("s", "t", departure_time_s=300.0)
    assert fast_path.edge_ids == ("a", "b")
    assert slow_path.edge_ids == ("c", "d")
    assert slow_path.travel_time_s > fast_path.travel_time_s
    closure = CostView(
        edges, forecast_version="closed", closed_edge_ids=("c",),
        travel_time_provider=ForecastAwareTravelTime(fast_a, edges, current_speed_mps=current, known_closed_edge_ids=("c",)),
    )
    assert closure.shortest_path("s", "t").edge_ids == ("a", "b")


def test_qpso_adapter_keeps_forecast_version_in_routing_state():
    from tests.step6.test_step6_sumo import load_base_scenario
    from src.optim.forecast_routing import optimize_with_forecast
    from src.optim.qpso import AdaptiveQPSOConfig

    scenario = load_base_scenario()
    # The adapter test uses a tiny population and budget; route feasibility is
    # delegated to the same Step7 evaluator used by production experiments.
    rows = tuple((1.0, 1.0) for _ in scenario.edges)
    forecast = Forecast(
        scenario_id=scenario.scenario_id, episode_id="e", forecast_version="qpso-f1",
        issued_at_s=0.0, target_times_s=(300.0, 600.0),
        edge_ids=tuple(edge.edge_id for edge in scenario.edges), prediction=rows,
        target_unit="speed_ratio", model_version="v1",
    )
    config = AdaptiveQPSOConfig(dimensions=2 * len(scenario.requests), population_size=4, max_evaluations=4, seed=3)
    result = optimize_with_forecast(scenario, forecast, qpso_config=config)
    assert result.forecast_version == "qpso-f1"
    assert result.engine.evaluator.cost_view.forecast_version == "qpso-f1"
    assert result.optimization.evaluations == 4
