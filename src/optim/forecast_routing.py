"""Step 14 adapter wiring forecast-aware costs into Step 7 and QPSO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from src.contracts.forecast import Forecast
from src.contracts.scenario import Scenario
from src.optim.qpso import AdaptiveQPSO, AdaptiveQPSOConfig, RouteFitnessOracle
from src.routing.forecast_cost import ForecastCostPolicy, forecast_cost_view
from src.routing.route_encoding import RepairConfig, Step7RouteEngine
from src.routing.route_evaluator import RouteEvaluationConfig, RouteEvaluator


@dataclass(frozen=True)
class ForecastQPSOResult:
    """QPSO output plus the exact forecast version used for every evaluation."""

    optimization: object
    oracle: RouteFitnessOracle
    engine: Step7RouteEngine
    forecast_version: str
    cost_policy: ForecastCostPolicy


def optimize_with_forecast(
    scenario: Scenario,
    forecast: Forecast,
    *,
    planning_time_s: float | None = None,
    current_speed_mps: Mapping[str, float] | None = None,
    known_closed_edge_ids: Iterable[str] = (),
    cost_policy: ForecastCostPolicy | None = None,
    evaluation_config: RouteEvaluationConfig | None = None,
    repair_config: RepairConfig | None = None,
    qpso_config: AdaptiveQPSOConfig | None = None,
) -> ForecastQPSOResult:
    """Run QPSO using only a causal forecast and current known state.

    The resulting ``RouteFitnessOracle`` is the same adapter used by the
    existing optimizer experiments; the only changed input is the immutable
    CostView supplied to its RouteEvaluator.
    """
    policy = cost_policy or ForecastCostPolicy()
    view = forecast_cost_view(
        scenario,
        forecast,
        current_speed_mps=current_speed_mps,
        known_closed_edge_ids=known_closed_edge_ids,
        policy=policy,
    )
    evaluator = RouteEvaluator(
        scenario,
        config=evaluation_config,
        cost_view=view,
        closed_edge_ids=known_closed_edge_ids,
    )
    engine = Step7RouteEngine(scenario, evaluator, repair_config=repair_config)
    start = float(forecast.issued_at_s if planning_time_s is None else planning_time_s)
    config = qpso_config or AdaptiveQPSOConfig(
        dimensions=engine.encoder.dimension,
        population_size=max(4, min(12, 2 * len(scenario.requests))),
        max_evaluations=48,
        seed=26137,
    )
    if config.dimensions != engine.encoder.dimension:
        raise ValueError("qpso_config dimensions must equal the route encoder dimension")
    oracle = RouteFitnessOracle(engine, planning_time_s=start, repair=True)
    optimization = AdaptiveQPSO(config, oracle).optimize()
    return ForecastQPSOResult(optimization, oracle, engine, str(forecast.forecast_version), policy)

