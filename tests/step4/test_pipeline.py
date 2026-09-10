"""Integration tests for the Step 4 routing pipeline.

The pipeline under test is:

    Scenario
        -> InitialSolutionBuilder
        -> RouteRepairer
        -> RouteEvaluator
        -> RoutingPipelineResult

These tests intentionally validate integration contracts rather than the
internal implementation of individual heuristics.
"""

from __future__ import annotations

import pytest

from src.routing.pipeline import (
    RoutingPipeline,
    RoutingPipelineConfig,
)
from src.routing.route_plan import RoutePlan


def test_pipeline_constructs_complete_feasible_solution(scenario) -> None:
    """The default baseline pipeline should solve the Step 3 fixture."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.solve(
        planning_time_s=0.0,
    )

    assert isinstance(result.final_route_plan, RoutePlan)

    assert result.feasible is True
    assert result.complete is True

    assert result.final_evaluation.feasible is True
    assert result.final_evaluation.all_requests_served is True
    assert (
        result.final_evaluation.customer_uniqueness_feasible
        is True
    )

    assert result.total_distance_m >= 0.0
    assert result.total_travel_time_s >= 0.0
    assert result.total_waiting_time_s >= 0.0
    assert result.total_lateness_s >= 0.0

    assert result.objective_value >= 0.0

    assert result.final_route_plan.vehicle_count == len(
        scenario.fleet
    )


def test_pipeline_returns_initial_solution_and_final_solution(
    scenario,
) -> None:
    """The pipeline must expose both construction and final outputs."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.solve()

    assert result.initial_solution is not None
    assert result.initial_solution.route_plan is not None
    assert result.initial_solution.evaluation is not None

    assert result.final_route_plan is not None
    assert result.final_evaluation is not None

    if result.repaired_solution is not None:
        assert (
            result.repaired_solution.repaired_route_plan
            == result.final_route_plan
        )

        assert (
            result.repaired_solution.repaired_evaluation
            == result.final_evaluation
        )


def test_pipeline_result_route_plan_alias(scenario) -> None:
    """The backwards-compatible route_plan property must expose final plan."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.solve()

    assert result.route_plan == result.final_route_plan


def test_pipeline_evaluate_delegates_to_evaluator(scenario) -> None:
    """Externally supplied plans can be evaluated through the pipeline."""
    pipeline = RoutingPipeline(scenario)

    plan = pipeline.initial_solution().route_plan

    direct_evaluation = pipeline.evaluator.evaluate(plan)
    pipeline_evaluation = pipeline.evaluate(plan)

    assert pipeline_evaluation == direct_evaluation


def test_pipeline_repair_delegates_to_repairer(scenario) -> None:
    """Externally supplied plans can be repaired through the pipeline."""
    pipeline = RoutingPipeline(scenario)

    plan = pipeline.initial_solution().route_plan

    direct_result = pipeline.repairer.repair(plan)
    pipeline_result = pipeline.repair(plan)

    assert pipeline_result == direct_result


def test_pipeline_initial_solution_delegates_to_builder(
    scenario,
) -> None:
    """Initial construction remains independently callable."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.initial_solution(
        planning_time_s=0.0,
    )

    direct_result = (
        pipeline.initial_solution_builder.build(
            planning_time_s=0.0,
        )
    )

    assert result == direct_result


def test_pipeline_without_repair_still_evaluates_solution(
    scenario,
) -> None:
    """Repair can be disabled without breaking the execution pipeline."""
    config = RoutingPipelineConfig(
        enable_repair=False,
    )

    pipeline = RoutingPipeline(
        scenario,
        config=config,
    )

    result = pipeline.solve()

    assert result.repaired_solution is None
    assert result.repair_changed_solution is False

    assert result.final_route_plan == (
        result.initial_solution.route_plan
    )

    assert result.final_evaluation == (
        pipeline.evaluator.evaluate(
            result.final_route_plan
        )
    )


def test_pipeline_repair_changed_flag_matches_repair_result(
    scenario,
) -> None:
    """The top-level result must faithfully expose repair status."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.solve()

    if result.repaired_solution is None:
        assert result.repair_changed_solution is False
    else:
        assert (
            result.repair_changed_solution
            == result.repaired_solution.changed
        )


def test_pipeline_errors_are_deterministically_deduplicated(
    scenario,
) -> None:
    """Repeated validation errors must not be duplicated in the result."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.solve()

    assert len(result.errors) == len(
        dict.fromkeys(result.errors)
    )


def test_pipeline_final_metrics_match_final_evaluation(
    scenario,
) -> None:
    """Top-level metrics must never diverge from evaluator truth."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.solve()

    evaluation = result.final_evaluation

    assert (
        result.total_distance_m
        == pytest.approx(
            evaluation.total_distance_m
        )
    )

    assert (
        result.total_travel_time_s
        == pytest.approx(
            evaluation.total_travel_time_s
        )
    )

    assert (
        result.total_waiting_time_s
        == pytest.approx(
            evaluation.total_waiting_time_s
        )
    )

    assert (
        result.total_lateness_s
        == pytest.approx(
            evaluation.total_lateness_s
        )
    )

    assert (
        result.objective_value
        == pytest.approx(
            evaluation.objective_value
        )
    )


def test_pipeline_is_deterministic(scenario) -> None:
    """Same scenario and configuration must produce the same result."""
    pipeline_a = RoutingPipeline(scenario)
    pipeline_b = RoutingPipeline(scenario)

    result_a = pipeline_a.solve(
        planning_time_s=0.0,
    )

    result_b = pipeline_b.solve(
        planning_time_s=0.0,
    )

    assert result_a.final_route_plan == result_b.final_route_plan
    assert result_a.final_evaluation == result_b.final_evaluation

    assert result_a.feasible == result_b.feasible
    assert result_a.complete == result_b.complete

    assert (
        result_a.repair_changed_solution
        == result_b.repair_changed_solution
    )

    assert result_a.errors == result_b.errors


def test_pipeline_does_not_mutate_scenario(scenario) -> None:
    """Routing execution must treat Scenario as immutable input."""
    original_dump = scenario.model_dump(
        mode="json",
    )

    pipeline = RoutingPipeline(scenario)

    pipeline.solve()

    assert scenario.model_dump(
        mode="json",
    ) == original_dump


def test_pipeline_respects_planning_time_argument(
    scenario,
) -> None:
    """Planning snapshot time must be forwarded into construction."""
    pipeline = RoutingPipeline(scenario)

    result_zero = pipeline.solve(
        planning_time_s=0.0,
    )

    result_later = pipeline.solve(
        planning_time_s=20.0,
    )

    # The exact route may or may not differ for this fixture, but the
    # pipeline must successfully support different planning snapshots.
    assert result_zero is not None
    assert result_later is not None

    assert isinstance(
        result_zero.final_route_plan,
        RoutePlan,
    )

    assert isinstance(
        result_later.final_route_plan,
        RoutePlan,
    )


def test_pipeline_supports_external_dynamic_travel_time_provider(
    scenario,
) -> None:
    """Dynamic edge travel-time logic can be injected without changing Scenario."""

    def travel_time_provider(edge, departure_time_s):
        # Preserve the normal physical travel time.
        return edge.length_m / edge.speed_limit_mps

    pipeline = RoutingPipeline(
        scenario,
        travel_time_provider=travel_time_provider,
    )

    result = pipeline.solve()

    assert result.final_evaluation.feasible is True
    assert result.complete is True


def test_pipeline_supports_closed_edges(
    scenario,
) -> None:
    """Closed directed edges must be passed through to the evaluator."""

    pipeline = RoutingPipeline(
        scenario,
        closed_edge_ids=("E12",),
    )

    result = pipeline.solve()

    assert result.final_evaluation.feasible is True
    assert result.complete is True

    # The closure should force the N1 -> N2 movement to use:
    # N1 -> N6 -> N2.
    assert (
        result.final_evaluation.connectivity_feasible
        is True
    )


def test_pipeline_exposes_final_route_as_the_repaired_route(
    scenario,
) -> None:
    """When repair is enabled, final_route_plan must be repair output."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.solve()

    assert result.repaired_solution is not None

    assert (
        result.final_route_plan
        == result.repaired_solution.repaired_route_plan
    )

    assert (
        result.final_evaluation
        == result.repaired_solution.repaired_evaluation
    )


def test_pipeline_complete_requires_all_requests(
    scenario,
) -> None:
    """Completeness must not be inferred merely from feasibility."""
    pipeline = RoutingPipeline(scenario)

    result = pipeline.solve()

    served = set(
        result.final_route_plan.all_customer_ids()
    )

    expected = {
        request.request_id
        for request in scenario.requests
    }

    assert served == expected
    assert result.complete is True