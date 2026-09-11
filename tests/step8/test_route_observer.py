from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.optim.common import FitnessResult
from src.optim.instrumentation import OptimizationTraceRecorder
from src.optim.route_observer import (
    InstrumentedRouteFitnessOracle,
    ObservedRouteEvaluation,
    RouteObservationError,
    RoutePopulationTraceAdapter,
)


def plan(*routes):
    from src.routing.route_plan import RoutePlan, VehicleRoute

    return RoutePlan.from_routes(
        tuple(
            VehicleRoute(
                vehicle_id=vehicle_id,
                customer_ids=tuple(customer_ids),
            )
            for vehicle_id, customer_ids in routes
        )
    )


@dataclass
class FakeRouteEngine:
    candidate: object
    calls: int = 0

    def evaluate_keys(
        self,
        position,
        *,
        commitments=None,
        planning_time_s=0.0,
        repair=True,
    ):
        self.calls += 1
        return self.candidate


class FakeOracle:
    def __init__(self, candidate, repair_penalty=0.0):
        self.route_engine = FakeRouteEngine(candidate)
        self.commitments = None
        self.planning_time_s = 0.0
        self.repair = True
        self.repair_penalty = repair_penalty
        self._last_candidate = None

    @property
    def last_candidate(self):
        return self._last_candidate

    @staticmethod
    def _repair_distance(candidate):
        from src.optim.diversity import route_distance

        if candidate.repair_result is None:
            return 0.0

        return route_distance(
            candidate.repair_result.original_route_plan,
            candidate.repair_result.repaired_route_plan,
        ).structure

    def __call__(self, position):
        candidate = self.route_engine.evaluate_keys(
            position,
            commitments=self.commitments,
            planning_time_s=self.planning_time_s,
            repair=self.repair,
        )

        self._last_candidate = candidate

        repair_distance = self._repair_distance(candidate)

        result = FitnessResult(
            fitness=42.0 + self.repair_penalty * repair_distance,
            feasible=True,
            repair_distance=repair_distance,
        )

        return result


def make_candidate():
    decoded = plan(
        (1, (101, 102)),
        (2, (103,)),
    )

    repaired = plan(
        (1, (101,)),
        (2, (102, 103)),
    )

    repair_result = SimpleNamespace(
        original_route_plan=decoded,
        repaired_route_plan=repaired,
    )

    evaluation = SimpleNamespace(
        objective_value=42.0,
        feasible=True,
    )

    return SimpleNamespace(
        decoded=SimpleNamespace(
            route_plan=decoded,
        ),
        repaired_plan=repaired,
        repaired_evaluation=evaluation,
        stored_keys=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        repair_result=repair_result,
    )


def test_wrapper_preserves_original_fitness_result():
    candidate = make_candidate()
    oracle = FakeOracle(candidate)
    wrapped = InstrumentedRouteFitnessOracle(oracle)

    result = wrapped(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )

    assert result.fitness == 42.0
    assert result.feasible is True
    assert result.repair_distance > 0.0

    assert wrapped.calls == 1
    assert len(wrapped.observations) == 1
    assert wrapped.last().candidate is candidate


def test_wrapper_calls_underlying_oracle_exactly_once():
    candidate = make_candidate()
    oracle = FakeOracle(candidate)
    wrapped = InstrumentedRouteFitnessOracle(oracle)

    wrapped(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )

    assert oracle.route_engine.calls == 1
    assert wrapped.calls == 1


def test_wrapper_preserves_repair_penalty():
    candidate = make_candidate()

    baseline_oracle = FakeOracle(candidate)
    penalized_oracle = FakeOracle(
        candidate,
        repair_penalty=10.0,
    )

    baseline = InstrumentedRouteFitnessOracle(
        baseline_oracle
    )
    penalized = InstrumentedRouteFitnessOracle(
        penalized_oracle
    )

    position = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    baseline_result = baseline(position)
    penalized_result = penalized(position)

    assert penalized_result.repair_distance > 0.0
    assert penalized_result.fitness > baseline_result.fitness


def test_observation_is_immutable():
    candidate = make_candidate()

    observation = ObservedRouteEvaluation(
        position=(0.1, 0.2),
        candidate=candidate,
        fitness_result=FitnessResult(fitness=5.0),
    )

    with pytest.raises(AttributeError):
        observation.position = (0.3, 0.4)


def test_adapter_preserves_exact_decoded_and_repaired_plans():
    candidate = make_candidate()
    oracle = FakeOracle(candidate)
    wrapped = InstrumentedRouteFitnessOracle(oracle)

    wrapped(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )

    recorder = OptimizationTraceRecorder(
        lower_bound=0.0,
        upper_bound=1.0,
    )

    adapter = RoutePopulationTraceAdapter(recorder)

    population = adapter.record_population(
        iteration=0,
        evaluations=1,
        observations=wrapped.observations,
    )

    observed = population.candidates[0]

    assert observed.decoded_plan is candidate.decoded.route_plan
    assert observed.repaired_plan is candidate.repaired_plan
    assert observed.fitness == 42.0
    assert observed.repair_distance > 0.0


def test_adapter_does_not_call_oracle():
    candidate = make_candidate()
    oracle = FakeOracle(candidate)
    wrapped = InstrumentedRouteFitnessOracle(oracle)

    wrapped(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )

    recorder = OptimizationTraceRecorder()

    adapter = RoutePopulationTraceAdapter(recorder)

    before = oracle.route_engine.calls

    adapter.record_population(
        iteration=0,
        evaluations=1,
        observations=wrapped.observations,
    )

    assert oracle.route_engine.calls == before


def test_empty_population_is_rejected():
    recorder = OptimizationTraceRecorder()
    adapter = RoutePopulationTraceAdapter(recorder)

    with pytest.raises(RouteObservationError):
        adapter.record_population(
            iteration=0,
            evaluations=0,
            observations=(),
        )


def test_clear_discards_observations_only():
    candidate = make_candidate()
    oracle = FakeOracle(candidate)
    wrapped = InstrumentedRouteFitnessOracle(oracle)

    wrapped(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )

    assert wrapped.calls == 1
    assert oracle.route_engine.calls == 1

    wrapped.clear()

    assert wrapped.calls == 0
    assert wrapped.observations == ()
    assert oracle.route_engine.calls == 1