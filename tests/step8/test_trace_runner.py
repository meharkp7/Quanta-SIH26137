from __future__ import annotations

from types import SimpleNamespace

from src.optim.common import FitnessResult
from src.optim.instrumentation import OptimizationTraceRecorder
from src.optim.qpso import AdaptiveQPSOConfig
from src.optim.route_observer import (
    InstrumentedRouteFitnessOracle,
)
from src.optim.trace_runner import (
    QPSOTraceRunner,
    run_qpso_with_trace,
)
from src.routing.route_plan import RoutePlan, VehicleRoute


def make_plan(route_variant: int) -> RoutePlan:
    """
    Every plan contains exactly the same customers.

    Only the ordering changes, which gives the diversity machinery a
    legitimate phenotype difference without violating its invariant.
    """

    if route_variant % 3 == 0:
        customers = (101, 102, 103, 104)
    elif route_variant % 3 == 1:
        customers = (101, 103, 102, 104)
    else:
        customers = (104, 102, 101, 103)

    return RoutePlan.from_routes(
        (
            VehicleRoute.from_sequence(
                vehicle_id=1,
                customer_ids=customers,
            ),
        )
    )


class FakeCandidate:
    def __init__(
        self,
        route_variant: int,
        repair_distance: float = 0.0,
    ) -> None:
        plan = make_plan(route_variant)

        self.decoded = SimpleNamespace(
            route_plan=plan,
        )
        self.repaired_plan = plan
        self.route_plan = plan
        self.repair_distance = repair_distance


class FakeOracle:
    """
    Deterministic route oracle used only to test trace plumbing.

    It intentionally preserves the same customer set for every candidate.
    """

    def __init__(self) -> None:
        self.last_candidate = None
        self.calls = 0

    def __call__(self, position):
        self.calls += 1

        route_variant = self.calls

        self.last_candidate = FakeCandidate(
            route_variant=route_variant,
            repair_distance=0.1 * route_variant,
        )

        fitness = float(sum(position))

        return FitnessResult(
            fitness=fitness,
            feasible=True,
            route_signature=(
                (1, tuple(make_plan(route_variant).all_customer_ids())),
            ),
            repair_distance=0.1 * route_variant,
        )


def make_config(
    *,
    population_size: int = 3,
    max_evaluations: int = 9,
) -> AdaptiveQPSOConfig:
    return AdaptiveQPSOConfig(
        dimensions=2,
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=population_size,
        max_evaluations=max_evaluations,
        seed=7,
    )


def make_initial_population():
    return (
        (0.1, 0.1),
        (0.2, 0.2),
        (0.3, 0.3),
    )


def test_runner_captures_real_evaluation_stream():
    oracle = FakeOracle()

    result = run_qpso_with_trace(
        make_config(
            population_size=3,
            max_evaluations=9,
        ),
        oracle,
        initial_population=make_initial_population(),
    )

    assert result.optimization_result.evaluations == 9
    assert len(result.observations) == 9
    assert oracle.calls == 9

    assert result.trace.iterations == (0, 1, 2)


def test_initial_population_is_iteration_zero():
    oracle = FakeOracle()

    result = run_qpso_with_trace(
        make_config(
            population_size=3,
            max_evaluations=3,
        ),
        oracle,
        initial_population=make_initial_population(),
    )

    first = result.trace.observations[0]

    assert first.iteration == 0
    assert first.evaluations == 3
    assert len(first.candidates) == 3

    assert first.candidates[0].position == (0.1, 0.1)
    assert first.candidates[1].position == (0.2, 0.2)
    assert first.candidates[2].position == (0.3, 0.3)


def test_population_batches_preserve_exact_positions():
    oracle = FakeOracle()

    initial_population = make_initial_population()

    result = run_qpso_with_trace(
        make_config(
            population_size=3,
            max_evaluations=6,
        ),
        oracle,
        initial_population=initial_population,
    )

    observations = result.observations

    first_batch = result.trace.observations[0]
    second_batch = result.trace.observations[1]

    assert [
        candidate.position
        for candidate in first_batch.candidates
    ] == list(initial_population)

    assert [
        candidate.position
        for candidate in second_batch.candidates
    ] == [
        observations[3].position,
        observations[4].position,
        observations[5].position,
    ]


def test_trace_evaluation_counts_are_cumulative():
    oracle = FakeOracle()

    result = run_qpso_with_trace(
        make_config(
            population_size=2,
            max_evaluations=6,
        ),
        oracle,
        initial_population=(
            (0.1, 0.1),
            (0.2, 0.2),
        ),
    )

    assert result.trace.observations[0].evaluations == 2
    assert result.trace.observations[1].evaluations == 4
    assert result.trace.observations[2].evaluations == 6

    assert len(result.trace.observations) == 3


def test_partial_final_population_is_preserved():
    oracle = FakeOracle()

    result = run_qpso_with_trace(
        make_config(
            population_size=3,
            max_evaluations=8,
        ),
        oracle,
        initial_population=make_initial_population(),
    )

    assert result.optimization_result.evaluations == 8
    assert len(result.observations) == 8

    final_population = result.trace.observations[-1]

    assert final_population.iteration == 2
    assert final_population.evaluations == 8
    assert len(final_population.candidates) == 2


def test_runner_does_not_re_evaluate_candidates():
    oracle = FakeOracle()

    result = run_qpso_with_trace(
        make_config(
            population_size=3,
            max_evaluations=9,
        ),
        oracle,
        initial_population=make_initial_population(),
    )

    assert oracle.calls == result.optimization_result.evaluations
    assert len(result.observations) == oracle.calls


def test_existing_instrumented_oracle_is_reused():
    base_oracle = FakeOracle()
    instrumented = InstrumentedRouteFitnessOracle(base_oracle)

    result = run_qpso_with_trace(
        make_config(
            population_size=3,
            max_evaluations=6,
        ),
        instrumented,
        initial_population=make_initial_population(),
    )

    assert result.optimization_result.evaluations == 6
    assert instrumented.calls == 6
    assert len(result.observations) == 6


def test_custom_recorder_is_supported():
    oracle = FakeOracle()

    recorder = OptimizationTraceRecorder(
        lower_bound=0.0,
        upper_bound=1.0,
    )

    result = run_qpso_with_trace(
        make_config(
            population_size=3,
            max_evaluations=6,
        ),
        oracle,
        initial_population=make_initial_population(),
        recorder=recorder,
    )

    assert result.trace == recorder.snapshot()


def test_trace_final_evaluation_matches_optimizer():
    oracle = FakeOracle()

    result = run_qpso_with_trace(
        make_config(
            population_size=3,
            max_evaluations=7,
        ),
        oracle,
        initial_population=make_initial_population(),
    )

    assert (
        result.trace.observations[-1].evaluations
        == result.optimization_result.evaluations
    )