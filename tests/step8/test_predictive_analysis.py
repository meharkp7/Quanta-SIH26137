import pytest

from src.optim.predictive_analysis import (
    PredictiveAnalysisError,
    SearchState,
    analyze_predictive_relationships,
    build_horizon_samples,
)


def make_state(
    iteration: int,
    best_fitness: float,
    *,
    coordinate_diversity: float = 0.5,
    decoded_assignment_diversity: float = 0.5,
    decoded_precedence_diversity: float = 0.5,
    decoded_structure_diversity: float = 0.5,
    repaired_assignment_diversity: float = 0.5,
    repaired_precedence_diversity: float = 0.5,
    repaired_structure_diversity: float = 0.5,
    repair_pressure: float = 0.1,
    feasible_rate: float = 1.0,
) -> SearchState:
    return SearchState(
        iteration=iteration,
        evaluations=iteration + 1,
        best_fitness=best_fitness,
        coordinate_diversity=coordinate_diversity,
        decoded_assignment_diversity=decoded_assignment_diversity,
        decoded_precedence_diversity=decoded_precedence_diversity,
        decoded_structure_diversity=decoded_structure_diversity,
        repaired_assignment_diversity=repaired_assignment_diversity,
        repaired_precedence_diversity=repaired_precedence_diversity,
        repaired_structure_diversity=repaired_structure_diversity,
        repair_pressure=repair_pressure,
        feasible_rate=feasible_rate,
    )


def test_search_state_validates_normalized_observables():
    with pytest.raises(PredictiveAnalysisError):
        make_state(
            iteration=0,
            best_fitness=10.0,
            coordinate_diversity=1.1,
        )


def test_search_state_rejects_nonfinite_fitness():
    with pytest.raises(PredictiveAnalysisError):
        make_state(
            iteration=0,
            best_fitness=float("inf"),
        )


def test_horizon_label_uses_future_best_fitness():
    states = (
        make_state(0, 100.0),
        make_state(1, 90.0),
        make_state(2, 80.0),
    )

    samples = build_horizon_samples(
        states,
        horizons=(1,),
    )

    assert len(samples) == 2

    assert samples[0].iteration == 0
    assert samples[0].horizon == 1
    assert samples[0].best_fitness == 100.0
    assert samples[0].future_best_fitness == 90.0
    assert samples[0].future_improvement == 10.0

    assert samples[1].iteration == 1
    assert samples[1].future_improvement == 10.0


def test_horizon_label_is_zero_when_no_future_improvement():
    states = (
        make_state(0, 100.0),
        make_state(1, 100.0),
    )

    samples = build_horizon_samples(
        states,
        horizons=(1,),
    )

    assert len(samples) == 1
    assert samples[0].future_improvement == 0.0
    assert samples[0].improved is False


def test_states_are_sorted_by_iteration():
    states = (
        make_state(2, 80.0),
        make_state(0, 100.0),
        make_state(1, 90.0),
    )

    samples = build_horizon_samples(
        states,
        horizons=(1,),
    )

    assert tuple(
        sample.iteration
        for sample in samples
    ) == (0, 1)


def test_duplicate_iterations_are_rejected():
    states = (
        make_state(0, 100.0),
        make_state(0, 90.0),
    )

    with pytest.raises(PredictiveAnalysisError):
        build_horizon_samples(
            states,
            horizons=(1,),
        )


def test_irregular_iteration_spacing_does_not_create_false_horizon_labels():
    states = (
        make_state(0, 100.0),
        make_state(2, 80.0),
    )

    samples = build_horizon_samples(
        states,
        horizons=(1,),
    )

    assert samples == ()


def test_multiple_horizons_are_independently_constructed():
    states = tuple(
        make_state(
            iteration=index,
            best_fitness=100.0 - index,
        )
        for index in range(6)
    )

    samples = build_horizon_samples(
        states,
        horizons=(1, 5),
    )

    assert sum(sample.horizon == 1 for sample in samples) == 5
    assert sum(sample.horizon == 5 for sample in samples) == 1


def test_spearman_detects_monotonic_predictive_signal():
    states = (
        make_state(
            iteration=0,
            best_fitness=100.0,
            coordinate_diversity=0.00,
        ),
        make_state(
            iteration=1,
            best_fitness=90.0,
            coordinate_diversity=0.25,
        ),
        make_state(
            iteration=2,
            best_fitness=70.0,
            coordinate_diversity=0.50,
        ),
        make_state(
            iteration=3,
            best_fitness=40.0,
            coordinate_diversity=0.75,
        ),
        make_state(
            iteration=4,
            best_fitness=0.0,
            coordinate_diversity=1.00,
        ),
    )

    result = analyze_predictive_relationships(
        states,
        horizons=(1,),
    )

    coordinate = next(
        item
        for item in result.correlations
        if item.signal == "coordinate_diversity"
        and item.horizon == 1
    )

    assert coordinate.sample_count == 4
    assert coordinate.coefficient == pytest.approx(1.0)


def test_constant_signal_has_zero_correlation():
    states = tuple(
        make_state(
            iteration=index,
            best_fitness=100.0 - index,
            coordinate_diversity=0.5,
        )
        for index in range(5)
    )

    result = analyze_predictive_relationships(
        states,
        horizons=(1,),
    )

    coordinate = next(
        item
        for item in result.correlations
        if item.signal == "coordinate_diversity"
    )

    assert coordinate.coefficient == 0.0


def test_analysis_contains_all_requested_signals():
    states = tuple(
        make_state(
            iteration=index,
            best_fitness=100.0 - index,
        )
        for index in range(5)
    )

    result = analyze_predictive_relationships(
        states,
        horizons=(1,),
    )

    signals = {
        item.signal
        for item in result.correlations
    }

    assert signals == {
        "coordinate_diversity",
        "decoded_assignment_diversity",
        "decoded_precedence_diversity",
        "decoded_structure_diversity",
        "repaired_assignment_diversity",
        "repaired_precedence_diversity",
        "repaired_structure_diversity",
        "repair_pressure",
        "feasible_rate",
    }


def test_strongest_signal_returns_largest_absolute_relationship():
    states = (
        make_state(
            iteration=0,
            best_fitness=100.0,
            coordinate_diversity=0.00,
            repair_pressure=1.00,
        ),
        make_state(
            iteration=1,
            best_fitness=90.0,
            coordinate_diversity=0.25,
            repair_pressure=0.75,
        ),
        make_state(
            iteration=2,
            best_fitness=70.0,
            coordinate_diversity=0.50,
            repair_pressure=0.50,
        ),
        make_state(
            iteration=3,
            best_fitness=40.0,
            coordinate_diversity=0.75,
            repair_pressure=0.25,
        ),
        make_state(
            iteration=4,
            best_fitness=0.0,
            coordinate_diversity=1.00,
            repair_pressure=0.00,
        ),
    )

    result = analyze_predictive_relationships(
        states,
        horizons=(1,),
    )

    strongest = result.strongest_signal(horizon=1)

    assert strongest is not None
    assert strongest.absolute_coefficient == pytest.approx(1.0)
    assert strongest.signal in {
        "coordinate_diversity",
        "repair_pressure",
    }


def test_invalid_horizon_is_rejected():
    states = (
        make_state(0, 100.0),
        make_state(1, 90.0),
    )

    with pytest.raises(PredictiveAnalysisError):
        build_horizon_samples(
            states,
            horizons=(0,),
        )


def test_empty_states_are_supported():
    result = analyze_predictive_relationships(
        (),
        horizons=(1, 5),
    )

    assert result.samples == ()
    assert result.correlations == ()
    assert result.horizons == (1, 5)