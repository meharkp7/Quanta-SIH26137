from __future__ import annotations

from typing import Sequence

from src.optim.instrumentation import OptimizationTrace
from src.optim.predictive_analysis import SearchState


class TraceBridgeError(ValueError):
    """Raised when an optimization trace cannot be converted."""


def trace_to_search_states(
    trace: OptimizationTrace,
) -> tuple[SearchState, ...]:
    """
    Convert immutable optimization observations into predictive-analysis
    search states.

    This is a pure projection:
    - no objective evaluations
    - no route decoding
    - no repair
    - no optimizer state changes
    """

    states: list[SearchState] = []

    for observation in trace.observations:
        states.append(
            SearchState(
                iteration=observation.iteration,
                evaluations=observation.evaluations,
                best_fitness=observation.best_fitness,
                coordinate_diversity=(
                    observation.diversity.genotype_coordinate
                ),
                decoded_assignment_diversity=(
                    observation.diversity.decoded_assignment
                ),
                decoded_precedence_diversity=(
                    observation.diversity.decoded_precedence
                ),
                decoded_structure_diversity=(
                    observation.diversity.decoded_structure
                ),
                repaired_assignment_diversity=(
                    observation.diversity.repaired_assignment
                ),
                repaired_precedence_diversity=(
                    observation.diversity.repaired_precedence
                ),
                repaired_structure_diversity=(
                    observation.diversity.repaired_structure
                ),
                repair_pressure=(
                    observation.repair_pressure.structure
                ),
                feasible_rate=observation.feasible_rate,
            )
        )

    return tuple(states)


def trace_to_search_states_strict(
    trace: OptimizationTrace,
) -> tuple[SearchState, ...]:
    """
    Strict conversion requiring a usable optimization trajectory.
    """

    states = trace_to_search_states(trace)

    if not states:
        raise TraceBridgeError(
            "optimization trace contains no observations"
        )

    return states