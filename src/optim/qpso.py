from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
import random
from typing import Sequence

from src.optim.common import (
    FitnessOracle,
    FitnessResult,
    OptimizationConfig,
    OptimizationError,
    OptimizationResult,
    Vector,
    clamp_vector,
    coordinate_diversity,
    route_diversity,
)
from src.optim.diversity import route_distance
from src.optim.qpso_mechanisms import (
    AlphaMode,
    AttractorMode,
    MbestMode,
    RecoveryMode,
    QPSOMechanismConfig,
    compute_alpha,
    compute_attractor,
    compute_mbest,
    should_recover,
)

@dataclass(frozen=True)
class AdaptiveQPSOConfig(OptimizationConfig):
    """
    Route-oriented adaptive QPSO configuration.

    The core remains canonical QPSO. Adaptation changes the contraction-
    expansion coefficient and local-attractor composition rather than
    replacing the quantum update equation.
    """

    alpha_max: float = 0.95
    alpha_min: float = 0.45

    progress_weight: float = 0.45
    diversity_weight: float = 0.30
    stagnation_weight: float = 0.25

    pbest_weight: float = 0.45
    gbest_weight: float = 0.35
    mbest_weight: float = 0.20

    recovery_probability: float = 0.10
    recovery_scale: float = 0.35

    # Explicit mechanism selection for controlled ablations.
    # The default preserves the existing adaptive implementation.
    mechanisms: QPSOMechanismConfig = QPSOMechanismConfig(
        alpha_mode=AlphaMode.ADAPTIVE,
        attractor_mode=AttractorMode.BLENDED,
        mbest_mode=MbestMode.WEIGHTED,
        recovery_mode=RecoveryMode.ADAPTIVE_BURST,
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not (
            0.0 < self.alpha_min <= self.alpha_max < 1.5
        ):
            raise OptimizationError(
                "alpha bounds must satisfy 0 < alpha_min <= alpha_max < 1.5"
            )

        weights = (
            self.progress_weight,
            self.diversity_weight,
            self.stagnation_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise OptimizationError(
                "adaptation weights cannot be negative"
            )

        if abs(sum(weights) - 1.0) > 1e-9:
            raise OptimizationError(
                "progress/diversity/stagnation weights must sum to 1"
            )

        attractor_weights = (
            self.pbest_weight,
            self.gbest_weight,
            self.mbest_weight,
        )

        if any(weight < 0.0 for weight in attractor_weights):
            raise OptimizationError(
                "attractor weights cannot be negative"
            )

        if abs(sum(attractor_weights) - 1.0) > 1e-9:
            raise OptimizationError(
                "attractor weights must sum to 1"
            )

        if not 0.0 <= self.recovery_probability <= 1.0:
            raise OptimizationError(
                "recovery_probability must be in [0, 1]"
            )

        if not 0.0 <= self.recovery_scale <= 1.0:
            raise OptimizationError(
                "recovery_scale must be in [0, 1]"
            )

        if not isinstance(self.mechanisms, QPSOMechanismConfig):
            raise OptimizationError(
                "mechanisms must be a QPSOMechanismConfig"
            )


@dataclass
class _Particle:
    position: list[float]
    pbest_position: list[float]
    pbest_result: FitnessResult


class AdaptiveQPSO:
    """
    Adaptive quantum-behaved particle swarm optimizer.

    Objective convention:
        lower fitness is better.

    The implementation deliberately separates:
      * genotype-space search,
      * decoded phenotype diversity,
      * objective evaluation.

    This makes it possible to ablate route-level diversity without changing
    the underlying evaluator.
    """

    def __init__(
        self,
        config: AdaptiveQPSOConfig,
        oracle: FitnessOracle,
        initial_population: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.config = config
        self.oracle = oracle
        self._rng = random.Random(config.seed)

        self._initial_population = (
            None
            if initial_population is None
            else tuple(tuple(float(x) for x in row) for row in initial_population)
        )

        if self._initial_population is not None:
            if len(self._initial_population) != config.population_size:
                raise OptimizationError(
                    "initial_population size must equal population_size"
                )

            for position in self._initial_population:
                if len(position) != config.dimensions:
                    raise OptimizationError(
                        "initial_population dimension mismatch"
                    )

    def _initial_positions(self) -> list[list[float]]:
        if self._initial_population is not None:
            return [
                list(
                    clamp_vector(
                        position,
                        self.config.lower_bound,
                        self.config.upper_bound,
                    )
                )
                for position in self._initial_population
            ]

        return [
            [
                self._rng.uniform(
                    self.config.lower_bound,
                    self.config.upper_bound,
                )
                for _ in range(self.config.dimensions)
            ]
            for _ in range(self.config.population_size)
        ]

    def _weighted_mbest(
        self,
        particles: Sequence[_Particle],
    ) -> Vector:
        """Compute the configured personal-best mean attractor."""
        return compute_mbest(
            particles,
            dimensions=self.config.dimensions,
            mode=self.config.mechanisms.mbest_mode,
        )

    def _adaptive_alpha(
        self,
        *,
        coordinate_div: float,
        route_div: float,
        progress: float,
        stagnation: int,
    ) -> float:
        """Compute the configured contraction-expansion coefficient."""
        return compute_alpha(
            config=self.config,
            coordinate_div=coordinate_div,
            route_div=route_div,
            progress=progress,
            stagnation=stagnation,
            mode=self.config.mechanisms.alpha_mode,
        )

    def _attractor(
        self,
        particle: _Particle,
        global_best: _Particle,
        mbest: Sequence[float],
    ) -> Vector:
        """Compute the configured local/global/mean-best attractor."""
        return compute_attractor(
            particle=particle,
            global_best=global_best,
            mbest=mbest,
            dimensions=self.config.dimensions,
            mode=self.config.mechanisms.attractor_mode,
            pbest_weight=self.config.pbest_weight,
            gbest_weight=self.config.gbest_weight,
            mbest_weight=self.config.mbest_weight,
        )

    def _recover(
        self,
        position: Sequence[float],
        mbest: Sequence[float],
    ) -> list[float]:
        """
        Bounded quantum exploration burst.

        Recovery mechanics are selected by the configured recovery mode.
        """
        if self.config.mechanisms.recovery_mode is RecoveryMode.NONE:
            return list(position)

        scale = (
            self.config.recovery_scale
            * (
                self.config.upper_bound
                - self.config.lower_bound
            )
        )

        return [
            value
            + self._rng.gauss(0.0, scale)
            + self._rng.uniform(
                -0.25 * scale,
                0.25 * scale,
            ) * (mbest[d] - value)
            for d, value in enumerate(position)
        ]

    def _evaluate_population(
        self,
        positions: Sequence[Sequence[float]],
        particles: list[_Particle],
        evaluations: int,
        *,
        update_pbests: bool,
    ) -> tuple[list[FitnessResult], int]:
        results: list[FitnessResult] = []

        for index, position in enumerate(positions):
            if evaluations >= self.config.max_evaluations:
                break

            result = self.oracle(position)
            evaluations += 1
            results.append(result)

            if update_pbests and result.fitness < (
                particles[index].pbest_result.fitness
                - self.config.tolerance
            ):
                particles[index].pbest_position = list(position)
                particles[index].pbest_result = result

        return results, evaluations

    def optimize(self) -> OptimizationResult:
        positions = self._initial_positions()

        particles: list[_Particle] = []

        evaluations = 0

        # Initial population is always evaluated in full.
        for position in positions:
            result = self.oracle(position)
            evaluations += 1

            particles.append(
                _Particle(
                    position=list(position),
                    pbest_position=list(position),
                    pbest_result=result,
                )
            )

        global_best = min(
            particles,
            key=lambda particle: particle.pbest_result.fitness,
        )

        history_best: list[float] = [
            global_best.pbest_result.fitness
        ]
        history_mean: list[float] = [
            sum(
                particle.pbest_result.fitness
                for particle in particles
            )
            / len(particles)
        ]
        history_coordinate: list[float] = [
            coordinate_diversity(
                [particle.position for particle in particles]
            )
        ]
        history_route: list[float] = [
            route_diversity(
                [particle.pbest_result for particle in particles]
            )
        ]

        iterations = 0
        stagnation = 0
        previous_best = global_best.pbest_result.fitness

        while evaluations < self.config.max_evaluations:
            iterations += 1

            mbest = self._weighted_mbest(particles)

            current_coordinate_div = coordinate_diversity(
                [particle.position for particle in particles]
            )
            current_route_div = route_diversity(
                [particle.pbest_result for particle in particles]
            )

            improvement = max(
                0.0,
                previous_best
                - global_best.pbest_result.fitness,
            )

            scale = max(abs(previous_best), 1.0)
            progress = min(1.0, improvement / scale)

            if global_best.pbest_result.fitness < (
                previous_best - self.config.tolerance
            ):
                stagnation = 0
            else:
                stagnation += 1

            alpha = self._adaptive_alpha(
                coordinate_div=current_coordinate_div,
                route_div=current_route_div,
                progress=progress,
                stagnation=stagnation,
            )

            previous_best = global_best.pbest_result.fitness

            candidate_positions: list[list[float]] = []

            for particle in particles:
                attractor = self._attractor(
                    particle,
                    global_best,
                    mbest,
                )

                candidate: list[float] = []

                for d in range(self.config.dimensions):
                    u = max(
                        self._rng.random(),
                        1e-15,
                    )

                    sign = (
                        -1.0
                        if self._rng.random() < 0.5
                        else 1.0
                    )

                    distance = abs(
                        attractor[d] - particle.position[d]
                    )

                    value = (
                        attractor[d]
                        + sign
                        * alpha
                        * distance
                        * __import__("math").log(1.0 / u)
                    )

                    candidate.append(value)

                if should_recover(
                    config=self.config,
                    stagnation=stagnation,
                    rng=self._rng,
                ):
                    candidate = self._recover(
                        candidate,
                        mbest,
                    )

                candidate_positions.append(
                    list(
                        clamp_vector(
                            candidate,
                            self.config.lower_bound,
                            self.config.upper_bound,
                        )
                    )
                )

            # Hard budget: evaluate only as many particles as remain.
            remaining = self.config.max_evaluations - evaluations

            for index in range(
                min(len(candidate_positions), remaining)
            ):
                result = self.oracle(candidate_positions[index])
                evaluations += 1

                particles[index].position = candidate_positions[index]

                if result.fitness < (
                    particles[index].pbest_result.fitness
                    - self.config.tolerance
                ):
                    particles[index].pbest_position = list(
                        candidate_positions[index]
                    )
                    particles[index].pbest_result = result

                if result.fitness < (
                    global_best.pbest_result.fitness
                    - self.config.tolerance
                ):
                    global_best = particles[index]

            history_best.append(
                global_best.pbest_result.fitness
            )

            history_mean.append(
                sum(
                    particle.pbest_result.fitness
                    for particle in particles
                )
                / len(particles)
            )

            history_coordinate.append(
                coordinate_diversity(
                    [particle.position for particle in particles]
                )
            )

            history_route.append(
                route_diversity(
                    [particle.pbest_result for particle in particles]
                )
            )

            if evaluations >= self.config.max_evaluations:
                break

        return OptimizationResult(
            best_position=tuple(global_best.pbest_position),
            best_fitness=global_best.pbest_result.fitness,
            best_result=global_best.pbest_result,
            evaluations=evaluations,
            iterations=iterations,
            history_best=tuple(history_best),
            history_mean=tuple(history_mean),
            history_coordinate_diversity=tuple(
                history_coordinate
            ),
            history_route_diversity=tuple(history_route),
            seed=self.config.seed,
        )


class RouteFitnessOracle(FitnessOracle):
    """
    Adapter from Step 7's route engine to the optimizer boundary.
    """

    def __init__(
        self,
        route_engine,
        *,
        commitments=None,
        planning_time_s: float = 0.0,
        repair: bool = True,
        repair_penalty: float = 0.0,
    ) -> None:
        self.route_engine = route_engine
        self.commitments = commitments
        self.planning_time_s = planning_time_s
        self.repair = repair
        self.repair_penalty = float(repair_penalty)
        self._last_candidate = None

        if not isfinite(self.repair_penalty):
            raise OptimizationError(
                "repair_penalty must be finite"
            )

        if self.repair_penalty < 0.0:
            raise OptimizationError(
                "repair_penalty must be non-negative"
            )

        super().__init__(self._evaluate_route)

    @staticmethod
    def _route_signature(candidate):
        return tuple(
            (
                vehicle_route.vehicle_id,
                tuple(vehicle_route.customer_ids),
            )
            for vehicle_route in candidate.repaired_plan.vehicle_routes
        )

    def _evaluate_route(
        self,
        position: Vector,
    ) -> FitnessResult:
        candidate = self.route_engine.evaluate_keys(
            position,
            commitments=self.commitments,
            planning_time_s=self.planning_time_s,
            repair=self.repair,
        )
        self._last_candidate = candidate

        evaluation = candidate.repaired_evaluation

        repair_distance = 0.0

        if candidate.repair_result is not None:
            repair_distance = route_distance(
                candidate.repair_result.original_route_plan,
                candidate.repair_result.repaired_route_plan,
            ).structure

        penalty = self.repair_penalty * repair_distance

        return FitnessResult(
            fitness=float(evaluation.objective_value) + penalty,
            feasible=bool(evaluation.feasible),
            route_signature=self._route_signature(candidate),
            repair_distance=float(repair_distance),
        )

    @property
    def last_candidate(self):
        """Most recently evaluated RouteCandidate, for observational tooling."""

        return self._last_candidate