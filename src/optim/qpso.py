from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
import random
from typing import Callable, Sequence

from src.optim.common import (
    FitnessOracle,
    FitnessResult,
    OptimizationConfig,
    OptimizationError,
    OptimizationResult,
    Vector,
    clamp_vector,
    coordinate_diversity,
    fitness_better,
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
    Configuration for the adaptive QPSO optimizer.

    The optimizer operates over a bounded continuous search space. For the
    routing problem, the resulting vectors are interpreted by the route
    encoding and repair layer.

    Selection is feasibility-first:

        feasible > infeasible

    and only then:

        lower fitness > higher fitness
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

    mechanisms: QPSOMechanismConfig = QPSOMechanismConfig(
        alpha_mode=AlphaMode.ADAPTIVE,
        attractor_mode=AttractorMode.BLENDED,
        mbest_mode=MbestMode.WEIGHTED,
        recovery_mode=RecoveryMode.ADAPTIVE_BURST,
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if not (
            0.0
            < float(self.alpha_min)
            <= float(self.alpha_max)
            < 1.5
        ):
            raise OptimizationError(
                "alpha bounds must satisfy "
                "0 < alpha_min <= alpha_max < 1.5"
            )

        adaptation_weights = (
            float(self.progress_weight),
            float(self.diversity_weight),
            float(self.stagnation_weight),
        )

        if any(
            not isfinite(weight)
            for weight in adaptation_weights
        ):
            raise OptimizationError(
                "adaptation weights must be finite"
            )

        if any(
            weight < 0.0
            for weight in adaptation_weights
        ):
            raise OptimizationError(
                "adaptation weights cannot be negative"
            )

        if abs(
            sum(adaptation_weights) - 1.0
        ) > 1e-9:
            raise OptimizationError(
                "progress/diversity/stagnation weights must sum to 1"
            )

        attractor_weights = (
            float(self.pbest_weight),
            float(self.gbest_weight),
            float(self.mbest_weight),
        )

        if any(
            not isfinite(weight)
            for weight in attractor_weights
        ):
            raise OptimizationError(
                "attractor weights must be finite"
            )

        if any(
            weight < 0.0
            for weight in attractor_weights
        ):
            raise OptimizationError(
                "attractor weights cannot be negative"
            )

        if abs(
            sum(attractor_weights) - 1.0
        ) > 1e-9:
            raise OptimizationError(
                "attractor weights must sum to 1"
            )

        if not (
            0.0
            <= float(self.recovery_probability)
            <= 1.0
        ):
            raise OptimizationError(
                "recovery_probability must be in [0, 1]"
            )

        if not (
            0.0
            <= float(self.recovery_scale)
            <= 1.0
        ):
            raise OptimizationError(
                "recovery_scale must be in [0, 1]"
            )

        if not isinstance(
            self.mechanisms,
            QPSOMechanismConfig,
        ):
            raise OptimizationError(
                "mechanisms must be a QPSOMechanismConfig"
            )


@dataclass
class _Particle:
    """
    Mutable internal QPSO particle state.
    """

    position: list[float]
    pbest_position: list[float]
    pbest_result: FitnessResult


class AdaptiveQPSO:
    """
    Quantum-behaved particle swarm optimizer.

    Canonical QPSO update:

        m_d = mean_p(P_pd)

        c_pd =
            phi_pd P_pd
            + (1 - phi_pd) B_d

        X'_pd =
            c_pd
            ± alpha |m_d - X_pd| ln(1 / u_pd)

    where:

        phi_pd ~ U(0, 1)
        u_pd   ~ U(0, 1)
        sign   ~ {-1, +1}

    The optimizer owns the seeded random-number generator.

    The fitness boundary is intentionally duck-typed. Any callable object
    that returns FitnessResult is accepted. This is required for instrumented
    oracle wrappers used by the trace infrastructure.
    """

    def __init__(
        self,
        config: AdaptiveQPSOConfig,
        oracle: Callable[[Sequence[float]], FitnessResult],
        initial_population: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if not isinstance(
            config,
            AdaptiveQPSOConfig,
        ):
            raise OptimizationError(
                "config must be an AdaptiveQPSOConfig"
            )

        # IMPORTANT:
        # Do not require isinstance(oracle, FitnessOracle).
        #
        # The tracing layer wraps FitnessOracle instances with
        # InstrumentedRouteFitnessOracle. The wrapper intentionally exposes
        # the same callable contract without necessarily inheriting from
        # FitnessOracle.
        if not callable(oracle):
            raise OptimizationError(
                "oracle must be callable and return FitnessResult"
            )

        self.config = config
        self.oracle = oracle

        self._rng = random.Random(
            config.seed
        )

        self._initial_population = (
            None
            if initial_population is None
            else tuple(
                tuple(
                    float(value)
                    for value in row
                )
                for row in initial_population
            )
        )

        if self._initial_population is not None:
            self._validate_initial_population(
                self._initial_population
            )

        self._current_stagnation = 0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_initial_population(
        self,
        population: Sequence[Sequence[float]],
    ) -> None:
        if len(population) != self.config.population_size:
            raise OptimizationError(
                "initial_population size must equal population_size"
            )

        for index, position in enumerate(population):
            if len(position) != self.config.dimensions:
                raise OptimizationError(
                    "initial_population dimension mismatch "
                    f"at index {index}: expected "
                    f"{self.config.dimensions}, got {len(position)}"
                )

            for dimension, value in enumerate(position):
                if not isfinite(
                    float(value)
                ):
                    raise OptimizationError(
                        "initial_population contains a "
                        f"non-finite value at "
                        f"[{index}][{dimension}]"
                    )

    @staticmethod
    def _validate_fitness_result(
        result: FitnessResult,
    ) -> None:
        if not isinstance(
            result,
            FitnessResult,
        ):
            raise OptimizationError(
                "fitness oracle must return FitnessResult"
            )

        if not isfinite(
            float(result.fitness)
        ):
            raise OptimizationError(
                "fitness oracle returned non-finite fitness"
            )

        if not isfinite(
            float(result.repair_distance)
        ):
            raise OptimizationError(
                "fitness oracle returned non-finite repair distance"
            )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initial_positions(
        self,
    ) -> list[list[float]]:
        """
        Construct the initial population.
        """
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
                for _ in range(
                    self.config.dimensions
                )
            ]
            for _ in range(
                self.config.population_size
            )
        ]

    def _initialize_particles(
        self,
    ) -> tuple[list[_Particle], int]:
        positions = self._initial_positions()

        particles: list[_Particle] = []
        evaluations = 0

        for position in positions:
            if (
                evaluations
                >= self.config.max_evaluations
            ):
                raise OptimizationError(
                    "evaluation budget exhausted during "
                    "initial population evaluation"
                )

            result = self.oracle(
                position
            )

            self._validate_fitness_result(
                result
            )

            evaluations += 1

            particles.append(
                _Particle(
                    position=list(position),
                    pbest_position=list(position),
                    pbest_result=result,
                )
            )

        return particles, evaluations

    # ------------------------------------------------------------------
    # Mechanisms
    # ------------------------------------------------------------------

    def _weighted_mbest(
        self,
        particles: Sequence[_Particle],
    ) -> Vector:
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
        alpha = compute_alpha(
            config=self.config,
            coordinate_div=coordinate_div,
            route_div=route_div,
            progress=progress,
            stagnation=stagnation,
            mode=self.config.mechanisms.alpha_mode,
        )

        if not isfinite(alpha):
            raise OptimizationError(
                "computed QPSO alpha is non-finite"
            )

        if alpha < 0.0:
            raise OptimizationError(
                "computed QPSO alpha cannot be negative"
            )

        return alpha

    def _sample_phi(
        self,
    ) -> Vector:
        """
        Generate independent coordinate-wise phi values.

        The values are kept strictly inside (0, 1), avoiding degenerate
        endpoint interpolation while preserving the intended Uniform(0, 1)
        distribution for practical purposes.
        """
        epsilon = 1e-12

        return tuple(
            min(
                1.0 - epsilon,
                max(
                    epsilon,
                    self._rng.random(),
                ),
            )
            for _ in range(
                self.config.dimensions
            )
        )

    def _sample_u(
        self,
    ) -> float:
        """
        Generate the logarithmic QPSO random variable.
        """
        epsilon = 1e-15

        return max(
            epsilon,
            self._rng.random(),
        )

    def _sample_sign(
        self,
    ) -> float:
        """
        Generate the QPSO direction.
        """
        return (
            -1.0
            if self._rng.random() < 0.5
            else 1.0
        )

    def _attractor(
        self,
        particle: _Particle,
        global_best: _Particle,
        mbest: Sequence[float],
        *,
        phi: Sequence[float] | None = None,
    ) -> Vector:
        return compute_attractor(
            particle=particle,
            global_best=global_best,
            mbest=mbest,
            dimensions=self.config.dimensions,
            mode=self.config.mechanisms.attractor_mode,
            pbest_weight=self.config.pbest_weight,
            gbest_weight=self.config.gbest_weight,
            mbest_weight=self.config.mbest_weight,
            phi=phi,
        )

    def _recover(
        self,
        position: Sequence[float],
        mbest: Sequence[float],
    ) -> list[float]:
        """
        Apply a bounded exploratory recovery burst.
        """
        if (
            self.config.mechanisms.recovery_mode
            is RecoveryMode.NONE
        ):
            return list(position)

        if len(position) != self.config.dimensions:
            raise OptimizationError(
                "recovery position dimension mismatch"
            )

        if len(mbest) != self.config.dimensions:
            raise OptimizationError(
                "recovery mean-best dimension mismatch"
            )

        search_range = (
            self.config.upper_bound
            - self.config.lower_bound
        )

        scale = (
            self.config.recovery_scale
            * search_range
        )

        recovered: list[float] = []

        for dimension, value in enumerate(position):
            direction = (
                float(mbest[dimension])
                - float(value)
            )

            attraction_component = (
                self._rng.uniform(
                    -0.25 * scale,
                    0.25 * scale,
                )
                * direction
            )

            gaussian_component = self._rng.gauss(
                0.0,
                scale,
            )

            recovered.append(
                float(value)
                + attraction_component
                + gaussian_component
            )

        return recovered

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _better(
        self,
        candidate: FitnessResult,
        incumbent: FitnessResult,
    ) -> bool:
        return fitness_better(
            candidate,
            incumbent,
            tolerance=self.config.tolerance,
        )

    def _best_particle_index(
        self,
        particles: Sequence[_Particle],
    ) -> int:
        if not particles:
            raise OptimizationError(
                "cannot select best particle from empty population"
            )

        best_index = 0

        for index in range(
            1,
            len(particles),
        ):
            if self._better(
                particles[index].pbest_result,
                particles[best_index].pbest_result,
            ):
                best_index = index

        return best_index

    @staticmethod
    def _copy_particle(
        particle: _Particle,
    ) -> _Particle:
        """
        Detach the global-best snapshot from mutable particle state.
        """
        return _Particle(
            position=list(
                particle.position
            ),
            pbest_position=list(
                particle.pbest_position
            ),
            pbest_result=particle.pbest_result,
        )

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidate(
        self,
        *,
        particle: _Particle,
        global_best: _Particle,
        mbest: Sequence[float],
        alpha: float,
    ) -> list[float]:
        """
        Generate one candidate using the configured QPSO mechanism.
        """
        if (
            self.config.mechanisms.attractor_mode
            is AttractorMode.CANONICAL
        ):
            phi = self._sample_phi()

            attractor = self._attractor(
                particle,
                global_best,
                mbest,
                phi=phi,
            )
        else:
            attractor = self._attractor(
                particle,
                global_best,
                mbest,
            )

        candidate: list[float] = []

        for dimension in range(
            self.config.dimensions
        ):
            u = self._sample_u()
            sign = self._sample_sign()

            distance = abs(
                float(
                    mbest[dimension]
                )
                - float(
                    particle.position[dimension]
                )
            )

            displacement = (
                sign
                * alpha
                * distance
                * log(
                    1.0 / u
                )
            )

            value = (
                float(attractor[dimension])
                + displacement
            )

            if not isfinite(value):
                raise OptimizationError(
                    "QPSO generated a non-finite candidate"
                )

            candidate.append(value)

        if should_recover(
            config=self.config,
            stagnation=self._current_stagnation,
            rng=self._rng,
        ):
            candidate = self._recover(
                candidate,
                mbest,
            )

        return list(
            clamp_vector(
                candidate,
                self.config.lower_bound,
                self.config.upper_bound,
            )
        )

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """
        Execute the optimizer until the exact evaluation budget is consumed.
        """
        particles, evaluations = (
            self._initialize_particles()
        )

        global_best = self._copy_particle(
            particles[
                self._best_particle_index(
                    particles
                )
            ]
        )

        history_best: list[float] = [
            float(
                global_best.pbest_result.fitness
            )
        ]

        history_mean: list[float] = [
            sum(
                float(
                    particle.pbest_result.fitness
                )
                for particle in particles
            )
            / len(particles)
        ]

        history_coordinate: list[float] = [
            coordinate_diversity(
                [
                    particle.position
                    for particle in particles
                ]
            )
        ]

        history_route: list[float] = [
            route_diversity(
                [
                    particle.pbest_result
                    for particle in particles
                ]
            )
        ]

        iterations = 0
        stagnation = 0
        progress = 0.0

        self._current_stagnation = 0

        while (
            evaluations
            < self.config.max_evaluations
        ):
            iterations += 1

            previous_best = (
                global_best.pbest_result
            )

            mbest = self._weighted_mbest(
                particles
            )

            coordinate_div = coordinate_diversity(
                [
                    particle.position
                    for particle in particles
                ]
            )

            route_div = route_diversity(
                [
                    particle.pbest_result
                    for particle in particles
                ]
            )

            alpha = self._adaptive_alpha(
                coordinate_div=coordinate_div,
                route_div=route_div,
                progress=progress,
                stagnation=stagnation,
            )

            self._current_stagnation = stagnation

            # Generate one candidate for every particle.
            candidate_positions = [
                self._generate_candidate(
                    particle=particle,
                    global_best=global_best,
                    mbest=mbest,
                    alpha=alpha,
                )
                for particle in particles
            ]

            remaining = (
                self.config.max_evaluations
                - evaluations
            )

            evaluated_count = min(
                len(candidate_positions),
                remaining,
            )

            for index in range(
                evaluated_count
            ):
                candidate_position = (
                    candidate_positions[index]
                )

                result = self.oracle(
                    candidate_position
                )

                self._validate_fitness_result(
                    result
                )

                evaluations += 1

                particle = particles[index]

                # Current state always moves to the newly evaluated point.
                particle.position = list(
                    candidate_position
                )

                # Personal-best update uses feasibility-first dominance.
                if self._better(
                    result,
                    particle.pbest_result,
                ):
                    particle.pbest_position = list(
                        candidate_position
                    )
                    particle.pbest_result = result

                # Global-best update uses the same rule.
                if self._better(
                    result,
                    global_best.pbest_result,
                ):
                    global_best = _Particle(
                        position=list(
                            candidate_position
                        ),
                        pbest_position=list(
                            candidate_position
                        ),
                        pbest_result=result,
                    )

            current_best = (
                global_best.pbest_result
            )

            # ----------------------------------------------------------
            # Progress is computed AFTER candidate evaluation.
            # ----------------------------------------------------------
            if self._better(
                current_best,
                previous_best,
            ):
                if (
                    not previous_best.feasible
                    and current_best.feasible
                ):
                    progress = 1.0
                else:
                    improvement = (
                        float(
                            previous_best.fitness
                        )
                        - float(
                            current_best.fitness
                        )
                    )

                    denominator = max(
                        abs(
                            float(
                                previous_best.fitness
                            )
                        ),
                        1.0,
                    )

                    progress = min(
                        1.0,
                        max(
                            0.0,
                            improvement
                            / denominator,
                        ),
                    )

                stagnation = 0
            else:
                progress = 0.0
                stagnation += 1

            self._current_stagnation = stagnation

            # ----------------------------------------------------------
            # Post-iteration trajectory state.
            # ----------------------------------------------------------
            history_best.append(
                float(
                    global_best.pbest_result.fitness
                )
            )

            history_mean.append(
                sum(
                    float(
                        particle.pbest_result.fitness
                    )
                    for particle in particles
                )
                / len(particles)
            )

            history_coordinate.append(
                coordinate_diversity(
                    [
                        particle.position
                        for particle in particles
                    ]
                )
            )

            history_route.append(
                route_diversity(
                    [
                        particle.pbest_result
                        for particle in particles
                    ]
                )
            )

            if (
                evaluations
                >= self.config.max_evaluations
            ):
                break

        # The optimizer must never silently diverge from the oracle's count.
        oracle_calls = getattr(
            self.oracle,
            "calls",
            None,
        )

        if (
            oracle_calls is not None
            and evaluations != oracle_calls
        ):
            raise OptimizationError(
                "optimizer/oracle evaluation accounting mismatch: "
                f"optimizer={evaluations}, "
                f"oracle={oracle_calls}"
            )

        return OptimizationResult(
            best_position=tuple(
                global_best.pbest_position
            ),
            best_fitness=float(
                global_best.pbest_result.fitness
            ),
            best_result=global_best.pbest_result,
            evaluations=evaluations,
            iterations=iterations,
            history_best=tuple(
                history_best
            ),
            history_mean=tuple(
                history_mean
            ),
            history_coordinate_diversity=tuple(
                history_coordinate
            ),
            history_route_diversity=tuple(
                history_route
            ),
            seed=self.config.seed,
        )


class RouteFitnessOracle(FitnessOracle):
    """
    Adapter between the route engine and the generic optimizer contract.
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
        if route_engine is None:
            raise OptimizationError(
                "route_engine cannot be None"
            )

        planning_time_s = float(
            planning_time_s
        )

        if not isfinite(
            planning_time_s
        ):
            raise OptimizationError(
                "planning_time_s must be finite"
            )

        if planning_time_s < 0.0:
            raise OptimizationError(
                "planning_time_s must be non-negative"
            )

        repair_penalty = float(
            repair_penalty
        )

        if not isfinite(
            repair_penalty
        ):
            raise OptimizationError(
                "repair_penalty must be finite"
            )

        if repair_penalty < 0.0:
            raise OptimizationError(
                "repair_penalty must be non-negative"
            )

        self.route_engine = route_engine
        self.commitments = commitments
        self.planning_time_s = planning_time_s
        self.repair = bool(repair)
        self.repair_penalty = repair_penalty

        self._last_candidate = None

        super().__init__(
            self._evaluate_route
        )

    @staticmethod
    def _route_signature(
        candidate,
    ) -> tuple:
        plan = getattr(
            candidate,
            "repaired_plan",
            None,
        )

        if plan is None:
            plan = getattr(
                candidate,
                "route_plan",
                None,
            )

        if plan is None:
            raise OptimizationError(
                "RouteCandidate does not expose repaired_plan "
                "or route_plan"
            )

        signatures = []

        for vehicle_route in (
            plan.vehicle_routes
        ):
            customer_ids = getattr(
                vehicle_route,
                "customer_ids",
                None,
            )

            if customer_ids is None:
                customer_ids = getattr(
                    vehicle_route,
                    "customer_order",
                    None,
                )

            if customer_ids is None:
                raise OptimizationError(
                    "vehicle route does not expose customer_ids "
                    "or customer_order"
                )

            signatures.append(
                (
                    int(
                        vehicle_route.vehicle_id
                    ),
                    tuple(
                        int(customer_id)
                        for customer_id in customer_ids
                    ),
                )
            )

        return tuple(
            signatures
        )

    @staticmethod
    def _repair_distance(
        candidate,
    ) -> float:
        direct_distance = getattr(
            candidate,
            "repair_distance",
            None,
        )

        if direct_distance is not None:
            value = float(
                direct_distance
            )

            if not isfinite(value):
                raise OptimizationError(
                    "RouteCandidate repair_distance is non-finite"
                )

            if value < 0.0:
                raise OptimizationError(
                    "RouteCandidate repair_distance is negative"
                )

            return value

        repair_result = getattr(
            candidate,
            "repair_result",
            None,
        )

        if repair_result is None:
            return 0.0

        original_plan = getattr(
            repair_result,
            "original_route_plan",
            None,
        )

        repaired_plan = getattr(
            repair_result,
            "repaired_route_plan",
            None,
        )

        if (
            original_plan is None
            or repaired_plan is None
        ):
            return 0.0

        distance = float(
            route_distance(
                original_plan,
                repaired_plan,
            ).structure
        )

        if not isfinite(distance):
            raise OptimizationError(
                "derived repair distance is non-finite"
            )

        if distance < 0.0:
            raise OptimizationError(
                "derived repair distance is negative"
            )

        return distance

    def _evaluate_route(
        self,
        position: Vector,
    ) -> FitnessResult:
        if len(position) == 0:
            raise OptimizationError(
                "route candidate position cannot be empty"
            )

        if any(
            not isfinite(float(value))
            for value in position
        ):
            raise OptimizationError(
                "route candidate position contains a non-finite value"
            )

        candidate = (
            self.route_engine.evaluate_keys(
                position,
                commitments=self.commitments,
                planning_time_s=self.planning_time_s,
                repair=self.repair,
            )
        )

        if candidate is None:
            raise OptimizationError(
                "route_engine.evaluate_keys returned None"
            )

        self._last_candidate = candidate

        evaluation = getattr(
            candidate,
            "repaired_evaluation",
            None,
        )

        if evaluation is None:
            evaluation = getattr(
                candidate,
                "evaluation",
                None,
            )

        if evaluation is None:
            raise OptimizationError(
                "RouteCandidate does not expose repaired_evaluation "
                "or evaluation"
            )

        objective_value = float(
            evaluation.objective_value
        )

        if not isfinite(
            objective_value
        ):
            raise OptimizationError(
                "route evaluation objective_value is non-finite"
            )

        feasible = bool(
            evaluation.feasible
        )

        repair_distance = (
            self._repair_distance(
                candidate
            )
        )

        fitness = (
            objective_value
            + self.repair_penalty
            * repair_distance
        )

        if not isfinite(
            fitness
        ):
            raise OptimizationError(
                "route fitness is non-finite"
            )

        return FitnessResult(
            fitness=fitness,
            feasible=feasible,
            route_signature=(
                self._route_signature(
                    candidate
                )
            ),
            repair_distance=repair_distance,
        )

    @property
    def last_candidate(self):
        """
        Most recently evaluated route candidate.
        """
        return self._last_candidate