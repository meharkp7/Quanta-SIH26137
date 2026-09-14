"""
Step 16 — PPO environment for dynamic traffic-routing decisions.

This module defines the five-action RL environment and its observation/reward
contract.

Important architectural boundary
---------------------------------
PPO chooses a SCOPE ACTION.

QPSO performs the actual combinatorial route search.

SUMO provides realized simulation outcomes.

The environment never exposes:
    - future event times
    - future simulator labels
    - hidden random seeds
    - future observations

Decision interval:
    60 simulated seconds.

This first implementation deliberately separates:
    1. observation construction
    2. scope/action handling
    3. QPSO solving
    4. simulation advancement
    5. reward accounting

The simulator adapter is injectable so the environment can be unit-tested
without replacing the production SUMO path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.contracts.decision import ScopeAction
from src.routing.route_plan import RoutePlan
from src.optim.qpso import (
    AdaptiveQPSO,
    AdaptiveQPSOConfig,
    RouteFitnessOracle,
)
from src.routing.evaluator_state import CommitmentSnapshot
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_evaluator import RouteEvaluator
from src.runtime.scope_actions import (
    JobImpact,
    ScopeActionConfig,
    ScopeActionSelector,
    ScopeSelection,
    build_commitment_snapshot,
)


# ============================================================================
# Action mapping
# ============================================================================

ACTION_TO_INDEX: dict[ScopeAction, int] = {
    ScopeAction.KEEP: 0,
    ScopeAction.LOCAL: 1,
    ScopeAction.VEHICLE: 2,
    ScopeAction.REGIONAL: 3,
    ScopeAction.GLOBAL: 4,
}

INDEX_TO_ACTION: dict[int, ScopeAction] = {
    value: key
    for key, value in ACTION_TO_INDEX.items()
}


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class PPOEnvConfig:
    """
    Configuration for the Step-16 environment.

    QPSO evaluation counts are intentionally small by default because this
    environment is first being profiled for throughput.
    """

    decision_interval_s: float = 60.0

    episode_duration_s: float = 600.0

    qpso_particles: int = 8
    qpso_evaluations: int = 16
    qpso_seed: int = 26137

    # Initial reward weights.
    # These are development configuration, not claimed optimal values.
    operating_cost_weight: float = 1.0
    congestion_weight: float = 1.0
    service_failure_weight: float = 10.0
    churn_weight: float = 0.5
    computation_weight: float = 0.1
    unfinished_work_weight: float = 10.0

    # Normalization constants prevent one component from dominating solely
    # because of units.
    operating_cost_scale: float = 1000.0
    congestion_scale: float = 100.0
    computation_scale: float = 1.0

    local_max_jobs: int = 10

    observation_size: int = 32

    def __post_init__(self) -> None:
        if self.decision_interval_s <= 0:
            raise ValueError(
                "decision_interval_s must be positive"
            )

        if self.episode_duration_s <= 0:
            raise ValueError(
                "episode_duration_s must be positive"
            )

        if self.qpso_particles <= 0:
            raise ValueError(
                "qpso_particles must be positive"
            )

        if self.qpso_evaluations < self.qpso_particles:
            raise ValueError(
                "qpso_evaluations must be >= qpso_particles"
            )


# ============================================================================
# Reward
# ============================================================================


@dataclass(frozen=True)
class RewardComponents:
    """
    Fully decomposed reward.

    Keeping these components separate is required for later analysis.
    """

    operating_cost: float = 0.0
    congestion_exposure: float = 0.0
    service_failures: float = 0.0
    churn: float = 0.0
    computation: float = 0.0
    unfinished_work: float = 0.0

    @property
    def total_penalty(self) -> float:
        return (
            self.operating_cost
            + self.congestion_exposure
            + self.service_failures
            + self.churn
            + self.computation
            + self.unfinished_work
        )

    @property
    def reward(self) -> float:
        return -self.total_penalty

    def as_dict(self) -> dict[str, float]:
        return {
            "operating_cost": self.operating_cost,
            "congestion_exposure": self.congestion_exposure,
            "service_failures": self.service_failures,
            "churn": self.churn,
            "computation": self.computation,
            "unfinished_work": self.unfinished_work,
            "total_penalty": self.total_penalty,
            "reward": self.reward,
        }


# ============================================================================
# Decision record
# ============================================================================


@dataclass(frozen=True)
class DecisionRecord:
    """Audit record for one PPO environment decision."""

    decision_index: int
    sim_time_s: float

    requested_action: ScopeAction
    executed_action: ScopeAction

    mutable_request_ids: tuple[str, ...]
    affected_vehicle_ids: tuple[str, ...]

    overridden: bool
    override_reason: str | None

    qpso_called: bool
    qpso_evaluations: int

    elapsed_s: float

    reward: RewardComponents

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "sim_time_s": self.sim_time_s,
            "requested_action": self.requested_action.value,
            "executed_action": self.executed_action.value,
            "mutable_request_ids": list(
                self.mutable_request_ids
            ),
            "affected_vehicle_ids": list(
                self.affected_vehicle_ids
            ),
            "overridden": self.overridden,
            "override_reason": self.override_reason,
            "qpso_called": self.qpso_called,
            "qpso_evaluations": self.qpso_evaluations,
            "elapsed_s": self.elapsed_s,
            "reward": self.reward.as_dict(),
        }


# ============================================================================
# Simulator protocol
# ============================================================================


class SimulatorBackend:
    """
    Minimal simulator interface required by the PPO environment.

    A production backend must advance the actual simulator.

    The backend returns realized measurements only; it must not expose future
    labels or future event information.
    """

    def reset(
        self,
        scenario: Any,
        route_plan: RoutePlan,
    ) -> None:
        raise NotImplementedError

    def advance(
        self,
        duration_s: float,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @property
    def sim_time_s(self) -> float:
        raise NotImplementedError

    @property
    def done(self) -> bool:
        raise NotImplementedError


# ============================================================================
# Deterministic observation encoder
# ============================================================================


class RoutingObservationEncoder:
    """
    Converts the current visible state into a fixed-size float vector.

    The encoder intentionally accepts summaries rather than future labels.

    Layout
    ------
    0   normalized simulation time
    1   remaining work fraction
    2   affected route fraction
    3   mean deadline slack
    4   minimum deadline slack
    5   mean vehicle load
    6   max vehicle load
    7   active event count
    8   recent KEEP
    9   recent LOCAL
    10  recent VEHICLE
    11  recent REGIONAL
    12  recent GLOBAL

    13–17 action one-hot / last executed action

    18–27 forecast summary
    28–31 reserved normalized state summaries

    The final four slots are intentionally populated from current-state
    summaries rather than hidden simulator truth.
    """

    def __init__(
        self,
        *,
        observation_size: int = 32,
        episode_duration_s: float = 600.0,
    ) -> None:
        if observation_size < 32:
            raise ValueError(
                "observation_size must be >= 32"
            )

        self.observation_size = observation_size
        self.episode_duration_s = episode_duration_s

    def encode(
        self,
        *,
        sim_time_s: float,
        remaining_work: float,
        total_work: float,
        affected_route_fraction: float,
        deadline_slacks_s: Sequence[float],
        vehicle_loads: Sequence[float],
        vehicle_capacities: Sequence[float],
        active_event_count: int,
        last_action: ScopeAction,
        forecast_summary: Sequence[float] = (),
        recent_action_counts: Mapping[ScopeAction, int] | None = None,
        recent_route_change_fraction: float = 0.0,
        recent_congestion: float = 0.0,
        mean_speed_ratio: float = 1.0,
    ) -> np.ndarray:

        x = np.zeros(
            self.observation_size,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Current operational state.
        # --------------------------------------------------------------

        x[0] = np.clip(
            sim_time_s / self.episode_duration_s,
            0.0,
            1.0,
        )

        if total_work > 0:
            x[1] = np.clip(
                remaining_work / total_work,
                0.0,
                1.0,
            )

        x[2] = np.clip(
            affected_route_fraction,
            0.0,
            1.0,
        )

        finite_slacks = [
            float(v)
            for v in deadline_slacks_s
            if np.isfinite(v)
        ]

        if finite_slacks:
            x[3] = np.clip(
                np.mean(finite_slacks) / 3600.0,
                0.0,
                1.0,
            )

            x[4] = np.clip(
                min(finite_slacks) / 3600.0,
                0.0,
                1.0,
            )

        ratios = []

        for load, capacity in zip(
            vehicle_loads,
            vehicle_capacities,
        ):
            if capacity > 0:
                ratios.append(
                    float(load) / float(capacity)
                )

        if ratios:
            x[5] = np.clip(
                np.mean(ratios),
                0.0,
                1.0,
            )

            x[6] = np.clip(
                max(ratios),
                0.0,
                1.0,
            )

        x[7] = np.clip(
            active_event_count / 10.0,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Recent action distribution.
        # --------------------------------------------------------------

        recent_action_counts = (
            recent_action_counts
            or {}
        )

        total_recent = max(
            1,
            sum(
                recent_action_counts.values()
            ),
        )

        for action, index in ACTION_TO_INDEX.items():
            x[8 + index] = np.clip(
                recent_action_counts.get(
                    action,
                    0,
                )
                / total_recent,
                0.0,
                1.0,
            )

        # Last executed action.
        x[13 + ACTION_TO_INDEX[last_action]] = 1.0

        # --------------------------------------------------------------
        # Forecast / uncertainty summary.
        # --------------------------------------------------------------

        for index, value in enumerate(
            forecast_summary[:10]
        ):
            x[18 + index] = np.clip(
                float(value),
                -10.0,
                10.0,
            )

        # --------------------------------------------------------------
        # Recent/current summaries.
        # --------------------------------------------------------------

        x[28] = np.clip(
            recent_route_change_fraction,
            0.0,
            1.0,
        )

        x[29] = np.clip(
            recent_congestion,
            0.0,
            1.0,
        )

        x[30] = np.clip(
            mean_speed_ratio,
            0.0,
            1.0,
        )

        x[31] = np.clip(
            1.0 - x[1],
            0.0,
            1.0,
        )

        # Any future extension remains zero-padded.
        return x


# ============================================================================
# PPO Environment
# ============================================================================


class TrafficRoutingPPOEnv(gym.Env):
    """
    Gymnasium-compatible traffic-routing control environment.

    Five discrete actions:

        0 KEEP
        1 LOCAL
        2 VEHICLE
        3 REGIONAL
        4 GLOBAL

    PPO selects the action.

    The environment:
        1. builds the scope;
        2. applies the safety layer;
        3. invokes real QPSO for non-KEEP actions;
        4. validates the candidate;
        5. advances the simulator by 60 seconds;
        6. computes realized reward;
        7. returns the next observation.

    No PPO training happens inside this class.
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        scenario: Any,
        initial_plan: RoutePlan,
        *,
        simulator: SimulatorBackend,
        config: PPOEnvConfig | None = None,
        job_impacts: Sequence[JobImpact] = (),
        forecast_provider: Callable[
            [float],
            Sequence[float],
        ]
        | None = None,
        affected_vehicle_provider: Callable[
            [float],
            Sequence[str],
        ]
        | None = None,
        affected_zone_provider: Callable[
            [float],
            Sequence[str],
        ]
        | None = None,
    ) -> None:

        super().__init__()

        self.scenario = scenario
        self.initial_plan = initial_plan
        self.simulator = simulator

        self.config = (
            config
            if config is not None
            else PPOEnvConfig()
        )

        self.job_impacts = tuple(
            job_impacts
        )

        self.forecast_provider = (
            forecast_provider
        )

        self.affected_vehicle_provider = (
            affected_vehicle_provider
        )

        self.affected_zone_provider = (
            affected_zone_provider
        )

        self.scope_selector = ScopeActionSelector(
            config=ScopeActionConfig(
                local_max_jobs=self.config.local_max_jobs,
            )
        )

        self.observation_encoder = (
            RoutingObservationEncoder(
                observation_size=self.config.observation_size,
                episode_duration_s=self.config.episode_duration_s,
            )
        )

        # --------------------------------------------------------------
        # Gym spaces.
        # --------------------------------------------------------------

        self.action_space = spaces.Discrete(
            len(ACTION_TO_INDEX)
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.config.observation_size,
            ),
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Runtime state.
        # --------------------------------------------------------------

        self.current_plan = initial_plan

        self._sim_time_s = 0.0
        self._decision_index = 0

        self._last_action = ScopeAction.KEEP

        self._recent_action_counts: dict[
            ScopeAction,
            int,
        ] = {
            action: 0
            for action in ScopeAction
        }

        self._decision_log: list[
            DecisionRecord
        ] = []

        self._total_work = max(
            1,
            len(scenario.requests),
        )

        self._remaining_work = float(
            self._total_work
        )

        self._previous_remaining_work = (
            self._remaining_work
        )

    # ------------------------------------------------------------------
    # Gym reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        self.current_plan = self.initial_plan

        self._sim_time_s = 0.0
        self._decision_index = 0

        self._last_action = ScopeAction.KEEP

        self._recent_action_counts = {
            action: 0
            for action in ScopeAction
        }

        self._decision_log = []

        self._remaining_work = float(
            self._total_work
        )

        self._previous_remaining_work = (
            self._remaining_work
        )

        self.simulator.reset(
            self.scenario,
            self.current_plan,
        )

        observation = self._observation()

        info = {
            "sim_time_s": self._sim_time_s,
            "decision_index": 0,
            "action_names": {
                index: action.value
                for index, action
                in INDEX_TO_ACTION.items()
            },
        }

        return observation, info

    # ------------------------------------------------------------------
    # Gym step
    # ------------------------------------------------------------------

    def step(
        self,
        action: int,
    ):
        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                f"invalid action index: {action!r}"
            )

        requested_action = (
            INDEX_TO_ACTION[int(action)]
        )

        wall_start = perf_counter()

        # --------------------------------------------------------------
        # Current visible state.
        # --------------------------------------------------------------

        affected_vehicles = (
            tuple(
                self.affected_vehicle_provider(
                    self._sim_time_s
                )
            )
            if self.affected_vehicle_provider
            else ()
        )

        affected_zones = (
            tuple(
                self.affected_zone_provider(
                    self._sim_time_s
                )
            )
            if self.affected_zone_provider
            else ()
        )

        # --------------------------------------------------------------
        # Build exact scope.
        # --------------------------------------------------------------

        selection = self.scope_selector.select(
            requested_action,
            jobs=self.job_impacts,
            affected_vehicle_ids=(
                affected_vehicles
            ),
            affected_zone_ids=affected_zones,
        )

        # --------------------------------------------------------------
        # KEEP: no QPSO.
        # --------------------------------------------------------------

        qpso_called = False
        qpso_evaluations = 0

        executed_action = selection.action

        if (
            selection.action
            == ScopeAction.KEEP
        ):
            next_plan = self.current_plan

        else:
            qpso_called = True

            commitments = (
                build_commitment_snapshot(
                    self.scenario,
                    self.current_plan,
                    selection,
                    planning_time_s=self._sim_time_s,
                )
            )

            evaluator = RouteEvaluator(
                self.scenario,
            )

            engine = Step7RouteEngine(
                self.scenario,
                evaluator,
            )

            oracle = RouteFitnessOracle(
                engine,
                commitments=commitments,
                planning_time_s=self._sim_time_s,
                repair=True,
            )

            qpso_config = (
                AdaptiveQPSOConfig(
                    dimensions=(
                        engine.encoder.dimension
                    ),
                    lower_bound=0.0,
                    upper_bound=1.0,
                    population_size=(
                        self.config.qpso_particles
                    ),
                    max_evaluations=(
                        self.config.qpso_evaluations
                    ),
                    seed=self.config.qpso_seed
                    + self._decision_index,
                )
            )

            incumbent = (
                engine.encoder.encode(
                    self.current_plan,
                    commitments=commitments,
                )
            )

            population = [
                list(incumbent.keys)
                for _ in range(
                    self.config.qpso_particles
                )
            ]

            optimizer = AdaptiveQPSO(
                qpso_config,
                oracle,
                initial_population=population,
            )

            result = optimizer.optimize()

            qpso_evaluations = (
                result.evaluations
            )

            candidate = (
                engine.evaluate_keys(
                    result.best_position,
                    commitments=commitments,
                    planning_time_s=self._sim_time_s,
                    repair=True,
                )
            )

            if (
                candidate.repaired_evaluation.feasible
            ):
                next_plan = (
                    candidate.repaired_plan
                )
            else:
                # Safety fallback: retain the current
                # legal incumbent rather than dispatch
                # an invalid route.
                next_plan = self.current_plan
                executed_action = (
                    ScopeAction.KEEP
                )

        # --------------------------------------------------------------
        # Advance the REAL simulator.
        # --------------------------------------------------------------

        sim_result = self.simulator.advance(
            self.config.decision_interval_s
        )

        self._sim_time_s = float(
            self.simulator.sim_time_s
        )

        # --------------------------------------------------------------
        # Derive realized reward.
        # --------------------------------------------------------------

        reward_components = (
            self._reward_from_simulation(
                sim_result,
                qpso_elapsed_s=(
                    perf_counter()
                    - wall_start
                    if qpso_called
                    else 0.0
                ),
                route_changed=(
                    next_plan != self.current_plan
                ),
            )
        )

        self.current_plan = next_plan

        self._last_action = executed_action

        self._recent_action_counts[
            executed_action
        ] += 1

        elapsed_s = (
            perf_counter()
            - wall_start
        )

        override_reason = (
            selection.override_reason
            if selection.overridden
            else None
        )

        if (
            executed_action
            != requested_action
            and override_reason is None
        ):
            override_reason = (
                "Safety fallback retained the "
                "last validated incumbent."
            )

        record = DecisionRecord(
            decision_index=self._decision_index,
            sim_time_s=self._sim_time_s,
            requested_action=requested_action,
            executed_action=executed_action,
            mutable_request_ids=(
                selection.request_ids
            ),
            affected_vehicle_ids=(
                selection.affected_vehicle_ids
            ),
            overridden=(
                executed_action
                != requested_action
                or selection.overridden
            ),
            override_reason=override_reason,
            qpso_called=qpso_called,
            qpso_evaluations=qpso_evaluations,
            elapsed_s=elapsed_s,
            reward=reward_components,
        )

        self._decision_log.append(record)

        self._decision_index += 1

        terminated = (
            self.simulator.done
            or self._remaining_work <= 0
        )

        truncated = (
            self._sim_time_s
            >= self.config.episode_duration_s
        ) and not terminated

        observation = self._observation()

        info = {
            "requested_action": requested_action.value,
            "executed_action": executed_action.value,
            "overridden": record.overridden,
            "override_reason": override_reason,
            "mutable_request_ids": list(
                selection.request_ids
            ),
            "affected_vehicle_ids": list(
                selection.affected_vehicle_ids
            ),
            "qpso_called": qpso_called,
            "qpso_evaluations": qpso_evaluations,
            "decision_elapsed_s": elapsed_s,
            "sim_time_s": self._sim_time_s,
            "reward_components": (
                reward_components.as_dict()
            ),
            "decision_index": (
                self._decision_index - 1
            ),
        }

        return (
            observation,
            float(
                reward_components.reward
            ),
            bool(terminated),
            bool(truncated),
            info,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        forecast_summary = ()

        if self.forecast_provider:
            forecast_summary = (
                tuple(
                    self.forecast_provider(
                        self._sim_time_s
                    )
                )
            )

        # The current implementation derives these from the simulator's
        # latest realized state summary. Missing values remain conservative.
        state = getattr(
            self.simulator,
            "latest_state",
            {},
        )

        deadline_slacks = state.get(
            "deadline_slacks_s",
            (),
        )

        vehicle_loads = state.get(
            "vehicle_loads",
            (),
        )

        vehicle_capacities = state.get(
            "vehicle_capacities",
            (),
        )

        affected_fraction = float(
            state.get(
                "affected_route_fraction",
                0.0,
            )
        )

        active_events = int(
            state.get(
                "active_event_count",
                0,
            )
        )

        recent_congestion = float(
            state.get(
                "congestion_exposure",
                0.0,
            )
        )

        mean_speed_ratio = float(
            state.get(
                "mean_speed_ratio",
                1.0,
            )
        )

        route_change_fraction = float(
            state.get(
                "route_change_fraction",
                0.0,
            )
        )

        return self.observation_encoder.encode(
            sim_time_s=self._sim_time_s,
            remaining_work=self._remaining_work,
            total_work=float(
                self._total_work
            ),
            affected_route_fraction=(
                affected_fraction
            ),
            deadline_slacks_s=(
                deadline_slacks
            ),
            vehicle_loads=vehicle_loads,
            vehicle_capacities=(
                vehicle_capacities
            ),
            active_event_count=(
                active_events
            ),
            last_action=self._last_action,
            forecast_summary=(
                forecast_summary
            ),
            recent_action_counts=(
                self._recent_action_counts
            ),
            recent_route_change_fraction=(
                route_change_fraction
            ),
            recent_congestion=(
                recent_congestion
            ),
            mean_speed_ratio=(
                mean_speed_ratio
            ),
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward_from_simulation(
        self,
        sim_result: Mapping[str, Any],
        *,
        qpso_elapsed_s: float,
        route_changed: bool,
    ) -> RewardComponents:

        operating_cost = float(
            sim_result.get(
                "incremental_operating_cost",
                sim_result.get(
                    "operating_cost",
                    0.0,
                ),
            )
        )

        congestion = float(
            sim_result.get(
                "congestion_exposure",
                0.0,
            )
        )

        service_failures = float(
            sim_result.get(
                "service_failures",
                0.0,
            )
        )

        unfinished = float(
            sim_result.get(
                "remaining_work",
                self._remaining_work,
            )
        )

        self._previous_remaining_work = (
            self._remaining_work
        )

        self._remaining_work = max(
            0.0,
            unfinished,
        )

        churn = (
            1.0
            if route_changed
            else 0.0
        )

        computation = float(
            qpso_elapsed_s
        )

        return RewardComponents(
            operating_cost=(
                self.config.operating_cost_weight
                * operating_cost
                / self.config.operating_cost_scale
            ),
            congestion_exposure=(
                self.config.congestion_weight
                * congestion
                / self.config.congestion_scale
            ),
            service_failures=(
                self.config.service_failure_weight
                * service_failures
            ),
            churn=(
                self.config.churn_weight
                * churn
            ),
            computation=(
                self.config.computation_weight
                * computation
                / self.config.computation_scale
            ),
            unfinished_work=(
                self.config.unfinished_work_weight
                * unfinished
                if unfinished <= 0
                else 0.0
            ),
        )

    # ------------------------------------------------------------------
    # Audit access
    # ------------------------------------------------------------------

    @property
    def decision_log(
        self,
    ) -> tuple[DecisionRecord, ...]:
        return tuple(
            self._decision_log
        )