"""
Step 16 — PPO environment for dynamic traffic-routing decisions.

Architecture
------------
PPO selects a routing SCOPE ACTION.

    PPO
      |
      v
  action mask
      |
      v
 safety override
      |
      v
 scope selector
      |
      v
 QPSO / KEEP
      |
      v
 route validation
      |
      v
 SUMO / simulator
      |
      v
 realized reward
      |
      v
 next observation

Important boundaries
--------------------
* PPO does not implement routing optimization.
* QPSO remains the route optimizer.
* The safety layer is independent of PPO rewards.
* Future simulator truth is never inserted into the observation.
* Forecasting architecture is not coupled to this environment.
* PPO may consume a PPORuntime later through the `policy_runtime` hook.
* A legacy 32-dimensional observation is retained for compatibility and
  interface testing until the structured representation adapter is wired.

Decision interval
-----------------
60 simulated seconds by default.

This module is an environment/orchestration boundary. It does not train PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from random import Random
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.contracts.decision import ScopeAction, ScopeDecision
from src.routing.route_plan import RoutePlan
from src.optim.qpso import (
    AdaptiveQPSO,
    AdaptiveQPSOConfig,
    RouteFitnessOracle,
)
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_evaluator import RouteEvaluator
from src.runtime.scope_actions import (
    JobImpact,
    ScopeActionConfig,
    ScopeActionSelector,
    ScopeSelection,
    build_commitment_snapshot,
)

from src.learning.action_mask import ActionMask
from src.learning.ppo_forecast import PPOForecastSignal, coerce_ppo_forecast_signal

try:
    from src.learning.ppo_runtime import (
        PPOPolicyDecision,
        PPORuntime,
    )
except ImportError:  # pragma: no cover - compatibility path
    PPOPolicyDecision = Any  # type: ignore[misc,assignment]
    PPORuntime = Any  # type: ignore[misc,assignment]
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
    index: action
    for action, index in ACTION_TO_INDEX.items()
}


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class PPOEnvConfig:
    """
    Configuration for one PPO environment.

    Reward weights are development defaults. They are not claimed to be
    optimal and must be kept fixed/reported during benchmark comparisons.
    """

    # ------------------------------------------------------------------
    # Environment timing
    # ------------------------------------------------------------------

    decision_interval_s: float = 60.0
    episode_duration_s: float = 600.0

    # ------------------------------------------------------------------
    # QPSO budget
    # ------------------------------------------------------------------

    qpso_particles: int = 8
    qpso_evaluations: int = 16
    qpso_seed: int = 26137

    # ------------------------------------------------------------------
    # Reward weights
    # ------------------------------------------------------------------

    operating_cost_weight: float = 1.0
    congestion_weight: float = 1.0
    service_failure_weight: float = 10.0
    churn_weight: float = 0.5
    computation_weight: float = 0.1
    unfinished_work_weight: float = 10.0

    # ------------------------------------------------------------------
    # Reward normalization
    # ------------------------------------------------------------------

    operating_cost_scale: float = 1000.0
    congestion_scale: float = 100.0
    computation_scale: float = 1.0

    # ------------------------------------------------------------------
    # Scope configuration
    # ------------------------------------------------------------------

    local_max_jobs: int = 10

    # ------------------------------------------------------------------
    # Legacy observation contract
    # ------------------------------------------------------------------

    observation_size: int = 32

    def __post_init__(self) -> None:
        numeric_positive = (
            "decision_interval_s",
            "episode_duration_s",
            "operating_cost_scale",
            "congestion_scale",
            "computation_scale",
        )

        for field_name in numeric_positive:
            value = float(getattr(self, field_name))

            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{field_name} must be finite and positive"
                )

        if self.qpso_particles <= 0:
            raise ValueError(
                "qpso_particles must be positive"
            )

        if self.qpso_evaluations < self.qpso_particles:
            raise ValueError(
                "qpso_evaluations must be >= qpso_particles"
            )

        if self.local_max_jobs <= 0:
            raise ValueError(
                "local_max_jobs must be positive"
            )

        if self.observation_size < 32:
            raise ValueError(
                "observation_size must be >= 32"
            )

        reward_weights = (
            "operating_cost_weight",
            "congestion_weight",
            "service_failure_weight",
            "churn_weight",
            "computation_weight",
            "unfinished_work_weight",
        )

        for field_name in reward_weights:
            value = float(getattr(self, field_name))

            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field_name} must be finite and non-negative"
                )


# ============================================================================
# Reward
# ============================================================================


@dataclass(frozen=True)
class RewardComponents:
    """
    Fully decomposed interval reward.

    Every field represents a penalty contribution after normalization and
    weighting. The final reward is the negative total penalty.
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
            "operating_cost": float(self.operating_cost),
            "congestion_exposure": float(
                self.congestion_exposure
            ),
            "service_failures": float(
                self.service_failures
            ),
            "churn": float(self.churn),
            "computation": float(self.computation),
            "unfinished_work": float(
                self.unfinished_work
            ),
            "total_penalty": float(
                self.total_penalty
            ),
            "reward": float(self.reward),
        }


# ============================================================================
# Decision audit record
# ============================================================================


@dataclass(frozen=True)
class DecisionRecord:
    """Immutable audit record for one environment decision."""

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
            "sim_time_s": float(self.sim_time_s),
            "requested_action": self.requested_action.value,
            "executed_action": self.executed_action.value,
            "mutable_request_ids": list(
                self.mutable_request_ids
            ),
            "affected_vehicle_ids": list(
                self.affected_vehicle_ids
            ),
            "overridden": bool(self.overridden),
            "override_reason": self.override_reason,
            "qpso_called": bool(self.qpso_called),
            "qpso_evaluations": int(
                self.qpso_evaluations
            ),
            "elapsed_s": float(self.elapsed_s),
            "reward": self.reward.as_dict(),
        }


# ============================================================================
# Simulator interface
# ============================================================================


class SimulatorBackend:
    """
    Minimal simulator interface required by the PPO environment.

    Production implementation may wrap SUMO/TraCI.

    `advance()` must return measurements realized during the interval only.
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
# Legacy observation encoder
# ============================================================================


class RoutingObservationEncoder:
    """
    Fixed-size compatibility observation.

    This encoder intentionally remains model-independent.

    Layout
    ------
    0
        normalized simulation time

    1
        remaining work fraction

    2
        affected route fraction

    3
        mean deadline slack

    4
        minimum deadline slack

    5
        mean vehicle load

    6
        maximum vehicle load

    7
        active event count

    8-12
        recent action distribution

    13-17
        last executed action one-hot

    18-27
        forecast/uncertainty summary

    28
        recent route-change fraction

    29
        recent congestion

    30
        mean speed ratio

    31
        completed-work fraction

    Remaining dimensions, if any, are zero padded.
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

        if episode_duration_s <= 0:
            raise ValueError(
                "episode_duration_s must be positive"
            )

        self.observation_size = int(
            observation_size
        )
        self.episode_duration_s = float(
            episode_duration_s
        )

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
        recent_action_counts: Mapping[
            ScopeAction,
            int,
        ]
        | None = None,
        recent_route_change_fraction: float = 0.0,
        recent_congestion: float = 0.0,
        mean_speed_ratio: float = 1.0,
    ) -> np.ndarray:

        x = np.zeros(
            self.observation_size,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Time
        # --------------------------------------------------------------

        x[0] = np.clip(
            float(sim_time_s)
            / self.episode_duration_s,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Work
        # --------------------------------------------------------------

        total = max(
            float(total_work),
            1.0,
        )

        remaining = max(
            0.0,
            float(remaining_work),
        )

        remaining_fraction = np.clip(
            remaining / total,
            0.0,
            1.0,
        )

        x[1] = remaining_fraction

        # --------------------------------------------------------------
        # Affected route fraction
        # --------------------------------------------------------------

        x[2] = np.clip(
            float(affected_route_fraction),
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Deadline slack
        # --------------------------------------------------------------

        finite_slacks = [
            float(value)
            for value in deadline_slacks_s
            if np.isfinite(float(value))
        ]

        if finite_slacks:
            mean_slack = float(
                np.mean(finite_slacks)
            )
            min_slack = float(
                np.min(finite_slacks)
            )

            # 600 seconds is a stable development normalization.
            x[3] = np.clip(
                mean_slack / 600.0,
                -10.0,
                10.0,
            )

            x[4] = np.clip(
                min_slack / 600.0,
                -10.0,
                10.0,
            )

        # --------------------------------------------------------------
        # Vehicle load
        # --------------------------------------------------------------

        loads = [
            float(value)
            for value in vehicle_loads
            if np.isfinite(float(value))
        ]

        capacities = [
            float(value)
            for value in vehicle_capacities
            if np.isfinite(float(value))
        ]

        if loads:
            x[5] = np.clip(
                float(np.mean(loads)),
                0.0,
                1.0,
            )

            if capacities:
                ratios = []

                for index, load in enumerate(
                    loads
                ):
                    if index >= len(capacities):
                        break

                    capacity = max(
                        capacities[index],
                        1e-9,
                    )

                    ratios.append(
                        load / capacity
                    )

                if ratios:
                    x[6] = np.clip(
                        float(np.max(ratios)),
                        0.0,
                        1.0,
                    )

        # --------------------------------------------------------------
        # Events
        # --------------------------------------------------------------

        x[7] = np.clip(
            float(active_event_count) / 10.0,
            0.0,
            1.0,
        )

        # --------------------------------------------------------------
        # Recent action distribution
        # --------------------------------------------------------------

        recent_action_counts = (
            recent_action_counts
            or {}
        )

        total_recent = max(
            1,
            sum(
                max(
                    0,
                    int(value),
                )
                for value in recent_action_counts.values()
            ),
        )

        for action, index in ACTION_TO_INDEX.items():
            x[8 + index] = np.clip(
                float(
                    recent_action_counts.get(
                        action,
                        0,
                    )
                )
                / total_recent,
                0.0,
                1.0,
            )

        # --------------------------------------------------------------
        # Last executed action
        # --------------------------------------------------------------

        if last_action not in ACTION_TO_INDEX:
            raise ValueError(
                f"unsupported last_action={last_action!r}"
            )

        x[
            13 + ACTION_TO_INDEX[last_action]
        ] = 1.0

        # --------------------------------------------------------------
        # Forecast / uncertainty summary
        # --------------------------------------------------------------

        for index, value in enumerate(
            forecast_summary[:10]
        ):
            numeric = float(value)

            if not np.isfinite(numeric):
                numeric = 0.0

            x[18 + index] = np.clip(
                numeric,
                -10.0,
                10.0,
            )

        # --------------------------------------------------------------
        # Current-state summaries
        # --------------------------------------------------------------

        x[28] = np.clip(
            float(recent_route_change_fraction),
            0.0,
            1.0,
        )

        x[29] = np.clip(
            float(recent_congestion),
            0.0,
            1.0,
        )

        x[30] = np.clip(
            float(mean_speed_ratio),
            0.0,
            1.0,
        )

        x[31] = np.clip(
            1.0 - remaining_fraction,
            0.0,
            1.0,
        )

        if not np.all(
            np.isfinite(x)
        ):
            raise RuntimeError(
                "observation encoder produced non-finite values"
            )

        return x


# ============================================================================
# PPO environment
# ============================================================================


class TrafficRoutingPPOEnv(gym.Env):
    """
    Gymnasium environment for dynamic traffic-routing control.

    PPO's action space:

        0 KEEP
        1 LOCAL
        2 VEHICLE
        3 REGIONAL
        4 GLOBAL

    One environment step:

        observation
            ↓
        action
            ↓
        feasibility mask
            ↓
        Step-16 safety layer
            ↓
        Step-15 exact scope
            ↓
        QPSO if required
            ↓
        candidate validation
            ↓
        simulator +60 s
            ↓
        realized reward
            ↓
        next observation

    The environment does not train the policy.
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
            PPOForecastSignal | Sequence[float],
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
        action_mask_provider: Callable[
            [float],
            Any,
        ]
        | None = None,
        policy_runtime: PPORuntime | None = None,
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

        self.action_mask_provider = (
            action_mask_provider
        )

        # Optional PPO runtime.
        #
        # This is intentionally NOT required for the environment's legacy
        # observation interface. It becomes the integration boundary when
        # the structured representation adapter is ready.
        self.policy_runtime = policy_runtime

        # --------------------------------------------------------------
        # Step-15 scope selector
        # --------------------------------------------------------------

        self.scope_selector = ScopeActionSelector(
            config=ScopeActionConfig(
                local_max_jobs=(
                    self.config.local_max_jobs
                ),
            )
        )

        # --------------------------------------------------------------
        # Observation
        # --------------------------------------------------------------

        self.observation_encoder = (
            RoutingObservationEncoder(
                observation_size=(
                    self.config.observation_size
                ),
                episode_duration_s=(
                    self.config.episode_duration_s
                ),
            )
        )

        # --------------------------------------------------------------
        # Gym spaces
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
        # Runtime state
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

        # Authoritative causal routing decision for the most recent step.
        #
        # This is deliberately separate from `decision_log`, whose
        # DecisionRecord contains realized reward/audit information and must
        # not be inserted into DecisionMemory.
        self._last_scope_decision: ScopeDecision | None = None

        self._total_work = max(
            1,
            len(
                getattr(
                    scenario,
                    "requests",
                    (),
                )
            ),
        )

        self._remaining_work = float(
            self._total_work
        )

        self._previous_remaining_work = (
            self._remaining_work
        )

        self._episode_started = False

        self._last_qpso_diagnostics: dict[str, Any] = {}

    # ==================================================================
    # Gym reset
    # ==================================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reset the simulator and environment state.

        Returns the initial causal observation and metadata.
        """

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
        self._last_scope_decision = None

        self._remaining_work = float(
            self._total_work
        )

        self._previous_remaining_work = (
            self._remaining_work
        )

        self._episode_started = True
        self._last_qpso_diagnostics = {}

        self.simulator.reset(
            self.scenario,
            self.current_plan,
        )

        simulator_time = float(
            self.simulator.sim_time_s
        )

        if not np.isfinite(simulator_time):
            raise RuntimeError(
                "simulator returned non-finite time after reset"
            )

        if simulator_time < 0.0:
            raise RuntimeError(
                "simulator returned negative time after reset"
            )

        self._sim_time_s = simulator_time

        observation = self._observation()

        info = {
            "sim_time_s": float(
                self._sim_time_s
            ),
            "decision_index": 0,
            "action_names": {
                index: action.value
                for index, action
                in INDEX_TO_ACTION.items()
            },
            "action_mask": self._action_mask().tolist(),
        }

        return observation, info

    # ==================================================================
    # Gym step
    # ==================================================================

    def step(
        self,
        action: int,
    ):
        """
        Execute one 60-second control interval.
        """

        if not self._episode_started:
            raise RuntimeError(
                "step() called before reset()"
            )

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
        phase_start = wall_start
        timing = {
            "affected_state_seconds": 0.0,
            "action_mask_seconds": 0.0,
            "scope_selection_seconds": 0.0,
            "qpso_seconds": 0.0,
            "simulator_advance_seconds": 0.0,
            "reward_seconds": 0.0,
            "state_commit_seconds": 0.0,
        }

        # --------------------------------------------------------------
        # Current visible affected state
        # --------------------------------------------------------------

        affected_vehicles = self._affected_vehicle_ids()

        affected_zones = self._affected_zone_ids()
        timing["affected_state_seconds"] = perf_counter() - phase_start
        phase_start = perf_counter()

        # --------------------------------------------------------------
        # Feasibility mask
        # --------------------------------------------------------------

        action_mask = self._action_mask(
            affected_vehicle_ids=affected_vehicles,
            affected_zone_ids=affected_zones,
        )
        timing["action_mask_seconds"] = perf_counter() - phase_start
        phase_start = perf_counter()

        requested_index = ACTION_TO_INDEX[
            requested_action
        ]

        if action_mask[requested_index] <= 0.0:
            # This is a policy proposal failure, not a silent route change.
            #
            # We retain the requested action for audit and deterministically
            # fall back to the canonical feasible action.
            safe_index = self._canonical_feasible_action(
                action_mask
            )

            requested_action_index = requested_index

            requested_action = (
                INDEX_TO_ACTION[
                    requested_action_index
                ]
            )

            masked_execution_action = (
                INDEX_TO_ACTION[
                    safe_index
                ]
            )
        else:
            masked_execution_action = (
                requested_action
            )

        # --------------------------------------------------------------
        # Build exact Step-15 scope
        # --------------------------------------------------------------

        selection = self.scope_selector.select(
            masked_execution_action,
            jobs=self.job_impacts,
            affected_vehicle_ids=(
                affected_vehicles
            ),
            affected_zone_ids=(
                affected_zones
            ),
        )
        timing["scope_selection_seconds"] = perf_counter() - phase_start
        phase_start = perf_counter()

        # --------------------------------------------------------------
        # QPSO / KEEP
        # --------------------------------------------------------------

        # Scope decisions are timestamped at the point the action is
        # resolved, before the simulator advances to the next observation.
        decision_time_s = float(self._sim_time_s)

        qpso_called = False
        qpso_evaluations = 0
        qpso_elapsed_s = 0.0

        self._last_qpso_diagnostics = {}

        executed_action = selection.action
        next_plan = self.current_plan

        if (
            selection.action
            != ScopeAction.KEEP
        ):
            qpso_called = True

            qpso_start = perf_counter()

            try:
                next_plan, qpso_evaluations = (
                    self._solve_scope(
                        selection
                    )
                )
            except Exception as exc:
                # A failed optimizer call must never result in dispatching
                # an unvalidated candidate.
                next_plan = self.current_plan
                executed_action = (
                    ScopeAction.KEEP
                )

                qpso_elapsed_s = (
                    perf_counter()
                    - qpso_start
                )

                override_reason = (
                    "QPSO execution failure: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                qpso_elapsed_s = (
                    perf_counter()
                    - qpso_start
                )

                override_reason = None

                if next_plan is self.current_plan:
                    executed_action = (
                        ScopeAction.KEEP
                    )

        else:
            override_reason = None

        timing["qpso_seconds"] = float(qpso_elapsed_s)
        phase_start = perf_counter()

        # --------------------------------------------------------------
        # Advance simulator
        # --------------------------------------------------------------

        sim_result = self.simulator.advance(
            self.config.decision_interval_s
        )

        if not isinstance(
            sim_result,
            Mapping,
        ):
            raise TypeError(
                "simulator.advance() must return a Mapping"
            )

        timing["simulator_advance_seconds"] = perf_counter() - phase_start
        phase_start = perf_counter()

        new_sim_time = float(
            self.simulator.sim_time_s
        )

        if not np.isfinite(
            new_sim_time
        ):
            raise RuntimeError(
                "simulator returned non-finite time"
            )

        if new_sim_time <= self._sim_time_s:
            raise RuntimeError(
                "simulator time did not advance: "
                f"previous={self._sim_time_s}, "
                f"current={new_sim_time}"
            )

        self._sim_time_s = new_sim_time

        # --------------------------------------------------------------
        # Realized reward
        # --------------------------------------------------------------

        route_changed = (
            next_plan != self.current_plan
        )

        reward_components = (
            self._reward_from_simulation(
                sim_result,
                qpso_elapsed_s=(
                    qpso_elapsed_s
                ),
                route_changed=(
                    route_changed
                ),
            )
        )
        timing["reward_seconds"] = perf_counter() - phase_start
        phase_start = perf_counter()

        # --------------------------------------------------------------
        # Commit state
        # --------------------------------------------------------------

        self.current_plan = next_plan

        self._last_action = executed_action

        self._recent_action_counts[
            executed_action
        ] += 1
        timing["state_commit_seconds"] = perf_counter() - phase_start

        elapsed_s = (
            perf_counter()
            - wall_start
        )

        if not np.isfinite(
            elapsed_s
        ):
            raise RuntimeError(
                "decision wall-clock time is non-finite"
            )

        # --------------------------------------------------------------
        # Determine override
        # --------------------------------------------------------------

        was_mask_override = (
            masked_execution_action
            != requested_action
        )

        was_execution_override = (
            executed_action
            != masked_execution_action
        )

        # ScopeDecision defines an override causally: the action that was
        # requested must differ from the action that was actually executed.
        # A selector may internally mark an override path even when the final
        # action remains unchanged (for example, KEEP -> KEEP); that must not
        # be serialized as an overridden ScopeDecision.
        overridden = (
            requested_action
            != executed_action
        )

        if (
            selection.overridden
            and selection.override_reason
        ):
            final_override_reason = (
                selection.override_reason
            )
        elif was_mask_override:
            final_override_reason = (
                "Requested action was infeasible "
                "under the current action mask."
            )
        elif override_reason is not None:
            final_override_reason = (
                override_reason
            )
        elif was_execution_override:
            final_override_reason = (
                "Candidate route was not accepted; "
                "retained the validated incumbent."
            )
        else:
            final_override_reason = None

        # ScopeDecision requires override_reason to be present exactly when
        # the executed action differs from the requested action.  Internal
        # selector/repair paths can produce a reason even when the final
        # action is unchanged (for example KEEP -> KEEP), so do not leak
        # that internal reason into the causal decision record.
        if not overridden:
            final_override_reason = None

        # --------------------------------------------------------------
        # Causal scope decision
        # --------------------------------------------------------------
        #
        # DecisionMemory requires ScopeDecision, not DecisionRecord.
        # The latter is intentionally a post-execution audit record carrying
        # realized reward and QPSO information.
        #
        # If the final execution falls back to KEEP after a failed/rejected
        # replanning attempt, KEEP has no mutable scope by contract.
        if executed_action == ScopeAction.KEEP:
            decision_request_ids: tuple[str, ...] = ()
        else:
            decision_request_ids = tuple(
                selection.request_ids
            )

        self._last_scope_decision = ScopeDecision(
            scenario_id=str(
                self.scenario.scenario_id
            ),
            state_version=(
                f"state-v{self._decision_index}"
            ),
            decision_time_s=decision_time_s,
            requested_action=requested_action,
            executed_action=executed_action,
            affected_vehicle_ids=tuple(
                selection.affected_vehicle_ids
            ),
            mutable_request_ids=decision_request_ids,
            budget_seconds=0.0,
            overridden=bool(
                overridden
            ),
            override_reason=(
                final_override_reason
            ),
            selection_reason=str(
                selection.reason
            ),
            selected_by="ppo_policy",
            decision_version="step17.2c-v1",
        )

        # --------------------------------------------------------------
        # Audit record
        # --------------------------------------------------------------

        record = DecisionRecord(
            decision_index=self._decision_index,
            sim_time_s=self._sim_time_s,
            requested_action=requested_action,
            executed_action=executed_action,
            mutable_request_ids=tuple(
                selection.request_ids
            ),
            affected_vehicle_ids=tuple(
                selection.affected_vehicle_ids
            ),
            overridden=bool(
                overridden
            ),
            override_reason=(
                final_override_reason
            ),
            qpso_called=bool(
                qpso_called
            ),
            qpso_evaluations=int(
                qpso_evaluations
            ),
            elapsed_s=float(
                elapsed_s
            ),
            reward=reward_components,
        )

        self._decision_log.append(
            record
        )

        # --------------------------------------------------------------
        # Termination
        # --------------------------------------------------------------

        terminated = bool(
            self.simulator.done
            or self._remaining_work <= 0.0
        )

        truncated = bool(
            self._sim_time_s
            >= self.config.episode_duration_s
            and not terminated
        )

        observation = self._observation()

        info = {
            "requested_action": (
                requested_action.value
            ),
            "proposed_action": (
                masked_execution_action.value
            ),
            "executed_action": (
                executed_action.value
            ),
            "overridden": bool(
                overridden
            ),
            "override_reason": (
                final_override_reason
            ),
            "mutable_request_ids": list(
                selection.request_ids
            ),
            "affected_vehicle_ids": list(
                selection.affected_vehicle_ids
            ),
            "qpso_called": bool(
                qpso_called
            ),
            "qpso_evaluations": int(
                qpso_evaluations
            ),
            "qpso_diagnostics": dict(
                self._last_qpso_diagnostics
            ),
            "decision_elapsed_s": float(
                elapsed_s
            ),
            "qpso_elapsed_s": float(
                qpso_elapsed_s
            ),
            "timing": {k: float(v) for k, v in timing.items()},
            "sim_time_s": float(
                self._sim_time_s
            ),
            "decision_index": int(
                self._decision_index
            ),
            "action_mask": action_mask.copy(),
            "reward_components": (
                reward_components.as_dict()
            ),
            "remaining_work": float(
                self._remaining_work
            ),
            "route_changed": bool(
                route_changed
            ),
        }

        self._decision_index += 1

        return (
            observation,
            float(
                reward_components.reward
            ),
            terminated,
            truncated,
            info,
        )

    # ==================================================================
    # Scope solving
    # ==================================================================

    @staticmethod
    def _build_qpso_initial_population(
        *,
        incumbent_keys: Sequence[float],
        population_size: int,
        seed: int,
    ) -> list[list[float]]:
        """Build a deterministic warm-start population with diversity.

        Particle zero is the exact incumbent. Remaining particles are local
        perturbations in the encoder's normalized [0, 1] search space. A
        bounded perturbation keeps the warm start local while ensuring QPSO
        has non-zero initial coordinate diversity.
        """
        if population_size <= 0:
            raise ValueError(
                "population_size must be positive"
            )

        incumbent = [
            float(value)
            for value in incumbent_keys
        ]

        population = [list(incumbent)]

        if population_size == 1:
            return population

        rng = Random(int(seed))

        # The first n coordinates are assignment preferences. With two
        # vehicles, the decoder boundary is 0.5; therefore a +/-0.10
        # perturbation around canonical keys (0.25 / 0.75) can never change
        # a vehicle assignment. That creates apparent continuous diversity
        # while still decoding to the same discrete route. Use a bounded
        # exploratory scale for assignment coordinates and a smaller local
        # scale for ordering coordinates. The decoder/repair/evaluator remain
        # authoritative for feasibility.
        customer_count = len(incumbent) // 2
        assignment_scale = 0.35
        order_scale = 0.15

        for particle_index in range(1, population_size):
            particle: list[float] = []

            for dimension, value in enumerate(incumbent):
                scale = (
                    assignment_scale
                    if dimension < customer_count
                    else order_scale
                )

                # Alternate the broad assignment direction across particles
                # so the deterministic warm start explores both sides of the
                # discrete assignment boundary instead of relying only on
                # random chance.
                if dimension < customer_count:
                    direction = (
                        -1.0
                        if (particle_index + dimension) % 2 == 0
                        else 1.0
                    )
                    offset = direction * rng.uniform(0.05, scale)
                else:
                    offset = rng.uniform(-scale, scale)

                particle.append(
                    min(
                        1.0,
                        max(
                            0.0,
                            value + offset,
                        ),
                    )
                )

            population.append(particle)

        return population

    def _solve_scope(
        self,
        selection: ScopeSelection,
    ) -> tuple[RoutePlan, int]:
        """
        Run the real Step-7/QPSO stack for one selected scope.

        Returns
        -------
        next_plan
            Validated candidate plan, or the current plan if no valid
            candidate is produced.

        evaluations
            Number of QPSO oracle evaluations reported by the optimizer.
        """

        commitments = (
            build_commitment_snapshot(
                self.scenario,
                self.current_plan,
                selection,
                planning_time_s=(
                    self._sim_time_s
                ),
            )
        )

        closed_edge_ids = tuple(
            getattr(
                self.simulator,
                "closed_edge_ids",
                (),
            )
        )

        evaluator = RouteEvaluator(
            self.scenario,
            closed_edge_ids=closed_edge_ids,
        )

        engine = Step7RouteEngine(
            self.scenario,
            evaluator,
        )

        oracle = RouteFitnessOracle(
            engine,
            commitments=commitments,
            planning_time_s=(
                self._sim_time_s
            ),
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
                seed=(
                    self.config.qpso_seed
                    + self._decision_index
                ),
            )
        )

        # --------------------------------------------------------------
        # Warm-start population from incumbent.
        # --------------------------------------------------------------

        incumbent = engine.encoder.encode(
            self.current_plan,
            commitments=commitments,
        )

        # Keep the incumbent as an exact warm-start particle, but do not
        # initialize every particle at the same point. Identical particles
        # collapse QPSO's coordinate/route diversity to zero and can make
        # the first search updates degenerate around the incumbent. The
        # remaining particles receive deterministic, bounded local
        # perturbations; route encoding/repair remains the authority for
        # feasibility.
        population = self._build_qpso_initial_population(
            incumbent_keys=incumbent.keys,
            population_size=self.config.qpso_particles,
            seed=(
                self.config.qpso_seed
                + self._decision_index
            ),
        )

        optimizer = AdaptiveQPSO(
            qpso_config,
            oracle,
            initial_population=population,
        )

        result = optimizer.optimize()

        evaluations = int(
            result.evaluations
        )

        # --------------------------------------------------------------
        # Final candidate evaluation
        # --------------------------------------------------------------

        candidate = engine.evaluate_keys(
            result.best_position,
            commitments=commitments,
            planning_time_s=(
                self._sim_time_s
            ),
            repair=True,
        )

        repaired_evaluation = (
            candidate.repaired_evaluation
        )

        if not repaired_evaluation.feasible:
            repair_result = candidate.repair_result
            self._last_qpso_diagnostics = {
                "stage": "repaired_evaluation",
                "feasible": False,
                "violations": [
                    {
                        "name": v.name,
                        "magnitude": float(v.magnitude),
                        "vehicle_id": v.vehicle_id,
                        "customer_id": v.customer_id,
                        "message": v.message,
                    }
                    for v in repaired_evaluation.violations
                ],
                "errors": list(repaired_evaluation.errors),
                "unserved_customer_ids": list(repaired_evaluation.unserved_customer_ids),
                "duplicate_customer_ids": list(repaired_evaluation.duplicate_customer_ids),
                "unknown_request_ids": list(repaired_evaluation.unknown_request_ids),
                "unknown_vehicle_ids": list(repaired_evaluation.unknown_vehicle_ids),
                "repair_failure_reason": (
                    repair_result.failure_reason
                    if repair_result is not None
                    else None
                ),
                "repair_unresolved_customer_ids": (
                    list(repair_result.unresolved_customer_ids)
                    if repair_result is not None
                    else []
                ),
            }
            return (
                self.current_plan,
                evaluations,
            )

        candidate_plan = (
            candidate.repaired_plan
        )

        # Final independent evaluator pass.
        final_evaluation = evaluator.evaluate(
            candidate_plan,
            commitments=commitments,
            planning_time_s=(
                self._sim_time_s
            ),
        )

        if not final_evaluation.feasible:
            self._last_qpso_diagnostics = {
                "stage": "final_evaluation",
                "feasible": False,
                "violations": [
                    {
                        "name": v.name,
                        "magnitude": float(v.magnitude),
                        "vehicle_id": v.vehicle_id,
                        "customer_id": v.customer_id,
                        "message": v.message,
                    }
                    for v in final_evaluation.violations
                ],
                "errors": list(final_evaluation.errors),
                "unserved_customer_ids": list(final_evaluation.unserved_customer_ids),
                "duplicate_customer_ids": list(final_evaluation.duplicate_customer_ids),
                "unknown_request_ids": list(final_evaluation.unknown_request_ids),
                "unknown_vehicle_ids": list(final_evaluation.unknown_vehicle_ids),
            }
            return (
                self.current_plan,
                evaluations,
            )

        # The environment's current_plan is authoritative only after the
        # candidate has been accepted by the live simulator.  Without this
        # handoff, QPSO would optimize a plan that SUMO never executes.
        apply_route_plan = getattr(
            self.simulator,
            "apply_route_plan",
            None,
        )

        if apply_route_plan is not None:
            try:
                applied = bool(
                    apply_route_plan(
                        candidate_plan,
                        commitments=commitments,
                        planning_time_s=self._sim_time_s,
                    )
                )
            except Exception:
                logger = __import__("logging").getLogger(__name__)
                logger.exception(
                    "Live route-plan application failed; "
                    "retaining incumbent route"
                )
                applied = False

            if not applied:
                self._last_qpso_diagnostics = {
                    "stage": "live_apply",
                    "feasible": True,
                    "apply_route_plan": False,
                }
                return (
                    self.current_plan,
                    evaluations,
                )

        return (
            candidate_plan,
            evaluations,
        )

    # ==================================================================
    # Affected-state providers
    # ==================================================================

    def _affected_vehicle_ids(
        self,
    ) -> tuple[str, ...]:
        if (
            self.affected_vehicle_provider
            is None
        ):
            return ()

        values = self.affected_vehicle_provider(
            self._sim_time_s
        )

        return self._unique_strings(
            values
        )

    def _affected_zone_ids(
        self,
    ) -> tuple[str, ...]:
        if (
            self.affected_zone_provider
            is None
        ):
            return ()

        values = self.affected_zone_provider(
            self._sim_time_s
        )

        return self._unique_strings(
            values
        )

    @staticmethod
    def _unique_strings(
        values: Sequence[Any]
        | Any,
    ) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            normalized = str(value)

            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(normalized)

        return tuple(result)

    # ==================================================================
    # Action mask
    # ==================================================================

    def _action_mask(
        self,
        *,
        affected_vehicle_ids: Sequence[str] = (),
        affected_zone_ids: Sequence[str] = (),
    ) -> np.ndarray:
        """
        Return the current five-action feasibility mask.

        A custom provider may return:
            * ActionMask
            * numpy array
            * sequence of 0/1 values

        The mask is feasibility information only. Route legality remains
        the responsibility of the selector/safety/route validator.
        """

        if self.action_mask_provider:
            raw_mask = self.action_mask_provider(
                self._sim_time_s
            )
            return self._normalize_action_mask(
                raw_mask
            )

        # Conservative default:
        # all semantic actions are available. Scope/safety/route validation
        # remains responsible for rejecting unsafe concrete plans.
        return np.ones(
            len(ACTION_TO_INDEX),
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_action_mask(
        raw_mask: Any,
    ) -> np.ndarray:
        if isinstance(
            raw_mask,
            ActionMask,
        ):
            values = getattr(
                raw_mask,
                "values",
                None,
            )

            if values is None:
                values = np.asarray(
                    [
                        float(
                            raw_mask.is_feasible(
                                action
                            )
                        )
                        for action in ScopeAction
                    ],
                    dtype=np.float32,
                )

            mask = np.asarray(
                values,
                dtype=np.float32,
            )
        else:
            mask = np.asarray(
                raw_mask,
                dtype=np.float32,
            )

        mask = mask.reshape(-1)

        if len(mask) != len(
            ACTION_TO_INDEX
        ):
            raise ValueError(
                "action mask must contain exactly "
                f"{len(ACTION_TO_INDEX)} values; "
                f"got {len(mask)}"
            )

        if not np.all(
            np.isfinite(mask)
        ):
            raise ValueError(
                "action mask contains non-finite values"
            )

        mask = (
            mask > 0.5
        ).astype(
            np.float32
        )

        if not np.any(mask):
            raise RuntimeError(
                "action mask contains no feasible actions"
            )

        mask.setflags(write=False)

        return mask

    @staticmethod
    def _canonical_feasible_action(
        action_mask: np.ndarray,
    ) -> int:
        """
        Deterministically choose the least invasive feasible action.

        Preference:
            KEEP → LOCAL → VEHICLE → REGIONAL → GLOBAL
        """

        for index in range(
            len(INDEX_TO_ACTION)
        ):
            if action_mask[index] > 0.0:
                return index

        raise RuntimeError(
            "no feasible action exists"
        )

    # ==================================================================
    # Observation
    # ==================================================================

    def _observation(self) -> np.ndarray:
        """
        Build the legacy compatibility observation.

        This deliberately does NOT call the GNN or Transformer.

        When `policy_runtime` is integrated, a structured causal state adapter
        can replace this observation path without changing the action/QPSO/
        simulator/reward contract.
        """

        forecast_summary: Sequence[
            float
        ] = ()

        if self.forecast_provider:
            forecast_summary = tuple(
                float(value)
                for value in coerce_ppo_forecast_signal(
                    self.forecast_provider(self._sim_time_s)
                ).forecast_features
            )

        state = getattr(
            self.simulator,
            "latest_state",
            {},
        )

        if state is None:
            state = {}

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

        observation = (
            self.observation_encoder.encode(
                sim_time_s=(
                    self._sim_time_s
                ),
                remaining_work=(
                    self._remaining_work
                ),
                total_work=float(
                    self._total_work
                ),
                affected_route_fraction=(
                    affected_fraction
                ),
                deadline_slacks_s=(
                    deadline_slacks
                ),
                vehicle_loads=(
                    vehicle_loads
                ),
                vehicle_capacities=(
                    vehicle_capacities
                ),
                active_event_count=(
                    active_events
                ),
                last_action=(
                    self._last_action
                ),
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
        )

        if not self.observation_space.contains(
            observation
        ):
            raise RuntimeError(
                "generated observation violates Gym observation space"
            )

        return observation

    # ==================================================================
    # Reward
    # ==================================================================

    def _reward_from_simulation(
        self,
        sim_result: Mapping[str, Any],
        *,
        qpso_elapsed_s: float,
        route_changed: bool,
    ) -> RewardComponents:
        """
        Convert realized simulator outcomes into the interval reward.

        IMPORTANT:
        `remaining_work` is a terminal-risk penalty. Positive unfinished
        work is penalized, while zero unfinished work receives zero penalty.

        No predicted future improvement contributes to reward.
        """

        operating_cost = self._finite_nonnegative(
            sim_result.get(
                "incremental_operating_cost",
                sim_result.get(
                    "operating_cost",
                    0.0,
                ),
            ),
            name="operating_cost",
        )

        congestion = self._finite_nonnegative(
            sim_result.get(
                "congestion_exposure",
                0.0,
            ),
            name="congestion_exposure",
        )

        service_failures = self._finite_nonnegative(
            sim_result.get(
                "service_failures",
                0.0,
            ),
            name="service_failures",
        )

        unfinished = self._finite_nonnegative(
            sim_result.get(
                "remaining_work",
                self._remaining_work,
            ),
            name="remaining_work",
        )

        self._previous_remaining_work = (
            self._remaining_work
        )

        self._remaining_work = min(
            unfinished,
            float(self._total_work),
        )

        churn = (
            1.0
            if route_changed
            else 0.0
        )

        computation = self._finite_nonnegative(
            qpso_elapsed_s,
            name="qpso_elapsed_s",
        )

        # --------------------------------------------------------------
        # Weighted normalized components
        # --------------------------------------------------------------

        operating_penalty = (
            self.config.operating_cost_weight
            * operating_cost
            / self.config.operating_cost_scale
        )

        congestion_penalty = (
            self.config.congestion_weight
            * congestion
            / self.config.congestion_scale
        )

        service_failure_penalty = (
            self.config.service_failure_weight
            * service_failures
        )

        churn_penalty = (
            self.config.churn_weight
            * churn
        )

        computation_penalty = (
            self.config.computation_weight
            * computation
            / self.config.computation_scale
        )

        # --------------------------------------------------------------
        # FIXED:
        #
        # The old implementation accidentally applied the unfinished-work
        # penalty when unfinished <= 0, which means it never penalized
        # unfinished work.
        #
        # The terminal penalty must increase with remaining work.
        # --------------------------------------------------------------

        unfinished_work_penalty = (
            self.config.unfinished_work_weight
            * (
                self._remaining_work
                / max(
                    float(self._total_work),
                    1.0,
                )
            )
        )

        components = RewardComponents(
            operating_cost=float(
                operating_penalty
            ),
            congestion_exposure=float(
                congestion_penalty
            ),
            service_failures=float(
                service_failure_penalty
            ),
            churn=float(
                churn_penalty
            ),
            computation=float(
                computation_penalty
            ),
            unfinished_work=float(
                unfinished_work_penalty
            ),
        )

        if not np.isfinite(
            components.total_penalty
        ):
            raise RuntimeError(
                "reward contains non-finite values"
            )

        return components

    @staticmethod
    def _finite_nonnegative(
        value: Any,
        *,
        name: str,
    ) -> float:
        numeric = float(value)

        if not np.isfinite(
            numeric
        ):
            raise ValueError(
                f"{name} must be finite"
            )

        if numeric < 0.0:
            raise ValueError(
                f"{name} must be non-negative"
            )

        return numeric

    # ==================================================================
    # PPO integration boundary
    # ==================================================================

    def policy_decide(
        self,
        policy_state: Any,
        *,
        action_mask: Any | None = None,
    ) -> PPOPolicyDecision:
        """
        Ask the optional PPO runtime for an action.

        This method is intentionally separate from `step()`.

        Why?
        ----
        The environment remains usable with ordinary Gymnasium training code,
        while the project's model-independent representation boundary can later
        feed `PPORuntime`.

        The GNN/Transformer implementation can therefore change without
        changing the environment's routing/QPSO/SUMO contract.
        """

        if self.policy_runtime is None:
            raise RuntimeError(
                "policy_runtime is not configured"
            )

        mask = (
            self._normalize_action_mask(
                action_mask
            )
            if action_mask is not None
            else self._action_mask()
        )

        return self.policy_runtime.decide(
            policy_state,
            action_mask=mask,
        )

    def policy_step(
        self,
        policy_state,
        controller,
        action_mask=None,
    ):
        """
        Execute one environment step using a PPOController.

        The controller produces the policy's requested action. The environment
        then executes that action through the normal `step()` path.

        This keeps policy inference separate from environment execution.

        Returns
        -------
        observation:
            Next environment observation.

        reward:
            Realized interval reward.

        terminated:
            Whether the episode naturally terminated.

        truncated:
            Whether the episode was time-truncated.

        info:
            Normal environment metadata plus PPO policy statistics.
        """

        if controller is None:
            raise ValueError(
                "controller is required"
            )

        resolved_action_mask = (
            self._action_mask()
            if action_mask is None
            else action_mask
        )
        
        decision = controller.decide(
            policy_state,
            action_mask=resolved_action_mask,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.step(
            decision.action_index
        )

        # Preserve the distinction between:
        #
        #   requested action = PPO output
        #
        #   executed action = downstream environment output
        #
        # This is required for correct rollout/audit semantics.
        info = dict(info)

        info.update(
            {
                "ppo_requested_action": (
                    decision.requested_action.value
                ),
                "ppo_action_index": int(
                    decision.action_index
                ),
                "ppo_log_probability": float(
                    decision.log_probability
                ),
                "ppo_value": float(
                    decision.value
                ),
            }
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ==================================================================
    # Audit access
    # ==================================================================

    @property
    def decision_log(
        self,
    ) -> tuple[DecisionRecord, ...]:
        """Return immutable oldest-to-newest decision history."""

        return tuple(
            self._decision_log
        )

    @property
    def last_scope_decision(
        self,
    ) -> ScopeDecision | None:
        """Return the authoritative causal scope decision for the latest step."""

        return self._last_scope_decision

    @property
    def sim_time_s(self) -> float:
        return float(
            self._sim_time_s
        )

    @property
    def remaining_work(self) -> float:
        return float(
            self._remaining_work
        )

    @property
    def last_action(self) -> ScopeAction:
        return self._last_action

    @property
    def decision_count(self) -> int:
        return int(
            self._decision_index
        )

    # ==================================================================
    # Gym render/close
    # ==================================================================

    def render(self):
        """
        Rendering is intentionally delegated to the platform/UI layer.

        The environment itself has no graphical renderer.
        """

        return None

    def close(self) -> None:
        """
        Give a production simulator a chance to close resources.

        The base SimulatorBackend does not require a close method, so this
        remains optional.
        """

        close_method = getattr(
            self.simulator,
            "close",
            None,
        )

        if callable(
            close_method
        ):
            close_method()
