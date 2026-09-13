from __future__ import annotations

from src.routing.route_plan import (
    RoutePlan,
    VehicleRoute,
)
from src.learning.ppo_env import (
    TrafficRoutingPPOEnv,
    SimulatorBackend,
)


class DeterministicSimulator(SimulatorBackend):

    def __init__(self):
        self._time = 0.0
        self._done = False
        self.latest_state = {}

    def reset(self, scenario, route_plan):
        self._time = 0.0
        self._done = False
        self.latest_state = {
            "deadline_slacks_s": [
                300.0,
                600.0,
            ],
            "vehicle_loads": [
                1.0,
                1.0,
            ],
            "vehicle_capacities": [
                5.0,
                5.0,
            ],
            "affected_route_fraction": 0.0,
            "active_event_count": 0,
            "congestion_exposure": 0.0,
            "mean_speed_ratio": 1.0,
            "route_change_fraction": 0.0,
        }

    def advance(self, duration_s):
        self._time += duration_s

        if self._time >= 180.0:
            self._done = True

        return {
            "incremental_operating_cost": 100.0,
            "congestion_exposure": 10.0,
            "service_failures": 0.0,
            "remaining_work": max(
                0.0,
                5.0 - self._time / 60.0,
            ),
        }

    @property
    def sim_time_s(self):
        return self._time

    @property
    def done(self):
        return self._done


def make_plan():
    return RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "V1",
                ["J1", "J2", "J3"],
            ),
            VehicleRoute.from_sequence(
                "V2",
                ["J4", "J5"],
            ),
        ]
    )


def test_environment_has_five_actions(scenario):
    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=DeterministicSimulator(),
    )

    assert env.action_space.n == 5

    observation, info = env.reset()

    assert observation.shape == (
        env.config.observation_size,
    )

    assert observation.dtype.name == "float32"


def test_one_keep_step_advances_exactly_60_seconds(
    scenario,
):
    simulator = DeterministicSimulator()

    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=simulator,
    )

    env.reset()

    observation, reward, terminated, truncated, info = (
        env.step(0)
    )

    assert simulator.sim_time_s == 60.0

    assert info["requested_action"] == "KEEP"
    assert info["executed_action"] == "KEEP"
    assert info["qpso_called"] is False

    assert isinstance(
        reward,
        float,
    )

    assert not terminated
    assert not truncated


def test_replanning_action_calls_real_qpso(
    scenario,
):
    simulator = DeterministicSimulator()

    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=simulator,
    )

    env.reset()

    observation, reward, terminated, truncated, info = (
        env.step(1)
    )

    assert info["requested_action"] == "LOCAL"

    assert info["qpso_called"] is True

    assert info["qpso_evaluations"] == (
        env.config.qpso_evaluations
    )

    assert simulator.sim_time_s == 60.0


def test_decision_log_records_requested_and_executed_action(
    scenario,
):
    simulator = DeterministicSimulator()

    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=simulator,
    )

    env.reset()

    env.step(0)

    assert len(
        env.decision_log
    ) == 1

    record = env.decision_log[0]

    assert record.requested_action.value == "KEEP"
    assert record.executed_action.value == "KEEP"

    assert record.reward.reward <= 0.0
    assert record.elapsed_s >= 0.0