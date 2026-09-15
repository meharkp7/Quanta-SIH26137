from __future__ import annotations

import json
import statistics
import time

import pytest

from tests.step16.test_ppo_env import (
    DeterministicSimulator,
    make_plan,
)

from src.learning.ppo_env import TrafficRoutingPPOEnv


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered))

    if lower == upper:
        return ordered[lower]

    fraction = index - lower

    return ordered[lower] + fraction * (
        ordered[upper] - ordered[lower]
    )


def benchmark_action(
    scenario,
    action: int,
    repetitions: int = 20,
) -> dict:
    timings = []

    for _ in range(repetitions):
        simulator = DeterministicSimulator()

        env = TrafficRoutingPPOEnv(
            scenario,
            make_plan(),
            simulator=simulator,
        )

        env.reset()

        start = time.perf_counter()
        env.step(action)
        elapsed = time.perf_counter() - start

        timings.append(elapsed)

    mean_s = statistics.mean(timings)

    return {
        "action": action,
        "repetitions": repetitions,
        "mean_s": mean_s,
        "median_s": statistics.median(timings),
        "p95_s": percentile(timings, 0.95),
        "min_s": min(timings),
        "max_s": max(timings),
        "steps_per_second": 1.0 / mean_s,
    }


def estimate_training_cost(
    seconds_per_step: float,
    intended_steps: int,
    parallel_envs: int = 1,
) -> dict:
    sequential_seconds = intended_steps * seconds_per_step

    effective_seconds = (
        sequential_seconds / parallel_envs
        if parallel_envs > 1
        else sequential_seconds
    )

    return {
        "intended_steps": intended_steps,
        "parallel_envs": parallel_envs,
        "seconds_per_step": seconds_per_step,
        "estimated_wall_clock_s": effective_seconds,
        "estimated_wall_clock_min": effective_seconds / 60.0,
        "estimated_wall_clock_hr": effective_seconds / 3600.0,
    }


def test_benchmark_ppo_environment_throughput(scenario):
    keep = benchmark_action(
        scenario,
        action=0,
        repetitions=20,
    )

    local = benchmark_action(
        scenario,
        action=1,
        repetitions=20,
    )

    print("\n=== PPO ENVIRONMENT THROUGHPUT ===")
    print(json.dumps(
        {
            "KEEP": keep,
            "LOCAL_QPSO": local,
        },
        indent=2,
    ))

    training = estimate_training_cost(
        seconds_per_step=local["mean_s"],
        intended_steps=10_000,
        parallel_envs=1,
    )

    print("\n=== TRAINING COST ESTIMATE ===")
    print(json.dumps(training, indent=2))

    assert keep["mean_s"] > 0.0
    assert local["mean_s"] > 0.0
    assert keep["p95_s"] >= keep["median_s"]
    assert local["p95_s"] >= local["median_s"]