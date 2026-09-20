"""Generate a fault-tolerant, reproducible Quanta SUMO training corpus.

Production properties
---------------------
* Uses the repository's real dynamic scenario generator, event generator,
  RoutingPipeline, SUMO/TraCI runner and causal audit.
* Never uses fixture routes, fake observations or a mock simulator.
* Uses deterministic seed streams separated by concern.
* Retries failed map generation with a *different deterministic configuration*
  rather than replaying the same failure.
* Retries failed SUMO episodes with a *different deterministic episode seed*
  and, when required, a deterministic traffic/event variant.
* Persists a checkpoint after every accepted episode so an interrupted run can
  continue without discarding completed work.
* Refuses infinite retry loops: every map/episode has a bounded retry budget.
* Records every failed attempt and its reason in the manifest.
* Keeps train/validation/test splits map-disjoint.

Default target: 2,000 real SUMO episodes on 20 base maps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.scenario import Scenario
from src.data.causal_audit import audit_episode
from src.data.causal_episodes import _finalize_existing_episode, _write_json
from src.data.dataset_generator import DynamicDatasetConfig, generate_dynamic_dataset
from src.data.dynamic_episodes import DynamicEpisodeConfig, _generate_events
from src.routing.initial_solution import InitialSolutionConfig
from src.routing.pipeline import RoutingPipeline, RoutingPipelineConfig
from src.sim.route_builder import RouteBuilder
from src.sim.sumo_causal import run_sumo_causal_episode
from src.sim.sumo_exporter import SumoExporter

TRAFFIC_REGIMES = ("normal", "morning_peak", "evening_peak", "corridor_surge")
EVENT_TYPES = (None, "incident", "closure", "multi_disruption")
WINDOWS = ("loose", "medium", "tight")
TOPOLOGIES = ("grid", "irregular")

MAP_RETRIES = 8
ROUTE_PLAN_RETRIES = 5
EPISODE_RETRIES = 8


# SUMO/TraCI parses random seeds as signed 32-bit integers. Keep a safety
# margin because the causal runner derives additional streams from the episode
# seed (currently +211, +307 and +419).
SUMO_SEED_MAX = 2_147_483_647
SUMO_SEED_HEADROOM = 1_024
SUMO_SEED_MODULUS = SUMO_SEED_MAX - SUMO_SEED_HEADROOM


def stable_seed(*parts: int) -> int:
    """Derive a deterministic positive seed accepted by SUMO.

    Every derived seed is strictly below 2^31 and leaves headroom for
    downstream deterministic stream offsets.
    """
    payload = ":".join(str(int(x)) for x in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    raw = int.from_bytes(digest[:8], "big")
    return 1 + (raw % SUMO_SEED_MODULUS)


def map_fingerprint(scenario: Scenario) -> str:
    nodes = {n.node_id: (float(n.x_m), float(n.y_m)) for n in scenario.nodes}
    roads = sorted(
        (
            nodes[e.from_node],
            nodes[e.to_node],
            float(e.length_m),
            float(e.speed_limit_mps),
            int(e.lane_count),
        )
        for e in scenario.edges
    )
    payload = json.dumps(roads, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def map_config(index: int, split: str, seed: int, attempt: int = 0) -> DynamicDatasetConfig:
    """Build a deterministic map config; later attempts deliberately alter geometry."""
    topology = TOPOLOGIES[index % len(TOPOLOGIES)]

    # Retry ladder: denser network + modestly less dispersed customers.
    density_bonus = min(attempt, 4)
    if topology == "grid":
        rows = 10 + (index % 4) + density_bonus
        cols = 10 + ((index * 3) % 5) + density_bonus
        junctions = rows * cols
    else:
        rows = 10
        cols = 10
        junctions = 110 + (index % 5) * 14 + density_bonus * 18

    extent = 5000.0 + (index % 4) * 1000.0
    spread = max(0.55, 0.72 + (index % 4) * 0.07 - attempt * 0.035)
    cluster = max(0.25, 0.35 + (index % 5) * 0.12 - attempt * 0.025)
    jitter = max(0.02, 0.04 + (index % 4) * 0.04 - attempt * 0.005)
    one_way = max(0.02, 0.05 + (index % 4) * 0.04 - attempt * 0.005)

    # Keep the declared snap gate at 500 m. We repair generation by changing
    # the geometry/customer distribution, not by silently weakening quality.
    map_seed = stable_seed(seed, 0x4D4150, index, attempt)

    return DynamicDatasetConfig(
        customer_count=40 + (index % 5) * 10,
        road_junction_count=junctions,
        extent_m=extent,
        topology_family=topology,
        grid_rows=rows,
        grid_cols=cols,
        jitter_fraction=jitter,
        one_way_fraction=one_way,
        max_customer_snap_m=500.0,
        window_profile=WINDOWS[index % len(WINDOWS)],
        window_slack_s=600.0 + (index % 4) * 300.0,
        service_duration_s=60.0,
        vehicle_capacity=30.0,
        fleet_size=None,
        customer_spread=spread,
        cluster_strength=cluster,
        minimum_road_segment_m=30.0,
        demand_min=1,
        demand_max=8,
        seed=map_seed,
        dataset_split=split,
    )


def episode_config(
    local_index: int,
    map_index: int,
    seed: int,
    duration_s: int,
    attempt: int = 0,
) -> DynamicEpisodeConfig:
    """Deterministically vary the exogenous stream on retry."""
    regime_index = (local_index + map_index + attempt) % len(TRAFFIC_REGIMES)
    event_index = (local_index * 3 + map_index + attempt) % len(EVENT_TYPES)
    regime = TRAFFIC_REGIMES[regime_index]
    event_type = EVENT_TYPES[event_index]

    # On retries, reduce only the complexity of the exogenous event pattern;
    # never manufacture observations or bypass SUMO.
    if attempt >= 4 and event_type == "multi_disruption":
        event_type = "incident"

    if event_type == "multi_disruption":
        event_count = 2 + ((local_index + map_index + attempt) % 2)
    else:
        event_count = 1 if event_type else 0

    # corpus_v2 density/supervision policy, gated on QUANTA_CORPUS_V2=1 so every
    # existing caller reproduces legacy behavior unless the v2 driver opts in
    # (the env var propagates automatically to sharded worker processes):
    #   * background departure gap 1.0-2.5 s (legacy 3.0-6.0 s) -> ~2.5x more
    #     background probe vehicles per episode -> richer per-bucket speed
    #     samples (observation missing fraction) and realized_traversal labels.
    #   * emit_closed_edge_speed_zero=True -> closed-edge buckets become valid
    #     speed-0 labels instead of missing rows (see CausalSumoLogger).
    # Episode duration stays a caller argument (--duration-s 3600 for v2 gives
    # 29 issue slots vs 9 at 2400 s; the loader derives issue times from data).
    v2 = os.environ.get("QUANTA_CORPUS_V2") == "1"
    return DynamicEpisodeConfig(
        duration_s=duration_s,
        interval_s=60,
        warmup_s=min(300, duration_s // 5),
        regime=regime,
        event_type=event_type,
        event_count=event_count,
        event_duration_s=240 + ((local_index + map_index + attempt) % 4) * 120,
        reveal_lead_s=60 + ((local_index + 2 * map_index + attempt) % 3) * 60,
        observation_missing_fraction=0.02 + ((local_index + map_index) % 4) * 0.01,
        observation_delay_s=((local_index + map_index) % 3) * 10,
        speed_noise_fraction=0.02 + ((local_index + map_index) % 3) * 0.01,
        occupancy_noise=0.01 + ((local_index + map_index) % 3) * 0.005,
        seed=stable_seed(seed, 0x45504953, map_index, local_index, attempt),
        backend="sumo",
        background_min_gap_s=1.0 if v2 else None,
        background_max_gap_s=2.5 if v2 else None,
        emit_closed_edge_speed_zero=v2,
    )


def build_split_map_indices(total_maps: int) -> dict[str, list[int]]:
    if total_maps < 5:
        raise ValueError("production corpus requires at least 5 base maps")
    train = max(1, int(round(total_maps * 0.80)))
    validation = max(1, int(round(total_maps * 0.10)))
    test = total_maps - train - validation
    if test < 1:
        test = 1
        train -= 1
    return {
        "train": list(range(0, train)),
        "validation": list(range(train, train + validation)),
        "test": list(range(train + validation, total_maps)),
    }


def _safe_remove(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _attempt_record(
    *,
    kind: str,
    index: int,
    attempt: int,
    seed: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "index": index,
        "attempt": attempt,
        "seed": int(seed),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _save_checkpoint(
    output: Path,
    *,
    requested_episodes: int,
    maps: int,
    episodes_per_map: int,
    duration_s: int,
    seed: int,
    map_records: dict[int, dict[str, Any]],
    episode_records: list[dict[str, Any]],
    failed_attempts: list[dict[str, Any]],
    started: float,
) -> None:
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in episode_records:
        counts[record["split"]] = counts.get(record["split"], 0) + 1

    checkpoint = {
        "schema_version": "quanta-real-sumo-corpus-v3-checkpoint",
        "requested_episodes": requested_episodes,
        "generated_episodes": len(episode_records),
        "base_maps": maps,
        "episodes_per_map": episodes_per_map,
        "duration_s": duration_s,
        "interval_s": 60,
        "seed": seed,
        "backend": "sumo",
        "split_episode_counts": counts,
        "map_records": map_records,
        "episodes": episode_records,
        "failed_attempts": failed_attempts,
        "updated_wall_clock_s": time.perf_counter() - started,
    }
    _write_json(output / "corpus_checkpoint.json", checkpoint)


def _scenario_from_map_dir(map_dir: Path) -> Scenario:
    scenario_path = map_dir / "scenario.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"missing scenario.json: {scenario_path}")
    return Scenario.model_validate_json(scenario_path.read_text(encoding="utf-8"))


def _generate_map_with_recovery(
    *,
    output: Path,
    map_index: int,
    split: str,
    seed: int,
    failed_attempts: list[dict[str, Any]],
) -> tuple[Scenario, DynamicDatasetConfig, str, int]:
    map_dir = output / "maps" / f"map_{map_index:03d}"

    for attempt in range(MAP_RETRIES):
        cfg = map_config(map_index, split, seed, attempt)
        tmp_dir = output / "maps" / f".map_{map_index:03d}.attempt_{attempt:02d}"
        _safe_remove(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            generate_dynamic_dataset(tmp_dir, cfg)
            scenario = _scenario_from_map_dir(tmp_dir)
            fingerprint = map_fingerprint(scenario)
            if len(scenario.requests) != cfg.customer_count:
                raise RuntimeError(
                    f"customer-count contract failed: expected {cfg.customer_count}, "
                    f"got {len(scenario.requests)}"
                )
            if not scenario.nodes or not scenario.edges:
                raise RuntimeError("generated scenario has empty road graph")

            # Commit only a validated scenario.
            _safe_remove(map_dir)
            tmp_dir.rename(map_dir)
            return _scenario_from_map_dir(map_dir), cfg, fingerprint, attempt
        except Exception as exc:
            failed_attempts.append(
                _attempt_record(
                    kind="map_generation",
                    index=map_index,
                    attempt=attempt,
                    seed=cfg.seed,
                    error=exc,
                )
            )
            print(
                f"MAP {map_index:03d} generation attempt {attempt + 1}/{MAP_RETRIES} "
                f"failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            _safe_remove(tmp_dir)

    raise RuntimeError(
        f"map {map_index} exhausted {MAP_RETRIES} deterministic generation attempts"
    )


def _route_strategy(attempt: int) -> tuple[str, RoutingPipelineConfig]:
    """Return a deterministic, materially different route-construction strategy."""
    strategies = (
        ("earliest_deadline", RoutingPipelineConfig()),
        ("nearest_feasible", RoutingPipelineConfig(
            initial_solution=InitialSolutionConfig(
                heuristic="nearest_feasible",
                customer_ordering="nearest_feasible",
            )
        )),
        ("earliest_release", RoutingPipelineConfig(
            initial_solution=InitialSolutionConfig(
                heuristic="earliest_release",
                customer_ordering="earliest_release",
            )
        )),
        ("largest_demand", RoutingPipelineConfig(
            initial_solution=InitialSolutionConfig(
                heuristic="largest_demand",
                customer_ordering="largest_demand",
            )
        )),
        ("customer_id", RoutingPipelineConfig(
            initial_solution=InitialSolutionConfig(
                heuristic="customer_id",
                customer_ordering="customer_id",
            )
        )),
    )
    return strategies[attempt % len(strategies)]


def _preflight_physical_route_plan(
    *,
    scenario: Scenario,
    route_plan: Any,
    map_dir: Path,
) -> Path:
    """Compile and materialize the route plan before any episode is attempted.

    This catches structural SUMO failures (empty customer legs, disconnected
    physical routes, illegal transitions, bad background corridors) at map
    preparation time rather than burning episode retries on an immutable map
    defect.
    """
    network_dir = map_dir / "network"
    routes_dir = map_dir / "route_preflight"
    network_dir.mkdir(parents=True, exist_ok=True)
    _safe_remove(routes_dir)
    routes_dir.mkdir(parents=True, exist_ok=True)

    exporter = SumoExporter(scenario, network_dir)
    net_path = exporter.export()
    if not net_path.exists():
        raise RuntimeError(f"SUMO network export did not produce {net_path}")

    builder = RouteBuilder(
        scenario,
        route_plan,
        {"edge_mapping": exporter.edge_mapping},
        background_duration_s=0.0,
    )
    route_path = builder.build(routes_dir)
    if not route_path.exists() or route_path.stat().st_size == 0:
        raise RuntimeError("route preflight produced an empty route file")
    return route_path


def _build_route_plan_with_recovery(
    *,
    scenario: Scenario,
    map_index: int,
    map_dir: Path,
    failed_attempts: list[dict[str, Any]],
) -> tuple[Any, str, int]:
    """Build a physically SUMO-valid route plan with bounded escalation."""
    last_exc: Exception | None = None
    for attempt in range(ROUTE_PLAN_RETRIES):
        strategy_name, pipeline_config = _route_strategy(attempt)
        try:
            result = RoutingPipeline(
                scenario,
                config=pipeline_config,
            ).solve(planning_time_s=0.0)
            if not result.feasible:
                raise RuntimeError(
                    f"routing pipeline infeasible: {result.errors!r}"
                )
            if not result.complete:
                raise RuntimeError(
                    f"routing pipeline incomplete: {result.errors!r}"
                )
            route_plan = result.final_route_plan
            if route_plan is None:
                raise RuntimeError("routing pipeline returned no route plan")

            _preflight_physical_route_plan(
                scenario=scenario,
                route_plan=route_plan,
                map_dir=map_dir,
            )
            return route_plan, strategy_name, attempt
        except Exception as exc:
            last_exc = exc
            failed_attempts.append(
                _attempt_record(
                    kind="route_plan_preflight",
                    index=map_index,
                    attempt=attempt,
                    seed=stable_seed(0x524F555445, map_index, attempt),
                    error=exc,
                )
            )
            print(
                f"  MAP {map_index:03d} route strategy {attempt + 1}/{ROUTE_PLAN_RETRIES} "
                f"({strategy_name}) failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    raise RuntimeError(
        f"map {map_index} exhausted {ROUTE_PLAN_RETRIES} distinct route strategies; "
        f"last error: {last_exc}"
    )

def _episode_success(
    *,
    scenario: Scenario,
    ep_dir: Path,
    eid: str,
    split: str,
    ep_cfg: DynamicEpisodeConfig,
) -> dict[str, Any]:
    finalized = _finalize_existing_episode(
        scenario,
        ep_dir,
        eid,
        split,
        ep_cfg,
    )
    audit = audit_episode(ep_dir)
    if not audit["ok"]:
        raise RuntimeError(f"causal audit failed for {eid}: {audit}")
    return {
        "label_coverage": finalized.label_coverage,
        "audit": audit,
    }


def _run_episode_with_recovery(
    *,
    scenario: Scenario,
    route_plan: Any,
    output: Path,
    eid: str,
    episode_number: int,
    local_index: int,
    map_index: int,
    split: str,
    seed: int,
    duration_s: int,
    failed_attempts: list[dict[str, Any]],
    route_strategy: str,
    route_attempt: int,
    event_generator: Callable[..., list] | None = None,
) -> tuple[dict[str, Any], DynamicEpisodeConfig, int]:
    ep_dir = output / "episodes" / eid
    network_cache_dir = output / "maps" / f"map_{map_index:03d}" / "network"
    network_cache_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(EPISODE_RETRIES):
        ep_cfg = episode_config(local_index, map_index, seed, duration_s, attempt)
        _safe_remove(ep_dir)
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_start = time.perf_counter()
        try:
            # The route plan remains the validated map-level routing plan.
            # Exogenous retry streams are changed deterministically. SUMO
            # remains authoritative for realized trajectories.
            run_sumo_causal_episode(
                scenario=scenario,
                output_dir=ep_dir,
                config=ep_cfg,
                episode_id=eid,
                split=split,
                events=(
                    event_generator(
                        scenario=scenario,
                        route_plan=route_plan,
                        config=ep_cfg,
                        episode_id=eid,
                    )
                    if event_generator is not None
                    else _generate_events(scenario, ep_cfg, eid)
                ),
                traffic_only=False,
                route_plan=route_plan,
                step_length_s=2.0,
                network_cache_dir=network_cache_dir,
            )
            finalized = _episode_success(
                scenario=scenario,
                ep_dir=ep_dir,
                eid=eid,
                split=split,
                ep_cfg=ep_cfg,
            )
            elapsed = time.perf_counter() - ep_start
            record = {
                "episode_id": eid,
                "split": split,
                "map_index": map_index,
                "scenario_id": scenario.scenario_id,
                "seed": ep_cfg.seed,
                "retry_attempt": attempt,
                "route_strategy": route_strategy,
                "route_strategy_attempt": route_attempt,
                "regime": ep_cfg.regime,
                "event_type": ep_cfg.event_type,
                "event_count": ep_cfg.event_count,
                "duration_s": ep_cfg.duration_s,
                "interval_s": ep_cfg.interval_s,
                "backend": "sumo",
                "wall_clock_s": elapsed,
                "episode_dir": str(ep_dir.relative_to(output)),
                "label_coverage": finalized["label_coverage"],
            }
            return record, ep_cfg, attempt
        except Exception as exc:
            failed_attempts.append(
                _attempt_record(
                    kind="episode_runtime",
                    index=episode_number,
                    attempt=attempt,
                    seed=ep_cfg.seed,
                    error=exc,
                )
            )
            print(
                f"  EP {episode_number:04d} attempt {attempt + 1}/{EPISODE_RETRIES} "
                f"failed ({type(exc).__name__}): {exc}",
                flush=True,
            )
            # Leave diagnostics from the last attempt in a bounded sidecar;
            # the next attempt gets a clean episode directory.
            diag = ep_dir / "retry_failure.txt"
            try:
                diag.write_text(
                    "".join(traceback.format_exception(exc)),
                    encoding="utf-8",
                )
            except Exception:
                pass

    last_failure = failed_attempts[-1] if failed_attempts else {}
    raise RuntimeError(
        f"episode {eid} exhausted {EPISODE_RETRIES} deterministic runtime attempts; "
        f"last_error={last_failure.get('error_type')}: {last_failure.get('error')}"
    )


def generate_corpus(
    output: Path,
    *,
    episodes: int = 2000,
    maps: int = 20,
    episodes_per_map: int | None = None,
    duration_s: int = 2400,
    seed: int = 26137,
    overwrite: bool = False,
) -> dict[str, Any]:
    if episodes < 1000:
        raise ValueError("production corpus requires at least 1000 episodes")
    if maps < 5:
        raise ValueError("production corpus requires at least 5 base maps")
    if episodes_per_map is None:
        if episodes % maps:
            raise ValueError(
                "episodes must be divisible by maps when --episodes-per-map is omitted"
            )
        episodes_per_map = episodes // maps
    if maps * episodes_per_map != episodes:
        raise ValueError(
            f"episodes must equal maps * episodes_per_map; got {episodes} != "
            f"{maps} * {episodes_per_map}"
        )
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    output = output.resolve()
    checkpoint_path = output / "corpus_checkpoint.json"
    manifest_path = output / "corpus_manifest.json"

    if overwrite:
        _safe_remove(output)
    output.mkdir(parents=True, exist_ok=True)

    split_maps = build_split_map_indices(maps)
    map_records: dict[int, dict[str, Any]] = {}
    episode_records: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    started = time.perf_counter()

    # Resume only from a structurally compatible checkpoint.
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint:
        compatible = (
            checkpoint.get("requested_episodes") == episodes
            and checkpoint.get("base_maps") == maps
            and checkpoint.get("episodes_per_map") == episodes_per_map
            and checkpoint.get("duration_s") == duration_s
            and checkpoint.get("seed") == seed
        )
        if compatible:
            for k, v in checkpoint.get("map_records", {}).items():
                map_records[int(k)] = v
            episode_records = list(checkpoint.get("episodes", []))
            failed_attempts = list(checkpoint.get("failed_attempts", []))
            print(
                f"RESUME: {len(episode_records)}/{episodes} accepted episodes already recorded",
                flush=True,
            )
        else:
            print("Checkpoint exists but configuration differs; starting clean.", flush=True)
            _safe_remove(checkpoint_path)

    accepted_ids = {r["episode_id"] for r in episode_records}
    map_fingerprints = {
        r["map_fingerprint"]
        for r in map_records.values()
        if r.get("map_fingerprint")
    }

    for map_index in range(maps):
        split = next(name for name, ids in split_maps.items() if map_index in ids)

        # Reuse a validated map on resume; otherwise generate with recovery.
        if map_index in map_records:
            map_dir = output / "maps" / f"map_{map_index:03d}"
            scenario = _scenario_from_map_dir(map_dir)
            cfg = map_config(map_index, split, seed, int(map_records[map_index].get("generation_attempt", 0)))
            fingerprint = map_records[map_index]["map_fingerprint"]
        else:
            scenario, cfg, fingerprint, map_attempt = _generate_map_with_recovery(
                output=output,
                map_index=map_index,
                split=split,
                seed=seed,
                failed_attempts=failed_attempts,
            )
            if fingerprint in map_fingerprints:
                raise RuntimeError(f"map fingerprint collision detected for map {map_index}")
            map_fingerprints.add(fingerprint)
            map_records[map_index] = {
                "split": split,
                "scenario_id": scenario.scenario_id,
                "scenario_path": str(
                    (output / "maps" / f"map_{map_index:03d}" / "scenario.json").relative_to(output)
                ),
                "map_fingerprint": fingerprint,
                "customers": len(scenario.requests),
                "nodes": len(scenario.nodes),
                "edges": len(scenario.edges),
                "seed": cfg.seed,
                "generation_attempt": map_attempt,
            }
            _save_checkpoint(
                output,
                requested_episodes=episodes,
                maps=maps,
                episodes_per_map=episodes_per_map,
                duration_s=duration_s,
                seed=seed,
                map_records=map_records,
                episode_records=episode_records,
                failed_attempts=failed_attempts,
                started=started,
            )

        print(
            f"MAP {map_index + 1:02d}/{maps}: {split} "
            f"customers={len(scenario.requests)} edges={len(scenario.edges)}",
            flush=True,
        )

        map_dir = output / "maps" / f"map_{map_index:03d}"
        try:
            route_plan, route_strategy, route_attempt = _build_route_plan_with_recovery(
                scenario=scenario,
                map_index=map_index,
                map_dir=map_dir,
                failed_attempts=failed_attempts,
            )
        except RuntimeError as route_exc:
            # A map can be perfectly valid at the scenario level but still be
            # unsuitable for SUMO after physical route resolution. Escalate
            # deterministically to a fresh map rather than retrying the same
            # immutable route plan forever.
            failed_attempts.append({
                "kind": "map_route_exhausted",
                "index": map_index,
                "error_type": type(route_exc).__name__,
                "error": str(route_exc),
            })
            scenario, cfg, fingerprint, map_attempt = _generate_map_with_recovery(
                output=output,
                map_index=map_index,
                split=split,
                seed=stable_seed(seed, 0x4D4150524F555445, map_index, 1),
                failed_attempts=failed_attempts,
            )
            if fingerprint in map_fingerprints:
                raise RuntimeError(f"map fingerprint collision detected after route recovery for map {map_index}")
            map_fingerprints.add(fingerprint)
            map_records[map_index] = {
                "split": split,
                "scenario_id": scenario.scenario_id,
                "scenario_path": str((output / "maps" / f"map_{map_index:03d}" / "scenario.json").relative_to(output)),
                "map_fingerprint": fingerprint,
                "customers": len(scenario.requests),
                "nodes": len(scenario.nodes),
                "edges": len(scenario.edges),
                "seed": cfg.seed,
                "generation_attempt": map_attempt,
                "route_recovery": True,
            }
            map_dir = output / "maps" / f"map_{map_index:03d}"
            route_plan, route_strategy, route_attempt = _build_route_plan_with_recovery(
                scenario=scenario,
                map_index=map_index,
                map_dir=map_dir,
                failed_attempts=failed_attempts,
            )

        for local_index in range(episodes_per_map):
            episode_number = map_index * episodes_per_map + local_index + 1
            eid = f"ep-{episode_number:04d}"
            if eid in accepted_ids:
                continue

            try:
                record, ep_cfg, attempt = _run_episode_with_recovery(
                    scenario=scenario,
                    route_plan=route_plan,
                    output=output,
                    eid=eid,
                    episode_number=episode_number,
                    local_index=local_index,
                    map_index=map_index,
                    split=split,
                    seed=seed,
                    duration_s=duration_s,
                    failed_attempts=failed_attempts,
                    route_strategy=route_strategy,
                    route_attempt=route_attempt,
                )
            except RuntimeError as episode_exc:
                message = str(episode_exc)
                structural = (
                    "Customer stop resolved without physical edges" in message
                    or "no physical route" in message
                    or "Logical vehicle route has customers but no physical route" in message
                    or ("route plan" in message.lower() and "infeasible" in message.lower())
                )
                if not structural:
                    raise

                print(
                    f"  EP {episode_number:04d}: structural route failure; "
                    "escalating from episode retry to deterministic map regeneration",
                    flush=True,
                )
                failed_attempts.append({
                    "kind": "episode_escalated_to_map",
                    "index": episode_number,
                    "error_type": type(episode_exc).__name__,
                    "error": message,
                })

                recovered = False
                last_map_exc: Exception | None = None
                for map_recovery_attempt in range(1, MAP_RETRIES + 1):
                    try:
                        recovery_seed = stable_seed(
                            seed,
                            0x4D41505245434F56,
                            map_index,
                            episode_number,
                            map_recovery_attempt,
                        )
                        scenario, cfg, fingerprint, map_attempt = _generate_map_with_recovery(
                            output=output,
                            map_index=map_index,
                            split=split,
                            seed=recovery_seed,
                            failed_attempts=failed_attempts,
                        )
                        if fingerprint in map_fingerprints:
                            raise RuntimeError(
                                f"map fingerprint collision after recovery attempt {map_recovery_attempt}"
                            )
                        map_fingerprints.add(fingerprint)
                        map_records[map_index] = {
                            "split": split,
                            "scenario_id": scenario.scenario_id,
                            "scenario_path": str((output / "maps" / f"map_{map_index:03d}" / "scenario.json").relative_to(output)),
                            "map_fingerprint": fingerprint,
                            "customers": len(scenario.requests),
                            "nodes": len(scenario.nodes),
                            "edges": len(scenario.edges),
                            "seed": cfg.seed,
                            "generation_attempt": map_attempt,
                            "route_recovery": True,
                            "route_recovery_attempt": map_recovery_attempt,
                        }
                        map_dir = output / "maps" / f"map_{map_index:03d}"
                        route_plan, route_strategy, route_attempt = _build_route_plan_with_recovery(
                            scenario=scenario,
                            map_index=map_index,
                            map_dir=map_dir,
                            failed_attempts=failed_attempts,
                        )
                        recovered = True
                        print(
                            f"  MAP {map_index:03d} recovered with new scenario "
                            f"after route failure (map-recovery {map_recovery_attempt})",
                            flush=True,
                        )
                        break
                    except Exception as exc:
                        last_map_exc = exc
                        failed_attempts.append({
                            "kind": "map_recovery",
                            "index": map_index,
                            "attempt": map_recovery_attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
                        print(
                            f"  MAP {map_index:03d} recovery {map_recovery_attempt}/{MAP_RETRIES} "
                            f"failed: {type(exc).__name__}: {exc}",
                            flush=True,
                        )

                if not recovered:
                    raise RuntimeError(
                        f"episode {eid} triggered structural map failure and "
                        f"all {MAP_RETRIES} map recoveries failed; last_error={last_map_exc}"
                    ) from episode_exc

                record, ep_cfg, attempt = _run_episode_with_recovery(
                    scenario=scenario,
                    route_plan=route_plan,
                    output=output,
                    eid=eid,
                    episode_number=episode_number,
                    local_index=local_index,
                    map_index=map_index,
                    split=split,
                    seed=stable_seed(seed, 0x4550495343414C45, map_index, local_index, episode_number),
                    duration_s=duration_s,
                    failed_attempts=failed_attempts,
                )

            episode_records.append(record)
            accepted_ids.add(eid)

            _save_checkpoint(
                output,
                requested_episodes=episodes,
                maps=maps,
                episodes_per_map=episodes_per_map,
                duration_s=duration_s,
                seed=seed,
                map_records=map_records,
                episode_records=episode_records,
                failed_attempts=failed_attempts,
                started=started,
            )

            if episode_number == 1 or episode_number % 10 == 0 or episode_number == episodes:
                total_elapsed = time.perf_counter() - started
                rate = len(episode_records) / max(total_elapsed, 1e-9)
                eta = (episodes - len(episode_records)) / max(rate, 1e-9)
                print(
                    f"EP {episode_number:04d}/{episodes} map={map_index:03d} "
                    f"regime={ep_cfg.regime} event={ep_cfg.event_type or 'none'} "
                    f"attempt={attempt} wall={record['wall_clock_s']:.1f}s "
                    f"rate={rate:.2f}/s eta={eta / 3600:.2f}h",
                    flush=True,
                )

    if len(episode_records) != episodes:
        raise RuntimeError(
            f"corpus incomplete: accepted {len(episode_records)} / {episodes} episodes"
        )

    counts = {split: 0 for split in split_maps}
    for record in episode_records:
        counts[record["split"]] += 1

    fingerprints = [r["map_fingerprint"] for r in map_records.values()]
    if len(fingerprints) != len(set(fingerprints)):
        raise RuntimeError("base-map fingerprint collision detected")

    manifest = {
        "schema_version": "quanta-real-sumo-corpus-v3",
        "requested_episodes": episodes,
        "generated_episodes": len(episode_records),
        "base_maps": maps,
        "episodes_per_map": episodes_per_map,
        "duration_s": duration_s,
        "interval_s": 60,
        "seed": seed,
        "backend": "sumo",
        "source": (
            "Quanta dynamic scenario generator + dynamic event generator + "
            "RoutingPipeline + SUMO/TraCI"
        ),
        "fixture_route_planner_used": False,
        "fake_observation_generator_used": False,
        "fixture_backend_used": False,
        "split_rule": "map-disjoint train/validation/test split assigned before episode generation",
        "retry_policy": {
            "map_max_attempts": MAP_RETRIES,
            "route_plan_max_strategies": ROUTE_PLAN_RETRIES,
            "episode_max_attempts": EPISODE_RETRIES,
            "retry_changes_seed": True,
            "retry_changes_configuration": True,
            "infinite_retry": False,
        },
        "split_episode_counts": counts,
        "map_records": map_records,
        "episodes": episode_records,
        "failed_attempts": failed_attempts,
        "created_wall_clock_s": time.perf_counter() - started,
    }
    _write_json(manifest_path, manifest)
    _write_json(checkpoint_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--maps", type=int, default=20)
    parser.add_argument("--episodes-per-map", type=int, default=None)
    parser.add_argument("--duration-s", "--duration", dest="duration_s", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=26137)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_corpus(
        args.output,
        episodes=args.episodes,
        maps=args.maps,
        episodes_per_map=args.episodes_per_map,
        duration_s=args.duration_s,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "generated_episodes": manifest["generated_episodes"],
                "base_maps": manifest["base_maps"],
                "split_episode_counts": manifest["split_episode_counts"],
                "failed_attempts": len(manifest["failed_attempts"]),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())