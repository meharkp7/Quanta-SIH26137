"""Generate a fault-tolerant, reproducible Quanta SUMO training corpus on
real New Delhi road geometry instead of synthetic grids.

This is a sibling of `generate_real_training_corpus.py`, not a replacement:
every episode-execution guarantee that script provides (real SUMO/TraCI,
no fixture routes, no fake observations, deterministic seed streams,
bounded retries, resumable checkpointing, map-disjoint splits) is reused
here verbatim by importing its internals. The only thing this script
changes is where a map's road network and demand come from:

    generate_real_training_corpus.py:  DynamicDatasetConfig -> synthetic grid
    generate_delhi_corpus.py:          a real OSM extract   -> generate_osm_scenario

Each zone in `--zones-file` becomes exactly one base map (one
`generate_osm_scenario` call), and `--episodes-per-map` SUMO episodes are
run against it with the same traffic-regime / event / observation-noise
variation `generate_real_training_corpus.py` uses for its synthetic maps.

SHARDING
--------
Per-zone map setup (OSM ingestion, depot/SCC search, route-plan preflight,
SUMO network export via netconvert) is a one-time cost per zone that does
NOT parallelize across the episodes of a single zone (they already share
it via `network_cache_dir`) but DOES fully parallelize across zones, since
different zones' map directories, episode directories, and episode IDs
are provably disjoint (global map index is baked into every path and every
episode number). `_process_zone_assignments` is the engine this file's
`generate_osm_corpus` and `generate_delhi_corpus_sharded.py` both call; it
takes an explicit, pre-computed (global_map_index, zone, split) assignment
list rather than deriving the split internally, which is what makes it
safe to hand a subset of zones to one worker process while another worker
handles a different subset, writing into the same shared output directory
concurrently with zero path collisions.

Zones file format (JSON):
    [
      {"name": "connaught_place", "graphml": "osm_extracts/connaught_place.graphml"},
      {"name": "chandni_chowk",   "graphml": "osm_extracts/chandni_chowk.graphml"},
      ...
    ]

This script does not acquire OSM data and never calls Overpass/Nominatim.
Acquire each zone's .graphml first (see docs/osm_ingestion.md and the
acquisition snippet in this repo's README/chat history), then point
--zones-file at the manifest above.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.scenario import Scenario
from src.data.causal_episodes import _finalize_existing_episode, _write_json
from src.data.causal_audit import audit_episode
from src.data.dynamic_episodes import DynamicEpisodeConfig, DynamicEvent, _generate_events
from src.data.osm_demand_generator import (
    OSMDemandConfig,
    OSMDemandGenerationError,
    generate_osm_scenario,
)
from src.data.osm_ingestion import OSMNetwork, load_graphml
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import RoutePlan
from src.sim.sumo_causal import run_sumo_causal_episode

# Reused verbatim from the synthetic corpus generator: episode execution,
# fault tolerance, and checkpointing do not depend on where the base map
# came from.
from scripts.generate_real_training_corpus import (
    EPISODE_RETRIES,
    MAP_RETRIES,
    ROUTE_PLAN_RETRIES,
    WINDOWS,
    _attempt_record,
    _build_route_plan_with_recovery,
    _load_checkpoint,
    _safe_remove,
    _save_checkpoint,
    _scenario_from_map_dir,
    build_split_map_indices,
    episode_config,
    map_fingerprint,
    stable_seed,
)


@dataclass(frozen=True, slots=True)
class ZoneSpec:
    name: str
    graphml: Path


def load_zone_specs(zones_file: Path) -> tuple[ZoneSpec, ...]:
    data = json.loads(zones_file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{zones_file} must contain a non-empty JSON list of zones")

    zones = []
    seen_names = set()
    for entry in data:
        name = str(entry["name"])
        graphml = Path(entry["graphml"])
        if name in seen_names:
            raise ValueError(f"duplicate zone name in {zones_file}: {name!r}")
        if not graphml.exists():
            raise FileNotFoundError(
                f"zone {name!r} references {graphml}, which does not exist. "
                "Acquire it first (see docs/osm_ingestion.md) -- this script "
                "never downloads OSM data itself."
            )
        seen_names.add(name)
        zones.append(ZoneSpec(name=name, graphml=graphml))

    return tuple(zones)


def osm_map_config(
    zone_index: int,
    attempt: int,
    seed: int,
) -> OSMDemandConfig:
    """Deterministic per-zone demand policy, widening its safety margin and
    shrinking customer_count on retry rather than retrying the identical
    (and therefore identically infeasible/unreachable) configuration.

    customer_count intentionally spans the same 40-80 range
    `generate_real_training_corpus.py`'s synthetic `map_config` uses, so a
    model trained on both corpora sees comparable problem sizes and the
    two are fairly comparable in ablations.
    """

    customer_shrink = attempt * 5
    customer_count = max(15, 40 + (zone_index % 5) * 10 - customer_shrink)

    vehicle_capacity = 30.0 + (zone_index % 3) * 10.0  # 30 / 40 / 50
    demand_min, demand_max = 1.0, 8.0

    # Real total demand for this exact seed can only be known after
    # drawing it, but its expectation is customer_count * mean(demand).
    # Bin-packing (first-fit-decreasing, see osm_demand_generator.py)
    # needs headroom above that expectation, and more of it on retry.
    expected_demand = customer_count * (demand_min + demand_max) / 2.0
    safety_margin = 1.35 + attempt * 0.2
    fleet_size = max(
        2, math.ceil(expected_demand * safety_margin / vehicle_capacity)
    )

    # Rotate through the same window tightness levels as the synthetic
    # generator for comparability; unlike that generator this maps
    # directly to slack seconds rather than a named profile, since
    # osm_demand_generator.py takes time_window_slack_s directly.
    slack_by_profile = {"loose": 1800.0, "medium": 900.0, "tight": 450.0}
    profile = WINDOWS[zone_index % len(WINDOWS)]
    time_window_slack_s = slack_by_profile[profile]

    map_seed = stable_seed(seed, 0x4F534D5A4F4E45, zone_index, attempt)  # "OSMZONE"

    return OSMDemandConfig(
        customer_count=customer_count,
        fleet_size=fleet_size,
        vehicle_capacity=vehicle_capacity,
        demand_min=demand_min,
        demand_max=demand_max,
        service_duration_s=60.0,
        time_window_slack_s=time_window_slack_s,
        seed=map_seed,
    )


def _generate_osm_map_with_recovery(
    *,
    output: Path,
    map_index: int,
    zone: ZoneSpec,
    split: str,
    seed: int,
    failed_attempts: list[dict[str, Any]],
) -> tuple[Scenario, OSMDemandConfig, str, int]:
    map_dir = output / "maps" / f"map_{map_index:03d}"

    network: OSMNetwork = load_graphml(str(zone.graphml))

    for attempt in range(MAP_RETRIES):
        cfg = osm_map_config(map_index, attempt, seed)
        tmp_dir = output / "maps" / f".map_{map_index:03d}.attempt_{attempt:02d}"
        _safe_remove(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = generate_osm_scenario(
                network,
                cfg,
                scenario_id=f"{zone.name}-map{map_index:03d}",
                source_name=f"osm:{zone.name}",
                dataset_split=split,
            )
            scenario = result.scenario
            fingerprint = map_fingerprint(scenario)

            if len(scenario.requests) != cfg.customer_count:
                raise RuntimeError(
                    f"customer-count contract failed: expected {cfg.customer_count}, "
                    f"got {len(scenario.requests)}"
                )
            if not scenario.nodes or not scenario.edges:
                raise RuntimeError("generated scenario has empty road graph")

            _write_json(tmp_dir / "scenario.json", scenario.model_dump(mode="json"))
            _write_json(
                tmp_dir / "osm_zone.json",
                {
                    "zone": zone.name,
                    "graphml": str(zone.graphml),
                    "depot_node_id": result.depot_node_id,
                    "customer_node_ids": list(result.customer_node_ids),
                    "unreachable_candidates_skipped": (
                        result.unreachable_candidates_skipped
                    ),
                },
            )

            _safe_remove(map_dir)
            tmp_dir.rename(map_dir)
            return _scenario_from_map_dir(map_dir), cfg, fingerprint, attempt
        except (OSMDemandGenerationError, RuntimeError) as exc:
            failed_attempts.append(
                _attempt_record(
                    kind="osm_map_generation",
                    index=map_index,
                    attempt=attempt,
                    seed=cfg.seed,
                    error=exc,
                )
            )
            print(
                f"MAP {map_index:03d} ({zone.name}) generation attempt "
                f"{attempt + 1}/{MAP_RETRIES} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            _safe_remove(tmp_dir)

    raise RuntimeError(
        f"map {map_index} (zone={zone.name!r}) exhausted {MAP_RETRIES} deterministic "
        "generation attempts"
    )



def _event_type_for_index(event_type: str, index: int) -> str:
    if event_type == "multi_disruption":
        return "incident" if index % 2 == 0 else "closure"
    return event_type


def _generate_decision_relevant_events(
    *,
    scenario: Scenario,
    ep_cfg: DynamicEpisodeConfig,
    episode_id: str,
    baseline_dir: Path,
) -> list[DynamicEvent]:
    """Generate disruptions from *observed baseline traffic*, not geometry alone.

    The baseline SUMO rollout tells us which controlled vehicles actually visit
    which parent roads and when.  Events are then placed around those real
    traversals, making the resulting episode a genuine decision opportunity.
    """
    if ep_cfg.event_type is None or ep_cfg.event_count == 0:
        return []

    trajectory_path = baseline_dir / "trajectories.csv"
    if not trajectory_path.exists():
        raise RuntimeError("baseline decision-event generation: trajectories.csv missing")

    controlled_ids = {str(v.vehicle_id) for v in scenario.fleet}
    edge_by_id = {str(e.edge_id): e for e in scenario.edges}
    candidates: list[tuple[float, str, str]] = []
    with trajectory_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            vid = str(row.get("trip_id", ""))
            edge_id = str(row.get("edge_id", ""))
            if vid not in controlled_ids or edge_id not in edge_by_id:
                continue
            try:
                entry = float(row["entry_time_s"])
                exit_time = float(row["exit_time_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if exit_time <= ep_cfg.warmup_s or entry >= ep_cfg.duration_s:
                continue
            parent = str(edge_by_id[edge_id].parent_road_id)
            candidates.append((max(entry, float(ep_cfg.warmup_s)), vid, parent))

    if not candidates:
        raise RuntimeError("baseline decision-event generation: no controlled traffic candidates")

    # Generate more candidates than the requested event count for
    # multi-disruption episodes. Some baseline-relevant roads can still fail
    # the independent reroute-feasibility check, so they are candidates to
    # discard rather than reasons to contaminate an otherwise useful episode.
    rng = random.Random(ep_cfg.seed + 911)
    rng.shuffle(candidates)
    target_count = int(ep_cfg.event_count)
    candidate_count = (
        min(len(candidates), max(target_count * 3, target_count + 2))
        if target_count > 1
        else target_count
    )

    selected: list[tuple[float, str, str]] = []
    used_parents: set[str] = set()
    for entry, vid, parent in candidates:
        if parent in used_parents:
            continue
        selected.append((entry, vid, parent))
        used_parents.add(parent)
        if len(selected) >= candidate_count:
            break
    if len(selected) < candidate_count:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= candidate_count:
                break

    events: list[DynamicEvent] = []
    for i, (entry, _vid, parent) in enumerate(selected[:candidate_count]):
        # Start shortly before/at the observed traversal, while preserving a
        # useful reveal lead and avoiding the warm-up period.
        latest_start = max(
            ep_cfg.warmup_s + 60,
            ep_cfg.duration_s - ep_cfg.event_duration_s - 60,
        )
        start = int(min(max(entry - min(60.0, ep_cfg.reveal_lead_s), ep_cfg.warmup_s + 60), latest_start))
        end = min(ep_cfg.duration_s, start + ep_cfg.event_duration_s)
        events.append(DynamicEvent(
            event_id=f"{episode_id}-EV{i:03d}",
            event_type=_event_type_for_index(ep_cfg.event_type, i),
            generation_time_s=0,
            reveal_time_s=max(0, start - ep_cfg.reveal_lead_s),
            effect_start_s=start,
            effect_end_s=end,
            affected_parent_road_ids=(parent,),
            severity=rng.uniform(0.65, 1.0),
        ))
    return sorted(events, key=lambda e: (e.effect_start_s, e.event_id))

def _decision_relevance_audit(
    *,
    scenario: Scenario,
    route_plan: Any,
    ep_dir: Path,
    events: list[DynamicEvent],
    baseline_dir: Path | None = None,
) -> dict[str, Any]:
    """Require declared disruptions to affect controlled traffic in SUMO.

    A route/geometry intersection alone is insufficient: the controlled
    vehicle must actually traverse an affected edge during the event window.
    We also require a feasible route when those affected edges are unavailable,
    providing a conservative check that rerouting is possible.
    """
    if not events:
        return {
            "required": False,
            "accepted": True,
            "event_count": 0,
            "accepted_event_count": 0,
            "events": [],
            "affected_vehicle_count": 0,
            "affected_vehicle_ids": [],
            "affected_job_count": 0,
            "baseline_traversal_confirmed": False,
            "alternative_route_available": False,
        }

    # Relevance is a counterfactual property of the baseline: for a closure,
    # the causal rollout may correctly divert the vehicle and therefore no
    # longer contain a traversal of the closed edge.  We must not use that
    # post-intervention trajectory to decide whether the event was relevant.
    trajectory_path = (baseline_dir / "trajectories.csv") if baseline_dir else (ep_dir / "trajectories.csv")
    if not trajectory_path.exists():
        raise RuntimeError("decision relevance gate: baseline trajectories.csv missing")
    with trajectory_path.open(encoding="utf-8", newline="") as fh:
        trajectories = list(csv.DictReader(fh))

    edge_by_id = {str(edge.edge_id): edge for edge in scenario.edges}
    controlled_ids = {str(vehicle.vehicle_id) for vehicle in scenario.fleet}
    routes_by_vehicle = {str(route.vehicle_id): route for route in route_plan.vehicle_routes}
    reports: list[dict[str, Any]] = []
    all_affected_vehicles: set[str] = set()
    all_affected_jobs: set[tuple[str, str]] = set()

    for event in events:
        parents = {str(x) for x in event.affected_parent_road_ids}
        affected_edges = {
            edge_id for edge_id, edge in edge_by_id.items()
            if str(edge.parent_road_id) in parents
        }
        impacted: set[str] = set()
        overlap_rows = 0
        for row in trajectories:
            vid = str(row.get("trip_id", ""))
            if vid not in controlled_ids or str(row.get("edge_id", "")) not in affected_edges:
                continue
            try:
                entry = float(row["entry_time_s"])
                exit_time = float(row["exit_time_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if entry < float(event.effect_end_s) and exit_time > float(event.effect_start_s):
                impacted.add(vid)
                overlap_rows += 1

        alternative = False
        alternative_error = ""
        if impacted:
            # Check the affected controlled vehicles independently. Requiring
            # the entire fleet to remain feasible would incorrectly reject an
            # otherwise useful event merely because an unrelated vehicle also
            # happens to use the closed road.
            errors = []
            for vehicle_id in sorted(impacted):
                route = routes_by_vehicle.get(vehicle_id)
                if route is None:
                    errors.append(f"missing logical route for {vehicle_id}")
                    continue
                try:
                    single_vehicle_plan = RoutePlan.from_routes((route,))
                    evaluation = RouteEvaluator(
                        scenario,
                        closed_edge_ids=affected_edges,
                    ).evaluate(single_vehicle_plan, planning_time_s=0.0)
                    if evaluation.feasible:
                        alternative = True
                        break
                    errors.append(
                        f"{vehicle_id}: "
                        + ("; ".join(str(x) for x in getattr(evaluation, "errors", ()))
                           or "route evaluator rejected affected-edge closure")
                    )
                except Exception as exc:
                    errors.append(f"{vehicle_id}: {type(exc).__name__}: {exc}")
            alternative_error = "; ".join(errors)

        jobs = set()
        for vid in impacted:
            route = routes_by_vehicle.get(vid)
            for customer_id in getattr(route, "customer_ids", ()):
                jobs.add((vid, str(customer_id)))
        all_affected_vehicles.update(impacted)
        all_affected_jobs.update(jobs)

        accepted = bool(impacted and jobs and alternative)
        reasons = []
        if not impacted:
            reasons.append("no controlled vehicle traversed the affected edge during the effect window")
        if not jobs:
            reasons.append("no jobs are assigned to the affected controlled vehicle")
        if not alternative:
            reasons.append(
                "no feasible reroute with affected edges unavailable"
                + (f": {alternative_error}" if alternative_error else "")
            )
        reports.append({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "affected_parent_road_ids": sorted(parents),
            "affected_edge_count": len(affected_edges),
            "temporal_overlap_rows": overlap_rows,
            "affected_vehicle_ids": sorted(impacted),
            "affected_vehicle_count": len(impacted),
            "affected_job_count": len(jobs),
            "baseline_traversal_confirmed": bool(impacted),
            "alternative_route_available": alternative,
            "accepted": accepted,
            "rejection_reason": "; ".join(reasons),
        })

    return {
        "required": True,
        "accepted": all(r["accepted"] for r in reports),
        "event_count": len(events),
        "accepted_event_count": sum(r["accepted"] for r in reports),
        "events": reports,
        "affected_vehicle_count": len(all_affected_vehicles),
        "affected_vehicle_ids": sorted(all_affected_vehicles),
        "affected_job_count": len(all_affected_jobs),
        "baseline_traversal_confirmed": bool(all_affected_vehicles),
        "alternative_route_available": all(r["alternative_route_available"] for r in reports),
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
) -> tuple[dict[str, Any], DynamicEpisodeConfig, int]:
    """Delhi-specific episode runner with the decision-relevance gate."""
    ep_dir = output / "episodes" / eid
    network_cache_dir = output / "maps" / f"map_{map_index:03d}" / "network"
    network_cache_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(EPISODE_RETRIES):
        ep_cfg = episode_config(local_index, map_index, seed, duration_s, attempt)
        _safe_remove(ep_dir)
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_start = time.perf_counter()
        try:
            if ep_cfg.event_type is None or ep_cfg.event_count == 0:
                events = []
                run_sumo_causal_episode(
                    scenario=scenario,
                    output_dir=ep_dir,
                    config=ep_cfg,
                    episode_id=eid,
                    split=split,
                    events=events,
                    traffic_only=False,
                    route_plan=route_plan,
                    step_length_s=2.0,
                    network_cache_dir=network_cache_dir,
                )
            else:
                # First obtain actual baseline traffic.  Event roads/times are
                # sampled from this rollout, so the disruption is temporally
                # coupled to controlled traffic rather than merely to OSM geometry.
                baseline_dir = ep_dir / ".baseline"
                _safe_remove(baseline_dir)
                baseline_dir.mkdir(parents=True, exist_ok=True)
                baseline_cfg = replace(ep_cfg, event_type=None, event_count=0)
                run_sumo_causal_episode(
                    scenario=scenario,
                    output_dir=baseline_dir,
                    config=baseline_cfg,
                    episode_id=f"{eid}-baseline",
                    split=split,
                    events=[],
                    traffic_only=False,
                    route_plan=route_plan,
                    step_length_s=2.0,
                    network_cache_dir=network_cache_dir,
                )
                candidate_events = _generate_decision_relevant_events(
                    scenario=scenario,
                    ep_cfg=ep_cfg,
                    episode_id=eid,
                    baseline_dir=baseline_dir,
                )

                # Preflight every candidate against the baseline before
                # spending another SUMO rollout on it. For multi-disruption
                # episodes we deliberately over-generate candidates and keep
                # only those that are genuinely decision-relevant *and*
                # independently reroutable.
                candidate_relevance = _decision_relevance_audit(
                    scenario=scenario,
                    route_plan=route_plan,
                    ep_dir=ep_dir,
                    events=candidate_events,
                    baseline_dir=baseline_dir,
                )
                accepted_candidates = [
                    event
                    for event, report in zip(
                        candidate_events, candidate_relevance["events"]
                    )
                    if report["accepted"]
                ]
                required_event_count = int(ep_cfg.event_count)
                if len(accepted_candidates) < required_event_count:
                    raise RuntimeError(
                        "decision event candidate gate rejected episode: "
                        f"found {len(accepted_candidates)} feasible relevant events, "
                        f"required {required_event_count}; "
                        + json.dumps(candidate_relevance, sort_keys=True)
                    )

                # Preserve the configured event count in the final episode:
                # rejected candidate disruptions are dropped, never weakened
                # or forced into the causal simulation.
                events = accepted_candidates[:required_event_count]
                run_sumo_causal_episode(
                    scenario=scenario,
                    output_dir=ep_dir,
                    config=ep_cfg,
                    episode_id=eid,
                    split=split,
                    events=events,
                    traffic_only=False,
                    route_plan=route_plan,
                    step_length_s=2.0,
                    network_cache_dir=network_cache_dir,
                )
            relevance = _decision_relevance_audit(
                scenario=scenario,
                route_plan=route_plan,
                ep_dir=ep_dir,
                events=events,
                baseline_dir=(baseline_dir if ep_cfg.event_type is not None and ep_cfg.event_count > 0 else None),
            )
            if ep_cfg.event_type is not None and ep_cfg.event_count > 0:
                _safe_remove(baseline_dir)
            if not relevance["accepted"]:
                raise RuntimeError(
                    "decision relevance gate rejected episode: "
                    + json.dumps(relevance, sort_keys=True)
                )

            finalized = _finalize_existing_episode(scenario, ep_dir, eid, split, ep_cfg)
            audit = audit_episode(ep_dir)
            if not audit["ok"]:
                raise RuntimeError(f"causal audit failed for {eid}: {audit}")
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
                "label_coverage": finalized.label_coverage,
                "decision_relevance": relevance,
                "causal_audit": audit,
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
            try:
                (ep_dir / "retry_failure.txt").write_text(
                    "".join(traceback.format_exception(exc)), encoding="utf-8"
                )
            except Exception:
                pass

    last_failure = failed_attempts[-1] if failed_attempts else {}
    raise RuntimeError(
        f"episode {eid} exhausted {EPISODE_RETRIES} deterministic runtime attempts; "
        f"last_error={last_failure.get('error_type')}: {last_failure.get('error')}"
    )

def _process_zone_assignments(
    output: Path,
    *,
    zone_assignments: list[tuple[int, "ZoneSpec", str]],
    episodes_per_map: int,
    duration_s: int,
    seed: int,
    checkpoint_path: Path,
    manifest_path: Path,
    total_maps_for_manifest: int,
    total_episodes_for_manifest: int,
    progress_label: str = "",
) -> dict[str, Any]:
    """Run map-generation + route-planning + episode execution for exactly
    the given (global_map_index, zone, split) assignments, checkpointing to
    `checkpoint_path` and writing a final manifest to `manifest_path`.

    This is the reusable engine behind both the sequential
    `generate_osm_corpus` (called with ALL zones, the default checkpoint/
    manifest paths) and `generate_delhi_corpus_sharded.py` (called once per
    shard, each with its own disjoint zone subset and its own checkpoint/
    manifest path under the same shared output root). It deliberately does
    NOT compute train/validation/test splits itself -- the caller supplies
    them already resolved per zone, so a shard covering only zones 5-9 of
    30 total zones still gets the SAME split each zone would have gotten
    in a full sequential run, rather than an incorrect split computed from
    a 5-zone-only view.

    `total_maps_for_manifest` / `total_episodes_for_manifest` are the
    GLOBAL corpus totals (not this call's local zone/episode count) purely
    for manifest metadata continuity; they do not affect what gets
    generated.
    """

    episodes_here = len(zone_assignments) * episodes_per_map

    map_records: dict[int, dict[str, Any]] = {}
    episode_records: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    started = time.perf_counter()

    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint:
        compatible = (
            checkpoint.get("requested_episodes_here") == episodes_here
            and checkpoint.get("episodes_per_map") == episodes_per_map
            and checkpoint.get("duration_s") == duration_s
            and checkpoint.get("seed") == seed
            and checkpoint.get("zone_map_indices")
            == [a[0] for a in zone_assignments]
        )
        if compatible:
            for k, v in checkpoint.get("map_records", {}).items():
                map_records[int(k)] = v
            episode_records = list(checkpoint.get("episodes", []))
            failed_attempts = list(checkpoint.get("failed_attempts", []))
            print(
                f"{progress_label}RESUME: {len(episode_records)}/{episodes_here} "
                "accepted episodes already recorded",
                flush=True,
            )
        else:
            print(
                f"{progress_label}Checkpoint exists but configuration differs; "
                "starting clean.",
                flush=True,
            )
            _safe_remove(checkpoint_path)

    accepted_ids = {r["episode_id"] for r in episode_records}
    map_fingerprints = {
        r["map_fingerprint"] for r in map_records.values() if r.get("map_fingerprint")
    }

    def _save_local_checkpoint() -> None:
        counts = {"train": 0, "validation": 0, "test": 0}
        for record in episode_records:
            counts[record["split"]] = counts.get(record["split"], 0) + 1
        checkpoint_payload = {
            "schema_version": "quanta-real-sumo-osm-corpus-shard-v1",
            "requested_episodes_here": episodes_here,
            "generated_episodes_here": len(episode_records),
            "episodes_per_map": episodes_per_map,
            "duration_s": duration_s,
            "seed": seed,
            "zone_map_indices": [a[0] for a in zone_assignments],
            "split_episode_counts": counts,
            "map_records": map_records,
            "episodes": episode_records,
            "failed_attempts": failed_attempts,
            "updated_wall_clock_s": time.perf_counter() - started,
        }
        _write_json(checkpoint_path, checkpoint_payload)

    for map_index, zone, split in zone_assignments:
        if map_index in map_records:
            map_dir = output / "maps" / f"map_{map_index:03d}"
            scenario = _scenario_from_map_dir(map_dir)
            fingerprint = map_records[map_index]["map_fingerprint"]
        else:
            scenario, cfg, fingerprint, map_attempt = _generate_osm_map_with_recovery(
                output=output,
                map_index=map_index,
                zone=zone,
                split=split,
                seed=seed,
                failed_attempts=failed_attempts,
            )
            if fingerprint in map_fingerprints:
                raise RuntimeError(f"map fingerprint collision detected for map {map_index}")
            map_fingerprints.add(fingerprint)
            map_records[map_index] = {
                "split": split,
                "zone": zone.name,
                "scenario_id": scenario.scenario_id,
                "scenario_path": str(
                    (output / "maps" / f"map_{map_index:03d}" / "scenario.json").relative_to(
                        output
                    )
                ),
                "map_fingerprint": fingerprint,
                "customers": len(scenario.requests),
                "nodes": len(scenario.nodes),
                "edges": len(scenario.edges),
                "seed": cfg.seed,
                "generation_attempt": map_attempt,
            }
            _save_local_checkpoint()

        print(
            f"{progress_label}MAP {map_index:03d} ({zone.name}): {split} "
            f"customers={len(scenario.requests)} edges={len(scenario.edges)}",
            flush=True,
        )

        map_dir = output / "maps" / f"map_{map_index:03d}"

        def _regenerate_map(recovery_seed: int) -> tuple[Scenario, str]:
            new_scenario, new_cfg, new_fingerprint, new_attempt = (
                _generate_osm_map_with_recovery(
                    output=output,
                    map_index=map_index,
                    zone=zone,
                    split=split,
                    seed=recovery_seed,
                    failed_attempts=failed_attempts,
                )
            )
            # A recovery regenerates DEMAND on the same OSM topology. Therefore
            # the physical map fingerprint is expected to remain identical to
            # the current map's fingerprint. It is only a real collision when
            # the regenerated fingerprint belongs to a DIFFERENT base map.
            existing_fingerprint = map_records.get(map_index, {}).get("map_fingerprint")
            if (
                new_fingerprint in map_fingerprints
                and new_fingerprint != existing_fingerprint
            ):
                raise RuntimeError(
                    f"map fingerprint collision after recovery for map {map_index}: "
                    f"fingerprint already belongs to another base map"
                )
            map_fingerprints.add(new_fingerprint)
            map_records[map_index] = {
                "split": split,
                "zone": zone.name,
                "scenario_id": new_scenario.scenario_id,
                "scenario_path": str(
                    (output / "maps" / f"map_{map_index:03d}" / "scenario.json").relative_to(
                        output
                    )
                ),
                "map_fingerprint": new_fingerprint,
                "customers": len(new_scenario.requests),
                "nodes": len(new_scenario.nodes),
                "edges": len(new_scenario.edges),
                "seed": new_cfg.seed,
                "generation_attempt": new_attempt,
                "route_recovery": True,
            }
            return new_scenario, new_fingerprint

        try:
            route_plan, route_strategy, route_attempt = _build_route_plan_with_recovery(
                scenario=scenario,
                map_index=map_index,
                map_dir=map_dir,
                failed_attempts=failed_attempts,
            )
        except RuntimeError as route_exc:
            failed_attempts.append(
                {
                    "kind": "map_route_exhausted",
                    "index": map_index,
                    "error_type": type(route_exc).__name__,
                    "error": str(route_exc),
                }
            )
            scenario, _fingerprint = _regenerate_map(
                stable_seed(seed, 0x4D4150524F555445, map_index, 1)
            )
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
                    f"{progress_label}  EP {episode_number:04d}: structural route failure; "
                    "escalating to deterministic OSM map regeneration",
                    flush=True,
                )
                failed_attempts.append(
                    {
                        "kind": "episode_escalated_to_map",
                        "index": episode_number,
                        "error_type": type(episode_exc).__name__,
                        "error": message,
                    }
                )

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
                        scenario, _fingerprint = _regenerate_map(recovery_seed)
                        map_records[map_index]["route_recovery_attempt"] = (
                            map_recovery_attempt
                        )
                        map_dir = output / "maps" / f"map_{map_index:03d}"
                        route_plan, route_strategy, route_attempt = (
                            _build_route_plan_with_recovery(
                                scenario=scenario,
                                map_index=map_index,
                                map_dir=map_dir,
                                failed_attempts=failed_attempts,
                            )
                        )
                        recovered = True
                        print(
                            f"{progress_label}  MAP {map_index:03d} recovered with new demand "
                            f"after route failure (map-recovery {map_recovery_attempt})",
                            flush=True,
                        )
                        break
                    except Exception as exc:
                        last_map_exc = exc
                        failed_attempts.append(
                            {
                                "kind": "map_recovery",
                                "index": map_index,
                                "attempt": map_recovery_attempt,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        print(
                            f"{progress_label}  MAP {map_index:03d} recovery {map_recovery_attempt}/"
                            f"{MAP_RETRIES} failed: {type(exc).__name__}: {exc}",
                            flush=True,
                        )

                if not recovered:
                    raise RuntimeError(
                        f"episode {eid} triggered structural map failure and all "
                        f"{MAP_RETRIES} map recoveries failed; last_error={last_map_exc}"
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
                    seed=stable_seed(
                        seed, 0x4550495343414C45, map_index, local_index, episode_number
                    ),
                    duration_s=duration_s,
                    failed_attempts=failed_attempts,
                    route_strategy=route_strategy,
                    route_attempt=route_attempt,
                )

            episode_records.append(record)
            accepted_ids.add(eid)
            _save_local_checkpoint()

            total_elapsed = time.perf_counter() - started
            rate = len(episode_records) / max(total_elapsed, 1e-9)
            eta = (episodes_here - len(episode_records)) / max(rate, 1e-9)
            print(
                f"{progress_label}EP {episode_number:04d} (local "
                f"{len(episode_records)}/{episodes_here}) map={map_index:03d} "
                f"({zone.name}) regime={ep_cfg.regime} "
                f"event={ep_cfg.event_type or 'none'} attempt={attempt} "
                f"wall={record['wall_clock_s']:.1f}s rate={rate:.2f}/s "
                f"eta={eta / 3600:.2f}h",
                flush=True,
            )

    if len(episode_records) != episodes_here:
        raise RuntimeError(
            f"{progress_label}zone-assignment batch incomplete: accepted "
            f"{len(episode_records)} / {episodes_here} episodes"
        )

    counts = {"train": 0, "validation": 0, "test": 0}
    for record in episode_records:
        counts[record["split"]] += 1

    manifest = {
        "schema_version": "quanta-real-sumo-osm-corpus-v1",
        "requested_episodes": total_episodes_for_manifest,
        "generated_episodes": len(episode_records),
        "base_maps": total_maps_for_manifest,
        "episodes_per_map": episodes_per_map,
        "duration_s": duration_s,
        "interval_s": 60,
        "seed": seed,
        "backend": "sumo",
        "source": (
            "Real OpenStreetMap road geometry per zone + osm_demand_generator "
            "+ dynamic event generator + RoutingPipeline + SUMO/TraCI"
        ),
        "zones": [
            {"index": idx, "name": z.name, "graphml": str(z.graphml)}
            for idx, z, _split in zone_assignments
        ],
        "fixture_route_planner_used": False,
        "fake_observation_generator_used": False,
        "fixture_backend_used": False,
        "synthetic_road_network_used": False,
        "split_rule": (
            "map-disjoint (zone-disjoint) train/validation/test split assigned "
            "before episode generation"
        ),
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
    return manifest


def generate_osm_corpus(
    output: Path,
    *,
    zones: tuple[ZoneSpec, ...],
    episodes_per_map: int,
    duration_s: int = 2400,
    seed: int = 26137,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Sequential entrypoint: process ALL zones in a single call.

    Unchanged in behavior and signature from the original single-process
    version -- this now delegates its per-zone loop to
    `_process_zone_assignments`, which is also what
    `generate_delhi_corpus_sharded.py` calls per shard, but the output
    (manifest/checkpoint at the corpus root, full-corpus semantics) is
    identical to before.
    """

    maps = len(zones)
    if maps < 5:
        raise ValueError("production corpus requires at least 5 zones")
    if episodes_per_map < 1:
        raise ValueError("episodes_per_map must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    episodes = maps * episodes_per_map

    output = output.resolve()
    checkpoint_path = output / "corpus_checkpoint.json"
    manifest_path = output / "corpus_manifest.json"

    if overwrite:
        _safe_remove(output)
    output.mkdir(parents=True, exist_ok=True)

    split_maps = build_split_map_indices(maps)
    split_by_index: dict[int, str] = {}
    for split_name, indices in split_maps.items():
        for idx in indices:
            split_by_index[idx] = split_name

    zone_assignments = [
        (index, zone, split_by_index[index]) for index, zone in enumerate(zones)
    ]

    manifest = _process_zone_assignments(
        output,
        zone_assignments=zone_assignments,
        episodes_per_map=episodes_per_map,
        duration_s=duration_s,
        seed=seed,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        total_maps_for_manifest=maps,
        total_episodes_for_manifest=episodes,
    )
    # Keep the legacy checkpoint filename in sync with the manifest, as
    # before (some tooling/resume logic may inspect either file).
    _write_json(checkpoint_path, manifest)

    fingerprints = [r["map_fingerprint"] for r in manifest["map_records"].values()]
    if len(fingerprints) != len(set(fingerprints)):
        raise RuntimeError("base-map fingerprint collision detected")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--zones-file", type=Path, required=True)
    parser.add_argument("--episodes-per-map", type=int, default=50)
    parser.add_argument("--duration-s", "--duration", dest="duration_s", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=26137)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zones = load_zone_specs(args.zones_file)

    manifest = generate_osm_corpus(
        args.output,
        zones=zones,
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
