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
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.scenario import Scenario
from src.data.causal_episodes import _write_json
from src.data.dynamic_episodes import _generate_events
from src.data.osm_demand_generator import (
    OSMDemandConfig,
    OSMDemandGenerationError,
    generate_osm_scenario,
)
from src.data.osm_ingestion import OSMNetwork, load_graphml

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
    _run_episode_with_recovery,
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
            if new_fingerprint in map_fingerprints:
                raise RuntimeError(
                    f"map fingerprint collision after recovery for map {map_index}"
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