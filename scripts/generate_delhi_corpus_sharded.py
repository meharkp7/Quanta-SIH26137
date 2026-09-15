"""Sharded driver for `generate_delhi_corpus.py`.

Why sharding helps here specifically
-------------------------------------
Profiling a real smoke run (5 zones, episodes_per_map=2) showed the
bottleneck is NOT per-episode SUMO execution (~7.6s/episode, already
amortized across episodes_per_map via network_cache_dir) — it's one-time
per-zone map setup (OSM ingestion, depot/SCC search, route-plan preflight,
SUMO network export via netconvert), at roughly 100+ seconds/zone. That
cost is paid once per zone regardless of episodes_per_map, and at the
production target (30 zones x 50 episodes/map = 1500 episodes,
duration_s=2400) it does NOT amortize away on its own, because there are
still 30 of these one-time costs to pay sequentially.

Different zones are fully independent: distinct map directories, distinct
episode-ID ranges (global map_index is baked into every episode number),
distinct route plans. There is no shared mutable state between zones
except the corpus-level checkpoint/manifest file, which this script
avoids by giving each shard its own checkpoint/manifest under
`<output>/.shard_state/` and merging only after every shard completes.
That means shards can safely write into the SAME output directory
concurrently — no separate per-shard staging directory or merge-by-copy
step is needed, only a merge of the small JSON manifests.

This turns the 30-zone one-time setup cost from fully sequential into
parallel-across-N-workers, roughly dividing total wall clock by
min(num_shards, cpu_count) for the setup-dominated portion of the run.

Usage
-----
    python scripts/generate_delhi_corpus_sharded.py corpus_delhi \\
        --zones-file delhi_zones.json \\
        --episodes-per-map 50 \\
        --duration-s 2400 \\
        --shards 8

Resumability
------------
Safe to Ctrl-C and re-run with identical arguments. Each shard's own
`_process_zone_assignments` checkpoint (reused unchanged from
generate_delhi_corpus.py) resumes exactly where it left off, and shard
assignment (which zones go to which shard) is a deterministic function of
zone order and --shards, so re-running reconstructs the same partition.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.causal_episodes import _write_json

from scripts.generate_delhi_corpus import ZoneSpec, load_zone_specs
from scripts.generate_real_training_corpus import (
    EPISODE_RETRIES,
    MAP_RETRIES,
    ROUTE_PLAN_RETRIES,
    _safe_remove,
    build_split_map_indices,
)


def partition_zones_round_robin(
    num_zones: int,
    num_shards: int,
) -> list[list[int]]:
    """Round-robin (not contiguous) partition of global zone indices.

    Round-robin spreads zones of any given "kind" (e.g. if zones were
    listed in a naturally clustered order) evenly across shards, which
    gives better load balance than contiguous blocks when per-zone cost
    is not perfectly uniform (bigger zones = more nodes/edges = slower
    ingestion and netconvert) and its distribution across the input list
    is unknown ahead of time.
    """

    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")

    shards: list[list[int]] = [[] for _ in range(num_shards)]
    for global_index in range(num_zones):
        shards[global_index % num_shards].append(global_index)

    return shards


def _run_shard(
    output_str: str,
    shard_id: int,
    zone_assignments_plain: list[tuple[int, str, str, str]],
    episodes_per_map: int,
    duration_s: int,
    seed: int,
    total_maps: int,
    total_episodes: int,
) -> dict[str, Any]:
    """Picklable worker entrypoint.

    Takes only plain, JSON-safe data (strings, ints, tuples of strings) —
    never a live Scenario/OSMNetwork/ZoneSpec object — and reconstructs
    everything it needs locally inside the worker process. This mirrors
    the pattern already established for this project's other
    process-pool workers (see src/optim/parallel_oracle.py): some of this
    codebase's objects (pydantic Scenario models, objects holding
    threading locks) are not picklable, so the safe and portable approach
    is to cross the process boundary with plain data only and rebuild
    live objects on the other side.
    """

    # Imported inside the worker so a process that never runs a shard
    # (e.g. --dry-run in the orchestrator) never pays this import cost.
    from scripts.generate_delhi_corpus import (
        ZoneSpec as _ZoneSpec,
        _process_zone_assignments,
    )

    output = Path(output_str)

    zone_assignments = [
        (map_index, _ZoneSpec(name=name, graphml=Path(graphml)), split)
        for map_index, name, graphml, split in zone_assignments_plain
    ]

    shard_state_dir = output / ".shard_state"
    shard_state_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = shard_state_dir / f"shard_{shard_id:03d}_checkpoint.json"
    manifest_path = shard_state_dir / f"shard_{shard_id:03d}_manifest.json"

    label = f"[shard {shard_id:02d}] "

    try:
        return _process_zone_assignments(
            output,
            zone_assignments=zone_assignments,
            episodes_per_map=episodes_per_map,
            duration_s=duration_s,
            seed=seed,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            total_maps_for_manifest=total_maps,
            total_episodes_for_manifest=total_episodes,
            progress_label=label,
        )
    except Exception as exc:
        # Re-raise with full traceback text preserved in the message,
        # since a bare exception crossing back through
        # ProcessPoolExecutor's pickling can lose some traceback detail.
        raise RuntimeError(
            f"shard {shard_id} failed: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        ) from exc


def generate_osm_corpus_sharded(
    output: Path,
    *,
    zones: tuple[ZoneSpec, ...],
    episodes_per_map: int,
    duration_s: int = 2400,
    seed: int = 26137,
    num_shards: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    maps = len(zones)
    if maps < 5:
        raise ValueError("production corpus requires at least 5 zones")
    if episodes_per_map < 1:
        raise ValueError("episodes_per_map must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    episodes = maps * episodes_per_map

    # Never create more shards than there are zones — an empty shard
    # would just be wasted process-startup overhead.
    num_shards = max(1, min(num_shards, maps))

    output = output.resolve()

    if overwrite:
        _safe_remove(output)
    output.mkdir(parents=True, exist_ok=True)

    split_maps = build_split_map_indices(maps)
    split_by_index: dict[int, str] = {}
    for split_name, indices in split_maps.items():
        for idx in indices:
            split_by_index[idx] = split_name

    shard_indices = partition_zones_round_robin(maps, num_shards)

    shard_assignments: list[list[tuple[int, str, str, str]]] = []
    for shard_zone_indices in shard_indices:
        assignment = [
            (
                global_index,
                zones[global_index].name,
                str(zones[global_index].graphml),
                split_by_index[global_index],
            )
            for global_index in shard_zone_indices
        ]
        shard_assignments.append(assignment)

    print(
        f"Partitioned {maps} zones into {num_shards} shard(s): "
        f"{[len(a) for a in shard_assignments]} zones each.",
        flush=True,
    )

    results: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}

    orchestrator_started = time.perf_counter()

    if num_shards == 1:
        results[0] = _run_shard(
            str(output),
            0,
            shard_assignments[0],
            episodes_per_map,
            duration_s,
            seed,
            maps,
            episodes,
        )
    else:
        with ProcessPoolExecutor(max_workers=num_shards) as executor:
            futures = {
                executor.submit(
                    _run_shard,
                    str(output),
                    shard_id,
                    assignment,
                    episodes_per_map,
                    duration_s,
                    seed,
                    maps,
                    episodes,
                ): shard_id
                for shard_id, assignment in enumerate(shard_assignments)
                if assignment
            }

            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    results[shard_id] = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors[shard_id] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[shard {shard_id:02d}] FAILED: {exc}",
                        flush=True,
                    )

    orchestrator_elapsed = time.perf_counter() - orchestrator_started

    if errors:
        raise RuntimeError(
            f"{len(errors)}/{num_shards} shard(s) failed: {errors}. "
            "Re-run the same command to resume the shards that "
            "succeeded and retry the failed ones from their own "
            "checkpoints."
        )

    merged_map_records: dict[int, dict[str, Any]] = {}
    merged_episode_records: list[dict[str, Any]] = []
    merged_failed_attempts: list[dict[str, Any]] = []

    for shard_id in sorted(results):
        shard_manifest = results[shard_id]
        for key, value in shard_manifest["map_records"].items():
            merged_map_records[int(key)] = value
        merged_episode_records.extend(shard_manifest["episodes"])
        merged_failed_attempts.extend(shard_manifest["failed_attempts"])

    if len(merged_episode_records) != episodes:
        raise RuntimeError(
            f"sharded corpus incomplete after merge: "
            f"{len(merged_episode_records)}/{episodes} episodes"
        )

    fingerprints = [r["map_fingerprint"] for r in merged_map_records.values()]
    if len(fingerprints) != len(set(fingerprints)):
        raise RuntimeError("base-map fingerprint collision across shards")

    counts = {"train": 0, "validation": 0, "test": 0}
    for record in merged_episode_records:
        counts[record["split"]] += 1

    manifest = {
        "schema_version": "quanta-real-sumo-osm-corpus-v1",
        "requested_episodes": episodes,
        "generated_episodes": len(merged_episode_records),
        "base_maps": maps,
        "episodes_per_map": episodes_per_map,
        "duration_s": duration_s,
        "interval_s": 60,
        "seed": seed,
        "backend": "sumo",
        "source": (
            "Real OpenStreetMap road geometry per zone + osm_demand_generator "
            "+ dynamic event generator + RoutingPipeline + SUMO/TraCI "
            "(sharded across zones)"
        ),
        "zones": [
            {"index": i, "name": z.name, "graphml": str(z.graphml)}
            for i, z in enumerate(zones)
        ],
        "fixture_route_planner_used": False,
        "fake_observation_generator_used": False,
        "fixture_backend_used": False,
        "synthetic_road_network_used": False,
        "split_rule": (
            "map-disjoint (zone-disjoint) train/validation/test split assigned "
            "before episode generation, computed globally before sharding"
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
        "map_records": merged_map_records,
        "episodes": merged_episode_records,
        "failed_attempts": merged_failed_attempts,
        "num_shards": num_shards,
        "shard_zone_counts": [len(a) for a in shard_assignments],
        "orchestrator_wall_clock_s": orchestrator_elapsed,
        "sum_of_shard_wall_clock_s": sum(
            m.get("created_wall_clock_s", 0.0) for m in results.values()
        ),
        "created_wall_clock_s": orchestrator_elapsed,
    }

    manifest_path = output / "corpus_manifest.json"
    checkpoint_path = output / "corpus_checkpoint.json"
    _write_json(manifest_path, manifest)
    _write_json(checkpoint_path, manifest)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--zones-file", type=Path, required=True)
    parser.add_argument("--episodes-per-map", type=int, default=50)
    parser.add_argument(
        "--duration-s", "--duration", dest="duration_s", type=int, default=2400
    )
    parser.add_argument("--seed", type=int, default=26137)
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes, one per shard of "
            "zones. Match this to your CPU core count (or slightly "
            "below, to leave headroom for the OS/other processes). "
            "Capped automatically at the number of zones."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zones = load_zone_specs(args.zones_file)

    manifest = generate_osm_corpus_sharded(
        args.output,
        zones=zones,
        episodes_per_map=args.episodes_per_map,
        duration_s=args.duration_s,
        seed=args.seed,
        num_shards=args.shards,
        overwrite=args.overwrite,
    )

    print(
        json.dumps(
            {
                "generated_episodes": manifest["generated_episodes"],
                "base_maps": manifest["base_maps"],
                "split_episode_counts": manifest["split_episode_counts"],
                "failed_attempts": len(manifest["failed_attempts"]),
                "num_shards": manifest["num_shards"],
                "orchestrator_wall_clock_s": round(
                    manifest["orchestrator_wall_clock_s"], 1
                ),
                "sum_of_shard_wall_clock_s": round(
                    manifest["sum_of_shard_wall_clock_s"], 1
                ),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())