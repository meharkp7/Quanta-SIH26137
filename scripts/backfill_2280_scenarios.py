"""Backfill missing scenario.json files for artifacts/delhi_2280.

Background: the delhi_2280 run generated 1615/2280 episodes on 15 base maps,
but the per-map directories contain only network/netconvert.log -- the
scenario.json (+ osm_zone.json) files were lost before commit. Without them,
src/learning/loader.load_pilot_windows cannot build training windows.

This script restores them WITHOUT rerunning SUMO, using two mechanisms:

1. Maps 0-4 (connaught_place, chandni_chowk, karol_bagh, dwarka_sector12,
   nehru_place): byte-source is corpus_delhi/maps/map_00X/scenario.json.
   The pilot was generated with identical (zone, map_index, seed, attempt),
   proven by equal map_fingerprint + nodes/edges/customers/seeds in both
   manifests. Copy + verify fingerprint.

2. Maps 5,6,7,8,10,11,12,14,15,16: deterministic re-emit via
   load_graphml + osm_map_config(map_index, attempt, seed=26137) +
   generate_osm_scenario -- the exact call path
   scripts/generate_delhi_corpus._generate_osm_map_with_recovery uses.
   Pure-Python, CPU-only, no torch, no SUMO/netconvert. Verified on
   pitampura-map016: re-emit reproduces the manifest fingerprint exactly
   in ~15 s.

Every written file is verified against the manifest record (scenario_id,
nodes, edges, customers, map_fingerprint) before/after write. Any mismatch
aborts that map with a non-zero exit, leaving existing files untouched.

Usage:
    python scripts/backfill_2280_scenarios.py --dry-run   # default: verify only
    python scripts/backfill_2280_scenarios.py --apply     # write missing files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "artifacts" / "delhi_2280"
PILOT = ROOT / "corpus_delhi"
ZONES_FILE = ROOT / "configs" / "delhi_zones_19.json"
MANIFEST = OUTPUT / "corpus_manifest.json"

# Maps whose scenario.json is bit-sourced from the pilot corpus (verified
# identical fingerprints + dims + seeds in both manifests).
PILOT_SOURCED = (0, 1, 2, 3, 4)


def map_fingerprint_of(scenario) -> str:
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


def check_record(scenario, record: dict, label: str) -> list[str]:
    errors = []
    if scenario.scenario_id != record["scenario_id"]:
        errors.append(f"{label}: scenario_id {scenario.scenario_id!r} != {record['scenario_id']!r}")
    if len(scenario.nodes) != record["nodes"]:
        errors.append(f"{label}: nodes {len(scenario.nodes)} != {record['nodes']}")
    if len(scenario.edges) != record["edges"]:
        errors.append(f"{label}: edges {len(scenario.edges)} != {record['edges']}")
    if len(scenario.requests) != record["customers"]:
        errors.append(f"{label}: customers {len(scenario.requests)} != {record['customers']}")
    if map_fingerprint_of(scenario) != record["map_fingerprint"]:
        errors.append(f"{label}: fingerprint mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    apply = args.apply

    from src.contracts.scenario import Scenario
    from src.data.causal_episodes import _write_json

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    zones = json.loads(ZONES_FILE.read_text(encoding="utf-8"))
    seed = int(manifest["seed"])
    failures: list[str] = []

    for key in sorted(manifest["map_records"], key=int):
        map_index = int(key)
        record = manifest["map_records"][key]
        map_dir = OUTPUT / "maps" / f"map_{map_index:03d}"
        target = map_dir / "scenario.json"
        zone_name = record["zone"]
        assert zones[map_index]["name"] == zone_name, f"zone order drift at {map_index}"

        if map_index in PILOT_SOURCED:
            src = PILOT / "maps" / f"map_{map_index:03d}" / "scenario.json"
            scenario = Scenario.model_validate_json(src.read_text(encoding="utf-8"))
            errors = check_record(scenario, record, f"map_{map_index:03d}(pilot-copy)")
            if errors:
                failures.extend(errors)
                continue
            if apply:
                map_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                # Re-verify the written file.
                reread = Scenario.model_validate_json(target.read_text(encoding="utf-8"))
                failures.extend(check_record(reread, record, f"map_{map_index:03d}(written)"))
                print(f"map_{map_index:03d} ({zone_name}): copied from pilot, verified", flush=True)
            else:
                print(f"map_{map_index:03d} ({zone_name}): pilot copy would verify OK", flush=True)
            continue

        # Deterministic re-emit (SUMO-free).
        from src.data.osm_ingestion import load_graphml
        from src.data.osm_demand_generator import generate_osm_scenario
        from scripts.generate_delhi_corpus import osm_map_config

        attempt = int(record.get("generation_attempt", 0))
        if record.get("route_recovery"):
            failures.append(f"map_{map_index:03d}: route_recovery demand not reproducible by attempt alone")
            continue
        t0 = time.perf_counter()
        network = load_graphml(str(ROOT / zones[map_index]["graphml"]))
        cfg = osm_map_config(map_index, attempt, seed)
        if cfg.seed != record["seed"]:
            failures.append(
                f"map_{map_index:03d}: config seed {cfg.seed} != manifest {record['seed']}"
            )
            continue
        result = generate_osm_scenario(
            network,
            cfg,
            scenario_id=record["scenario_id"],
            source_name=f"osm:{zone_name}",
            dataset_split=record["split"],
        )
        errors = check_record(result.scenario, record, f"map_{map_index:03d}(re-emit)")
        dt = time.perf_counter() - t0
        if errors:
            failures.extend(errors)
            continue
        if apply:
            map_dir.mkdir(parents=True, exist_ok=True)
            _write_json(target, result.scenario.model_dump(mode="json"))
            _write_json(
                map_dir / "osm_zone.json",
                {
                    "zone": zone_name,
                    "graphml": zones[map_index]["graphml"],
                    "depot_node_id": result.depot_node_id,
                    "customer_node_ids": list(result.customer_node_ids),
                    "unreachable_candidates_skipped": result.unreachable_candidates_skipped,
                },
            )
            print(f"map_{map_index:03d} ({zone_name}): re-emitted + verified in {dt:.1f}s", flush=True)
        else:
            print(f"map_{map_index:03d} ({zone_name}): re-emit would verify OK ({dt:.1f}s)", flush=True)

    if failures:
        print("FAILURES:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    print(f"done ({'applied' if apply else 'dry-run, nothing written'})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
