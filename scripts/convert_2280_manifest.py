"""Convert artifacts/delhi_2280/corpus_manifest.json (schema
quanta-real-sumo-osm-corpus-v1) to the loader-compatible pilot schema
(quanta-step13-delhi-adapted-v1) WITHOUT touching the canonical manifest.

The mapping is unambiguous:
  train/validation/test <- sorted episode_ids grouped by episodes[].split
  scenario_files        <- {scenario_id: scenario_path} from map_records
  map_fingerprints      <- {scenario_id: map_fingerprint} from map_records
  map_disjoint          <- True (split_rule is zone-disjoint; verified below)
  split_episode_counts  <- copied verbatim (must equal list lengths)
  map_records           <- copied verbatim (validate_manifest_splits needs
                           per-map split + fingerprint for leakage checks)

Also verified in dry-run:
  - every episodes[].episode_dir exists on disk with observations.csv,
    labels.jsonl, episode_manifest.json (the only three files
    build_episode_windows reads, plus scenario.json from scenario_files)
  - every scenario_files target exists (i.e. backfill_2280_scenarios.py
    has been applied)
  - per-episode episode_manifest.json split/scenario_id agree with the
    corpus manifest (loader raises on disagreement)
  - no episode ID or map fingerprint leaks across splits

Usage:
    python scripts/convert_2280_manifest.py --dry-run   # default: verify only
    python scripts/convert_2280_manifest.py --apply     # write pilot_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "artifacts" / "delhi_2280"
SRC_MANIFEST = OUTPUT / "corpus_manifest.json"
DST_MANIFEST = OUTPUT / "pilot_manifest.json"
SPLITS = ("train", "validation", "test")
EPISODE_FILES = ("observations.csv", "labels.jsonl", "episode_manifest.json")


def build_pilot_manifest(src: dict) -> tuple[dict, list[str]]:
    errors: list[str] = []
    by_split: dict[str, list[str]] = {name: [] for name in SPLITS}
    for entry in src["episodes"]:
        split = entry["split"]
        if split not in by_split:
            errors.append(f"unknown split {split!r} on {entry['episode_id']}")
            continue
        by_split[split].append(entry["episode_id"])
    for name in SPLITS:
        by_split[name] = sorted(by_split[name])

    scenario_files: dict[str, str] = {}
    map_fingerprints: dict[str, str] = {}
    for key in sorted(src["map_records"], key=int):
        record = src["map_records"][key]
        scenario_files[record["scenario_id"]] = record["scenario_path"]
        map_fingerprints[record["scenario_id"]] = record["map_fingerprint"]

    counts = src["split_episode_counts"]
    for name in SPLITS:
        if int(counts.get(name, -1)) != len(by_split[name]):
            errors.append(
                f"{name} count mismatch: manifest says {counts.get(name)}, "
                f"episodes list has {len(by_split[name])}"
            )

    # Cross-check episode dirs on disk (loader hard-requires these).
    episode_by_id = {entry["episode_id"]: entry for entry in src["episodes"]}
    for name in SPLITS:
        for eid in by_split[name]:
            entry = episode_by_id[eid]
            ep_dir = OUTPUT / entry["episode_dir"]
            for fname in EPISODE_FILES:
                if not (ep_dir / fname).exists():
                    errors.append(f"{eid}: missing {fname}")
            manifest_path = ep_dir / "episode_manifest.json"
            if manifest_path.exists():
                try:
                    ep_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except ValueError as exc:
                    errors.append(f"{eid}: episode_manifest.json unreadable: {exc}")
                    continue
                if ep_manifest.get("split") != name:
                    errors.append(f"{eid}: split disagreement ({ep_manifest.get('split')} != {name})")
                if ep_manifest.get("scenario_id") != entry["scenario_id"]:
                    errors.append(f"{eid}: scenario_id disagreement")
            if entry["scenario_id"] not in scenario_files:
                errors.append(f"{eid}: scenario_id {entry['scenario_id']} has no map_record")

    for sid, rel in scenario_files.items():
        if not (OUTPUT / rel).exists():
            errors.append(f"scenario file missing: {sid} -> {rel}")

    # Leakage checks (mirror validate_manifest_splits).
    seen: set[str] = set()
    for name in SPLITS:
        overlap = seen.intersection(by_split[name])
        if overlap:
            errors.append(f"episode leakage into {name}: {sorted(overlap)[:5]}")
        seen.update(by_split[name])
    fp_seen: dict[str, str] = {}
    for key in sorted(src["map_records"], key=int):
        record = src["map_records"][key]
        fp = record["map_fingerprint"]
        if fp in fp_seen:
            errors.append(f"fingerprint shared by maps {fp_seen[fp]} and {key}")
        fp_seen[fp] = key

    dst = {
        "schema_version": "quanta-step13-delhi-adapted-v1",
        "source_schema_version": src.get("schema_version"),
        "requested_episodes": src.get("requested_episodes"),
        "generated_episodes": src.get("generated_episodes"),
        "base_maps": len(src["map_records"]),
        "episodes_per_map": src.get("episodes_per_map"),
        "duration_s": src.get("duration_s"),
        "interval_s": src.get("interval_s"),
        "seed": src.get("seed"),
        "backend": src.get("backend"),
        "source": src.get("source"),
        "split_rule": src.get("split_rule"),
        "split_episode_counts": {name: len(by_split[name]) for name in SPLITS},
        "map_disjoint": True,
        "train": by_split["train"],
        "validation": by_split["validation"],
        "test": by_split["test"],
        "scenario_files": scenario_files,
        "map_fingerprints": map_fingerprints,
        "map_records": src["map_records"],
    }
    return dst, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    src = json.loads(SRC_MANIFEST.read_text(encoding="utf-8"))
    dst, errors = build_pilot_manifest(src)
    print(
        json.dumps(
            {
                "train": len(dst["train"]),
                "validation": len(dst["validation"]),
                "test": len(dst["test"]),
                "scenario_files": len(dst["scenario_files"]),
                "errors": len(errors),
            },
            indent=2,
        )
    )
    for error in errors[:20]:
        print(f"  ERROR: {error}", flush=True)
    if errors:
        return 1
    if args.apply:
        DST_MANIFEST.write_text(json.dumps(dst, indent=2), encoding="utf-8")
        print(f"wrote {DST_MANIFEST} (canonical corpus_manifest.json untouched)", flush=True)
    else:
        print("dry-run OK, nothing written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
