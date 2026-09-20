"""Add loader-compatible keys to a generated OSM corpus manifest, in place.

Merges the exact schema src.learning.loader expects into
<corpus>/corpus_manifest.json WITHOUT dropping the generator fields:

    train / validation / test <- sorted episode_id lists grouped by split
    scenario_files             <- {scenario_id: scenario_path} from map_records
    map_fingerprints           <- {scenario_id: map_fingerprint} from map_records
    map_disjoint               <- True (zone-disjoint split assigned before
                                  generation; verified fingerprint-disjoint)

This is the parameterized successor of scripts/adapt_delhi_manifest.py
(which is hardcoded to corpus_delhi and left untouched): corpus_v2 gets the
loader schema "from the start" by running this as a pipeline step right
after generation, instead of as a later bolt-on.

Usage:
    python3 scripts/adapt_corpus_manifest.py --corpus artifacts/corpus_v2

Idempotent: re-running on an already-adapted manifest re-validates and
rewrites the same keys. A backup of the pre-adapt manifest is kept at
<corpus>/corpus_manifest.pre_adapt_backup.json (first run only).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SPLITS = ("train", "validation", "test")


def adapt(corpus: Path) -> dict:
    manifest_path = corpus / "corpus_manifest.json"
    backup_path = corpus / "corpus_manifest.pre_adapt_backup.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required = {"map_records", "episodes", "split_episode_counts"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing required fields: {sorted(missing)}")

    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)

    map_records = manifest["map_records"]
    episode_records = manifest["episodes"]

    scenario_files: dict[str, str] = {}
    map_fingerprints: dict[str, str] = {}
    map_split: dict[int, str] = {}
    for raw_index, record in map_records.items():
        map_index = int(raw_index)
        scenario_id = record["scenario_id"]
        if not (corpus / record["scenario_path"]).exists():
            raise FileNotFoundError(
                f"missing scenario file for map {map_index}: {record['scenario_path']}"
            )
        scenario_files[scenario_id] = record["scenario_path"]
        map_fingerprints[scenario_id] = record["map_fingerprint"]
        map_split[map_index] = record["split"]

    split_ids: dict[str, list[str]] = {name: [] for name in SPLITS}
    seen_ids: set[str] = set()
    for record in episode_records:
        episode_id, split = record["episode_id"], record["split"]
        if split not in split_ids:
            raise ValueError(f"invalid split {split!r} for {episode_id}")
        if episode_id in seen_ids:
            raise ValueError(f"duplicate episode id: {episode_id}")
        seen_ids.add(episode_id)
        map_index = int(record["map_index"])
        if map_split.get(map_index) != split:
            raise ValueError(f"split mismatch for {episode_id}: episode={split}")
        if record["scenario_id"] not in scenario_files:
            raise ValueError(f"{episode_id} references unknown scenario")
        ep_manifest = json.loads(
            (corpus / "episodes" / episode_id / "episode_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if ep_manifest.get("split") != split:
            raise ValueError(f"episode-manifest split mismatch for {episode_id}")
        if ep_manifest.get("scenario_id") != record["scenario_id"]:
            raise ValueError(f"episode-manifest scenario mismatch for {episode_id}")
        split_ids[split].append(episode_id)
    for name in SPLITS:
        split_ids[name].sort()

    expected_counts = manifest["split_episode_counts"]
    for name in SPLITS:
        if len(split_ids[name]) != int(expected_counts[name]):
            raise ValueError(
                f"{name} count mismatch: {len(split_ids[name])} != {expected_counts[name]}"
            )

    # Map-disjointness: fingerprints unique globally and single-split per map.
    fps = [r["map_fingerprint"] for r in map_records.values()]
    if len(set(fps)) != len(fps):
        raise ValueError("map fingerprint collision across maps")
    fp_split: dict[str, str] = {}
    for record in map_records.values():
        fp = record["map_fingerprint"]
        if fp in fp_split and fp_split[fp] != record["split"]:
            raise ValueError(f"base-map leakage: fingerprint {fp[:12]} in two splits")
        fp_split[fp] = record["split"]

    manifest.update(
        {
            "source_schema_version": manifest.get("schema_version"),
            "schema_version": "quanta-step13-delhi-adapted-v1",
            "map_disjoint": True,
            "train": split_ids["train"],
            "validation": split_ids["validation"],
            "test": split_ids["test"],
            "scenario_files": scenario_files,
            "map_fingerprints": map_fingerprints,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "status": "ok",
        "corpus": str(corpus),
        "episodes": len(seen_ids),
        "train": len(split_ids["train"]),
        "validation": len(split_ids["validation"]),
        "test": len(split_ids["test"]),
        "maps": len(map_records),
        "map_disjoint": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adapt(args.corpus), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
