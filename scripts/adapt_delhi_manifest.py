from __future__ import annotations

import json
from pathlib import Path
import shutil


ROOT = Path("corpus_delhi")
MANIFEST = ROOT / "corpus_manifest.json"
BACKUP = ROOT / "corpus_manifest.osm_backup.json"


def main() -> None:
    if not ROOT.exists():
        raise FileNotFoundError(f"Missing corpus directory: {ROOT}")

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    required = {
        "map_records",
        "episodes",
        "split_episode_counts",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(
            f"Delhi manifest is missing required fields: {sorted(missing)}"
        )

    if not BACKUP.exists():
        shutil.copy2(MANIFEST, BACKUP)

    map_records = manifest["map_records"]
    episode_records = manifest["episodes"]

    if len(episode_records) != manifest["generated_episodes"]:
        raise ValueError(
            "Manifest episode count mismatch: "
            f"{len(episode_records)} != {manifest['generated_episodes']}"
        )

    # Build scenario lookup and map-fingerprint lookup.
    scenario_files: dict[str, str] = {}
    map_fingerprints: dict[str, str] = {}
    map_split: dict[int, str] = {}

    for raw_index, record in map_records.items():
        map_index = int(raw_index)

        scenario_id = record["scenario_id"]
        scenario_path = record["scenario_path"]
        fingerprint = record["map_fingerprint"]
        split = record["split"]

        absolute_scenario = ROOT / scenario_path
        if not absolute_scenario.exists():
            raise FileNotFoundError(
                f"Missing scenario file for map {map_index}: "
                f"{absolute_scenario}"
            )

        scenario_files[scenario_id] = scenario_path
        map_fingerprints[scenario_id] = fingerprint
        map_split[map_index] = split

    # Construct the exact schema expected by src.learning.loader.
    split_ids = {
        "train": [],
        "validation": [],
        "test": [],
    }

    seen_ids: set[str] = set()

    for record in episode_records:
        episode_id = record["episode_id"]
        split = record["split"]
        map_index = int(record["map_index"])
        scenario_id = record["scenario_id"]

        if split not in split_ids:
            raise ValueError(
                f"Invalid split {split!r} for episode {episode_id}"
            )

        if episode_id in seen_ids:
            raise ValueError(f"Duplicate episode ID: {episode_id}")
        seen_ids.add(episode_id)

        if map_index not in map_split:
            raise ValueError(
                f"Episode {episode_id} references unknown map {map_index}"
            )

        if map_split[map_index] != split:
            raise ValueError(
                f"Split mismatch for {episode_id}: "
                f"episode={split}, map={map_split[map_index]}"
            )

        if scenario_id not in scenario_files:
            raise ValueError(
                f"Episode {episode_id} references unknown scenario "
                f"{scenario_id!r}"
            )

        episode_dir = ROOT / "episodes" / episode_id
        episode_manifest = episode_dir / "episode_manifest.json"

        if not episode_manifest.exists():
            raise FileNotFoundError(
                f"Missing episode manifest: {episode_manifest}"
            )

        episode_data = json.loads(
            episode_manifest.read_text(encoding="utf-8")
        )

        if episode_data.get("split") != split:
            raise ValueError(
                f"Episode manifest split mismatch for {episode_id}: "
                f"{episode_data.get('split')!r} != {split!r}"
            )

        if episode_data.get("scenario_id") != scenario_id:
            raise ValueError(
                f"Episode manifest scenario mismatch for {episode_id}: "
                f"{episode_data.get('scenario_id')!r} != {scenario_id!r}"
            )

        split_ids[split].append(episode_id)

    # Stable deterministic ordering.
    for split in split_ids:
        split_ids[split].sort()

    expected_counts = manifest["split_episode_counts"]

    for split in ("train", "validation", "test"):
        actual = len(split_ids[split])
        expected = int(expected_counts[split])

        if actual != expected:
            raise ValueError(
                f"{split} episode count mismatch: "
                f"actual={actual}, expected={expected}"
            )

    if sum(len(v) for v in split_ids.values()) != len(episode_records):
        raise ValueError("Not all manifest episodes were assigned to a split")

    if len(seen_ids) != len(episode_records):
        raise ValueError("Episode leakage/duplication detected")

    # Verify map-disjointness.
    split_fingerprints: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    for raw_index, record in map_records.items():
        split = record["split"]
        fingerprint = record["map_fingerprint"]

        if fingerprint in split_fingerprints[split]:
            raise ValueError(
                f"Duplicate map fingerprint inside {split}: {fingerprint}"
            )

        split_fingerprints[split].add(fingerprint)

    for left in ("train", "validation", "test"):
        for right in ("train", "validation", "test"):
            if left >= right:
                continue

            overlap = (
                split_fingerprints[left]
                & split_fingerprints[right]
            )

            if overlap:
                raise ValueError(
                    f"Base-map leakage between {left} and {right}: "
                    f"{sorted(overlap)}"
                )

    adapted = dict(manifest)

    adapted.update(
        {
            "schema_version": "quanta-step13-delhi-adapted-v1",
            "source_schema_version": manifest.get("schema_version"),
            "map_disjoint": True,
            "train": split_ids["train"],
            "validation": split_ids["validation"],
            "test": split_ids["test"],
            "scenario_files": scenario_files,
            "map_fingerprints": map_fingerprints,
        }
    )

    MANIFEST.write_text(
        json.dumps(adapted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "corpus": str(ROOT),
                "episodes": len(seen_ids),
                "train": len(split_ids["train"]),
                "validation": len(split_ids["validation"]),
                "test": len(split_ids["test"]),
                "maps": len(map_records),
                "map_disjoint": True,
                "scenario_files": len(scenario_files),
                "backup": str(BACKUP),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
