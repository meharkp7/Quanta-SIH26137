"""Production audit checks for Step 11 causal datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_episode(path: str | Path) -> dict:
    root = Path(path)

    required = [
        "observations.csv",
        "edge_truth.csv",
        "trajectories.csv",
        "labels.jsonl",
        "issued_forecasts.jsonl",
        "events.json",
        "episode_manifest.json",
    ]

    missing = [
        name for name in required
        if not (root / name).is_file()
    ]

    if missing:
        return {
            "ok": False,
            "errors": [
                f"missing artifact: {name}"
                for name in missing
            ],
        }

    manifest = json.loads(
        (root / "episode_manifest.json").read_text()
    )

    errors: list[str] = []

    visibility = manifest.get("causal_visibility", {})

    if not visibility.get("future_effect_times_hidden"):
        errors.append(
            "manifest does not assert future event times are hidden"
        )

    if not visibility.get("future_labels_hidden"):
        errors.append(
            "manifest does not assert future labels are hidden"
        )

    # Observation contract:
    # future truth/event fields must never leak into observations.csv.
    import csv

    with (root / "observations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    forbidden = {
        "effect_start_s",
        "effect_end_s",
        "active_event_id",
        "true_speed_mps",
        "true_speed_ratio",
        "target_available_at_s",
    }

    if rows and forbidden.intersection(rows[0].keys()):
        errors.append(
            "future/truth fields leaked into observations.csv"
        )

    labels = [
        json.loads(line)
        for line in (root / "labels.jsonl").read_text().splitlines()
        if line.strip()
    ]

    for label in labels:
        if not label.get("missing") and label.get("value") is not None:
            if label.get("available_at_s") is None:
                errors.append(
                    "usable label has no availability timestamp"
                )

            if float(label["available_at_s"]) < float(label["target_time_s"]):
                errors.append(
                    "label availability precedes target time"
                )

    hashes = {
        name: sha256_file(root / name)
        for name in required
    }

    return {
        "ok": not errors,
        "errors": errors,
        "episode_id": manifest.get("episode_id"),
        "artifact_sha256": hashes,
    }
