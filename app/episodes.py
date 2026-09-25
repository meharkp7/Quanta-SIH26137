"""Read-only episode playback over the recorded corpus (artifacts/corpus_v2).

The control room loads real recorded SUMO episodes next to the live solver:
list/detail/frame/forecasts endpoints that stream only the small time-sorted
CSV blocks instead of the ~86 MB per-episode artifacts. Nothing here mutates
the corpus — every path is derived from a validated ``ep-<digits>`` id.

Design notes
------------
* ``corpus_manifest.json`` (5.8 MB, parsed once) is the episode index — no
  per-episode stat sweep on the hot list path.
* ``observations.csv`` / ``edge_truth.csv`` are sorted ascending by time, so
  the first frame request builds a ``{t: (start, end)}`` byte-offset index and
  later requests seek straight to the block (~400 KB) instead of rescanning
  27 MB. Both files cover every edge on every step (4778 rows/step for
  map_000), and the canonical edge order comes from the first block.
* Speeds are sparse by design (~3% of edges report per step — only roads with
  traffic); ``ratio``/``observed`` arrays carry ``null`` for the rest and the
  UI draws those dim gray ("no observation this step").
* Episode closures live in the frame's ``closed`` array for display only —
  they never enter the live solver's closure state, so playback cannot fire
  an automatic re-plan.
"""

from __future__ import annotations

import csv
import io
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = PROJECT_ROOT / "artifacts" / "corpus_v2"

_EPISODE_ID = re.compile(r"^ep-\d+$")

_SUMMARY_FIELDS = (
    "episode_id",
    "scenario_id",
    "split",
    "regime",
    "event_type",
    "event_count",
    "seed",
    "duration_s",
    "interval_s",
)


def _slim(record: dict) -> dict:
    return {key: record.get(key) for key in _SUMMARY_FIELDS}


@lru_cache(maxsize=1)
def _manifest_records() -> tuple[dict, ...]:
    """Every episode summary from corpus_manifest.json, parsed once."""
    data = json.loads((CORPUS_ROOT / "corpus_manifest.json").read_text(encoding="utf-8"))
    return tuple(_slim(row) for row in data.get("episodes", []))


def list_episodes(
    scenario_id: str = "",
    split: str = "",
    regime: str = "",
    with_events: bool = False,
    limit: int = 100,
) -> dict:
    rows = _manifest_records()
    if scenario_id:
        rows = tuple(row for row in rows if row.get("scenario_id") == scenario_id)
    if split:
        rows = tuple(row for row in rows if row.get("split") == split)
    if regime:
        rows = tuple(row for row in rows if row.get("regime") == regime)
    if with_events:
        rows = tuple(row for row in rows if (row.get("event_count") or 0) > 0)
    limit = max(1, min(int(limit or 100), 500))
    return {
        "scenario_id": scenario_id,
        "split": split,
        "regime": regime,
        "total": len(rows),
        "episodes": list(rows[:limit]),
    }


def _episode_dir(episode_id: str) -> Path:
    if not _EPISODE_ID.match(episode_id or ""):
        raise ValueError(f"Malformed episode id: {episode_id!r} (expected ep-<digits>)")
    path = CORPUS_ROOT / "episodes" / episode_id
    if not path.is_dir():
        raise ValueError(f"Unknown episode: {episode_id}")
    return path


@lru_cache(maxsize=32)
def _episode_json(episode_id: str, name: str) -> Any:
    path = _episode_dir(episode_id) / name
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


@lru_cache(maxsize=6)
def _edge_ids(episode_id: str) -> tuple[str, ...]:
    """Canonical edge order — every row of the first (earliest) truth block."""
    path = _episode_dir(episode_id) / "edge_truth.csv"
    ids: list[str] = []
    first_t: int | None = None
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        col = header.index("edge_id") if "edge_id" in header else 1
        for row in reader:
            if not row:
                continue
            try:
                step = int(float(row[0]))
            except (ValueError, IndexError):
                continue
            if first_t is None:
                first_t = step
            elif step != first_t:
                break
            ids.append(row[col])
    return tuple(ids)


@lru_cache(maxsize=12)
def _csv_name(episode_id: str, kind: str) -> str:
    return "edge_truth.csv" if kind == "truth" else "observations.csv"


@lru_cache(maxsize=12)
def _csv_header(episode_id: str, kind: str) -> tuple[str, ...]:
    path = _episode_dir(episode_id) / _csv_name(episode_id, kind)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return tuple(next(csv.reader(fh), []))


@lru_cache(maxsize=12)
def _time_index(episode_id: str, kind: str) -> dict[int, tuple[int, int]]:
    """``{t: (start, end)}`` byte offsets for a time-sorted corpus CSV."""
    path = _episode_dir(episode_id) / _csv_name(episode_id, kind)
    offsets: dict[int, tuple[int, int]] = {}
    start: int | None = None
    current: int | None = None
    pos = 0
    with path.open("rb") as fh:
        for raw in fh:
            end = pos + len(raw)
            comma = raw.find(b",")
            step: int | None = None
            if comma > 0:
                try:
                    step = int(float(raw[:comma].decode("utf-8", "replace")))
                except (ValueError, TypeError):
                    step = None
            if step is not None:
                if current is None:
                    current, start = step, pos
                elif step != current:
                    offsets[current] = (start, pos)  # type: ignore[arg-type]
                    current, start = step, pos
            pos = end
        if current is not None and start is not None:
            offsets[current] = (start, pos)
    return offsets


def _snap(index: dict[int, tuple[int, int]], t: int) -> int:
    if not index:
        raise ValueError("Episode has no recorded timesteps")
    if t in index:
        return t
    return min(index, key=lambda key: abs(key - t))


def _read_block(
    path: Path, bounds: tuple[int, int], fieldnames: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    start, end = bounds
    with path.open("rb") as fh:
        fh.seek(start)
        raw = fh.read(max(0, end - start))
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8", "replace")), fieldnames=list(fieldnames))
    return {row["edge_id"]: row for row in reader if row.get("edge_id")}


@lru_cache(maxsize=6)
def _trajectories(episode_id: str) -> tuple[tuple[str, str, float, float], ...]:
    """``(trip_id, edge_id, entry_s, exit_s)`` for every recorded traversal."""
    path = _episode_dir(episode_id) / "trajectories.csv"
    rows: list[tuple[str, str, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append(
                    (row["trip_id"], row["edge_id"], float(row["entry_time_s"]), float(row["exit_time_s"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
    return tuple(rows)


def _manifest(episode_id: str) -> dict:
    manifest = _episode_json(episode_id, "episode_manifest.json")
    if not isinstance(manifest, dict) or "episode_id" not in manifest:
        raise ValueError(f"Unknown episode: {episode_id}")
    return manifest


def episode_detail(episode_id: str) -> dict:
    _episode_dir(episode_id)
    manifest = _manifest(episode_id)
    interval = int(manifest.get("interval_s") or 60)
    duration = int(manifest.get("duration_s") or 0)
    times = list(range(interval, max(duration, interval) + 1, interval))
    events = _episode_json(episode_id, "events.json")
    runtime = _episode_json(episode_id, "runtime_events.json")
    return {
        "episode_id": episode_id,
        "scenario_id": manifest.get("scenario_id"),
        "split": manifest.get("split"),
        "regime": manifest.get("regime"),
        "seed": manifest.get("seed"),
        "backend": manifest.get("backend"),
        "interval_s": interval,
        "duration_s": duration,
        "warmup_s": manifest.get("warmup_s"),
        "horizons_s": manifest.get("horizons_s") or [],
        "manifest": manifest,
        "events": events if isinstance(events, list) else [],
        "runtime_events": runtime if isinstance(runtime, list) else [],
        "edge_ids": list(_edge_ids(episode_id)),
        "times": times,
    }


def episode_frame(episode_id: str, t: int) -> dict:
    """Observed speeds, ground-truth ratios, closures, and vehicles at time ``t``.

    ``t`` snaps to the nearest recorded timestep. All three arrays are aligned
    to ``episode_detail``'s ``edge_ids``.
    """
    directory = _episode_dir(episode_id)
    edge_ids = _edge_ids(episode_id)
    total = len(edge_ids)
    truth_index = _time_index(episode_id, "truth")
    obs_index = _time_index(episode_id, "obs")
    truth_step = _snap(truth_index, int(t))
    obs_step = _snap(obs_index, int(t))

    truth = _read_block(
        directory / _csv_name(episode_id, "truth"),
        truth_index[truth_step],
        _csv_header(episode_id, "truth"),
    )
    observed_rows = _read_block(
        directory / _csv_name(episode_id, "obs"),
        obs_index[obs_step],
        _csv_header(episode_id, "obs"),
    )

    ratio: list[float | None] = [None] * total
    observed: list[float | None] = [None] * total
    closed: list[int] = [0] * total
    reporting = 0
    observed_count = 0
    ratio_sum = 0.0
    observed_sum = 0.0

    for index, edge_id in enumerate(edge_ids):
        row = truth.get(edge_id)
        if row is not None:
            if (row.get("is_closed") or "").strip().lower() == "true":
                closed[index] = 1
            value = row.get("true_speed_ratio") or ""
            if value:
                parsed = float(value)
                ratio[index] = round(parsed, 3)
                reporting += 1
                ratio_sum += parsed
        row = observed_rows.get(edge_id)
        if row is not None:
            if (row.get("known_closed") or "").strip().lower() in ("1", "true"):
                closed[index] = 1
            value = row.get("observed_speed_mps") or ""
            if value:
                parsed = float(value)
                observed[index] = round(parsed, 2)
                observed_count += 1
                observed_sum += parsed

    clock = float(obs_step)
    vehicles = []
    fleet = 0
    background = 0
    for trip_id, edge_id, entry_s, exit_s in _trajectories(episode_id):
        if entry_s <= clock < exit_s:
            span = max(exit_s - entry_s, 1e-9)
            pct = min(1.0, max(0.0, (clock - entry_s) / span))
            kind = "fleet" if trip_id.startswith("osm-veh:") else "bg"
            if kind == "fleet":
                fleet += 1
            else:
                background += 1
            vehicles.append({"id": trip_id, "edge": edge_id, "pct": round(pct, 3), "kind": kind})

    return {
        "episode_id": episode_id,
        "t": obs_step,
        "observed": observed,
        "ratio": ratio,
        "closed": closed,
        "vehicles": vehicles,
        "summary": {
            "total": total,
            "reporting": reporting,
            "observed_count": observed_count,
            "closed": sum(closed),
            "mean_ratio": round(ratio_sum / reporting, 3) if reporting else None,
            "mean_observed_mps": round(observed_sum / observed_count, 2) if observed_count else None,
            "fleet": fleet,
            "bg": background,
        },
    }


@lru_cache(maxsize=4)
def _forecast_issues(episode_id: str) -> tuple[dict, ...]:
    """Per-issue network aggregates: predicted mean/p10/p90 vs realized truth.

    Predictions are per-edge m/s arrays (``speed_proxy``); truth means reuse the
    offset-indexed ``edge_truth`` blocks at each target time, restricted to the
    same edges the forecast marked valid so the comparison is honest.
    """
    directory = _episode_dir(episode_id)
    path = directory / "issued_forecasts.jsonl"
    if not path.is_file():
        return ()
    truth_index = _time_index(episode_id, "truth")
    truth_header = _csv_header(episode_id, "truth")
    truth_blocks: dict[int, dict[str, dict[str, str]]] = {}
    issues: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            edge_ids = record.get("edge_ids") or []
            predictions = record.get("prediction") or []
            mask = record.get("valid_mask") or []
            targets = record.get("target_times_s") or []
            pred_mean: list[float | None] = []
            pred_p10: list[float | None] = []
            pred_p90: list[float | None] = []
            truth_mean: list[float | None] = []
            valid_edges: list[int] = []
            for column, target_t in enumerate(targets):
                values: list[float] = []
                for row_index, row in enumerate(predictions):
                    if row_index >= len(mask) or column >= len(mask[row_index]) or not mask[row_index][column]:
                        continue
                    if column >= len(row) or row[column] is None:
                        continue
                    values.append(float(row[column]))
                values.sort()
                count = len(values)
                pred_mean.append(round(sum(values) / count, 2) if count else None)
                pred_p10.append(round(values[int(0.10 * (count - 1))], 2) if count else None)
                pred_p90.append(round(values[int(0.90 * (count - 1))], 2) if count else None)
                valid_edges.append(count)
                realized: list[float] = []
                if truth_index:
                    step = _snap(truth_index, int(round(float(target_t))))
                    if step not in truth_blocks:
                        truth_blocks[step] = _read_block(
                            directory / _csv_name(episode_id, "truth"),
                            truth_index[step],
                            truth_header,
                        )
                    block = truth_blocks[step]
                    for row_index, edge_id in enumerate(edge_ids):
                        if row_index >= len(mask) or column >= len(mask[row_index]) or not mask[row_index][column]:
                            continue
                        row = block.get(edge_id)
                        if row and (row.get("true_speed_mps") or ""):
                            realized.append(float(row["true_speed_mps"]))
                truth_mean.append(round(sum(realized) / len(realized), 2) if realized else None)
            issues.append(
                {
                    "issued_at_s": float(record.get("issued_at_s") or 0),
                    "target_times_s": [float(value) for value in targets],
                    "pred_mean": pred_mean,
                    "pred_p10": pred_p10,
                    "pred_p90": pred_p90,
                    "truth_mean": truth_mean,
                    "valid_edges": valid_edges,
                    "forecast_version": record.get("forecast_version"),
                    "target_kind": record.get("target_kind"),
                    "target_unit": record.get("target_unit") or "m/s",
                }
            )
    issues.sort(key=lambda issue: issue["issued_at_s"])
    return tuple(issues)


def episode_forecasts(episode_id: str) -> dict:
    _episode_dir(episode_id)
    issues = _forecast_issues(episode_id)
    return {
        "episode_id": episode_id,
        "issues": list(issues),
        "model_version": issues[0].get("forecast_version") if issues else None,
    }
