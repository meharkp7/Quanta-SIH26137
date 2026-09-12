"""Step 11 label-maturity rules.

Speed-proxy labels mature when the future minute bucket has been observed.
Realized traversal labels mature when a vehicle exits an edge and are aligned
to the *entry-time* bucket.  Sparse or unused edges are missing, never 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable, Mapping

from src.contracts.core_types import TargetKind


def minute_bucket(time_s: float, interval_s: int = 60) -> int:
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    return int(floor(float(time_s) / interval_s) * interval_s)


@dataclass(frozen=True)
class MatureLabel:
    """One edge/horizon label with an explicit availability time."""

    episode_id: str
    edge_id: str
    issue_time_s: int
    target_time_s: int
    target_kind: str
    value: float | None
    available_at_s: int | None
    missing: bool

    def usable_at(self, now_s: float) -> bool:
        if self.missing or self.available_at_s is None or self.value is None:
            return False
        return float(now_s) >= float(self.available_at_s)


def mature_speed_proxy_label(
    *,
    episode_id: str,
    edge_id: str,
    issue_time_s: int,
    target_time_s: int,
    observation_by_bucket: Mapping[tuple[str, int], Mapping[str, object]],
    interval_s: int = 60,
) -> MatureLabel:
    """A speed-ratio / travel-time proxy matures only when that bucket exists."""

    if target_time_s <= issue_time_s:
        raise ValueError("speed-proxy target must be strictly after issue time")
    bucket = minute_bucket(target_time_s, interval_s)
    row = observation_by_bucket.get((edge_id, bucket))
    if row is None:
        return MatureLabel(
            episode_id=episode_id,
            edge_id=edge_id,
            issue_time_s=issue_time_s,
            target_time_s=target_time_s,
            target_kind=TargetKind.SPEED_PROXY.value,
            value=None,
            available_at_s=None,
            missing=True,
        )
    missing = bool(int(row.get("missing") or 0))
    raw = row.get("observed_speed_mps")
    if missing or raw in (None, ""):
        return MatureLabel(
            episode_id=episode_id,
            edge_id=edge_id,
            issue_time_s=issue_time_s,
            target_time_s=target_time_s,
            target_kind=TargetKind.SPEED_PROXY.value,
            value=None,
            available_at_s=bucket,
            missing=True,
        )
    return MatureLabel(
        episode_id=episode_id,
        edge_id=edge_id,
        issue_time_s=issue_time_s,
        target_time_s=target_time_s,
        target_kind=TargetKind.SPEED_PROXY.value,
        value=float(raw),
        available_at_s=bucket,
        missing=False,
    )


def mature_traversal_label(
    *,
    episode_id: str,
    edge_id: str,
    issue_time_s: int,
    target_time_s: int,
    traversals: Iterable[Mapping[str, object]],
    interval_s: int = 60,
) -> MatureLabel:
    """Mean realized duration for traversals that *entered* in the target bucket."""

    if target_time_s <= issue_time_s:
        raise ValueError("traversal target must be strictly after issue time")
    bucket = minute_bucket(target_time_s, interval_s)
    durations: list[float] = []
    exits: list[float] = []
    for item in traversals:
        if str(item["edge_id"]) != edge_id:
            continue
        entry = float(item["entry_time_s"])
        exit_time = float(item["exit_time_s"])
        if minute_bucket(entry, interval_s) != bucket:
            continue
        duration = exit_time - entry
        if duration <= 0:
            continue
        durations.append(duration)
        exits.append(exit_time)
    if not durations:
        return MatureLabel(
            episode_id=episode_id,
            edge_id=edge_id,
            issue_time_s=issue_time_s,
            target_time_s=target_time_s,
            target_kind=TargetKind.REALIZED_TRAVERSAL.value,
            value=None,
            available_at_s=None,
            missing=True,
        )
    return MatureLabel(
        episode_id=episode_id,
        edge_id=edge_id,
        issue_time_s=issue_time_s,
        target_time_s=target_time_s,
        target_kind=TargetKind.REALIZED_TRAVERSAL.value,
        value=sum(durations) / len(durations),
        available_at_s=int(max(exits)),
        missing=False,
    )
