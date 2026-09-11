"""Versioned cache for directed routing paths.

Step 5B adds explicit version-aware path caching. The cache stores physical
path structure rather than time-dependent travel-time values.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from threading import RLock
from typing import Iterable, Protocol


TimeS = float
RoadNodeId = str
RoadEdgeId = str


class PathResultLike(Protocol):
    """Minimal interface required to cache a path result."""

    node_ids: tuple[RoadNodeId, ...]
    edge_ids: tuple[RoadEdgeId, ...]


@dataclass(frozen=True)
class PathCacheKey:
    """Complete state key for one cached routing query."""

    from_node: RoadNodeId
    to_node: RoadNodeId
    graph_version: str
    cost_version: str
    forecast_version: str
    departure_bucket: int


@dataclass(frozen=True)
class CachedPath:
    """Physical path structure retained by the cache."""

    node_ids: tuple[RoadNodeId, ...]
    edge_ids: tuple[RoadEdgeId, ...]

    @classmethod
    def from_result(
        cls,
        result: PathResultLike,
    ) -> "CachedPath":
        return cls(
            node_ids=tuple(result.node_ids),
            edge_ids=tuple(result.edge_ids),
        )


class PathCache:
    """Thread-safe versioned cache for physical routing paths."""

    def __init__(
        self,
        *,
        departure_bucket_s: TimeS = 60.0,
        max_entries: int | None = None,
    ) -> None:
        bucket = float(departure_bucket_s)

        if (
            not isfinite(bucket)
            or bucket <= 0.0
        ):
            raise ValueError(
                "departure_bucket_s must be finite "
                "and positive"
            )

        if (
            max_entries is not None
            and max_entries <= 0
        ):
            raise ValueError(
                "max_entries must be positive "
                "when provided"
            )

        self.departure_bucket_s = bucket
        self.max_entries = max_entries

        self._entries: dict[
            PathCacheKey,
            CachedPath,
        ] = {}

        self._hits = 0
        self._misses = 0

        self._lock = RLock()

    def make_key(
        self,
        *,
        from_node: RoadNodeId,
        to_node: RoadNodeId,
        graph_version: str,
        cost_version: str,
        forecast_version: str,
        departure_time_s: TimeS,
    ) -> PathCacheKey:
        """Create a versioned key for a routing query."""

        value = float(departure_time_s)

        if not isfinite(value):
            raise ValueError(
                "departure_time_s must be finite"
            )

        if value < 0.0:
            raise ValueError(
                "departure_time_s must be non-negative"
            )

        return PathCacheKey(
            from_node=from_node,
            to_node=to_node,
            graph_version=str(graph_version),
            cost_version=str(cost_version),
            forecast_version=str(
                forecast_version
            ),
            departure_bucket=int(
                floor(
                    value
                    / self.departure_bucket_s
                )
            ),
        )

    def get(
        self,
        key: PathCacheKey,
    ) -> CachedPath | None:
        with self._lock:
            cached = self._entries.get(key)

            if cached is None:
                self._misses += 1
                return None

            self._hits += 1
            return cached

    def put(
        self,
        key: PathCacheKey,
        path: CachedPath,
    ) -> None:
        with self._lock:
            if (
                self.max_entries is not None
                and key not in self._entries
                and len(self._entries)
                >= self.max_entries
            ):
                oldest_key = next(
                    iter(self._entries)
                )
                del self._entries[oldest_key]

            self._entries[key] = path

    def put_result(
        self,
        key: PathCacheKey,
        result: PathResultLike,
    ) -> None:
        self.put(
            key,
            CachedPath.from_result(result),
        )

    def invalidate(
        self,
        *,
        graph_version: str | None = None,
        cost_version: str | None = None,
        forecast_version: str | None = None,
    ) -> int:
        """Remove entries matching supplied version dimensions.

        If no version is supplied, the complete cache is cleared.
        """

        with self._lock:
            if (
                graph_version is None
                and cost_version is None
                and forecast_version is None
            ):
                removed = len(self._entries)
                self._entries.clear()
                return removed

            keys_to_remove = [
                key
                for key in self._entries
                if (
                    graph_version is None
                    or key.graph_version
                    == str(graph_version)
                )
                and (
                    cost_version is None
                    or key.cost_version
                    == str(cost_version)
                )
                and (
                    forecast_version is None
                    or key.forecast_version
                    == str(forecast_version)
                )
            ]

            for key in keys_to_remove:
                del self._entries[key]

            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached paths."""

        self.invalidate()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses

            if total == 0:
                return 0.0

            return self._hits / total

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            total = self._hits + self._misses

            return {
                "size": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "requests": total,
                "hit_rate": (
                    self._hits / total
                    if total
                    else 0.0
                ),
            }

    def keys(self) -> tuple[PathCacheKey, ...]:
        with self._lock:
            return tuple(
                self._entries.keys()
            )