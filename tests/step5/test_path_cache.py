from __future__ import annotations

from src.routing.path_cache import (
    CachedPath,
    PathCache,
)


def test_cache_key_contains_all_routing_state() -> None:
    cache = PathCache(
        departure_bucket_s=60.0,
    )

    key = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="graph-1",
        cost_version="cost-2",
        forecast_version="forecast-3",
        departure_time_s=125.0,
    )

    assert key.from_node == "N1"
    assert key.to_node == "N2"
    assert key.graph_version == "graph-1"
    assert key.cost_version == "cost-2"
    assert key.forecast_version == "forecast-3"
    assert key.departure_bucket == 2


def test_same_state_and_bucket_hits() -> None:
    cache = PathCache(
        departure_bucket_s=60.0,
    )

    key = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=10.0,
    )

    path = CachedPath(
        node_ids=("N1", "N2"),
        edge_ids=("E1",),
    )

    cache.put(key, path)

    assert cache.get(key) == path
    assert cache.hits == 1
    assert cache.misses == 0


def test_missing_key_is_miss() -> None:
    cache = PathCache()

    key = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=0.0,
    )

    assert cache.get(key) is None
    assert cache.misses == 1


def test_graph_version_changes_key() -> None:
    cache = PathCache()

    key1 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=0.0,
    )

    key2 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g2",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=0.0,
    )

    assert key1 != key2


def test_cost_version_changes_key() -> None:
    cache = PathCache()

    key1 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=0.0,
    )

    key2 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c2",
        forecast_version="f1",
        departure_time_s=0.0,
    )

    assert key1 != key2


def test_forecast_version_changes_key() -> None:
    cache = PathCache()

    key1 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=0.0,
    )

    key2 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f2",
        departure_time_s=0.0,
    )

    assert key1 != key2


def test_departure_bucket_changes_key() -> None:
    cache = PathCache(
        departure_bucket_s=60.0,
    )

    key1 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=59.0,
    )

    key2 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=60.0,
    )

    assert key1 != key2
    assert key1.departure_bucket == 0
    assert key2.departure_bucket == 1


def test_invalidation_by_version() -> None:
    cache = PathCache()

    key1 = cache.make_key(
        from_node="N1",
        to_node="N2",
        graph_version="g1",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=0.0,
    )

    key2 = cache.make_key(
        from_node="N2",
        to_node="N3",
        graph_version="g2",
        cost_version="c1",
        forecast_version="f1",
        departure_time_s=0.0,
    )

    path = CachedPath(
        node_ids=("N1", "N2"),
        edge_ids=("E1",),
    )

    cache.put(key1, path)
    cache.put(key2, path)

    assert cache.size == 2

    removed = cache.invalidate(
        graph_version="g1",
    )

    assert removed == 1
    assert cache.size == 1
    assert cache.get(key1) is None
    assert cache.get(key2) == path