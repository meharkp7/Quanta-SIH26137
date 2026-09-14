from src.routing.cost_view import CostView
from src.routing.path_builder import DirectedPathBuilder
from src.routing.path_cache import PathCache
from src.routing.poi_distance_matrix import (
    departure_time_grid,
    poi_nodes_from_scenario,
    precompute_poi_distance_matrix,
)
from src.routing.route_evaluator import RouteEvaluationConfig, RouteEvaluator
from tests.test_contracts import make_tiny_scenario


def test_poi_helpers_are_deterministic():
    scenario = make_tiny_scenario()
    assert poi_nodes_from_scenario(scenario) == ("c1", "c2", "depot")
    assert departure_time_grid(scenario) == (0.0, 3600.0)


def test_precompute_warms_existing_cache_and_preserves_shortest_path():
    scenario = make_tiny_scenario()
    cost_view = CostView(scenario.edges, graph_version=scenario.graph_version)
    cache = PathCache(departure_bucket_s=60.0, max_entries=100)
    builder = DirectedPathBuilder(
        scenario.edges,
        cost_view=cost_view,
        path_cache=cache,
    )

    cold = builder.shortest_path("depot", "c2", departure_time_s=0.0)
    cache.invalidate()
    stats = precompute_poi_distance_matrix(
        scenario,
        builder,
        departure_times_s=(0.0,),
    )
    warm = builder.shortest_path("depot", "c2", departure_time_s=0.0)

    assert stats.path_stats.searches == 3
    assert stats.path_stats.paths_cached == 2
    assert warm == cold
    assert cache.stats()["hits"] >= 1


def test_precompute_is_idempotent_at_result_level():
    scenario = make_tiny_scenario()
    cost_view = CostView(scenario.edges, graph_version=scenario.graph_version)
    cache = PathCache(departure_bucket_s=60.0, max_entries=100)
    builder = DirectedPathBuilder(
        scenario.edges,
        cost_view=cost_view,
        path_cache=cache,
    )

    first = precompute_poi_distance_matrix(
        scenario,
        builder,
        departure_times_s=(0.0,),
    )
    second = precompute_poi_distance_matrix(
        scenario,
        builder,
        departure_times_s=(0.0,),
    )
    assert first.path_stats == second.path_stats


def test_missing_cache_is_rejected():
    scenario = make_tiny_scenario()
    cost_view = CostView(scenario.edges, graph_version=scenario.graph_version)
    builder = DirectedPathBuilder(scenario.edges, cost_view=cost_view)
    try:
        precompute_poi_distance_matrix(
            scenario,
            builder,
            departure_times_s=(0.0,),
        )
    except RuntimeError as exc:
        assert "PathCache" in str(exc)
    else:
        raise AssertionError("expected PathCache requirement to be enforced")


def test_evaluator_exposes_explicit_precompute_and_opt_in_auto_warmup():
    scenario = make_tiny_scenario()
    evaluator = RouteEvaluator(scenario)
    assert evaluator.path_builder.path_cache is not None
    assert evaluator.path_builder.path_cache.stats()["size"] == 0

    stats = evaluator.precompute_poi_matrix()
    assert stats.poi_node_ids == ("c1", "c2", "depot")
    assert stats.path_stats.searches == 6

    warmed = RouteEvaluator(
        scenario,
        config=RouteEvaluationConfig(precompute_poi_matrix=True),
        cost_view=CostView(scenario.edges, graph_version=scenario.graph_version),
    )
    assert warmed.path_builder.path_cache.stats()["size"] > 0


def test_auto_precompute_rejects_legacy_provider_without_cost_view():
    scenario = make_tiny_scenario()

    def provider(edge, departure_time_s):
        del departure_time_s
        return edge.free_flow_time_s

    try:
        RouteEvaluator(
            scenario,
            config=RouteEvaluationConfig(precompute_poi_matrix=True),
            travel_time_provider=provider,
        )
    except ValueError as exc:
        assert "requires a CostView" in str(exc)
    else:
        raise AssertionError("expected explicit CostView requirement")
