from types import SimpleNamespace

from src.routing.route_encoding import BoundedRouteRepairer


def test_result_reports_instance_evaluation_metrics_without_static_self_error():
    repairer = BoundedRouteRepairer.__new__(BoundedRouteRepairer)
    repairer._evaluation_calls = 7
    repairer._evaluation_cache_hits = 3

    route_plan = SimpleNamespace()
    evaluation = SimpleNamespace(
        feasible=True,
        all_requests_served=True,
        duplicate_customer_ids=(),
        unknown_request_ids=(),
    )

    result = repairer._result(
        SimpleNamespace(route_plan=route_plan),
        route_plan,
        evaluation,
        [],
        2,
        1,
        0.0,
        False,
        False,
        (),
    )

    assert result.evaluation_calls == 7
    assert result.evaluation_cache_hits == 3
    assert result.feasible is True
    assert result.complete is True
