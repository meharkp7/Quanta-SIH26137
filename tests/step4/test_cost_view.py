from src.routing.cost_view import CostView
from tests.step6.test_step6_sumo import load_base_scenario

def test_cost_revision_cannot_reuse_stale_path():
    scenario = load_base_scenario()
    factor = [1]
    view = CostView(scenario.edges, graph_version="g1", travel_time_provider=lambda e,t: factor[0]*e.length_m/e.speed_limit_mps)
    before = view.shortest_path("N0", "N1")
    assert view.shortest_path("N0", "N1") is before
    factor[0] = 2
    view.invalidate(cost_version="c2")
    after = view.shortest_path("N0", "N1")
    assert after.travel_time_s == 2*before.travel_time_s
    assert after.distance_m == before.distance_m
