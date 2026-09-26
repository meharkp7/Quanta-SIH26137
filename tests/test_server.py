from __future__ import annotations

from app.server import app


def test_server_exposes_current_core_routes():
    # FastAPI >= 0.140 keeps a lazy `_IncludedRouter` entry (no `.path`) in
    # app.routes for include_router()'d sub-routers — skip wrapper objects.
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/api/health" in paths
    assert "/api/meta" in paths
    assert "/api/scenarios" in paths
    assert "/api/scenarios/{scenario_id}/closeable-edges" in paths
    assert "/api/episodes" in paths
    assert "/api/episodes/{episode_id}" in paths
    assert "/api/episodes/{episode_id}/frame" in paths
    assert "/api/episodes/{episode_id}/forecasts" in paths
    assert "/api/validate" in paths
    assert "/api/solve" in paths
    assert "/api/compare" in paths
    assert "/api/path" in paths
    assert "/api/loop" in paths
    assert "/api/sumo" in paths
    assert "/api/sumo/replay" in paths
    assert "/api/models/forecaster/status" in paths
    assert "/api/models/forecaster/train" in paths
    assert "/api/evidence" in paths
    assert "/api/here/status" in paths
    assert "/api/here/geocode" in paths
    assert "/api/here/traffic" in paths
    assert "/api/here/route" in paths
    assert "/" in paths


def test_here_layer_is_optional_and_never_fabricated():
    """The HERE endpoints answer honestly with or without a key."""
    from fastapi.testclient import TestClient

    client = TestClient(app)

    status = client.get("/api/here/status")
    assert status.status_code == 200
    body = status.json()
    assert isinstance(body["configured"], bool)
    if not body["configured"]:
        assert "HERE_API_KEY" in body["reason"]

    # Lenient read: a HERE problem must not break the graph payload.
    graph = client.get("/api/scenarios/S3_BASE", params={"live": "true"})
    assert graph.status_code == 200
    traffic = graph.json()["here_traffic"]
    assert traffic["requested"] is True
    assert traffic["applied"] is False  # synthetic fixture: nothing to overlay
    assert traffic["reason"]  # …and the reason says exactly why

    # HERE routing never returns a fake 200 — 409 (no key), 502 (HERE
    # down), or 400 (this fixture has no coordinates) are all honest.
    route = client.post("/api/here/route", json={"source": "N1", "target": "N2"})
    assert route.status_code in (400, 409, 502)
    assert route.json()["detail"]

    # Strict-by-default compute: asking for live traffic on a keyless
    # backend fails loudly instead of silently solving on static speeds.
    path = client.post(
        "/api/path", json={"source": "N1", "target": "N2", "live_traffic": True}
    )
    assert path.status_code in (400, 409)  # no key (409) or no geo map (400)
    assert "HERE" in path.json()["detail"]


def test_forecaster_status_reports_corpus_v2():
    from src.platform.service import PlatformService

    status = PlatformService().forecaster_status()
    assert status["corpus"]["manifest_present"] is True
    assert status["corpus"]["episodes"] == {
        "train": 1050,
        "validation": 350,
        "test": 350,
    }
    assert "CausalGNNTransformer" in status["model"]


def test_solve_methods_match_platform_service():
    from app.server import SolveMethod

    assert set(SolveMethod.__args__) == {
        "constructive",
        "qpso",
        "pso",
        "alns",
        "milp",
    }
