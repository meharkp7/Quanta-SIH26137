from __future__ import annotations

from app.server import app


def test_server_exposes_current_core_routes():
    paths = {route.path for route in app.routes}
    assert "/api/health" in paths
    assert "/api/meta" in paths
    assert "/api/scenarios" in paths
    assert "/api/scenarios/{scenario_id}/closeable-edges" in paths
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
    assert "/" in paths


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
