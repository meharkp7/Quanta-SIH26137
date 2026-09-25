"""Episode Explorer API tests — list/detail/frame/forecasts against the real corpus."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_episodes_list_indexes_corpus_manifest():
    response = client.get("/api/episodes", params={"limit": 5})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1750
    rows = payload["episodes"]
    assert len(rows) == 5
    row = rows[0]
    assert set(row) >= {"episode_id", "scenario_id", "split", "regime", "event_count", "duration_s", "interval_s"}
    assert all(r["episode_id"].startswith("ep-") for r in rows)


def test_episodes_filter_by_scenario_and_events():
    scenario = "connaught_place-map000"
    response = client.get(
        "/api/episodes",
        params={"scenario_id": scenario, "with_events": True, "limit": 3},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] > 0
    rows = payload["episodes"]
    assert len(rows) <= 3
    assert all(r["scenario_id"] == scenario for r in rows)
    assert all((r["event_count"] or 0) > 0 for r in rows)


def test_episode_detail_returns_alignment_and_events():
    detail = client.get("/api/episodes/ep-0002").json()  # known to carry events
    assert detail["episode_id"] == "ep-0002"
    assert detail["interval_s"] == 60
    assert detail["duration_s"] == 2400
    assert detail["edge_ids"], "canonical edge order must be present"
    assert len(detail["times"]) == detail["duration_s"] // detail["interval_s"]
    assert detail["times"] == sorted(detail["times"])
    assert detail["events"], "ep-0002 records runtime incidents"
    event = detail["events"][0]
    assert set(event) >= {"event_id", "event_type", "reveal_time_s", "effect_start_s", "effect_end_s"}
    # Frame arrays must align 1:1 with the canonical edge order.
    frame = client.get("/api/episodes/ep-0002/frame", params={"t": 600}).json()
    assert len(frame["ratio"]) == len(detail["edge_ids"])
    assert len(frame["observed"]) == len(detail["edge_ids"])
    assert len(frame["closed"]) == len(detail["edge_ids"])


def test_episode_frame_is_sparse_and_snaps_time():
    frame = client.get("/api/episodes/ep-0001/frame", params={"t": 1200}).json()
    assert frame["t"] == 1200
    summary = frame["summary"]
    assert summary["total"] > 1000
    # Corpus records speeds only on roads with traffic — sparse by design.
    assert 0 < summary["reporting"] < summary["total"]
    assert summary["mean_ratio"] is not None and 0 < summary["mean_ratio"] <= 2
    assert len(frame["vehicles"]) == summary["fleet"] + summary["bg"]
    assert all(v["kind"] in ("fleet", "bg") for v in frame["vehicles"])
    assert all(0 <= v["pct"] <= 1 for v in frame["vehicles"])
    # Time snaps to the nearest recorded step when off-grid.
    snapped = client.get("/api/episodes/ep-0001/frame", params={"t": 1237}).json()
    assert snapped["t"] in (1200, 1260)


def test_episode_frame_marks_recorded_closures():
    # ep-0003's recorded closure is active across t=780..1200 (750-1230 window).
    frame = client.get("/api/episodes/ep-0003/frame", params={"t": 900}).json()
    assert frame["summary"]["closed"] > 0, "ep-0003 records a closure at t=900"
    assert sum(frame["closed"]) == frame["summary"]["closed"]
    # Outside the effect window nothing is closed.
    quiet = client.get("/api/episodes/ep-0003/frame", params={"t": 600}).json()
    assert quiet["summary"]["closed"] == 0


def test_episode_forecasts_compare_pred_vs_truth():
    payload = client.get("/api/episodes/ep-0001/forecasts").json()
    issues = payload["issues"]
    assert len(issues) == 9
    issue = issues[0]
    assert issue["issued_at_s"] == pytest.approx(1020)
    assert issue["target_times_s"] == [1320.0, 1620.0, 1920.0]
    assert len(issue["pred_mean"]) == 3
    assert all(v is None or v > 0 for v in issue["pred_mean"])
    assert all(v is None or v > 0 for v in issue["pred_p10"])
    assert all(v is None or v > 0 for v in issue["pred_p90"])
    assert issue["truth_mean"], "realized truth must be attached to every issue"
    assert issue["valid_edges"][0] > 100
    assert payload["model_version"] == "persistence-v0"


def test_episode_unknown_or_malformed_ids_are_404():
    assert client.get("/api/episodes/ep-99999999").status_code == 404
    assert client.get("/api/episodes/not-an-episode").status_code == 404
    assert client.get("/api/episodes/ep-1/frame", params={"t": 60}).status_code == 404
    # Path traversal must be rejected before any filesystem access.
    assert client.get("/api/episodes/..%2F..%2Fetc").status_code in (404, 400)
