import shutil

import pytest

from src.runtime.loop import DemoLoop


def test_step10_builds_and_commits_a_valid_plan_without_manual_repair():
    state = DemoLoop().run(
        "S3_BASE",
        particles=4,
        evaluations=12,
        seed=3,
        execute_sumo=False,
    )

    assert state.forecast_mode == "persistence"
    assert state.scope_action == "KEEP"
    assert state.committed is True
    assert state.state_version.startswith("S3_BASE:t0:graph:")
    assert state.route_version == "step10-3-t0"
    assert state.solve["evaluation"]["feasible"] is True
    assert state.sumo_episode is None
    assert any("Plan committed" in note for note in state.notes)


@pytest.mark.sumo
def test_step10_runs_the_committed_plan_through_sumo(tmp_path):
    if shutil.which("sumo") is None:
        pytest.skip("SUMO executable is not installed")
    try:
        import traci  # noqa: F401
    except ImportError:
        pytest.skip("TraCI is not installed")

    state = DemoLoop().run(
        "S3_BASE",
        particles=4,
        evaluations=12,
        seed=3,
        execute_sumo=True,
        output_dir=tmp_path / "step10",
    )

    assert state.committed is True
    assert state.sumo_episode is not None
    assert state.sumo_episode["delivered_count"] == 5
    assert state.sumo_episode["teleport_events"] == 0
    assert any("SUMO advanced the committed plan" in note for note in state.notes)
