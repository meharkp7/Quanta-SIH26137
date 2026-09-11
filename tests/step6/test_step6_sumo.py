"""
tests/step6/test_step6_sumo.py
================================
Step 6 "done-when" verification tests.

These tests cover the acceptance criteria from the plan:
  ✓ Network export: nodes, edges, connection XML are valid and net.xml is produced
  ✓ Single-vehicle validation: one vehicle can traverse the fixture network
  ✓ SimState: delivery state machine transitions are correct
  ✓ IncidentManager: closure events are fired in the correct order
  ✓ No teleportation guard (structural check — full TraCI test needs SUMO running)

Tests that require a live SUMO process are marked with @pytest.mark.sumo
and are skipped unless SUMO is accessible.  All other tests are pure-Python
and run without SUMO.

Run all:
    pytest tests/step6/ -v

Run only SUMO-independent tests:
    pytest tests/step6/ -v -m "not sumo"

Run full integration (requires SUMO on PATH or SUMO_HOME set):
    pytest tests/step6/ -v -m sumo
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_BASE = PROJECT_ROOT / "fixtures" / "step3" / "base" / "scenario.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "step6_test"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_base_scenario():
    from contracts.scenario import Scenario
    data = json.loads(FIXTURE_BASE.read_text(encoding="utf-8"))
    return Scenario.model_validate(data)


def make_minimal_route_plan(scenario):
    """
    Build a minimal RoutePlan that sends V1 to J1→J2→J3 and V2 to J4→J5.

    Uses direct node-to-node edge paths from the fixture graph.
    """
    from contracts.routing import RoutePlan, VehicleRoute, StopLeg

    # V1 route: depot(N0) → J1(N1) → J2(N2) → J3(N3) → depot(N0)
    v1_legs = [
        StopLeg(
            from_stop_id="N0", to_stop_id="N1",
            physical_edge_ids=("E01",),
            travel_time_s=10.0, distance_m=100.0, congestion_exposure=0.0,
            departure_time_s=0.0, arrival_time_s=10.0,
        ),
        StopLeg(
            from_stop_id="N1", to_stop_id="N2",
            physical_edge_ids=("E12",),
            travel_time_s=10.0, distance_m=100.0, congestion_exposure=0.0,
            departure_time_s=20.0, arrival_time_s=30.0,
        ),
        StopLeg(
            from_stop_id="N2", to_stop_id="N3",
            physical_edge_ids=("E23",),
            travel_time_s=10.0, distance_m=100.0, congestion_exposure=0.0,
            departure_time_s=45.0, arrival_time_s=55.0,
        ),
        StopLeg(
            from_stop_id="N3", to_stop_id="N0",
            physical_edge_ids=("E34", "E45", "E51", "E10"),
            travel_time_s=40.0, distance_m=400.0, congestion_exposure=0.0,
            departure_time_s=70.0, arrival_time_s=110.0,
        ),
    ]

    # V2 route: depot(N0) → J4(N4) → J5(N5) → depot(N0)
    v2_legs = [
        StopLeg(
            from_stop_id="N0", to_stop_id="N4",
            physical_edge_ids=("E01", "E12", "E23", "E34"),
            travel_time_s=40.0, distance_m=400.0, congestion_exposure=0.0,
            departure_time_s=0.0, arrival_time_s=40.0,
        ),
        StopLeg(
            from_stop_id="N4", to_stop_id="N5",
            physical_edge_ids=("E45",),
            travel_time_s=10.0, distance_m=100.0, congestion_exposure=0.0,
            departure_time_s=50.0, arrival_time_s=60.0,
        ),
        StopLeg(
            from_stop_id="N5", to_stop_id="N0",
            physical_edge_ids=("E51", "E10"),
            travel_time_s=20.0, distance_m=200.0, congestion_exposure=0.0,
            departure_time_s=70.0, arrival_time_s=90.0,
        ),
    ]

    v1 = VehicleRoute(
        vehicle_id="V1",
        customer_order=("J1", "J2", "J3"),
        legs=tuple(v1_legs),
        expected_departure_s=0.0,
        expected_return_s=110.0,
        expected_driving_time_s=70.0,
        expected_waiting_time_s=0.0,
        expected_service_time_s=35.0,
        served_request_ids=("J1", "J2", "J3"),
        frozen_prefix_edge_ids=(),
    )
    v2 = VehicleRoute(
        vehicle_id="V2",
        customer_order=("J4", "J5"),
        legs=tuple(v2_legs),
        expected_departure_s=0.0,
        expected_return_s=90.0,
        expected_driving_time_s=70.0,
        expected_waiting_time_s=0.0,
        expected_service_time_s=20.0,
        served_request_ids=("J4", "J5"),
        frozen_prefix_edge_ids=(),
    )

    return RoutePlan(
        scenario_id="S3_BASE",
        state_version="v1",
        route_version="rv1",
        generated_at_s=0.0,
        vehicle_routes=(v1, v2),
        objective_value=0.0,
        objective_time=180.0,
        objective_distance=900.0,
        objective_congestion=0.0,
        route_change_penalty=0.0,
        changed_vehicle_ids=("V1", "V2"),
        generated_by="step6_test_fixture",
    )


# ===========================================================================
# Pure-Python tests (no SUMO process needed)
# ===========================================================================

class TestSumoExporterXml:
    """Validate that the exporter produces well-formed XML with correct IDs."""

    def test_nodes_xml_has_all_nodes(self, tmp_path):
        from sim.sumo_exporter import SumoExporter
        scenario = load_base_scenario()
        exp = SumoExporter(scenario, output_dir=tmp_path)

        # Patch netconvert to not actually run
        with patch("src.sim.sumo_exporter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            nod_path = exp._write_nodes()

        import xml.etree.ElementTree as ET
        tree = ET.parse(nod_path)
        node_ids = {el.get("id") for el in tree.getroot().findall("node")}
        expected = {n.node_id for n in scenario.nodes}
        assert node_ids == expected, f"Missing node IDs: {expected - node_ids}"

    def test_edges_xml_has_all_edges(self, tmp_path):
        from sim.sumo_exporter import SumoExporter
        scenario = load_base_scenario()
        exp = SumoExporter(scenario, output_dir=tmp_path)
        exp._write_nodes()   # populate node_mapping first
        edg_path = exp._write_edges()

        import xml.etree.ElementTree as ET
        tree = ET.parse(edg_path)
        edge_ids = {el.get("id") for el in tree.getroot().findall("edge")}
        expected = {e.edge_id for e in scenario.edges}
        assert edge_ids == expected, f"Missing edge IDs: {expected - edge_ids}"

    def test_edge_mapping_is_identity(self, tmp_path):
        """SUMO edge IDs must equal contract edge IDs (our naming convention)."""
        from sim.sumo_exporter import SumoExporter
        scenario = load_base_scenario()
        exp = SumoExporter(scenario, output_dir=tmp_path)
        exp._write_nodes()
        exp._write_edges()
        for contract_id, sumo_id in exp.edge_mapping.items():
            assert contract_id == sumo_id, (
                f"Edge mapping is not identity: {contract_id!r} -> {sumo_id!r}"
            )

    def test_connections_no_u_turns(self, tmp_path):
        """Connections XML must not contain any U-turn connections."""
        from sim.sumo_exporter import SumoExporter
        import xml.etree.ElementTree as ET

        scenario = load_base_scenario()
        exp = SumoExporter(scenario, output_dir=tmp_path)
        exp._write_nodes()
        exp._write_edges()
        con_path = exp._write_connections()

        # Build a from_node/to_node lookup by edge_id
        edge_map = {e.edge_id: e for e in scenario.edges}

        tree = ET.parse(con_path)
        for conn in tree.getroot().findall("connection"):
            from_eid = conn.get("from")
            to_eid = conn.get("to")
            if from_eid in edge_map and to_eid in edge_map:
                in_edge = edge_map[from_eid]
                out_edge = edge_map[to_eid]
                assert in_edge.from_node != out_edge.to_node, (
                    f"U-turn connection found: {from_eid} -> {to_eid}"
                )


class TestRouteBuilder:
    """Verify that RouteBuilder produces valid XML with correct stop structure."""

    def test_route_xml_has_both_vehicles(self, tmp_path):
        from sim.route_builder import RouteBuilder
        scenario = load_base_scenario()
        route_plan = make_minimal_route_plan(scenario)
        mapping = {
            "edge_mapping": {e.edge_id: e.edge_id for e in scenario.edges},
            "node_mapping": {n.node_id: n.node_id for n in scenario.nodes},
        }
        builder = RouteBuilder(scenario, route_plan, mapping)
        rou_path = builder.build(output_dir=tmp_path)

        import xml.etree.ElementTree as ET
        tree = ET.parse(rou_path)
        vehicle_ids = {el.get("id") for el in tree.getroot().findall("vehicle")}
        assert "V1" in vehicle_ids
        assert "V2" in vehicle_ids

    def test_stops_have_trip_ids(self, tmp_path):
        from sim.route_builder import RouteBuilder
        import xml.etree.ElementTree as ET

        scenario = load_base_scenario()
        route_plan = make_minimal_route_plan(scenario)
        mapping = {
            "edge_mapping": {e.edge_id: e.edge_id for e in scenario.edges},
            "node_mapping": {n.node_id: n.node_id for n in scenario.nodes},
        }
        builder = RouteBuilder(scenario, route_plan, mapping)
        rou_path = builder.build(output_dir=tmp_path)

        tree = ET.parse(rou_path)
        trip_ids = {
            stop.get("tripId")
            for veh in tree.getroot().findall("vehicle")
            for stop in veh.findall("stop")
            if stop.get("tripId")
        }
        # All customer IDs should appear as stop tripIds
        for request_id in ("J1", "J2", "J3", "J4", "J5"):
            assert request_id in trip_ids, f"{request_id} missing from stop tripIds"

    def test_background_trips_present(self, tmp_path):
        from sim.route_builder import RouteBuilder
        import xml.etree.ElementTree as ET

        scenario = load_base_scenario()
        route_plan = make_minimal_route_plan(scenario)
        mapping = {
            "edge_mapping": {e.edge_id: e.edge_id for e in scenario.edges},
            "node_mapping": {n.node_id: n.node_id for n in scenario.nodes},
        }
        builder = RouteBuilder(scenario, route_plan, mapping)
        rou_path = builder.build(output_dir=tmp_path)

        tree = ET.parse(rou_path)
        trips = tree.getroot().findall("trip")
        assert len(trips) >= 1, "Expected at least one background trip"


class TestSimState:
    """Verify the delivery state machine transitions."""

    def test_initial_state_all_pending(self):
        from sim.sim_state import SimState, RequestState
        scenario = load_base_scenario()
        state = SimState(scenario)
        for snap in state.snapshot_requests():
            assert snap.state == RequestState.PENDING

    def test_begin_service_transitions_to_onboard(self):
        from sim.sim_state import SimState, RequestState
        scenario = load_base_scenario()
        state = SimState(scenario)
        state.begin_service("V1", "J1", sim_time_s=10.0)
        snaps = {s.request_id: s for s in state.snapshot_requests()}
        assert snaps["J1"].state == RequestState.ONBOARD
        assert snaps["J1"].assigned_vehicle_id == "V1"
        assert snaps["J1"].service_start_s == 10.0

    def test_complete_service_transitions_to_delivered(self):
        from sim.sim_state import SimState, RequestState
        scenario = load_base_scenario()
        state = SimState(scenario)
        state.begin_service("V1", "J1", sim_time_s=10.0)
        state.complete_service("V1", "J1", sim_time_s=20.0)
        snaps = {s.request_id: s for s in state.snapshot_requests()}
        assert snaps["J1"].state == RequestState.DELIVERED
        assert snaps["J1"].service_end_s == 20.0

    def test_no_duplicate_service(self):
        """begin_service on already-ONBOARD request is a no-op (idempotent)."""
        from sim.sim_state import SimState, RequestState
        scenario = load_base_scenario()
        state = SimState(scenario)
        state.begin_service("V1", "J1", sim_time_s=10.0)
        # Call again — should not raise, should stay ONBOARD
        state.begin_service("V1", "J1", sim_time_s=11.0)
        snaps = {s.request_id: s for s in state.snapshot_requests()}
        assert snaps["J1"].state == RequestState.ONBOARD
        # service_start_s should still be the first call's value
        assert snaps["J1"].service_start_s == 10.0

    def test_capacity_guard(self):
        """Loading more than capacity is blocked — vehicle load stays consistent."""
        from sim.sim_state import SimState, RequestState
        scenario = load_base_scenario()
        state = SimState(scenario)
        # V2 has capacity 6. J1(2) + J2(2) + J3(3) = 7 > 6
        state.begin_service("V2", "J1", sim_time_s=5.0)
        state.begin_service("V2", "J2", sim_time_s=15.0)
        # J3 demand=3, would push load to 7 > capacity 6 → should be blocked
        state.begin_service("V2", "J3", sim_time_s=25.0)
        snaps = {s.request_id: s for s in state.snapshot_requests()}
        # J3 should remain PENDING (not accepted)
        assert snaps["J3"].state == RequestState.PENDING

    def test_late_service_flagged(self):
        from src.sim.sim_state import SimState
        scenario = load_base_scenario()
        state = SimState(scenario)
        # J1 latest_service_start_s = 30, serve it at t=50 → was_late
        state.begin_service("V1", "J1", sim_time_s=50.0)
        snaps = {s.request_id: s for s in state.snapshot_requests()}
        assert snaps["J1"].was_late is True

    def test_all_delivered_true_when_all_done(self):
        from src.sim.sim_state import SimState
        scenario = load_base_scenario()
        state = SimState(scenario)
        t = 0.0
        for req in scenario.requests:
            state.begin_service("V1", req.request_id, sim_time_s=t)
            state.complete_service("V1", req.request_id, sim_time_s=t + req.service_duration_s)
            t += 20.0
        assert state.all_delivered()

    def test_vehicle_load_decrements_on_delivery(self):
        from src.sim.sim_state import SimState
        scenario = load_base_scenario()
        state = SimState(scenario)
        state.begin_service("V1", "J1", sim_time_s=10.0)   # demand=2
        state.begin_service("V1", "J2", sim_time_s=30.0)   # demand=2
        veh = {v.vehicle_id: v for v in state.snapshot_vehicles()}
        assert veh["V1"].remaining_load == pytest.approx(4.0)

        state.complete_service("V1", "J1", sim_time_s=20.0)
        veh = {v.vehicle_id: v for v in state.snapshot_vehicles()}
        assert veh["V1"].remaining_load == pytest.approx(2.0)


class TestIncidentManager:
    """Verify road-closure event sequencing without live SUMO."""

    def _make_mock_traci(self, vehicles_on_edge: list[str] | None = None):
        """Create a mock traci object."""
        mock = MagicMock()
        mock.edge.getLaneNumber.return_value = 1
        mock.edge.getLastStepVehicleIDs.return_value = vehicles_on_edge or []
        mock.lane.setMaxSpeed.return_value = None
        return mock

    def test_no_events_before_announce_time(self):
        from sim.incidents import IncidentManager, IncidentConfig
        cfg = IncidentConfig(
            incident_id="INC_TEST",
            edge_id="E23",
            trigger_time_s=50.0,
            announce_lead_s=10.0,
        )
        mgr = IncidentManager([cfg], edge_mapping={"E23": "E23"})
        mock_traci = self._make_mock_traci()

        events = mgr.step(mock_traci, sim_time_s=30.0)
        assert events == []

    def test_announced_event_fires_at_announce_time(self):
        from sim.incidents import IncidentManager, IncidentConfig
        cfg = IncidentConfig(
            incident_id="INC_TEST",
            edge_id="E23",
            trigger_time_s=50.0,
            announce_lead_s=10.0,
        )
        mgr = IncidentManager([cfg], edge_mapping={"E23": "E23"})
        mock_traci = self._make_mock_traci()

        events = mgr.step(mock_traci, sim_time_s=40.0)
        event_types = [e[1] for e in events]
        assert "announced" in event_types

    def test_closed_event_fires_at_trigger_time(self):
        from sim.incidents import IncidentManager, IncidentConfig
        cfg = IncidentConfig(
            incident_id="INC_TEST",
            edge_id="E23",
            trigger_time_s=50.0,
            announce_lead_s=0.0,
        )
        mgr = IncidentManager([cfg], edge_mapping={"E23": "E23"})
        mock_traci = self._make_mock_traci(vehicles_on_edge=[])

        events = mgr.step(mock_traci, sim_time_s=50.0)
        event_types = [e[1] for e in events]
        assert "closed" in event_types

    def test_lane_permissions_block_new_entry_on_close(self):
        from sim.incidents import IncidentManager, IncidentConfig
        cfg = IncidentConfig(
            incident_id="INC_TEST",
            edge_id="E23",
            trigger_time_s=50.0,
            announce_lead_s=0.0,
        )
        mgr = IncidentManager([cfg], edge_mapping={"E23": "E23"})
        mock_traci = self._make_mock_traci(vehicles_on_edge=[])

        mgr.step(mock_traci, sim_time_s=50.0)
        mock_traci.lane.setAllowed.assert_called_once_with("E23_0", ["authority"])
        mock_traci.lane.setMaxSpeed.assert_not_called()

    def test_clearing_event_when_vehicles_still_on_edge(self):
        from sim.incidents import IncidentManager, IncidentConfig
        cfg = IncidentConfig(
            incident_id="INC_TEST",
            edge_id="E23",
            trigger_time_s=50.0,
            announce_lead_s=0.0,
        )
        mgr = IncidentManager([cfg], edge_mapping={"E23": "E23"})
        # Close with a vehicle on the edge
        mock_traci = self._make_mock_traci(vehicles_on_edge=["V1"])

        events = mgr.step(mock_traci, sim_time_s=50.0)
        event_types = [e[1] for e in events]
        assert "clearing" in event_types

    def test_cleared_event_when_edge_becomes_empty(self):
        from sim.incidents import IncidentManager, IncidentConfig
        cfg = IncidentConfig(
            incident_id="INC_TEST",
            edge_id="E23",
            trigger_time_s=50.0,
            announce_lead_s=0.0,
        )
        mgr = IncidentManager([cfg], edge_mapping={"E23": "E23"})

        # Step 1: close with vehicle on edge
        mock_with_veh = self._make_mock_traci(vehicles_on_edge=["V1"])
        mgr.step(mock_with_veh, sim_time_s=50.0)

        # Step 2: vehicle clears
        mock_empty = self._make_mock_traci(vehicles_on_edge=[])
        events = mgr.step(mock_empty, sim_time_s=55.0)
        event_types = [e[1] for e in events]
        assert "cleared" in event_types

    def test_is_edge_closed_reflects_state(self):
        from sim.incidents import IncidentManager, IncidentConfig
        cfg = IncidentConfig(
            incident_id="INC_TEST",
            edge_id="E23",
            trigger_time_s=50.0,
            announce_lead_s=0.0,
        )
        mgr = IncidentManager([cfg], edge_mapping={"E23": "E23"})
        mock_traci = self._make_mock_traci(vehicles_on_edge=[])

        assert not mgr.is_edge_closed("E23")
        mgr.step(mock_traci, sim_time_s=50.0)
        assert mgr.is_edge_closed("E23")

    def test_step6_closure_config_targets_e23(self):
        from src.sim.incidents import STEP6_CLOSURE_CONFIG
        assert STEP6_CLOSURE_CONFIG.edge_id == "E23"
        assert STEP6_CLOSURE_CONFIG.trigger_time_s == 50.0
        assert STEP6_CLOSURE_CONFIG.announce_lead_s == 10.0


# ===========================================================================
# Integration test requiring SUMO (marked, skipped if not available)
# ===========================================================================

def _sumo_available() -> bool:
    """Return True when a SUMO executable is available cross-platform."""
    import shutil

    if shutil.which("sumo") or shutil.which("sumo.exe"):
        return True

    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        return False

    home = Path(sumo_home)
    return any(
        (home / "bin" / name).is_file()
        for name in ("sumo", "sumo.exe")
    )


@pytest.mark.sumo
@pytest.mark.skipif(not _sumo_available(), reason="SUMO not installed / SUMO_HOME not set")
class TestFullEpisodeIntegration:
    """
    Full integration test: runs a real SUMO episode with the step-3 fixture.

    Verifies the done-when criteria:
      * Both vehicles execute without teleportation
      * Application state matches SUMO service events
      * Road closure fires correctly at t=50s
    """

    def test_episode_runs_without_teleportation(self, tmp_path):
        from sim.sumo_runner import run_episode
        scenario = load_base_scenario()
        route_plan = make_minimal_route_plan(scenario)

        result = run_episode(
            scenario,
            route_plan,
            output_dir=tmp_path / "episode",
            end_time_s=300.0,
            step_length_s=1.0,
        )

        assert result.teleport_events == 0, (
            f"Teleportation detected! ({result.teleport_events} events). "
            "Check route edges and network turn connections."
        )

    def test_episode_delivers_requests(self, tmp_path):
        from sim.sumo_runner import run_episode
        scenario = load_base_scenario()
        route_plan = make_minimal_route_plan(scenario)

        result = run_episode(
            scenario,
            route_plan,
            output_dir=tmp_path / "episode",
            end_time_s=300.0,
        )

        assert result.delivered_count == 5
        assert result.service_begin_events == result.service_complete_events == 5
        assert all(v["state"] == "finished" for v in result.vehicle_summaries)
        # Total must account for all 5 requests
        total = result.delivered_count + result.failed_count + result.pending_count
        assert total == 5

    def test_closure_event_fires(self, tmp_path):
        from sim.sumo_runner import run_episode
        scenario = load_base_scenario()
        route_plan = make_minimal_route_plan(scenario)

        result = run_episode(
            scenario,
            route_plan,
            output_dir=tmp_path / "episode",
            end_time_s=300.0,
        )

        incident_types = {evt[1] for evt in result.incident_events}
        assert "closed" in incident_types, (
            "Road closure event was never fired. "
            f"Events seen: {result.incident_events}"
        )
        # Edge E23 closure should fire at or after t=50s
        closed_events = [
            evt for evt in result.incident_events if evt[1] == "closed"
        ]
        assert len(closed_events) == 1

