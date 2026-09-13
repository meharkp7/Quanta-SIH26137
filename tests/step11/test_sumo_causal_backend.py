from pathlib import Path

from src.contracts.scenario import Scenario
from src.data.dynamic_episodes import DynamicEpisodeConfig, DynamicEvent
from src.sim.sumo_causal import _RuntimeEventTape, _event_edges


ROOT = Path(__file__).resolve().parents[2]


def load_scenario():
    return Scenario.model_validate_json(
        (ROOT / "fixtures/step3/base/scenario.json").read_text(encoding="utf-8")
    )


def test_event_edges_resolve_parent_roads_to_contract_edges():
    scenario = load_scenario()
    parent = scenario.edges[0].parent_road_id
    event = DynamicEvent(
        event_id="EV",
        event_type="closure",
        generation_time_s=0,
        reveal_time_s=10,
        effect_start_s=20,
        effect_end_s=40,
        affected_parent_road_ids=(parent,),
        severity=1.0,
    )
    edges = _event_edges(scenario, event)
    assert edges
    assert all(edge_id in {e.edge_id for e in scenario.edges} for edge_id in edges)


def test_runtime_event_tape_reveals_then_activates_then_restores():
    scenario = load_scenario()
    edge = next(e for e in scenario.edges if e.parent_road_id == "R23")
    event = DynamicEvent(
        event_id="EV",
        event_type="incident",
        generation_time_s=0,
        reveal_time_s=10,
        effect_start_s=20,
        effect_end_s=40,
        affected_parent_road_ids=(edge.parent_road_id,),
        severity=0.8,
    )

    class Lane:
        def __init__(self):
            self.allowed = ["passenger"]
            self.disallowed = []
            self.speed = 13.0

    class Edge:
        def getLaneNumber(self, _):
            return 1

    class FakeTraci:
        def __init__(self):
            self.edge = Edge()
            self.lane = LaneAPI()

    class LaneAPI:
        def __init__(self):
            self.l = Lane()

        def getAllowed(self, _):
            return list(self.l.allowed)

        def getDisallowed(self, _):
            return list(self.l.disallowed)

        def getMaxSpeed(self, _):
            return self.l.speed

        def setMaxSpeed(self, _, value):
            self.l.speed = value

        def setAllowed(self, _, value):
            self.l.allowed = list(value)

        def setDisallowed(self, _, value):
            self.l.disallowed = list(value)

    tape = _RuntimeEventTape(
        scenario,
        [event],
        {e.edge_id: e.edge_id for e in scenario.edges},
    )
    traci = FakeTraci()

    assert tape.step(traci, 5.0) == []
    assert ("EV", "announced") in tape.step(traci, 10.0)
    active = tape.step(traci, 20.0)
    assert ("EV", "incident_active") in active
    assert traci.lane.l.speed < 13.0
    restored = tape.step(traci, 40.0)
    assert ("EV", "reopened") in restored
    assert traci.lane.l.speed == 13.0
