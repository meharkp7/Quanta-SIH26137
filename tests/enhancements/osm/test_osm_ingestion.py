from __future__ import annotations

import pytest

from src.data.osm_ingestion import OSMIngestionConfig, OSMIngestionError, graph_to_quanta


class FakeGraph:
    def __init__(self, projected: bool = True):
        self.graph = {"crs": _CRS(projected)}
        self._nodes = {
            2: {"x": 100.0, "y": 200.0, "highway": "traffic_signals"},
            1: {"x": 0.0, "y": 0.0},
        }
        self._edges = [
            (1, 2, "a", {"length": 100.0, "maxspeed": "50", "lanes": "2", "highway": "primary"}),
            (2, 1, "b", {"length": 100.0, "maxspeed": "30 mph", "lanes": ["1"], "highway": "residential"}),
        ]

    def nodes(self, data=False):
        return self._nodes.items() if data else self._nodes.keys()

    def edges(self, keys=False, data=False):
        if keys and data:
            return iter(self._edges)
        raise TypeError


class _CRS:
    def __init__(self, projected: bool):
        self.is_projected = projected

    def __str__(self):
        return "EPSG:32644" if self.is_projected else "EPSG:4326"


def test_converts_directed_edges_and_attributes():
    result = graph_to_quanta(FakeGraph())
    assert len(result.nodes) == 2
    assert len(result.edges) == 2
    assert result.edges[0].length_m == 100.0
    assert result.edges[0].lane_count == 2
    assert result.edges[0].road_class.value == "arterial"
    assert result.nodes[-1].node_id == "osm:n:2"
    assert result.nodes[-1].signalized is True


def test_rejects_unprojected_graph_by_default():
    with pytest.raises(OSMIngestionError, match="not projected"):
        graph_to_quanta(FakeGraph(projected=False))


class _StringCrsGraph(FakeGraph):
    """Mimics a GraphML round trip: crs comes back as a plain string."""

    def __init__(self, crs_string: str):
        super().__init__(projected=True)
        self.graph = {"crs": crs_string}


def test_accepts_string_valued_projected_crs():
    # As produced by osmnx save_graphml -> load_graphml, e.g. Delhi's UTM zone.
    result = graph_to_quanta(_StringCrsGraph("epsg:32643"))
    assert result.projected is True
    assert result.source_crs == "epsg:32643"


def test_rejects_string_valued_unprojected_crs():
    with pytest.raises(OSMIngestionError, match="not projected"):
        graph_to_quanta(_StringCrsGraph("epsg:4326"))


def test_rejects_unparseable_string_crs():
    with pytest.raises(OSMIngestionError, match="not projected"):
        graph_to_quanta(_StringCrsGraph("not-a-real-crs"))


def test_rejects_missing_crs():
    graph = FakeGraph()
    graph.graph = {}
    with pytest.raises(OSMIngestionError, match="not projected"):
        graph_to_quanta(graph)


class _SelfLoopGraph(FakeGraph):
    """A real OSM extract shape: one legal edge plus one self-loop artifact."""

    def __init__(self):
        super().__init__(projected=True)
        self._edges = self._edges + [
            (1, 1, "loop", {"length": 5.0, "maxspeed": "30", "highway": "residential"}),
        ]


def test_self_loop_edges_are_skipped_by_default():
    result = graph_to_quanta(_SelfLoopGraph())
    assert result.self_loop_edges_skipped == 1
    assert all(e.from_node != e.to_node for e in result.edges)
    # the two legal edges from the base fixture are unaffected
    assert len(result.edges) == 2


def test_self_loop_edges_can_be_rejected_instead():
    config = OSMIngestionConfig(skip_self_loop_edges=False)
    with pytest.raises(OSMIngestionError, match="self-loop"):
        graph_to_quanta(_SelfLoopGraph(), config=config)


def test_rejects_missing_length_when_required():
    graph = FakeGraph()
    graph._edges[0][3].pop("length")
    with pytest.raises(OSMIngestionError, match="no valid length"):
        graph_to_quanta(graph)


def test_allows_explicit_length_fallback_policy():
    graph = FakeGraph()
    graph._edges[0][3].pop("length")
    result = graph_to_quanta(graph, config=OSMIngestionConfig(require_length=False))
    assert result.edges[0].length_m == pytest.approx(223.60679774997897)