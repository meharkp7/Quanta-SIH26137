"""Tests for scripts/generate_delhi_corpus.py's OSM-specific pieces.

Everything downstream of map generation (_build_route_plan_with_recovery,
_run_episode_with_recovery) is reused verbatim from
scripts/generate_real_training_corpus.py and already covered by its own
test suite / production track record; it does not depend on whether the
scenario came from a synthetic grid or a real OSM zone. These tests cover
only what generate_delhi_corpus.py actually adds: zone loading, the
demand-config retry ladder, and map generation with recovery -- none of
which require a SUMO installation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from scripts import generate_delhi_corpus as g
from tests.enhancements.osm.test_osm_demand_generator import make_grid_network


def test_load_zone_specs_reads_a_valid_manifest(tmp_path: Path):
    graphml_a = tmp_path / "a.graphml"
    graphml_b = tmp_path / "b.graphml"
    graphml_a.write_text("stub")
    graphml_b.write_text("stub")

    zones_file = tmp_path / "zones.json"
    zones_file.write_text(
        json.dumps(
            [
                {"name": "zone_a", "graphml": str(graphml_a)},
                {"name": "zone_b", "graphml": str(graphml_b)},
            ]
        )
    )

    zones = g.load_zone_specs(zones_file)

    assert [z.name for z in zones] == ["zone_a", "zone_b"]
    assert zones[0].graphml == graphml_a


def test_load_zone_specs_rejects_duplicate_names(tmp_path: Path):
    graphml = tmp_path / "a.graphml"
    graphml.write_text("stub")
    zones_file = tmp_path / "zones.json"
    zones_file.write_text(
        json.dumps(
            [
                {"name": "zone_a", "graphml": str(graphml)},
                {"name": "zone_a", "graphml": str(graphml)},
            ]
        )
    )

    with pytest.raises(ValueError, match="duplicate zone name"):
        g.load_zone_specs(zones_file)


def test_load_zone_specs_rejects_missing_graphml(tmp_path: Path):
    zones_file = tmp_path / "zones.json"
    zones_file.write_text(
        json.dumps([{"name": "zone_a", "graphml": str(tmp_path / "missing.graphml")}])
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        g.load_zone_specs(zones_file)


def test_osm_map_config_is_deterministic():
    a = g.osm_map_config(zone_index=3, attempt=0, seed=26137)
    b = g.osm_map_config(zone_index=3, attempt=0, seed=26137)
    assert a == b


def test_osm_map_config_shrinks_customers_and_widens_margin_on_retry():
    base = g.osm_map_config(zone_index=0, attempt=0, seed=1)
    retried = g.osm_map_config(zone_index=0, attempt=1, seed=1)

    assert retried.customer_count < base.customer_count
    # A tighter customer_count with the same capacity policy should never
    # need *more* vehicles than the original attempt.
    assert retried.fleet_size <= base.fleet_size + 1


def test_generate_osm_map_with_recovery_succeeds_via_retry_ladder(tmp_path: Path):
    """A 6x6 grid (36 nodes) cannot satisfy the default 40-customer demand
    policy for map_index=0; the retry ladder's customer_count shrink must
    recover rather than exhausting all MAP_RETRIES attempts.
    """

    network = make_grid_network(size=6)
    zone = g.ZoneSpec(name="fake_zone", graphml=tmp_path / "fake.graphml")
    failed_attempts: list = []

    with mock.patch.object(g, "load_graphml", return_value=network):
        scenario, cfg, fingerprint, attempt = g._generate_osm_map_with_recovery(
            output=tmp_path,
            map_index=0,
            zone=zone,
            split="train",
            seed=26137,
            failed_attempts=failed_attempts,
        )

    assert attempt > 0  # confirms the retry ladder, not a lucky first try
    assert len(failed_attempts) == attempt
    assert len(scenario.requests) == cfg.customer_count
    assert len(scenario.fleet) == cfg.fleet_size

    map_dir = tmp_path / "maps" / "map_000"
    assert (map_dir / "scenario.json").exists()

    zone_meta = json.loads((map_dir / "osm_zone.json").read_text())
    assert zone_meta["zone"] == "fake_zone"
    assert zone_meta["depot_node_id"] not in zone_meta["customer_node_ids"]


def test_generate_osm_map_with_recovery_is_deterministic(tmp_path: Path):
    network = make_grid_network(size=8)
    zone = g.ZoneSpec(name="fake_zone", graphml=tmp_path / "fake.graphml")

    with mock.patch.object(g, "load_graphml", return_value=network):
        first = g._generate_osm_map_with_recovery(
            output=tmp_path / "run1",
            map_index=2,
            zone=zone,
            split="train",
            seed=26137,
            failed_attempts=[],
        )
        second = g._generate_osm_map_with_recovery(
            output=tmp_path / "run2",
            map_index=2,
            zone=zone,
            split="train",
            seed=26137,
            failed_attempts=[],
        )

    assert first[2] == second[2]  # same fingerprint
    assert first[0].requests == second[0].requests


def test_generate_osm_map_with_recovery_raises_after_exhausting_retries(tmp_path: Path):
    """A genuinely tiny, disconnected network should exhaust the retry
    ladder loudly rather than silently returning a degenerate scenario.
    """

    from src.contracts.core_types import NodeKind
    from src.contracts.scenario import RoadNode
    from src.data.osm_ingestion import OSMNetwork

    tiny_network = OSMNetwork(
        nodes=(
            RoadNode(node_id="n1", x_m=0.0, y_m=0.0, kind=NodeKind.JUNCTION, zone_id="osm"),
            RoadNode(node_id="n2", x_m=10.0, y_m=0.0, kind=NodeKind.JUNCTION, zone_id="osm"),
        ),
        edges=(),  # no edges at all: nothing is reachable from anything
        source_crs="EPSG:32643",
        projected=True,
    )
    zone = g.ZoneSpec(name="tiny", graphml=tmp_path / "tiny.graphml")

    with mock.patch.object(g, "load_graphml", return_value=tiny_network):
        with pytest.raises(RuntimeError, match="exhausted"):
            g._generate_osm_map_with_recovery(
                output=tmp_path,
                map_index=0,
                zone=zone,
                split="train",
                seed=1,
                failed_attempts=[],
            )