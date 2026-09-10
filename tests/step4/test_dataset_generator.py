import json
from pathlib import Path

from src.data.dataset_generator import GeneratorConfig, generate_from_vrp, generate_scenario
from src.data.vrp_parser import parse_vrp


TINY_VRP = """NAME : step4_tiny\nTYPE : CVRP\nCOMMENT : 2 vehicles\nDIMENSION : 6\nEDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : 7\nNODE_COORD_SECTION\n1 0 0\n2 100 0\n3 200 50\n4 300 0\n5 400 50\n6 500 0\nDEMAND_SECTION\n1 0\n2 2\n3 2\n4 3\n5 1\n6 1\nDEPOT_SECTION\n1\n-1\nEOF\n"""


def test_parse_vrp(tmp_path: Path) -> None:
    path = tmp_path / "tiny.vrp"
    path.write_text(TINY_VRP, encoding="utf-8")
    instance = parse_vrp(path)
    assert instance.dimension == 6
    assert instance.depot_id == "1"
    assert instance.vehicle_count == 2
    assert sum(record.demand for record in instance.customers[1:]) == 9
    assert len(instance.source_checksum) == 64


def test_step4_generates_reproducibly(tmp_path: Path) -> None:
    source = tmp_path / "tiny.vrp"
    source.write_text(TINY_VRP, encoding="utf-8")
    config = GeneratorConfig(
        junction_count=49,
        grid_rows=7,
        grid_cols=7,
        max_customer_snap_m=500,
        one_way_fraction=0.1,
        seed=26137,
    )
    instance = parse_vrp(source)
    first = generate_scenario(instance, config)
    second = generate_scenario(instance, config)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert len(first.nodes) >= 49
    assert len(first.edges) > 0
    assert len(first.requests) == 5
    assert all(request.access_distance_m >= 0 for request in first.requests)
    assert all(request.latest_service_start_s >= request.earliest_service_start_s for request in first.requests)
    assert first.coordinate_transform is not None


def test_step4_writes_required_static_tables(tmp_path: Path) -> None:
    source = tmp_path / "tiny.vrp"
    output = tmp_path / "dataset"
    source.write_text(TINY_VRP, encoding="utf-8")
    generate_from_vrp(
        source,
        output,
        GeneratorConfig(junction_count=49, grid_rows=7, grid_cols=7, seed=26137),
    )
    for filename in (
        "scenario.json",
        "nodes.csv",
        "edges.csv",
        "requests.csv",
        "fleet.csv",
        "provenance_quality.json",
        "tiny.vrp",
    ):
        assert (output / filename).exists()


def test_dynamic_generator_is_reproducible_and_seed_sensitive(tmp_path):
    from src.data.dataset_generator import DynamicDatasetConfig, generate_dynamic_dataset

    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    config = DynamicDatasetConfig(customer_count=40, road_junction_count=100, seed=777)
    generate_dynamic_dataset(a, config)
    generate_dynamic_dataset(b, config)
    generate_dynamic_dataset(c, DynamicDatasetConfig(customer_count=40, road_junction_count=100, seed=778))

    assert (a / "scenario.json").read_bytes() == (b / "scenario.json").read_bytes()
    assert (a / "scenario.json").read_bytes() != (c / "scenario.json").read_bytes()


def test_dynamic_dataset_quality(tmp_path):
    from src.data.dataset_generator import DynamicDatasetConfig, generate_dynamic_dataset

    output = tmp_path / "dynamic"
    generate_dynamic_dataset(output, DynamicDatasetConfig(customer_count=60, road_junction_count=121, seed=2026))
    report = json.loads((output / "provenance_quality.json").read_text())
    scenario = json.loads((output / "scenario.json").read_text())

    assert report["generation_mode"] == "fully_dynamic_synthetic"
    assert all(report["quality_checks"].values())
    assert report["customer_count"] == 60
    assert report["road_node_count"] >= 121
    assert len(scenario["requests"]) == 60
    assert len(scenario["edges"]) > len(scenario["nodes"])
