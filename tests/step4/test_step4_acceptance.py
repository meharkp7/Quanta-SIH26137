from __future__ import annotations

from pathlib import Path

from src.data.dataset_generator import (
    GeneratorConfig,
    build_quality_report,
    generate_from_vrp,
    generate_scenario,
)
from src.data.vrp_parser import parse_vrp


TINY_VRP = """NAME : step4_acceptance
TYPE : CVRP
COMMENT : 2 vehicles
DIMENSION : 7
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 6
NODE_COORD_SECTION
1 0 0
2 100 0
3 200 50
4 300 0
5 400 50
6 500 0
7 250 200
DEMAND_SECTION
1 0
2 2
3 2
4 3
5 1
6 1
7 2
DEPOT_SECTION
1
-1
EOF
"""


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "acceptance.vrp"
    source.write_text(
        TINY_VRP,
        encoding="utf-8",
    )
    return source


def test_reference_schedule_is_independently_evaluated(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    scenario = generate_scenario(
        parse_vrp(source),
        GeneratorConfig(
            junction_count=64,
            grid_rows=8,
            grid_cols=8,
            seed=26137,
        ),
    )

    report = build_quality_report(
        scenario,
        parse_vrp(source),
    )

    evaluation = report[
        "reference_schedule_evaluation"
    ]

    assert evaluation["feasible"] is True
    assert evaluation["errors"] == []


def test_infeasible_stress_cases_are_explicitly_labelled(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    for stress_case in (
        "missed_window",
        "disconnected",
    ):
        output = (
            tmp_path / stress_case
        )

        generate_from_vrp(
            source,
            output,
            GeneratorConfig(
                junction_count=64,
                grid_rows=8,
                grid_cols=8,
                seed=26137,
                stress_case=stress_case,
            ),
        )

        import json

        report = json.loads(
            (
                output
                / "provenance_quality.json"
            ).read_text()
        )

        assert (
            report["stress_case"]
            == stress_case
        )

        assert (
            report["expected_feasible"]
            is False
        )

        assert (
            report[
                "reference_schedule_feasible"
            ]
            is False
        )


def test_required_step4_quality_checks_are_reported(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "normal"

    generate_from_vrp(
        source,
        output,
        GeneratorConfig(
            junction_count=64,
            grid_rows=8,
            grid_cols=8,
            seed=26137,
        ),
    )

    import json

    report = json.loads(
        (
            output
            / "provenance_quality.json"
        ).read_text()
    )

    assert (
        report["expected_feasible"]
        is True
    )

    assert (
        report["stress_case"]
        == "none"
    )

    assert (
        report[
            "reference_schedule_feasible"
        ]
        is True
    )

    assert report[
        "unreachable_requests"
    ] == []

    assert report[
        "disconnected_nodes"
    ] == []

    assert (
        report["customer_count"]
        == 6
    )

    assert (
        report["total_customer_demand"]
        == 11.0
    )

    assert report[
        "reference_schedule_evaluation"
    ]["errors"] == []


def test_original_coordinates_and_snap_distances_are_preserved(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    scenario = generate_scenario(
        parse_vrp(source),
        GeneratorConfig(
            junction_count=64,
            grid_rows=8,
            grid_cols=8,
            seed=26137,
        ),
    )

    by_original = {
        request.original_customer_id: request
        for request in scenario.requests
    }

    assert (
        by_original["2"].original_x
        == 100.0
    )

    assert (
        by_original["2"].original_y
        == 0.0
    )

    assert (
        by_original["2"].access_distance_m
        >= 0.0
    )

    assert (
        scenario.coordinate_transform
        is not None
    )

    assert (
        scenario.coordinate_transform.scale
        > 0.0
    )