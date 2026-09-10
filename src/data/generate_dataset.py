"""CLI for the Step 4 CVRP-to-road dataset pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_generator import DynamicDatasetConfig, GeneratorConfig, generate_dynamic_dataset, generate_from_vrp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Step 4 synthetic road dataset")
    sub = parser.add_subparsers(dest="mode", required=False)

    dynamic = sub.add_parser("dynamic", help="Generate a complete dataset without a .vrp input")
    dynamic.add_argument("output", type=Path)
    dynamic.add_argument("--customers", type=int, default=120)
    dynamic.add_argument("--junctions", type=int, default=196)
    dynamic.add_argument("--scale-m", type=float, default=8000.0)
    dynamic.add_argument("--family", choices=("grid", "irregular"), default="grid")
    dynamic.add_argument("--rows", type=int, default=14)
    dynamic.add_argument("--cols", type=int, default=14)
    dynamic.add_argument("--one-way-fraction", type=float, default=0.15)
    dynamic.add_argument("--min-segment-m", type=float, default=30.0)
    dynamic.add_argument("--max-snap-m", type=float, default=500.0)
    dynamic.add_argument("--window-profile", choices=("loose", "medium", "tight"), default="medium")
    dynamic.add_argument("--window-slack-s", type=float, default=900.0)
    dynamic.add_argument("--service-duration-s", type=float, default=60.0)
    dynamic.add_argument("--vehicle-capacity", type=float, default=30.0)
    dynamic.add_argument("--fleet-size", type=int, default=None)
    dynamic.add_argument("--seed", type=int, default=26137)
    dynamic.add_argument("--split", choices=("train", "validation", "test"), default="train")

    vrp = sub.add_parser("vrp", help="Derive a road dataset from a CVRP .vrp input")
    vrp.add_argument("input", type=Path)
    vrp.add_argument("output", type=Path)
    vrp.add_argument("--seed", type=int, default=26137)
    vrp.add_argument("--scale-m", type=float, default=8000.0)
    vrp.add_argument("--family", choices=("grid", "irregular"), default="grid")
    vrp.add_argument("--junctions", type=int, default=196)
    vrp.add_argument("--rows", type=int, default=14)
    vrp.add_argument("--cols", type=int, default=14)
    vrp.add_argument("--one-way-fraction", type=float, default=0.15)
    vrp.add_argument("--max-snap-m", type=float, default=500.0)
    vrp.add_argument("--window-profile", choices=("loose", "medium", "tight"), default="medium")
    vrp.add_argument("--window-slack-s", type=float, default=900.0)
    vrp.add_argument("--service-duration-s", type=float, default=60.0)
    vrp.add_argument("--fleet-size", type=int, default=None)
    vrp.add_argument("--reference-variant", type=int, choices=(0, 1), default=0)
    vrp.add_argument("--split", choices=("train", "validation", "test"), default="train")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode in (None, "dynamic"):
        if args.mode is None:
            parser.error("use: python -m src.data.generate_dataset dynamic <output> [options]")
        config = DynamicDatasetConfig(
            customer_count=args.customers, road_junction_count=args.junctions, extent_m=args.scale_m,
            topology_family=args.family, grid_rows=args.rows, grid_cols=args.cols,
            one_way_fraction=args.one_way_fraction, max_customer_snap_m=args.max_snap_m,
            minimum_road_segment_m=args.min_segment_m,
            window_profile=args.window_profile, window_slack_s=args.window_slack_s,
            service_duration_s=args.service_duration_s, vehicle_capacity=args.vehicle_capacity,
            fleet_size=args.fleet_size, seed=args.seed, dataset_split=args.split,
        )
        scenario = generate_dynamic_dataset(args.output, config)
        print(f"Generated dynamic Step 4 scenario: {scenario}")
        return 0

    config = GeneratorConfig(
        scale_extent_m=args.scale_m, topology_family=args.family, junction_count=args.junctions,
        grid_rows=args.rows, grid_cols=args.cols, one_way_fraction=args.one_way_fraction,
        max_customer_snap_m=args.max_snap_m, window_profile=args.window_profile,
        window_slack_s=args.window_slack_s, service_duration_s=args.service_duration_s,
        fleet_size=args.fleet_size, reference_variant=args.reference_variant, seed=args.seed,
        dataset_split=args.split,
    )
    scenario = generate_from_vrp(args.input, args.output, config)
    print(f"Generated Step 4 scenario from CVRP: {scenario}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
