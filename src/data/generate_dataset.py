"""CLI for the Quanta Step 4 dataset pipeline.

Supports:
- synthetic development datasets
- dynamic traffic episodes
- CVRP-to-road generation
- empirical real-traffic profiling
- empirical traffic-shock priors
- real-conditioned development episodes
- canonical real-traffic dataset conversion
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_generator import (
    DynamicDatasetConfig,
    GeneratorConfig,
    generate_dynamic_dataset,
    generate_from_vrp,
)
from .dynamic_episodes import DynamicEpisodeConfig, generate_episode
from .pilot_dataset import generate_pilot_dataset
from .metrla_spatial import (
    load_metrla_spatial,
    validate_metrla_spatial,
    write_metrla_spatial,
)
from .metrla_analysis import (
    analyze_sensor_statistics,
    analyze_spatial_statistics,
    analyze_temporal_profile,
    detect_shock_candidates,
    write_analysis,
)
from src.data.pemsbay_spatial import (
    load_pemsbay_spatial,
    validate_pemsbay_spatial,
    write_pemsbay_spatial,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and process Quanta traffic/routing datasets"
    )
    sub = parser.add_subparsers(dest="mode", required=False)
    metrla_spatial = sub.add_parser(
        "metrla-spatial",
        help="Build the METR-LA sensor spatial graph.",
    )

    metrla_spatial.add_argument(
        "sensor_ids",
        help="Path to graph_sensor_ids.txt",
    )

    metrla_spatial.add_argument(
        "locations",
        help="Path to graph_sensor_locations.csv",
    )

    metrla_spatial.add_argument(
        "distances",
        help="Path to distances_la_2012.csv",
    )

    metrla_spatial.add_argument(
        "output",
        help="Output directory for the spatial artifact.",
    )

    metrla_spatial.set_defaults(func=cmd_metrla_spatial)

    pemsbay_spatial = sub.add_parser(
        "pemsbay-spatial",
        help="Build the canonical PEMS-BAY spatial graph",
    )

    pemsbay_spatial.add_argument("locations")
    pemsbay_spatial.add_argument("distances")
    pemsbay_spatial.add_argument("output")
    # ------------------------------------------------------------------
    # METR-LA empirical analysis
    # ------------------------------------------------------------------
    analysis = sub.add_parser(
        "metrla-analysis",
        help="Analyze the canonical METR-LA traffic and spatial layers",
    )
    analysis.add_argument("observations", type=Path)
    analysis.add_argument("sensor_edges", type=Path)
    analysis.add_argument("output", type=Path)
    analysis.add_argument("--absolute-drop", type=float, default=10.0)
    analysis.add_argument("--relative-drop", type=float, default=0.20)

    # ------------------------------------------------------------------
    # Synthetic development dataset
    # ------------------------------------------------------------------
    dynamic = sub.add_parser(
        "dynamic",
        help="Generate a complete synthetic dataset without a .vrp input",
    )
    dynamic.add_argument("output", type=Path)
    dynamic.add_argument("--customers", type=int, default=120)
    dynamic.add_argument("--junctions", type=int, default=196)
    dynamic.add_argument("--scale-m", type=float, default=8000.0)
    dynamic.add_argument(
        "--family",
        choices=("grid", "irregular", "radial", "hybrid"),
        default="grid",
    )
    dynamic.add_argument("--rows", type=int, default=14)
    dynamic.add_argument("--cols", type=int, default=14)
    dynamic.add_argument("--one-way-fraction", type=float, default=0.15)
    dynamic.add_argument("--min-segment-m", type=float, default=30.0)
    dynamic.add_argument("--max-snap-m", type=float, default=500.0)
    dynamic.add_argument(
        "--window-profile",
        choices=("loose", "medium", "tight"),
        default="medium",
    )
    dynamic.add_argument("--window-slack-s", type=float, default=900.0)
    dynamic.add_argument("--service-duration-s", type=float, default=60.0)
    dynamic.add_argument("--vehicle-capacity", type=float, default=30.0)
    dynamic.add_argument("--fleet-size", type=int, default=None)
    dynamic.add_argument("--seed", type=int, default=26137)
    dynamic.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )

    # ------------------------------------------------------------------
    # Synthetic pilot
    # ------------------------------------------------------------------
    pilot = sub.add_parser(
        "pilot",
        help="Generate an auditable multi-topology dynamic-data pilot",
    )
    pilot.add_argument("output", type=Path)
    pilot.add_argument("--seed", type=int, default=26137)

    # ------------------------------------------------------------------
    # Synthetic causal episode
    # ------------------------------------------------------------------
    episode = sub.add_parser(
        "episode",
        help="Generate one synthetic causal dynamic traffic episode from scenario.json",
    )
    episode.add_argument("scenario", type=Path)
    episode.add_argument("output", type=Path)
    episode.add_argument("--duration-s", type=int, default=1800)
    episode.add_argument("--interval-s", type=int, default=60)
    episode.add_argument("--warmup-s", type=int, default=300)
    episode.add_argument(
        "--regime",
        choices=(
            "normal",
            "morning_peak",
            "evening_peak",
            "corridor_surge",
        ),
        default="normal",
    )
    episode.add_argument(
        "--event",
        choices=(
            "closure",
            "incident",
            "multi_disruption",
            "scheduled_closure",
        ),
        default=None,
    )
    episode.add_argument("--event-count", type=int, default=1)
    episode.add_argument("--event-duration-s", type=int, default=240)
    episode.add_argument("--reveal-lead-s", type=int, default=60)
    episode.add_argument("--missing-fraction", type=float, default=0.02)
    episode.add_argument("--seed", type=int, default=26137)

    # ------------------------------------------------------------------
    # CVRP -> road dataset
    # ------------------------------------------------------------------
    vrp = sub.add_parser(
        "vrp",
        help="Derive a road dataset from a CVRP .vrp input",
    )
    vrp.add_argument("input", type=Path)
    vrp.add_argument("output", type=Path)
    vrp.add_argument("--seed", type=int, default=26137)
    vrp.add_argument("--scale-m", type=float, default=8000.0)
    vrp.add_argument(
        "--family",
        choices=("grid", "irregular", "radial", "hybrid"),
        default="grid",
    )
    vrp.add_argument("--junctions", type=int, default=196)
    vrp.add_argument("--rows", type=int, default=14)
    vrp.add_argument("--cols", type=int, default=14)
    vrp.add_argument("--one-way-fraction", type=float, default=0.15)
    vrp.add_argument("--max-snap-m", type=float, default=500.0)
    vrp.add_argument(
        "--window-profile",
        choices=("loose", "medium", "tight"),
        default="medium",
    )
    vrp.add_argument("--window-slack-s", type=float, default=900.0)
    vrp.add_argument("--service-duration-s", type=float, default=60.0)
    vrp.add_argument("--fleet-size", type=int, default=None)
    vrp.add_argument(
        "--reference-variant",
        type=int,
        choices=(0, 1),
        default=0,
    )
    vrp.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    vrp.add_argument(
        "--stress-case",
        choices=(
            "none",
            "missed_window",
            "disconnected",
        ),
        default="none",
        help=(
            "Explicitly generate a labelled "
            "infeasible Step 4 stress case."
        ),
    )

    # ------------------------------------------------------------------
    # Real empirical traffic profile
    # ------------------------------------------------------------------
    empirical_profile = sub.add_parser(
        "empirical-profile",
        help="Build a canonical empirical traffic profile from a real dataset",
    )
    empirical_profile.add_argument("source", type=Path)
    empirical_profile.add_argument("output", type=Path)
    empirical_profile.add_argument("--name", type=str, default=None)
    empirical_profile.add_argument("--interval-s", type=int, default=300)
    empirical_profile.add_argument("--anomaly-z", type=float, default=4.0)

    # ------------------------------------------------------------------
    # Empirical traffic-shock priors
    # ------------------------------------------------------------------
    event_priors = sub.add_parser(
        "event-priors",
        help="Estimate empirical traffic-shock priors from a traffic profile",
    )
    event_priors.add_argument("profile", type=Path)
    event_priors.add_argument("output", type=Path)

    # ------------------------------------------------------------------
    # Real-conditioned development replay
    # ------------------------------------------------------------------
    empirical_episode = sub.add_parser(
        "empirical-episode",
        help=(
            "Generate a real-conditioned development replay episode "
            "from a routing scenario and empirical traffic profile"
        ),
    )
    empirical_episode.add_argument("scenario", type=Path)
    empirical_episode.add_argument("profile", type=Path)
    empirical_episode.add_argument("output", type=Path)
    empirical_episode.add_argument("--start-index", type=int, default=0)
    empirical_episode.add_argument("--steps", type=int, default=30)
    empirical_episode.add_argument("--seed", type=int, default=26137)

    # ------------------------------------------------------------------
    # Canonical real-data conversion
    # ------------------------------------------------------------------
    canonical = sub.add_parser(
        "canonical",
        help="Convert a real traffic dataset into the canonical Quanta representation",
    )
    canonical.add_argument("dataset_id", type=str)
    canonical.add_argument("source", type=Path)
    canonical.add_argument("output", type=Path)
    canonical.add_argument("--interval-s", type=int, default=None)

    causal_pilot = sub.add_parser(
        "causal-pilot",
        help="Generate the Step 11 six-episode causal pilot with matured labels",
    )
    causal_pilot.add_argument("scenario", type=Path)
    causal_pilot.add_argument("output", type=Path)
    causal_pilot.add_argument("--duration-s", type=int, default=2400)
    causal_pilot.add_argument("--interval-s", type=int, default=60)
    causal_pilot.add_argument("--warmup-s", type=int, default=300)
    causal_pilot.add_argument("--seed", type=int, default=26137)

    windows = sub.add_parser(
        "forecast-windows",
        help="Build Step 12 [B,L,E,F] windows and baselines from a causal pilot",
    )
    windows.add_argument("pilot", type=Path)
    windows.add_argument("output", type=Path)
    windows.add_argument("scenario", type=Path)

    return parser

def cmd_metrla_spatial(args) -> int:
    sensor_ids, locations, sensor_edges, profile = load_metrla_spatial(
        args.sensor_ids,
        args.locations,
        args.distances,
    )

    issues = validate_metrla_spatial(
        sensor_ids,
        locations,
        sensor_edges,
        profile,
    )

    if issues:
        print("METR-LA spatial validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    output = write_metrla_spatial(
        args.output,
        sensor_ids,
        locations,
        sensor_edges,
        profile,
        source_sensor_ids=args.sensor_ids,
        source_locations=args.locations,
        source_distances=args.distances,
    )

    print(f"Generated METR-LA spatial dataset: {output}")
    print()
    print(f"Traffic sensors:             {profile.sensor_count}")
    print(f"Sensor locations:            {profile.location_count}")
    print(
        f"Source distance graph nodes: {profile.distance_graph_node_count}"
    )
    print(
        f"Source distance rows:        {profile.distance_graph_row_count}"
    )
    print(
        f"Direct sensor pairs:         {profile.direct_sensor_pair_count}"
    )
    print(
        f"Non-self sensor edges:       {profile.non_self_sensor_pair_count}"
    )
    print(f"Self-pairs:                   {profile.self_pair_count}")
    print(
        f"Reciprocal sensor pairs:     "
        f"{profile.reciprocal_sensor_pair_count}"
    )
    print(
        f"Asymmetric reciprocal pairs: "
        f"{profile.asymmetric_sensor_pair_count}"
    )
    print(
        f"Minimum distance:            "
        f"{profile.minimum_non_self_distance}"
    )
    print(
        f"Maximum distance:            "
        f"{profile.maximum_non_self_distance}"
    )
    print()
    print("Validation: PASS")

    return 0

def _read_analysis_observations(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = __import__("csv").DictReader(handle)

        for row in reader:
            value = row["value"].strip()

            yield {
                "source_dataset": row["source_dataset"],
                "sample_index": int(row["sample_index"]),
                "timestamp": row["timestamp"],
                "sensor_id": row["sensor_id"],
                "metric": row["metric"],
                "value": None if value == "" else float(value),
            }

def cmd_pemsbay_spatial(args):
    profile, locations, edges = load_pemsbay_spatial(
        args.locations,
        args.distances,
    )

    issues = validate_pemsbay_spatial(
        profile,
        locations,
        edges,
    )

    if issues:
        print("PEMS-BAY spatial validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    output = write_pemsbay_spatial(
        profile,
        locations,
        edges,
        args.output,
    )

    print(f"Generated PEMS-BAY spatial dataset: {output}")
    print()
    print(f"Traffic sensors:             {profile.sensor_count}")
    print(f"Sensor locations:            {profile.location_count}")
    print(f"Source distance graph nodes: {profile.distance_graph_node_count}")
    print(f"Source distance rows:        {profile.distance_graph_row_count}")
    print(f"Direct sensor pairs:         {profile.sensor_edge_count + profile.self_pair_count}")
    print(f"Non-self sensor edges:       {profile.sensor_edge_count}")
    print(f"Self-pairs:                  {profile.self_pair_count}")
    print(f"Reciprocal pairs:            {profile.reciprocal_pair_count}")
    print(f"Asymmetric pairs:            {profile.asymmetric_pair_count}")
    print(f"Minimum distance:            {profile.min_distance}")
    print(f"Maximum distance:            {profile.max_distance}")
    print()
    print("Validation: PASS")

    return 0

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # No mode
    # ------------------------------------------------------------------
    if args.mode is None:
        parser.error(
            "use one of: dynamic, pilot, episode, vrp, empirical-profile, "
            "event-priors, empirical-episode, canonical, metrla-spatial, metrla-analysis"
        )

    if args.mode == "pemsbay-spatial":
        return cmd_pemsbay_spatial(args)
    # ------------------------------------------------------------------
    # METR-LA empirical analysis
    # ------------------------------------------------------------------
    if args.mode == "metrla-analysis":
        observations = list(_read_analysis_observations(args.observations))

        temporal = analyze_temporal_profile(observations)
        sensor_stats = analyze_sensor_statistics(observations)
        shocks = detect_shock_candidates(
            observations,
            absolute_drop=args.absolute_drop,
            relative_drop=args.relative_drop,
        )
        spatial = analyze_spatial_statistics(args.sensor_edges)

        output = write_analysis(
            args.output,
            temporal,
            sensor_stats,
            shocks,
            spatial,
        )

        print(f"Generated METR-LA empirical analysis: {output}")
        print()
        print(f"Time samples:          {temporal.sample_count}")
        print(f"Sensors:               {temporal.sensor_count}")
        print(f"Observed values:       {temporal.observed_count}")
        print(f"Missing values:        {temporal.missing_count}")
        print(f"Candidate shocks:      {len(shocks)}")
        print(f"Spatial edges:         {spatial.directed_edge_count}")
        print()
        print("Validation: PASS")
        return 0

    # ------------------------------------------------------------------
    # Synthetic dynamic dataset
    # ------------------------------------------------------------------
    if args.mode == "dynamic":
        config = DynamicDatasetConfig(
            customer_count=args.customers,
            road_junction_count=args.junctions,
            extent_m=args.scale_m,
            topology_family=args.family,
            grid_rows=args.rows,
            grid_cols=args.cols,
            one_way_fraction=args.one_way_fraction,
            max_customer_snap_m=args.max_snap_m,
            minimum_road_segment_m=args.min_segment_m,
            window_profile=args.window_profile,
            window_slack_s=args.window_slack_s,
            service_duration_s=args.service_duration_s,
            vehicle_capacity=args.vehicle_capacity,
            fleet_size=args.fleet_size,
            seed=args.seed,
            dataset_split=args.split,
        )

        scenario = generate_dynamic_dataset(args.output, config)
        print(f"Generated dynamic Step 4 scenario: {scenario}")
        return 0

    # ------------------------------------------------------------------
    # Synthetic pilot
    # ------------------------------------------------------------------
    if args.mode == "pilot":
        manifest = generate_pilot_dataset(
            args.output,
            seed=args.seed,
        )
        print(f"Generated dynamic pilot dataset: {manifest}")
        return 0

    # ------------------------------------------------------------------
    # Synthetic causal episode
    # ------------------------------------------------------------------
    if args.mode == "episode":
        from src.contracts.scenario import Scenario

        scenario = Scenario.model_validate_json(
            args.scenario.read_text(encoding="utf-8")
        )

        episode_config = DynamicEpisodeConfig(
            duration_s=args.duration_s,
            interval_s=args.interval_s,
            warmup_s=args.warmup_s,
            regime=args.regime,
            event_type=args.event,
            event_count=args.event_count,
            event_duration_s=args.event_duration_s,
            reveal_lead_s=args.reveal_lead_s,
            observation_missing_fraction=args.missing_fraction,
            seed=args.seed,
        )

        output = generate_episode(
            scenario,
            args.output,
            episode_config,
        )

        print(f"Generated dynamic episode: {output}")
        return 0

    # ------------------------------------------------------------------
    # CVRP -> road
    # ------------------------------------------------------------------
    if args.mode == "vrp":
        config = GeneratorConfig(
            scale_extent_m=args.scale_m,
            topology_family=args.family,
            junction_count=args.junctions,
            grid_rows=args.rows,
            grid_cols=args.cols,
            one_way_fraction=args.one_way_fraction,
            max_customer_snap_m=args.max_snap_m,
            window_profile=args.window_profile,
            window_slack_s=args.window_slack_s,
            service_duration_s=args.service_duration_s,
            fleet_size=args.fleet_size,
            reference_variant=args.reference_variant,
            seed=args.seed,
            dataset_split=args.split,
            stress_case=args.stress_case,
        )

        scenario = generate_from_vrp(
            args.input,
            args.output,
            config,
        )

        print(f"Generated Step 4 scenario from CVRP: {scenario}")
        return 0

    # ------------------------------------------------------------------
    # Real empirical profile
    # ------------------------------------------------------------------
    if args.mode == "empirical-profile":
        from .empirical_traffic import (
            TrafficProfileConfig,
            build_profile,
            load_real_traffic,
            write_profile,
        )

        config = TrafficProfileConfig(
            name=args.name,
            interval_s=args.interval_s,
            anomaly_z=args.anomaly_z,
        )

        traffic = load_real_traffic(args.source)
        profile = build_profile(
            traffic,
            config,
        )

        output = write_profile(
            profile,
            args.output,
        )

        print(f"Generated empirical traffic profile: {output}")
        return 0

    # ------------------------------------------------------------------
    # Empirical event/shock priors
    # ------------------------------------------------------------------
    if args.mode == "event-priors":
        from .event_calibration import (
            calibrate_event_priors,
            write_event_priors,
        )
        from .empirical_traffic import TrafficProfile

        profile = TrafficProfile.from_json(
            args.profile.read_text(encoding="utf-8")
        )

        priors = calibrate_event_priors(profile)

        output = write_event_priors(
            priors,
            args.output,
        )

        print(f"Generated empirical traffic-shock priors: {output}")
        return 0

    # ------------------------------------------------------------------
    # Real-conditioned development replay
    # ------------------------------------------------------------------
    if args.mode == "empirical-episode":
        from src.contracts.scenario import Scenario
        from .empirical_traffic import TrafficProfile
        from .real_conditioned_episode import generate_real_conditioned_episode

        scenario = Scenario.model_validate_json(
            args.scenario.read_text(encoding="utf-8")
        )

        profile = TrafficProfile.from_json(
            args.profile.read_text(encoding="utf-8")
        )

        output = generate_real_conditioned_episode(
            scenario=scenario,
            profile=profile,
            output_dir=args.output,
            start_index=args.start_index,
            steps=args.steps,
            seed=args.seed,
        )

        print(f"Generated real-conditioned development episode: {output}")
        return 0
    
    # ------------------------------------------------------------------
    # METR-LA spatial graph
    # ------------------------------------------------------------------
    if args.mode == "metrla-spatial":
        return cmd_metrla_spatial(args)

    # ------------------------------------------------------------------
    # Canonical real dataset
    # ------------------------------------------------------------------
    if args.mode == "canonical":
        from .canonical_traffic import (
            load_dataset,
            validate_dataset,
            write_canonical,
        )

        dataset = load_dataset(
            args.dataset_id,
            args.source,
            interval_s=args.interval_s,
        )

        validation = validate_dataset(dataset)

        if validation:
            print("Canonical dataset validation failed:")
            for issue in validation:
                print(f"  - {issue}")
            return 1

        output = write_canonical(
            dataset,
            args.output,
        )

        print(f"Generated canonical traffic dataset: {output}")
        print()
        print("Validation: PASS")

        return 0

    if args.mode == "causal-pilot":
        from src.contracts.scenario import Scenario
        from src.data.causal_episodes import generate_causal_pilot

        scenario = Scenario.model_validate_json(
            args.scenario.read_text(encoding="utf-8")
        )
        coverage = generate_causal_pilot(
            scenario,
            args.output,
            duration_s=args.duration_s,
            interval_s=args.interval_s,
            warmup_s=args.warmup_s,
            base_seed=args.seed,
        )
        print(f"Generated causal pilot: {args.output}")
        print(
            f"episodes={coverage['episodes']} "
            f"wall_s={coverage['wall_clock_s']} "
            f"disk={coverage['disk_bytes']}"
        )
        return 0

    if args.mode == "forecast-windows":
        from src.learning.build_windows import main as build_windows_main
        import sys

        sys.argv = [
            "forecast-windows",
            str(args.pilot),
            str(args.output),
            str(args.scenario),
        ]
        return build_windows_main()

    parser.error(f"unsupported mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())