"""CLI for importing and profiling real traffic observations."""
from __future__ import annotations

import argparse
from pathlib import Path

from .empirical_traffic import TrafficProfileConfig, build_profile, write_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an empirical traffic profile from real observations")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--interval-s", type=int, default=300)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--anomaly-z", type=float, default=3.5)
    args = parser.parse_args()
    name = args.name or args.source.stem
    cfg = TrafficProfileConfig(
        source_name=name,
        source_path=str(args.source),
        interval_s=args.interval_s,
        anomaly_z=args.anomaly_z,
    )
    profile = build_profile(cfg)
    out = write_profile(profile, args.output)
    print(f"Built empirical traffic profile: {out}")
    print(f"Sensors: {len(profile.sensor_ids)} | timesteps: {len(profile.timestamps)} | anomaly candidates: {len(profile.anomaly_events)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())