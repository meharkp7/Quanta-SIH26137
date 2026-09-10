"""Build a small, auditable dynamic-data pilot across topology/traffic families."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset_generator import DynamicDatasetConfig, generate_dynamic_dataset
from .dynamic_episodes import DynamicEpisodeConfig, generate_episode
from src.contracts.scenario import Scenario


def generate_pilot_dataset(root: str | Path, seed: int = 26137) -> Path:
    """Generate 12 independent pilot episodes: 4 topologies x 3 traffic regimes.

    This is deliberately small enough for manual inspection.  It is a synthetic
    traffic-field pilot until the SUMO backend is connected; records therefore
    carry ``backend=synthetic_field_v1`` and are not presented as simulator truth.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    families = ("grid", "irregular", "radial", "hybrid")
    regimes = ("normal", "morning_peak", "evening_peak")
    records: list[dict] = []
    index = 0
    for family in families:
        for regime in regimes:
            split = "test" if family == "radial" else ("validation" if regime == "evening_peak" else "train")
            scenario_dir = root / "scenarios" / f"scenario_{index:03d}"
            cfg = DynamicDatasetConfig(
                customer_count=120,
                road_junction_count=196,
                topology_family=family,
                seed=seed + index * 1009,
                dataset_split=split,
            )
            scenario_path = generate_dynamic_dataset(scenario_dir, cfg)
            scenario = Scenario.model_validate_json(scenario_path.read_text(encoding="utf-8"))
            event = None if regime == "normal" else ("incident" if regime == "morning_peak" else "closure")
            episode_dir = root / "episodes" / f"episode_{index:03d}"
            episode_cfg = DynamicEpisodeConfig(
                regime=regime,
                event_type=event,
                event_count=1 if event else 0,
                seed=seed + index * 1009 + 500,
            )
            generate_episode(scenario, episode_dir, episode_cfg, episode_id=f"episode_{index:03d}")
            records.append({
                "episode_id": f"episode_{index:03d}",
                "scenario_id": scenario.scenario_id,
                "topology_family": family,
                "traffic_regime": regime,
                "event_type": event,
                "split": split,
                "scenario_seed": cfg.seed,
                "episode_seed": episode_cfg.seed,
                "backend": episode_cfg.backend,
            })
            index += 1
    manifest = {
        "dataset_version": "dynamic-pilot-v1",
        "backend": "synthetic_field_v1",
        "episode_count": len(records),
        "families": list(families),
        "traffic_regimes": list(regimes),
        "episodes": records,
        "split_rule": "radial held out; evening-peak validation; remaining train",
        "note": "Replace synthetic edge-field truth with SUMO-derived episodes before final learned-model evaluation.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return root / "manifest.json"
