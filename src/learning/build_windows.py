"""CLI: build Step 12 windows from a Step 11 causal pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.contracts.scenario import Scenario
from src.learning.baselines import (
    PersistenceForecaster,
    TemporalOnlyForecaster,
    masked_speed_metrics,
)
from src.learning.inspect import write_inspect_example
from src.learning.loader import load_pilot_windows, write_fixture_tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()

    scenario = Scenario.model_validate_json(args.scenario.read_text(encoding="utf-8"))
    datasets = load_pilot_windows(args.pilot_dir, scenario)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_inspect_example(datasets["train"], args.output_dir / "inspect_example.json")
    datasets["train"].scaler.save(args.output_dir / "scaler.json")

    # Persist every split so the loader/evidence package contains train,
    # validation and untouched test tensors rather than only training data.
    for name, dataset in datasets.items():
        write_fixture_tensors(args.output_dir / name, dataset)

    train_batch = datasets["train"].batch()
    persistence_model = PersistenceForecaster()
    temporal_model = TemporalOnlyForecaster().fit(train_batch)
    baseline_report = {}
    for name, dataset in datasets.items():
        batch = dataset.batch()
        persistence = persistence_model.predict(batch)
        temporal = temporal_model.predict(batch)
        baseline_report[name] = {
            "persistence": masked_speed_metrics(
                batch.speed_targets, batch.speed_target_mask, persistence
            ),
            "temporal_only": masked_speed_metrics(
                batch.speed_targets, batch.speed_target_mask, temporal
            ),
        }

    report = {
        "schema_version": "step12-1.0",
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "input_shape": list(train_batch.shape),
        "feature_count": int(train_batch.shape[-1]),
        "history_minutes": 12,
        "horizons_minutes": [5, 10, 15],
        "scaler_fit_on": "train_only",
        "split_policy": "episode-level split before overlapping windows",
        "baseline_metrics": baseline_report,
    }
    (args.output_dir / "loader_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
