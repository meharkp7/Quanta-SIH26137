"""CLI: build Step 12 windows from a Step 11 causal pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.contracts.scenario import Scenario
from src.learning.baselines import PersistenceForecaster, TemporalOnlyForecaster
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
    write_fixture_tensors(args.output_dir, datasets["train"])
    datasets["train"].scaler.save(args.output_dir / "scaler.json")

    train_batch = datasets["train"].batch()
    persistence = PersistenceForecaster().predict(train_batch)
    temporal = TemporalOnlyForecaster().fit(train_batch).predict(train_batch)
    report = {
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "input_shape": list(train_batch.shape),
        "persistence_shape": list(persistence.shape),
        "temporal_only_shape": list(temporal.shape),
        "scaler_fit_on": "train",
    }
    (args.output_dir / "loader_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
