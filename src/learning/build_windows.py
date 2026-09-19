"""Build Step-12 causal windows, baselines, and an auditable evidence package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.contracts.scenario import Scenario
from src.learning.baselines import PersistenceForecaster, TemporalOnlyForecaster, masked_speed_metrics
from src.learning.inspect import write_inspect_example
from src.learning.loader import load_pilot_windows, write_fixture_tensors


def _split_report(batch, persistence, temporal, horizons=(5, 10, 15)) -> dict:
    report = {
        "scenario_ids": sorted(set(batch.scenario_ids)),
        "window_count": int(batch.features.shape[0]),
        "real_edge_count": int((~batch.edge_padding_mask).sum()),
        "per_horizon": {},
    }
    real_edges = int((~batch.edge_padding_mask).sum())
    for i, horizon in enumerate(horizons):
        target_mask = batch.speed_target_mask[:, :, i]
        persistence_valid = target_mask & np.isfinite(persistence[:, :, i])
        temporal_valid = target_mask & np.isfinite(temporal[:, :, i])
        report["per_horizon"][str(horizon)] = {
            "label_coverage": float(target_mask.sum() / real_edges) if real_edges else 0.0,
            "persistence": masked_speed_metrics(
                batch.speed_targets[:, :, i], persistence_valid, persistence[:, :, i]
            ),
            "temporal_only": masked_speed_metrics(
                batch.speed_targets[:, :, i], temporal_valid, temporal[:, :, i]
            ),
        }
    report["persistence"] = masked_speed_metrics(
        batch.speed_targets, batch.speed_target_mask,
        persistence,
    )
    report["temporal_only"] = masked_speed_metrics(
        batch.speed_targets, batch.speed_target_mask,
        temporal,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("scenario", type=Path, nargs="?", help="Legacy single-map pilot only")
    args = parser.parse_args()

    scenario = Scenario.model_validate_json(args.scenario.read_text(encoding="utf-8")) if args.scenario else None
    datasets = load_pilot_windows(args.pilot_dir, scenario)
    if not datasets["train"].windows:
        raise RuntimeError("training split contains no causal forecast windows")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_inspect_example(datasets["train"], args.output_dir / "inspect_example.json")
    if datasets["train"].scaler is None:
        raise RuntimeError("training scaler was not fitted")
    datasets["train"].scaler.save(args.output_dir / "scaler.json")

    for name, dataset in datasets.items():
        write_fixture_tensors(args.output_dir / name, dataset)

    train_batch = datasets["train"].batch()
    persistence_model = PersistenceForecaster()
    temporal_model = TemporalOnlyForecaster().fit(train_batch)
    np.savez_compressed(
        args.output_dir / "temporal_baseline.npz",
        weights=temporal_model.weights,
        ridge=np.asarray([temporal_model.ridge], dtype=np.float32),
        feature_dim=np.asarray([temporal_model.feature_dim], dtype=np.int64),
        horizon_count=np.asarray([temporal_model.horizon_count], dtype=np.int64),
    )

    baseline_report = {}
    for name, dataset in datasets.items():
        if not dataset.windows:
            baseline_report[name] = {
                "scenario_ids": [],
                "window_count": 0,
                "real_edge_count": 0,
                "per_horizon": {},
                "persistence": {"count": 0, "mae": None, "rmse": None},
                "temporal_only": {"count": 0, "mae": None, "rmse": None},
            }
            continue
        batch = dataset.batch()
        baseline_report[name] = _split_report(
            batch,
            persistence_model.predict(batch),
            temporal_model.predict(batch),
        )

    manifest = json.loads((args.pilot_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    train_batch = datasets["train"].batch()
    report = {
        "schema_version": "step12-2.0",
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "input_shape": list(train_batch.shape),
        "feature_count": int(train_batch.shape[-1]),
        "history_minutes": 12,
        "horizons_minutes": [5, 10, 15],
        "scaler_fit_on": "train_only",
        "split_policy": manifest["rule"],
        "baseline_models": ["persistence", "temporal_only"],
        "baseline_report": baseline_report,
    }
    (args.output_dir / "loader_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
