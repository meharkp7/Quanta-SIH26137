"""Train and evaluate the Step 13 ``forecaster_v1`` artifact."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from src.contracts.scenario import Scenario
from src.learning.baselines import PersistenceForecaster, TemporalOnlyForecaster, masked_speed_metrics
from src.learning.gnn_transformer import CausalGNNTransformer, build_graph_batch, forecast_loss
from src.learning.loader import ForecastBatch, load_pilot_windows
from src.learning.uncertainty import (
    calibrated_arrays,
    fit_residual_calibration,
    interval_metrics,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensor_batch(batch: ForecastBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
    padding = torch.as_tensor(batch.edge_padding_mask, dtype=torch.bool, device=device)
    return features, padding


def _evaluate(model, dataset, scenarios, device):
    batch = dataset.batch()
    graph = build_graph_batch(batch, scenarios, device=device)
    features, padding = _tensor_batch(batch, device)
    model.eval()
    with torch.no_grad():
        outputs = model(features, graph, padding)
    prediction = outputs["speed_ratio"].cpu().numpy()
    metrics = {}
    for horizon in range(prediction.shape[-1]):
        metric = masked_speed_metrics(
            batch.speed_targets[:, :, horizon],
            batch.speed_target_mask[:, :, horizon],
            prediction[:, :, horizon],
        )
        total = int(batch.speed_target_mask[:, :, horizon].sum())
        metric["label_coverage"] = float(metric["count"] / total) if total else 0.0
        metrics[str((horizon + 1) * 5)] = metric
    return metrics, outputs


def _breakdown_metrics(model, dataset, scenarios, device):
    """Report per-map and incident-window metrics on the same predictions."""
    batch = dataset.batch()
    graph = build_graph_batch(batch, scenarios, device=device)
    features, padding = _tensor_batch(batch, device)
    model.eval()
    with torch.no_grad():
        prediction = model(features, graph, padding)["speed_ratio"].cpu().numpy()
    groups = {"per_map": {}}
    scenario_ids = np.asarray(batch.scenario_ids)
    incident = np.asarray([
        bool(np.nan_to_num(window.features[..., 5], nan=0.0).max() > 0.0)
        for window in dataset.windows
    ])
    groups["incident"] = _group_metrics(batch, prediction, incident)
    for scenario_id in sorted(set(batch.scenario_ids)):
        groups["per_map"][scenario_id] = _group_metrics(
            batch, prediction, scenario_ids == scenario_id
        )
    return groups


def _group_metrics(batch, prediction, selected):
    selected = np.asarray(selected, dtype=bool)
    result = {}
    for horizon in range(prediction.shape[-1]):
        target = batch.speed_targets[selected, :, horizon]
        mask = batch.speed_target_mask[selected, :, horizon]
        metric = masked_speed_metrics(target, mask, prediction[selected, :, horizon])
        total = int(mask.sum())
        metric["label_coverage"] = float(metric["count"] / total) if total else 0.0
        result[str((horizon + 1) * 5)] = metric
    return result


def train_forecaster(pilot_dir: Path, output_dir: Path, *, epochs: int = 60, seed: int = 26137,
                     width: int = 32, device: str = "cpu") -> dict:
    """Train on Step 12 windows and write a reproducible artifact manifest."""
    seed_everything(seed)
    device_obj = torch.device(device)
    pilot_dir, output_dir = Path(pilot_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = json.loads((pilot_dir / "split_manifest.json").read_text(encoding="utf-8"))
    scenarios = {
        sid: Scenario.model_validate_json((pilot_dir / relative).read_text(encoding="utf-8"))
        for sid, relative in split_manifest.get("scenario_files", {}).items()
    }
    datasets = load_pilot_windows(pilot_dir)
    train_batch = datasets["train"].batch()
    graph = build_graph_batch(train_batch, scenarios, device=device_obj)
    model = CausalGNNTransformer(width=width).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train_features, train_padding = _tensor_batch(train_batch, device_obj)
    history = []
    best = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(train_features, graph, train_padding)
        losses = forecast_loss(outputs, train_batch, device=device_obj)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        val_metrics, _ = _evaluate(model, datasets["validation"], scenarios, device_obj)
        val_values = [m["mae"] for m in val_metrics.values() if m["mae"] is not None]
        val_mae = float(np.mean(val_values)) if val_values else float("inf")
        row = {"epoch": epoch, "train_loss": float(losses["total"].item()),
               "train_speed_loss": float(losses["speed"].item()),
               "train_traversal_loss": float(losses["traversal"].item()),
               "validation_speed_mae": val_mae}
        history.append(row)
        if val_mae < best:
            best = val_mae
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No model checkpoint was produced")
    model.load_state_dict(best_state)
    metrics = {}
    breakdowns = {}
    for name, dataset in datasets.items():
        metrics[name], _ = _evaluate(model, dataset, scenarios, device_obj)
        breakdowns[name] = _breakdown_metrics(model, dataset, scenarios, device_obj)
    # Calibrate only on the held-out validation split.  The test split is
    # used once below for the reported coverage and is never used to tune the
    # radius.
    _, validation_outputs = _evaluate(model, datasets["validation"], scenarios, device_obj)
    validation_batch = datasets["validation"].batch()
    calibration = fit_residual_calibration(
        validation_outputs["speed_ratio"].detach().cpu().numpy(),
        validation_batch.speed_targets,
        validation_batch.speed_target_mask,
        nominal_coverage=0.80,
    )
    interval_reports = {}
    for name, dataset in datasets.items():
        _, outputs = _evaluate(model, dataset, scenarios, device_obj)
        batch = dataset.batch()
        _, lower, upper = calibrated_arrays(outputs["speed_ratio"].detach().cpu().numpy(), calibration)
        interval_reports[name] = interval_metrics(
            lower, upper, batch.speed_targets, batch.speed_target_mask,
            nominal_coverage=calibration.nominal_coverage,
        )
    (output_dir / "uncertainty.json").write_text(json.dumps({
        **calibration.to_dict(),
        "validation": interval_reports["validation"],
        "test": interval_reports["test"],
    }, indent=2), encoding="utf-8")
    baseline_metrics = {}
    temporal = TemporalOnlyForecaster().fit(datasets["train"].batch())
    persistence = PersistenceForecaster()
    for name, dataset in datasets.items():
        batch = dataset.batch()
        baseline_metrics[name] = {
            "persistence": masked_speed_metrics(
                batch.speed_targets, batch.speed_target_mask, persistence.predict(batch)
            ),
            "temporal_only": masked_speed_metrics(
                batch.speed_targets, batch.speed_target_mask, temporal.predict(batch)
            ),
        }
    cutoff = max(
        int(np.nanmax(window.label_available_at_s))
        for window in datasets["train"].windows
        if np.isfinite(window.label_available_at_s).any()
    )
    torch.save(model.state_dict(), output_dir / "weights.pt")
    shutil.copy2(pilot_dir / "split_manifest.json", output_dir / "split_manifest.json")
    datasets["train"].scaler.save(output_dir / "scaler.json")
    manifest = {
        "artifact": "forecaster_v1", "architecture": model.config, "seed": seed,
        "device": str(device_obj), "training_cutoff_s": cutoff,
        "feature_schema": ["speed_ratio", "occupancy", "halting", "observation_age_min", "missing", "known_closed"],
        "target_schema": {"speed_ratio": [5, 10, 15], "traversal_time_s": [5, 10, 15]},
        "split_policy": split_manifest.get("rule"), "metrics": metrics,
        "breakdowns": breakdowns,
        "uncertainty": {
            **calibration.to_dict(),
            "validation": interval_reports["validation"],
            "test": interval_reports["test"],
        },
        "baseline_metrics": baseline_metrics,
        "train_windows": len(datasets["train"]), "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "training_curve.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=26137)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(train_forecaster(args.pilot, args.output, epochs=args.epochs,
                                       seed=args.seed, width=args.width), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
