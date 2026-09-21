"""Train and evaluate the Step 13 ``forecaster_v1`` artifact.

Training is performed in deterministic mini-batches of Step-13 windows.
The real SUMO episode corpus is loaded once, but the full corpus is never
materialised as one giant model tensor.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.contracts.scenario import Scenario
from src.learning.baselines import (
    PersistenceForecaster,
    TemporalOnlyForecaster,
    masked_speed_metrics,
)
from src.learning.gnn_transformer import (
    CausalGNNTransformer,
    build_graph_batch,
    forecast_loss,
)
from src.learning.loader import (
    ForecastBatch,
    load_corpus_manifest,
    load_corpus_scenarios,
    load_pilot_windows,
)
from src.learning.uncertainty import (
    calibrated_arrays,
    fit_residual_calibration,
    interval_metrics,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batch_from_windows(windows):
    """Use the existing WindowDataset batching contract."""
    from src.learning.loader import WindowDataset
    return WindowDataset(list(windows)).batch()


def _tensor_batch(
    batch: ForecastBatch,
    device: torch.device,
):
    features = torch.as_tensor(
        batch.features,
        dtype=torch.float32,
        device=device,
    )
    padding = torch.as_tensor(
        batch.edge_padding_mask,
        dtype=torch.bool,
        device=device,
    )
    return features, padding


def _scenario_subset(scenarios, batch):
    return {
        scenario_id: scenarios[scenario_id]
        for scenario_id in set(batch.scenario_ids)
    }


def _forward_batch(model, batch, scenarios, device):
    """Build sparse graph and run one mini-batch."""
    graph = build_graph_batch(
        batch,
        _scenario_subset(scenarios, batch),
        device=device,
    )
    features, padding = _tensor_batch(batch, device)
    outputs = model(features, graph, padding)
    return outputs


def _label_coverage(batch: ForecastBatch, mask: np.ndarray) -> float:
    """Return valid-label coverage over real (non-padding) edge cells."""
    from src.learning.evaluation import padding_aware_label_coverage

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim == 2:
        mask_array = mask_array[..., None]
    return padding_aware_label_coverage(mask_array, batch.edge_padding_mask)


def _iter_batches(dataset, batch_size, *, shuffle=False, rng=None):
    windows = list(dataset.windows)
    indices = np.arange(len(windows))

    if shuffle:
        if rng is None:
            rng = np.random.default_rng(26137)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        selected = [
            windows[int(i)]
            for i in indices[start:start + batch_size]
        ]
        yield _batch_from_windows(selected)


def _supervision_report(dataset):
    """Aggregate padding-aware supervision coverage across windows."""
    total_edges = sum(len(window.edge_ids) for window in dataset.windows)
    report = {"windows": len(dataset.windows), "edge_instances": total_edges, "horizons": {}}
    for index, horizon in enumerate((5, 10, 15)):
        speed_valid = sum(int(window.speed_target_mask[:, index].sum()) for window in dataset.windows)
        traversal_valid = sum(int(window.traversal_target_mask[:, index].sum()) for window in dataset.windows)
        report["horizons"][str(horizon)] = {
            "speed_valid_count": speed_valid,
            "speed_coverage": float(speed_valid / total_edges) if total_edges else 0.0,
            "traversal_valid_count": traversal_valid,
            "traversal_coverage": float(traversal_valid / total_edges) if total_edges else 0.0,
        }
    return report


def _global_edge_count(dataset) -> int:
    """Max edge count across windows (maps have different E)."""
    return max(len(window.edge_ids) for window in dataset.windows)


def _pad_edge_axis(array: np.ndarray, width: int, *, fill_value) -> np.ndarray:
    """Pad axis=1 (E) to `width`; exact for masked metrics (mask=False there)."""
    pad = width - array.shape[1]
    if pad <= 0:
        return array
    shape = [(0, 0)] * array.ndim
    shape[1] = (0, pad)
    return np.pad(array, shape, constant_values=fill_value)


def _predict_dataset(
    model,
    dataset,
    scenarios,
    device,
    *,
    batch_size,
    collect_outputs=False,
):
    """Chunked deterministic inference over a complete split."""
    predictions = []
    speed_targets = []
    speed_masks = []
    all_outputs = []

    model.eval()

    with torch.no_grad():
        for batch in _iter_batches(
            dataset,
            batch_size,
            shuffle=False,
        ):
            outputs = _forward_batch(
                model,
                batch,
                scenarios,
                device,
            )

            predictions.append(
                outputs["speed_ratio"].detach().cpu().numpy()
            )
            speed_targets.append(batch.speed_targets)
            speed_masks.append(batch.speed_target_mask)

            if collect_outputs:
                all_outputs.append(outputs)

    prediction = np.concatenate(
        [
            _pad_edge_axis(p, _global_edge_count(dataset), fill_value=0.0)
            for p in predictions
        ],
        axis=0,
    )
    targets = np.concatenate(
        [
            _pad_edge_axis(t, _global_edge_count(dataset), fill_value=np.nan)
            for t in speed_targets
        ],
        axis=0,
    )
    masks = np.concatenate(
        [
            _pad_edge_axis(m, _global_edge_count(dataset), fill_value=False)
            for m in speed_masks
        ],
        axis=0,
    )

    metrics = {}
    for horizon in range(prediction.shape[-1]):
        metric = masked_speed_metrics(
            targets[:, :, horizon],
            masks[:, :, horizon],
            prediction[:, :, horizon],
        )
        total_edges = sum(len(window.edge_ids) for window in dataset.windows)
        metric["label_coverage"] = (
            float(metric["count"] / total_edges)
            if total_edges
            else 0.0
        )
        metrics[str((horizon + 1) * 5)] = metric

    if not collect_outputs:
        return metrics, None

    # The uncertainty code expects the complete [N,E,H] prediction array.
    return metrics, {
        "speed_ratio": torch.as_tensor(
            prediction,
            dtype=torch.float32,
            device=device,
        ),
        "targets": targets,
        "masks": masks,
    }


def _evaluate(
    model,
    dataset,
    scenarios,
    device,
    *,
    batch_size,
):
    metrics, packed = _predict_dataset(
        model,
        dataset,
        scenarios,
        device,
        batch_size=batch_size,
        collect_outputs=True,
    )
    return metrics, packed


def _group_metrics(batch, prediction, selected):
    selected = np.asarray(selected, dtype=bool)
    result = {}

    for horizon in range(prediction.shape[-1]):
        target = batch.speed_targets[selected, :, horizon]
        mask = batch.speed_target_mask[selected, :, horizon]

        metric = masked_speed_metrics(
            target,
            mask,
            prediction[selected, :, horizon],
        )

        total_edges = int((~batch.edge_padding_mask[selected]).sum())
        metric["label_coverage"] = (
            float(metric["count"] / total_edges)
            if total_edges
            else 0.0
        )
        result[str((horizon + 1) * 5)] = metric

    return result


def _breakdown_metrics(
    model,
    dataset,
    scenarios,
    device,
    *,
    batch_size,
):
    """Report per-map and incident-window metrics without giant tensors."""
    predictions = []
    batches = []

    for batch in _iter_batches(
        dataset,
        batch_size,
        shuffle=False,
    ):
        outputs = _forward_batch(
            model,
            batch,
            scenarios,
            device,
        )
        predictions.append(
            outputs["speed_ratio"].detach().cpu().numpy()
        )
        batches.append(batch)

    prediction = np.concatenate(
        [
            _pad_edge_axis(p, _global_edge_count(dataset), fill_value=0.0)
            for p in predictions
        ],
        axis=0,
    )
    batch = _batch_from_windows(dataset.windows)

    groups = {"per_map": {}}

    scenario_ids = np.asarray(batch.scenario_ids)
    incident = np.asarray([
        bool(
            np.nan_to_num(
                window.features[..., 5],
                nan=0.0,
            ).max() > 0.0
        )
        for window in dataset.windows
    ])

    groups["incident"] = _group_metrics(
        batch,
        prediction,
        incident,
    )

    for scenario_id in sorted(set(batch.scenario_ids)):
        groups["per_map"][scenario_id] = _group_metrics(
            batch,
            prediction,
            scenario_ids == scenario_id,
        )

    return groups


def _calibration_inputs(
    model,
    dataset,
    scenarios,
    device,
    *,
    batch_size,
):
    predictions = []
    targets = []
    masks = []

    for batch in _iter_batches(
        dataset,
        batch_size,
        shuffle=False,
    ):
        outputs = _forward_batch(
            model,
            batch,
            scenarios,
            device,
        )
        predictions.append(
            outputs["speed_ratio"].detach().cpu().numpy()
        )
        targets.append(batch.speed_targets)
        masks.append(batch.speed_target_mask)

    width = _global_edge_count(dataset)
    return (
        np.concatenate(
            [_pad_edge_axis(p, width, fill_value=0.0) for p in predictions],
            axis=0,
        ),
        np.concatenate(
            [_pad_edge_axis(t, width, fill_value=np.nan) for t in targets],
            axis=0,
        ),
        np.concatenate(
            [_pad_edge_axis(m, width, fill_value=False) for m in masks],
            axis=0,
        ),
    )


def _batch_metrics_for_baselines(dataset):
    batch = _batch_from_windows(dataset.windows)
    persistence = PersistenceForecaster()
    return batch, persistence


def train_forecaster(
    pilot_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 60,
    seed: int = 26137,
    width: int = 32,
    device: str = "cpu",
    batch_size: int = 2,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    heads: int = 4,
    layers: int = 2,
    scheduler: str = "cosine",
    patience: int = 25,
    min_delta: float = 1e-4,
    grad_clip: float = 1.0,
    final_eval: bool = True,
    window_cache: str | None = None,
    datasets: dict | None = None,
    manifest_path: Path | None = None,
) -> dict:
    """Train on Step-13 windows and write a reproducible artifact manifest.

    Early stopping halts when validation MAE fails to improve by ``min_delta``
    for ``patience`` consecutive epochs (``patience=0`` disables it); the best
    checkpoint is always retained. ``scheduler`` is one of
    ``{"cosine", "plateau", "none"}``.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not (0.0 < lr <= 1.0):
        raise ValueError("lr must lie in (0, 1]")
    if not (0.0 <= weight_decay <= 1.0):
        raise ValueError("weight_decay must lie in [0, 1]")
    if not (0.0 <= dropout < 1.0):
        raise ValueError("dropout must lie in [0, 1)")
    if heads <= 0 or layers <= 0:
        raise ValueError("heads and layers must be positive")
    if scheduler not in ("cosine", "plateau", "none"):
        raise ValueError("scheduler must be cosine, plateau or none")
    if patience < 0:
        raise ValueError("patience cannot be negative")
    if min_delta < 0.0:
        raise ValueError("min_delta cannot be negative")
    if grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")

    seed_everything(seed)

    device_obj = torch.device(device)
    pilot_dir = Path(pilot_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest, _source_manifest_path = load_corpus_manifest(
        pilot_dir, manifest_path
    )
    scenarios = load_corpus_scenarios(pilot_dir, split_manifest)

    print(
        f"Loading causal forecast windows from {pilot_dir}...",
        flush=True,
    )
    if datasets is None:
        # Fresh load (parallel builds + optional NPZ cache, either schema).
        datasets = load_pilot_windows(
            pilot_dir, cache_dir=window_cache, manifest_path=manifest_path
        )
    else:
        # Shared preloaded datasets (e.g. sweep across configs): same bytes,
        # no reload. Scaler must already be fit on training data.
        if window_cache is not None:
            print(
                "Using shared preloaded datasets; window_cache not re-read.",
                flush=True,
            )

    if "validation" not in datasets:
        raise ValueError(
            "Forecaster training requires a separate validation split. "
            "Do not use the ten local training episodes as validation data; "
            "use them only for format and pipeline smoke tests."
        )

    train_windows = len(datasets["train"])
    validation_windows = len(datasets["validation"])
    test_windows = len(datasets.get("test", ()))

    supervision = {name: _supervision_report(dataset) for name, dataset in datasets.items()}

    print(
        "Windows loaded: "
        f"train={train_windows}, "
        f"validation={validation_windows}, "
        f"test={test_windows}",
        flush=True,
    )

    model = CausalGNNTransformer(
        width=width,
        heads=heads,
        layers=layers,
        dropout=dropout,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    lr_scheduler = None
    if scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

    history = []
    best = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    rng = np.random.default_rng(seed)

    steps_per_epoch = (
        (train_windows + batch_size - 1)
        // batch_size
    )

    print(
        f"Training on {device_obj} | "
        f"batch_size={batch_size} | "
        f"steps/epoch={steps_per_epoch}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        running_speed = 0.0
        running_traversal = 0.0
        count = 0

        for batch in _iter_batches(
            datasets["train"],
            batch_size,
            shuffle=True,
            rng=rng,
        ):
            optimizer.zero_grad(set_to_none=True)

            outputs = _forward_batch(
                model,
                batch,
                scenarios,
                device_obj,
            )

            losses = forecast_loss(
                outputs,
                batch,
                device=device_obj,
            )

            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
            optimizer.step()

            running_total += float(
                losses["total"].detach().cpu()
            )
            running_speed += float(
                losses["speed"].detach().cpu()
            )
            running_traversal += float(
                losses["traversal"].detach().cpu()
            )
            count += 1

        val_metrics, _ = _evaluate(
            model,
            datasets["validation"],
            scenarios,
            device_obj,
            batch_size=batch_size,
        )

        val_values = [
            metric["mae"]
            for metric in val_metrics.values()
            if metric["mae"] is not None
        ]
        val_mae = (
            float(np.mean(val_values))
            if val_values
            else float("inf")
        )

        row = {
            "epoch": epoch,
            "train_loss": running_total / max(count, 1),
            "train_speed_loss": running_speed / max(count, 1),
            "train_traversal_loss": running_traversal / max(count, 1),
            "validation_speed_mae": val_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "validation_speed_coverage": {
                horizon: val_metrics[horizon].get("label_coverage", 0.0)
                for horizon in val_metrics
            },
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train={row['train_loss']:.6f} | "
            f"speed={row['train_speed_loss']:.6f} | "
            f"travel={row['train_traversal_loss']:.6f} | "
            f"val_MAE={val_mae:.6f} | "
            f"lr={row['lr']:.2e}",
            flush=True,
        )

        if val_mae < best - min_delta:
            best = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            # Persist immediately: a later crash must not lose the best model.
            torch.save(best_state, output_dir / "best_weights.pt")
        else:
            epochs_without_improvement += 1

        if lr_scheduler is not None:
            if scheduler == "plateau":
                lr_scheduler.step(val_mae)
            else:
                lr_scheduler.step()

        if patience and epochs_without_improvement >= patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: no improvement "
                f"for {patience} epochs (best val_MAE={best:.6f} "
                f"at epoch {best_epoch}).",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("No model checkpoint was produced")

    model.load_state_dict(best_state)

    torch.save(
        model.state_dict(),
        output_dir / "weights.pt",
    )

    (output_dir / "training_curve.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    if not final_eval:
        # Screening mode: skip the expensive full-corpus eval/baselines/
        # calibration. Selection reads best val MAE from training_curve.json.
        manifest = {
            "artifact": "forecaster_v1_screening",
            "architecture": model.config,
            "seed": seed,
            "device": str(device_obj),
            "hyperparameters": {
                "epochs_requested": epochs,
                "epochs_run": len(history),
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "heads": heads,
                "layers": layers,
                "scheduler": scheduler,
                "grad_clip": grad_clip,
                "batch_size": batch_size,
            },
            "early_stopping": {
                "enabled": bool(patience),
                "patience": patience,
                "min_delta": min_delta,
                "stopped_early": stopped_early,
                "best_epoch": best_epoch,
                "best_validation_mae": best,
            },
            "train_windows": train_windows,
            "validation_windows": validation_windows,
            "test_windows": test_windows,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return manifest

    metrics = {}
    breakdowns = {}

    for name, dataset in datasets.items():
        metrics[name], _ = _evaluate(
            model,
            dataset,
            scenarios,
            device_obj,
            batch_size=batch_size,
        )
        breakdowns[name] = _breakdown_metrics(
            model,
            dataset,
            scenarios,
            device_obj,
            batch_size=batch_size,
        )

    validation_prediction, validation_targets, validation_masks = (
        _calibration_inputs(
            model,
            datasets["validation"],
            scenarios,
            device_obj,
            batch_size=batch_size,
        )
    )

    calibration = fit_residual_calibration(
        validation_prediction,
        validation_targets,
        validation_masks,
        nominal_coverage=0.80,
    )

    interval_reports = {}

    for name, dataset in datasets.items():
        prediction, targets, masks = _calibration_inputs(
            model,
            dataset,
            scenarios,
            device_obj,
            batch_size=batch_size,
        )

        _, lower, upper = calibrated_arrays(
            prediction,
            calibration,
        )

        interval_reports[name] = interval_metrics(
            lower,
            upper,
            targets,
            masks,
            nominal_coverage=calibration.nominal_coverage,
        )

    (output_dir / "uncertainty.json").write_text(
        json.dumps(
            {
                **calibration.to_dict(),
                "validation": interval_reports["validation"],
                "test": interval_reports.get("test"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    baseline_metrics = {}

    train_batch = _batch_from_windows(
        datasets["train"].windows
    )
    temporal = TemporalOnlyForecaster().fit(train_batch)
    persistence = PersistenceForecaster()

    for name, dataset in datasets.items():
        batch = _batch_from_windows(dataset.windows)

        baseline_metrics[name] = {
            "persistence": masked_speed_metrics(
                batch.speed_targets,
                batch.speed_target_mask,
                persistence.predict(batch),
            ),
            "temporal_only": masked_speed_metrics(
                batch.speed_targets,
                batch.speed_target_mask,
                temporal.predict(batch),
            ),
        }

    cutoff_values = [
        int(np.nanmax(window.label_available_at_s))
        for window in datasets["train"].windows
        if np.isfinite(window.label_available_at_s).any()
    ]

    if not cutoff_values:
        raise RuntimeError(
            "Training windows contain no finite label_available_at_s values"
        )

    cutoff = max(cutoff_values)

    # weights.pt already saved from the best in-RAM state above; re-save here
    # only if the file is missing (e.g. best_weights.pt recovery path).
    if not (output_dir / "weights.pt").exists():
        torch.save(
            model.state_dict(),
            output_dir / "weights.pt",
        )

    shutil.copy2(source_manifest_path, output_dir / source_manifest_path.name)

    datasets["train"].scaler.save(
        output_dir / "scaler.json"
    )

    manifest = {
        "artifact": "forecaster_v1",
        "corpus_manifest_file": source_manifest_path.name,
        "architecture": model.config,
        "seed": seed,
        "device": str(device_obj),
        "hyperparameters": {
            "epochs_requested": epochs,
            "epochs_run": len(history),
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "heads": heads,
            "layers": layers,
            "scheduler": scheduler,
            "grad_clip": grad_clip,
            "batch_size": batch_size,
            "window_cache": window_cache,
        },
        "early_stopping": {
            "enabled": bool(patience),
            "patience": patience,
            "min_delta": min_delta,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_validation_mae": best,
        },
        "training_cutoff_s": cutoff,
        "feature_schema": [
            "speed_ratio",
            "occupancy",
            "halting",
            "observation_age_min",
            "missing",
            "known_closed",
        ],
        "target_schema": {
            "speed_ratio": [5, 10, 15],
            "traversal_time_s": [5, 10, 15],
        },
        "split_policy": split_manifest.get("rule"),
        "metrics": metrics,
        "breakdowns": breakdowns,
        "uncertainty": {
            **calibration.to_dict(),
            "validation": interval_reports["validation"],
            "test": interval_reports["test"],
        },
        "baseline_metrics": baseline_metrics,
        "train_windows": train_windows,
        "validation_windows": validation_windows,
        "test_windows": test_windows,
        "batch_size": batch_size,
        "loss": {
            "speed_weight": 1.0,
            "traversal_weight": 1.0,
            "traversal_log_space": True,
            "traversal_scale_s": None,
            "rationale": "Traversal optimized in log1p space (seconds) so it cannot numerically dominate the dimensionless speed-ratio objective.",
        },
        "supervision": supervision,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    (output_dir / "training_curve.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=26137)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=("cosine", "plateau", "none"),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early-stopping patience in epochs (0 disables).",
    )
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--no-final-eval",
        action="store_true",
        help="Screening mode: skip full-corpus eval/baselines/calibration.",
    )
    parser.add_argument(
        "--window-cache",
        type=str,
        default=None,
        help="Per-episode NPZ window cache dir (repeat loads in minutes).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Torch device for training (mps on Apple Silicon, cpu fallback).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit corpus manifest; useful for a named smoke subset.",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            train_forecaster(
                args.pilot,
                args.output,
                epochs=args.epochs,
                seed=args.seed,
                width=args.width,
                batch_size=args.batch_size,
                device=args.device,
                final_eval=not args.no_final_eval,
                window_cache=args.window_cache,
                manifest_path=args.manifest,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
