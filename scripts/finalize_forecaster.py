"""Streamed final evaluation for a trained forecaster (no giant tensors).

The in-training final eval in ``src.learning.train_forecaster`` concatenates
complete [N, E, H] split arrays (plus a full-train dense batch for the
temporal baseline). On the 1750-episode corpus_v2 that memory spike gets the
process SIGKILLed after the last epoch: weights survive (``best_weights.pt``)
but no ``manifest.json``/``uncertainty.json`` is written.

This script rebuilds exactly those artifacts from saved weights, streaming
in small chunks with O(batch) peak memory:

- per-horizon MAE/RMSE/relative-MAE/mean-error/count/coverage accumulators
  (same float64 math as ``masked_speed_metrics``),
- split-conformal radii from valid-label residuals only (~2.6% of cells),
- ridge temporal baseline via streaming XTX/XTy accumulation (bit-identical
  to full-batch ``TemporalOnlyForecaster.fit``),
- per-map and incident-window breakdowns, baselines, scaler, manifest.

Usage:
    python3 scripts/finalize_forecaster.py <corpus_dir> <output_dir> [--device mps]

``<output_dir>`` must already contain ``best_weights.pt`` and
``training_status.json`` (written by the backend training launch); the model
config (width/heads/layers/seed) is read from the status file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from math import ceil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.learning.baselines import PersistenceForecaster, TemporalOnlyForecaster
from src.learning.gnn_transformer import CausalGNNTransformer
from src.learning.loader import WindowDataset, load_pilot_windows
from src.learning.scaling import FeatureScaler
from src.learning.schema import FEATURE_NAMES
from src.learning.train_forecaster import (
    _forward_batch,
    _iter_batches,
    _supervision_report,
    seed_everything,
)
from src.learning.uncertainty import ResidualCalibration

HORIZONS = (5, 10, 15)


class HorizonAccumulator:
    """Streaming equivalent of masked_speed_metrics (float64, same order)."""

    def __init__(self, horizons: int = 3) -> None:
        self.sum_abs = np.zeros(horizons, dtype=np.float64)
        self.sum_sq = np.zeros(horizons, dtype=np.float64)
        self.sum_rel = np.zeros(horizons, dtype=np.float64)
        self.sum_err = np.zeros(horizons, dtype=np.float64)
        self.count = np.zeros(horizons, dtype=np.int64)

    def add(self, targets: np.ndarray, mask: np.ndarray, prediction: np.ndarray) -> None:
        for h in range(self.sum_abs.shape[0]):
            y = np.asarray(targets[:, :, h], dtype=np.float64)
            m = np.asarray(mask[:, :, h], dtype=bool)
            p = np.asarray(prediction[:, :, h], dtype=np.float64)
            valid = m & np.isfinite(y) & np.isfinite(p)
            if not np.any(valid):
                continue
            err = p[valid] - y[valid]
            absolute = np.abs(err)
            scale = np.maximum(np.abs(y[valid]), 1e-6)
            self.sum_abs[h] += absolute.sum()
            self.sum_sq[h] += np.square(err).sum()
            self.sum_rel[h] += (absolute / scale).sum()
            self.sum_err[h] += err.sum()
            self.count[h] += int(valid.sum())

    def metrics(self, total_edges: int) -> dict:
        out = {}
        for i, horizon in enumerate(HORIZONS):
            count = int(self.count[i])
            if count == 0:
                out[str(horizon)] = {"count": 0, "mae": None, "rmse": None}
                continue
            out[str(horizon)] = {
                "count": count,
                "mae": float(self.sum_abs[i] / count),
                "rmse": float(np.sqrt(self.sum_sq[i] / count)),
                "relative_mae": float(self.sum_rel[i] / count),
                "mean_error": float(self.sum_err[i] / count),
                "label_coverage": float(count / total_edges) if total_edges else 0.0,
            }
        return out


def conformal_radii(residuals_per_horizon: list[list[float]], nominal: float = 0.80) -> ResidualCalibration:
    """Same order statistic as fit_residual_calibration, from 1D residuals."""
    radii, counts = [], []
    for residuals in residuals_per_horizon:
        counts.append(len(residuals))
        if not residuals:
            radii.append(0.0)
            continue
        ordered = np.sort(np.asarray(residuals, dtype=np.float64))
        rank = int(ceil((ordered.size + 1) * nominal)) - 1
        radii.append(float(ordered[min(max(rank, 0), ordered.size - 1)]))
    return ResidualCalibration(tuple(radii), nominal, tuple(counts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device", type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()

    corpus, output = args.corpus, args.output
    status = json.loads((output / "training_status.json").read_text(encoding="utf-8"))
    config = status["config"]
    seed_everything(int(config.get("seed", 26137)))
    device = torch.device(args.device)
    curve = json.loads((output / "training_curve.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((corpus / "corpus_manifest.json").read_text(encoding="utf-8"))

    print("Loading windows (cache hits expected)...", flush=True)
    datasets = load_pilot_windows(corpus, cache_dir=str(corpus / ".window_cache"))
    scaler = FeatureScaler.fit_windows(datasets["train"].windows, FEATURE_NAMES)
    datasets = {
        name: WindowDataset(ds.windows, scaler=scaler) for name, ds in datasets.items()
    }
    scaler.save(output / "scaler.json")

    from src.contracts.scenario import Scenario

    scenarios = {
        sid: Scenario.model_validate_json((corpus / rel).read_text(encoding="utf-8"))
        for sid, rel in split_manifest.get("scenario_files", {}).items()
    }

    model = CausalGNNTransformer(
        width=int(config["width"]),
        heads=int(config["heads"]),
        layers=int(config["layers"]),
    ).to(device)
    model.load_state_dict(
        torch.load(output / "best_weights.pt", map_location=device, weights_only=True)
    )
    model.eval()

    total_edges = {
        name: sum(len(w.edge_ids) for w in ds.windows) for name, ds in datasets.items()
    }

    # Pass 1: model metrics + validation residuals + temporal ridge statistics.
    model_acc = {name: HorizonAccumulator() for name in datasets}
    residuals: list[list[float]] = [[], [], []]
    ridge_xtx: np.ndarray | None = None
    ridge_xty: np.ndarray | None = None
    ridge_dim = 12 * len(FEATURE_NAMES) + 1
    persistence = PersistenceForecaster()
    temporal_probe = TemporalOnlyForecaster()

    with torch.no_grad():
        for name, ds in datasets.items():
            print(f"Pass 1: {name} ({len(ds.windows)} windows)...", flush=True)
            for batch in _iter_batches(ds, args.batch_size, shuffle=False):
                outputs = _forward_batch(
                    model, batch,
                    {sid: scenarios[sid] for sid in set(batch.scenario_ids)},
                    device,
                )
                pred = outputs["speed_ratio"].detach().cpu().numpy()
                model_acc[name].add(batch.speed_targets, batch.speed_target_mask, pred)
                if name == "validation":
                    for h in range(3):
                        y = batch.speed_targets[:, :, h]
                        m = batch.speed_target_mask[:, :, h]
                        valid = m & np.isfinite(y) & np.isfinite(pred[:, :, h])
                        residuals[h].extend(
                            np.abs(pred[:, :, h][valid].astype(np.float64)
                                   - y[valid].astype(np.float64)).tolist()
                        )
                if name == "train":
                    design, targets, masks = temporal_probe._xy(batch, require_targets=True)
                    x = design.astype(np.float64)
                    if ridge_xtx is None:
                        ridge_xtx = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
                        ridge_xty = np.zeros((x.shape[1], targets.shape[1]), dtype=np.float64)
                    rows = np.any(masks & np.isfinite(targets), axis=1)
                    ridge_xtx += x[rows].T @ x[rows]
                    for h in range(targets.shape[1]):
                        v = masks[:, h] & np.isfinite(targets[:, h])
                        if np.any(v):
                            ridge_xty[:, h] += x[v].T @ targets[v, h].astype(np.float64)

    calibration = conformal_radii(residuals)
    temporal = TemporalOnlyForecaster()
    system = ridge_xtx + temporal.ridge * np.eye(ridge_dim, dtype=np.float64)
    weights = np.zeros((ridge_dim, 3), dtype=np.float32)
    for h in range(3):
        weights[:, h] = np.linalg.solve(system, ridge_xty[:, h]).astype(np.float32)
    temporal.weights, temporal.feature_dim, temporal.horizon_count = weights, ridge_dim, 3

    # Pass 2: baselines + calibrated interval coverage.
    base_acc = {
        name: {"persistence": HorizonAccumulator(), "temporal_only": HorizonAccumulator()}
        for name in datasets
    }
    cover: dict[str, dict[str, list]] = {
        name: {"covered": [0, 0, 0], "width": [0.0, 0.0, 0.0], "count": [0, 0, 0]}
        for name in datasets
    }
    radii = np.asarray(calibration.radii, dtype=np.float64)
    with torch.no_grad():
        for name, ds in datasets.items():
            print(f"Pass 2: {name}...", flush=True)
            for batch in _iter_batches(ds, args.batch_size, shuffle=False):
                outputs = _forward_batch(
                    model, batch,
                    {sid: scenarios[sid] for sid in set(batch.scenario_ids)},
                    device,
                )
                pred = outputs["speed_ratio"].detach().cpu().numpy().astype(np.float64)
                base_acc[name]["persistence"].add(
                    batch.speed_targets, batch.speed_target_mask, persistence.predict(batch))
                base_acc[name]["temporal_only"].add(
                    batch.speed_targets, batch.speed_target_mask, temporal.predict(batch))
                for h in range(3):
                    y = batch.speed_targets[:, :, h].astype(np.float64)
                    m = batch.speed_target_mask[:, :, h] & np.isfinite(y)
                    p = pred[:, :, h]
                    valid = m & np.isfinite(p)
                    count = int(valid.sum())
                    if not count:
                        continue
                    lower, upper = p - radii[h], p + radii[h]
                    cover[name]["covered"][h] += int(((y >= lower) & (y <= upper) & valid).sum())
                    cover[name]["width"][h] += float((2.0 * radii[h]) * count)
                    cover[name]["count"][h] += count

    metrics = {name: acc.metrics(total_edges[name]) for name, acc in model_acc.items()}
    baseline_metrics = {
        name: {
            kind: acc.metrics(total_edges[name]) for kind, acc in kinds.items()
        }
        for name, kinds in base_acc.items()
    }
    interval_reports = {}
    for name in datasets:
        horizons = {}
        for i, horizon in enumerate(HORIZONS):
            count = cover[name]["count"][i]
            horizons[str(horizon)] = {
                "nominal_coverage": calibration.nominal_coverage,
                "actual_coverage": (cover[name]["covered"][i] / count) if count else 0.0,
                "mean_width": (cover[name]["width"][i] / count) if count else 0.0,
                "count": count,
            }
        interval_reports[name] = horizons

    # Breakdowns: per-map + incident windows, streamed with running offsets
    # (_iter_batches with shuffle=False yields dataset order).
    breakdowns = {}
    with torch.no_grad():
        for name, ds in datasets.items():
            print(f"Breakdowns: {name}...", flush=True)
            incident_mask = np.asarray([
                bool(np.nan_to_num(w.features[..., 5], nan=0.0).max() > 0.0)
                for w in ds.windows
            ])
            map_names = sorted(set(w.scenario_id for w in ds.windows))
            per_map = {sid: HorizonAccumulator() for sid in map_names}
            incident_acc = HorizonAccumulator()
            offset = 0
            for batch in _iter_batches(ds, args.batch_size, shuffle=False):
                outputs = _forward_batch(
                    model, batch,
                    {s: scenarios[s] for s in set(batch.scenario_ids)}, device)
                pred = outputs["speed_ratio"].detach().cpu().numpy()
                rows = len(batch.episode_ids)
                idx = np.arange(offset, offset + rows)
                for sid, acc in per_map.items():
                    sel = np.asarray(batch.scenario_ids) == sid
                    if np.any(sel):
                        acc.add(batch.speed_targets[sel], batch.speed_target_mask[sel], pred[sel])
                if np.any(incident_mask[idx]):
                    sel = incident_mask[idx]
                    incident_acc.add(batch.speed_targets[sel], batch.speed_target_mask[sel], pred[sel])
                offset += rows
            edge_totals = {}
            for sid in map_names:
                edge_totals[sid] = sum(len(w.edge_ids) for w in ds.windows if w.scenario_id == sid)
            incident_edges = int(sum(len(w.edge_ids) for w, flag in zip(ds.windows, incident_mask) if flag))
            breakdowns[name] = {
                "per_map": {sid: acc.metrics(edge_totals[sid]) for sid, acc in per_map.items()},
                "incident": incident_acc.metrics(incident_edges),
            }

    (output / "uncertainty.json").write_text(
        json.dumps({**calibration.to_dict(),
                    "validation": interval_reports["validation"],
                    "test": interval_reports["test"]}, indent=2), encoding="utf-8")

    cutoff_values = [
        int(np.nanmax(w.label_available_at_s)) for w in datasets["train"].windows
        if np.isfinite(w.label_available_at_s).any()
    ]
    cutoff = max(cutoff_values)
    shutil.copy2(corpus / "corpus_manifest.json", output / "corpus_manifest.json")

    best_val = min(r["validation_speed_mae"] for r in curve)
    best_epoch = min(curve, key=lambda r: r["validation_speed_mae"])["epoch"]
    manifest = {
        "artifact": "forecaster_v1",
        "architecture": model.config,
        "seed": config.get("seed", 26137),
        "device": args.device,
        "hyperparameters": {**config, "epochs_run": len(curve)},
        "early_stopping": {
            "enabled": bool(config.get("patience", 0)),
            "patience": config.get("patience", 0),
            "best_epoch": best_epoch,
            "best_validation_mae": best_val,
        },
        "training_cutoff_s": cutoff,
        "feature_schema": list(FEATURE_NAMES),
        "target_schema": {"speed_ratio": [5, 10, 15], "traversal_time_s": [5, 10, 15]},
        "split_policy": split_manifest.get("rule", split_manifest.get("split_rule")),
        "metrics": metrics,
        "breakdowns": breakdowns,
        "uncertainty": {**calibration.to_dict(),
                        "validation": interval_reports["validation"],
                        "test": interval_reports["test"]},
        "baseline_metrics": baseline_metrics,
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "batch_size": args.batch_size,
        "loss": {
            "speed_weight": 1.0, "traversal_weight": 1.0,
            "traversal_log_space": True, "traversal_scale_s": None,
            "rationale": "Traversal optimized in log1p space (seconds) so it cannot numerically dominate the dimensionless speed-ratio objective.",
        },
        "supervision": {n: _supervision_report(d) for n, d in datasets.items()},
        "finalized_by": "scripts/finalize_forecaster.py (streamed eval; training scaler refit via FeatureScaler.fit_windows on identical train windows)",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "test_mae": {
        h: metrics["test"][h]["mae"] for h in metrics["test"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
