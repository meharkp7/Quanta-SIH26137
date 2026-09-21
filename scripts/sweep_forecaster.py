"""Short-epoch hyperparameter screening for the GNN-Transformer forecaster.

Runs each config in CONFIGS (or a JSON list passed via --configs) for a few
epochs on the given device (MPS default on Apple Silicon) and ranks by best
validation MAE. Screening never touches the test split for selection — test
metrics are reported read-only for the winner.

Usage:
    python3 scripts/sweep_forecaster.py <corpus_dir> <output_base> [--epochs 5]

Outputs <output_base>/<config_name>/(train_forecaster artifacts) plus
<output_base>/sweep_report.json with the ranking.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

# Make `src.*` importable regardless of caller environment (PYTHONPATH,
# cwd, or direct `python3 scripts/...` invocation which puts scripts/ first).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.learning.train_forecaster import seed_everything, train_forecaster

DEFAULT_GRID = [
    {"width": 32, "lr": 2e-3, "weight_decay": 1e-4, "dropout": 0.1},
    {"width": 32, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1},
    {"width": 32, "lr": 2e-3, "weight_decay": 1e-3, "dropout": 0.2},
    {"width": 64, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1},
    {"width": 64, "lr": 2e-3, "weight_decay": 1e-4, "dropout": 0.2},
    {"width": 64, "lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.2},
]


def config_name(index: int, config: dict) -> str:
    parts = [f"{key}{value}" for key, value in sorted(config.items())]
    return f"cfg{index:02d}_" + "_".join(parts).replace(".", "p")


def run_sweep(
    corpus_dir: Path,
    output_base: Path,
    configs: list[dict],
    *,
    epochs: int,
    seed: int,
    device: str,
    batch_size: int,
    window_cache: str | None = None,
) -> dict:
    seed_everything(seed)
    output_base.mkdir(parents=True, exist_ok=True)
    # Load once, share across configs: the dominant cost is window building
    # (~1 h cold on 1750 eps), not training. Same bytes for every config.
    from src.learning.loader import load_pilot_windows

    print("Sweep: loading shared datasets once...", flush=True)
    shared = load_pilot_windows(corpus_dir, cache_dir=window_cache)
    rows = []
    for index, config in enumerate(configs):
        name = config_name(index, config)
        out = output_base / name
        print(f"[{index + 1}/{len(configs)}] {name}", flush=True)
        train_forecaster(
            corpus_dir,
            out,
            epochs=epochs,
            seed=seed,
            device=device,
            batch_size=batch_size,
            patience=0,  # fixed screening budget; selection by best val MAE
            final_eval=False,  # skip expensive full-corpus eval in screening
            window_cache=window_cache,
            datasets=shared,
            **config,
        )
        curve = json.loads((out / "training_curve.json").read_text())
        best_val = min(
            row["validation_speed_mae"]
            for row in curve
            if row["validation_speed_mae"] is not None
            and row["validation_speed_mae"] != float("inf")
        )
        rows.append(
            {
                "name": name,
                "config": config,
                "best_validation_mae": float(best_val),
                "mean_test_mae": None,  # test untouched during screening
            }
        )
    rows.sort(key=lambda r: r["best_validation_mae"])
    report = {
        "screening_epochs": epochs,
        "seed": seed,
        "device": device,
        "selection_split": "validation curve best (test untouched in screening)",
        "ranking": rows,
        "winner": rows[0]["name"] if rows else None,
    }
    (output_base / "sweep_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output_base", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=26137)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--window-cache",
        type=str,
        default=None,
        help="Per-episode NPZ window cache dir shared across configs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="JSON list of config dicts; defaults to DEFAULT_GRID.",
    )
    args = parser.parse_args()
    configs = json.loads(args.configs) if args.configs else DEFAULT_GRID
    print(
        json.dumps(
            run_sweep(
                args.corpus,
                args.output_base,
                configs,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
                batch_size=args.batch_size,
                window_cache=args.window_cache,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
