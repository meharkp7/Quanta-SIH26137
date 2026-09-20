"""Labels-side traversal winsorization REPORT for corpus_v2 (report only).

Raw labels in labels.jsonl are NEVER modified: heavy-tail handling must stay
an explicit, reviewable modeling choice (log1p / winsorize / robust loss),
not a silent mutation of stored semantics. This script records, per episode
(sidecar <ep>/label_winsor.json) and corpus-wide (label_winsor_report.json):

    raw         moments of valid realized_traversal labels (n, p50, p99, max, mean)
    cap_s       reporting winsorization cap (CLI, default 120 s)
    winsorized  same moments with values clipped at cap_s (computed, not stored)
    exceedance  fraction of valid labels above cap_s
    speed_zero  fraction of valid speed_proxy labels exactly 0.0
                (closed-edge supervision signal; ~0 in corpus_delhi)

Usage:
    python3 scripts/corpus_v2_label_report.py --corpus artifacts/corpus_v2 [--cap-s 120]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def episode_stats(ep_dir: Path, cap_s: float) -> dict:
    trav: list[float] = []
    n_speed_valid = 0
    n_speed_zero = 0
    n_labels = 0
    with (ep_dir / "labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            n_labels += 1
            d = json.loads(line)
            if d.get("missing") or d.get("value") is None:
                continue
            if d.get("target_kind") == "realized_traversal":
                trav.append(float(d["value"]))
            elif d.get("target_kind") == "speed_proxy":
                n_speed_valid += 1
                if float(d["value"]) == 0.0:
                    n_speed_zero += 1
    tv = np.asarray(trav, dtype=np.float64)
    if len(tv):
        clipped = np.minimum(tv, cap_s)
        raw = {
            "n": int(len(tv)),
            "mean": float(tv.mean()),
            "p50": float(np.median(tv)),
            "p99": float(np.quantile(tv, 0.99)),
            "max": float(tv.max()),
        }
        win = {
            "n": int(len(clipped)),
            "mean": float(clipped.mean()),
            "p50": float(np.median(clipped)),
            "p99": float(np.quantile(clipped, 0.99)),
            "max": float(clipped.max()),
        }
        exceed = float(np.mean(tv > cap_s))
    else:
        raw = {"n": 0, "mean": float("nan"), "p50": float("nan"),
               "p99": float("nan"), "max": float("nan")}
        win = dict(raw)
        exceed = 0.0
    return {
        "episode": ep_dir.name,
        "n_labels": n_labels,
        "traversal_raw": raw,
        "cap_s": cap_s,
        "traversal_winsorized_at_cap": win,
        "traversal_exceedance_fraction": exceed,
        "speed_valid": n_speed_valid,
        "speed_zero_fraction": (n_speed_zero / n_speed_valid) if n_speed_valid else 0.0,
        "note": "raw labels.jsonl unmodified; winsorized moments computed for reporting only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cap-s", type=float, default=120.0)
    args = parser.parse_args()

    ep_dirs = sorted((args.corpus / "episodes").iterdir())
    per_episode = []
    for ep_dir in ep_dirs:
        if not ep_dir.is_dir() or not (ep_dir / "labels.jsonl").exists():
            continue
        stats = episode_stats(ep_dir, args.cap_s)
        (ep_dir / "label_winsor.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        per_episode.append(stats)

    all_trav_max = [s["traversal_raw"]["max"] for s in per_episode if s["traversal_raw"]["n"]]
    report = {
        "corpus": str(args.corpus),
        "cap_s": args.cap_s,
        "episodes": len(per_episode),
        "traversal_raw_max_over_episodes": float(max(all_trav_max)) if all_trav_max else float("nan"),
        "traversal_raw_p99_median": float(np.median(
            [s["traversal_raw"]["p99"] for s in per_episode if s["traversal_raw"]["n"]]
        )) if per_episode else float("nan"),
        "traversal_exceedance_mean": float(np.mean(
            [s["traversal_exceedance_fraction"] for s in per_episode]
        )) if per_episode else 0.0,
        "speed_zero_fraction_mean": float(np.mean(
            [s["speed_zero_fraction"] for s in per_episode]
        )) if per_episode else 0.0,
        "per_episode": {s["episode"]: s for s in per_episode},
        "note": "raw labels.jsonl unmodified everywhere; adopt a winsorize/log1p transform in modeling after review",
    }
    out = args.corpus / "label_winsor_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: v for k, v in report.items() if k != "per_episode"}, indent=2
    ))
    print(f"wrote {out} + {len(per_episode)} per-episode sidecars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
