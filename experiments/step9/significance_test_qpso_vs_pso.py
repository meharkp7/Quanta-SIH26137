"""Statistical significance test for QPSO vs PSO paired comparisons.

Run from the repository root:

    python significance_test_qpso_vs_pso.py

Reads: artifacts/step8_research/r3_multi_instance/paired_runs.csv
(already produced by your Step 8/9 research runs — no new experiments
needed to use this script).

Why this matters
-----------------
`scale_summary.json` currently reports only win/loss/tie counts
("qpso_wins: 8, pso_wins: 2" at 30 customers). That is directional
evidence, not a significance claim, and it is easy for a judge or
reviewer to ask "is that difference real, or noise from 15 seeds?"

This script runs a paired Wilcoxon signed-rank test (appropriate here
because the same scenario/seed is solved by both optimizers, so the
comparison is paired, not independent) and reports the p-value and an
effect-size estimate alongside the existing win/loss counts, per
customer-count bucket. It also reports the count of usable (non-tied)
pairs, since Wilcoxon needs non-zero differences and repeat scenarios
sometimes yield identical results due to the fitness landscape being
flat at that scale.

Add scipy to requirements.txt if not already present:
    scipy>=1.11
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

DEFAULT_PATH = Path(
    "artifacts/step8_research/r3_multi_instance/paired_runs.csv"
)


def load_paired_diffs(path: Path) -> dict[int, list[float]]:
    """Group qpso_minus_pso differences by customer count."""

    diffs_by_scale: dict[int, list[float]] = defaultdict(list)

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            customers = int(row["customers"])
            diffs_by_scale[customers].append(
                float(row["qpso_minus_pso"])
            )

    return diffs_by_scale


def summarize(diffs: list[float]) -> dict:
    """Compute win/loss/tie counts and a Wilcoxon significance test.

    Convention: qpso_minus_pso < 0 means QPSO achieved a lower
    (better, since this is a minimization objective) fitness than PSO
    on that paired seed.
    """

    n = len(diffs)
    wins = sum(1 for d in diffs if d < 0)
    losses = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)

    nonzero = [d for d in diffs if d != 0]

    if len(nonzero) == 0:
        p_value = float("nan")
        statistic = float("nan")
    else:
        try:
            statistic, p_value = stats.wilcoxon(nonzero)
        except ValueError:
            # All-identical or too few non-zero diffs for the test.
            statistic, p_value = float("nan"), float("nan")

    mean_diff = statistics.mean(diffs)
    median_diff = statistics.median(diffs)

    return {
        "n": n,
        "qpso_wins": wins,
        "pso_wins": losses,
        "ties": ties,
        "usable_pairs": len(nonzero),
        "mean_qpso_minus_pso": mean_diff,
        "median_qpso_minus_pso": median_diff,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
    }


def required_sample_size_note(usable_pairs: int, p_value: float) -> str:
    """Plain-language guidance on whether more seeds are needed."""

    if usable_pairs < 20:
        return (
            f"Only {usable_pairs} non-tied pairs. Wilcoxon has low "
            "power below ~20 pairs — do not report this p-value as a "
            "conclusive result either way. Run more seeds before "
            "claiming or ruling out a QPSO advantage at this scale."
        )

    if p_value != p_value:  # NaN check
        return "Not enough variation in the data to run the test."

    if p_value < 0.05:
        return "Statistically significant at the conventional p<0.05 threshold."

    return (
        "NOT statistically significant at p<0.05. Do not claim a "
        "QPSO advantage at this scale without more paired seeds or "
        "a larger observed effect."
    )


def main() -> None:
    path = DEFAULT_PATH

    if not path.exists():
        raise SystemExit(
            f"Could not find {path}. Run this from the repository "
            f"root, or edit DEFAULT_PATH."
        )

    diffs_by_scale = load_paired_diffs(path)

    print(
        "customers | n | qpso_wins | pso_wins | ties | "
        "mean(qpso-pso) | wilcoxon_p"
    )
    print("-" * 78)

    for customers in sorted(diffs_by_scale):
        result = summarize(diffs_by_scale[customers])

        print(
            f"{customers:9d} | {result['n']:1d} | "
            f"{result['qpso_wins']:9d} | {result['pso_wins']:8d} | "
            f"{result['ties']:4d} | "
            f"{result['mean_qpso_minus_pso']:14.2f} | "
            f"{result['wilcoxon_p_value']:.4f}"
        )
        print(
            "   -> "
            + required_sample_size_note(
                result["usable_pairs"],
                result["wilcoxon_p_value"],
            )
        )

    print()
    print(
        "Recommendation: before writing 'QPSO outperforms PSO' anywhere "
        "in your submission, increase paired seeds per scale to at "
        "least 30-50 (ideally using the same protocol.json budget you "
        "already have) and re-run this script. Report the p-value next "
        "to the win/loss count in your benchmark tables."
    )


if __name__ == "__main__":
    main()