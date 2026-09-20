"""Step 22 evaluation skeleton (deliverable D10).

Paired comparisons with shared exogenous tapes, reproducible from saved
records.  This package currently covers the step8/step9 evidence schemas:

- ``compare``: per-seed paired tables, gap-vs-reference math, Markdown+JSON
  report writer.  Gaps use best_feasible / best_known_reference semantics
  from ``src.optim.references`` — a heuristic result is NEVER labelled
  optimal.
- ``replay``: deterministic read-only re-validation of stored routes through
  the independent validator path used by ``src.optim.qpso_vs_alns``.

What stays open for full Step 22 — see ``src/evaluation/NOT_YET.md``:
SUMO paired-episode tapes, forecast ablations, PPO-vs-rules comparisons,
scale sweeps, and the frozen test manifest.  This skeleton must not be
mistaken for completed evaluation.
"""
