# Step 22 — what this skeleton does NOT yet cover

This skeleton (`src/evaluation/compare.py` + `src/evaluation/replay.py`,
deliverable D10 seed) regenerates paired tables and replays stored step9
routes through the independent validator. It is NOT completed evaluation.

Open items for full Step 22:

1. SUMO paired-episode tapes — shared exogenous tapes (identical demand /
   closure / signal programs) replayed across policies; no tape format or
   runner exists yet.
2. Forecast ablations — evaluation with/without forecaster input; needs the
   frozen forecaster artifact plus an ablation switch in the runner.
3. PPO-vs-rules comparison — paired episodes of learned policy vs
   rule-based baseline on identical tapes; neither the policy checkpoint
   protocol nor the rules baseline harness exists here.
4. Scale sweeps — larger instances / longer horizons with matched budgets;
   current evidence covers only the 5-job fixture (seeds 1–3).
5. Frozen test manifest — a locked set of scenarios + seeds + budgets that
   final numbers must be reproduced from; not yet defined.
6. Reproducible figures — plots rendered from saved records (no figure
   script yet; the Markdown + JSON sidecar is the record to plot from).
