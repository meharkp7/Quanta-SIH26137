# Step 22 evaluation skeleton report

Certified optimum (exact MILP reference): **1575.0**

> Terminology: heuristic entries report best_feasible_objective and gap vs the certified reference. A zero gap means matches best-known reference, never optimal.

## Evidence: `/Users/meharkapoor7/Quanta-SIH26137/artifacts/step9_research/qpso_vs_alns_5job.json`

Fixture: `fixtures/step3/base/scenario.json`

| seed | qpso_best_feasible | qpso_gap | qpso_wall_s | qpso_agree | alns_best_feasible | alns_gap | alns_wall_s | alns_agree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1575.0 | 0.000000 | 2.618 | True | 1575.0 | 0.000000 | 0.158 | True |
| 2 | 1575.0 | 0.000000 | 2.940 | True | 1575.0 | 0.000000 | 0.158 | True |
| 3 | 1575.0 | 0.000000 | 4.509 | True | 1575.0 | 0.000000 | 0.165 | True |

## Headline

- n_evidence_files: 1
- n_validator_checks: 6
- n_validator_agree: 6
- n_zero_gap_matches: 6
- verdict: all stored validator verdicts agree and all feasible runs match the certified reference
