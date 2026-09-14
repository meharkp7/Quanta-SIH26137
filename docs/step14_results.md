# Step 14 results

The Step 13 `forecaster_v1` was evaluated on the six real SUMO episodes
(36 train, 9 validation, and 9 held-out test windows).  The uncertainty
policy is split conformal calibration on validation absolute residuals; test
labels are never used to tune its radii.

| split | nominal coverage | actual coverage | mean interval width | labels |
|---|---:|---:|---:|---:|
| validation | 80% | 80.7% | 0.4632 speed-ratio units | 540 |
| held-out test | 80% | 52.1% | 0.4650 speed-ratio units | 747 |

The held-out result is below nominal, so the intervals are measurable but not
calibrated for this map shift yet.  The implementation reports this failure
instead of presenting edge intervals as route-level confidence guarantees.

The forecast-aware provider is versioned and causal.  It interpolates between
issued horizons, uses current observed speed beyond the horizon by default,
decays reliance with forecast age and interval width, and lets known closures
remove edges from the legal graph.  The resulting `CostView` is passed through
`RouteEvaluator` → `Step7RouteEngine` → `RouteFitnessOracle` → QPSO.

In a controlled two-route graph, changing the forecast on the first route from
speed ratio 1.0 to 0.25 changed the selected shortest route from `(a, b)` to
`(c, d)`.  The Step 14 QPSO adapter also preserves `forecast_version` in the
routing state for every candidate evaluation.

The neural model's held-out speed metrics remain:

| horizon | MAE | RMSE |
|---:|---:|---:|
| 5 min | 0.3779 | 0.3998 |
| 10 min | 0.1396 | 0.1789 |
| 15 min | 0.2444 | 0.2942 |

For comparison, the existing temporal-only baseline has aggregate held-out
MAE 0.0372 and RMSE 0.0646.  The GNN–Transformer currently underperforms that
baseline, so it should not be promoted as a winning forecaster without more
map-diverse training data and recalibration.
