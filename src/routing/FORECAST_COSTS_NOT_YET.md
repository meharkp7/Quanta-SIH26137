# Step 14 remainder — NOT YET (forecast-aware routing costs)

Verified and closed by the Step 14 verifiable core (see
`artifacts/step14/forecast_cost_demo.json` and
`tests/step14/test_forecast_costs.py`):

- validation-residual intervals with nominal (80%) vs actual held-out
  coverage/width (validation ~80.0%, map-disjoint test ~76.4%);
- causal `Forecast` issue with `issued_at_s` / `forecast_version` /
  `model_version` metadata and pilot horizons [+5, +10, +15] s;
- horizon interpolation + declared beyond-horizon extension
  (`ForecastCostPolicy.extension`: `"current"` | `"last"`);
- validation-tuned stale/uncertain down-weighting, all constants in
  `ForecastCostPolicy` (`uncertainty_weight=0.35`,
  `stale_half_life_s=900.0`, `max_forecast_reliance=1.0`,
  `min_speed_ratio=0.10`);
- known closures override predicted accessibility (provider raises,
  `CostView` excludes the edge from the graph);
- FIFO-preserving sequential traversal evaluation (`RouteEvaluator`
  propagates entry times; `check_fifo` helper verifies no-overtaking);
- controlled proof that changing a forecast changes candidate costs AND
  the resulting route (E12 12.07 s -> 46.98 s flips N1->N2 from direct
  to via-N6; full 5-job plan stays feasible, objective 1590 -> 1790).

## Open

1. **Live PPO-loop integration.** `optimize_with_forecast` wires
   forecasts into QPSO/Step7 evaluation, but the rolling PPO controller
   does not yet re-issue forecasts online or feed updates back into
   scope selection. No latency measurement exists for issue -> cost ->
   route under a live loop.
2. **Quantile heads.** Intervals are split-conformal residuals around a
   point forecast (`residual-conformal-v1`), not learned quantile heads
   in `gnn_transformer.py`. Quantile outputs would give asymmetric,
   edge-conditional intervals.
3. **FIFO projection for steep recoveries.** `check_fifo` honestly
   reports violations when travel time drops faster than departures
   advance (near-total congestion clearing instantly); see
   `test_fifo_helper_detects_steep_recovery_violation`. A FIFO
   projection (e.g. greatest non-decreasing minorant on the arrival
   function) is needed before time-dependent Dijkstra is provably sound
   on such profiles.
4. **Latency measurement.** No serving-latency benchmark for forecast
   issue + `forecast_cost_view` construction + replan on Delhi-scale
   graphs.
5. **Recalibration cadence.** Radii are frozen from the pilot
   calibration split; no scheduled refit or drift monitor as new
   episodes arrive.
