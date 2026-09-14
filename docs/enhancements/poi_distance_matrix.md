# POI path precomputation enhancement

> **Enhancement only.** This module is intentionally separate from the
> numbered SIH26137 implementation-plan steps.

## Purpose

The routing stack already owns a versioned `PathCache`. This enhancement adds
an explicit warm-up operation that runs one time-dependent Dijkstra search per
POI and departure bucket, then stores the resulting physical paths in that
existing cache. Subsequent `shortest_path()` calls use the normal cache path;
no alternate routing API or correctness path is introduced.

The cache remains keyed by graph, cost, forecast, source, destination, and
departure bucket. Cached entries contain physical edge/node structure rather
than time-dependent travel-time totals, so the normal `CostView` reconstructs
costs for the actual departure time when a cached path is consumed.

## Production API

- `DirectedPathBuilder.precompute_poi_paths(...)`
  - low-level graph/cache operation;
  - validates the POI set and departure times;
  - requires a `PathCache`;
  - records deterministic preprocessing statistics.
- `precompute_poi_distance_matrix(...)`
  - scenario-aware orchestration;
  - derives POIs from vehicle start/depot nodes and request access nodes;
  - derives a deterministic departure-time set from request temporal anchors.
- `RouteEvaluator.precompute_poi_matrix()`
  - production entry point for an already constructed evaluator.
- `RouteEvaluationConfig.precompute_poi_matrix`
  - opt-in eager warm-up at evaluator construction.
  - defaults to `False` to preserve latency and memory behavior for small or
    sparse workloads.

Eager mode requires an explicit `CostView`. The legacy travel-time-provider
construction path is deliberately rejected for eager precomputation because
cache version metadata is part of the correctness boundary.

## Why eager mode is opt-in

Precomputation trades repeated shortest-path work for upfront graph searches
and cache memory. It is not universally faster. The benchmark therefore
reports construction/precompute time separately from first-evaluation and
steady-state evaluation time.

Run the enhancement benchmark from the repository root with:

```bash
python -m experiments.enhancements.benchmark_poi_precompute --sizes 5 10 20 --repeats 3
```

The output is written to `artifacts/enhancements/poi_precompute.csv`.

## Correctness guarantees

The enhancement does not modify `shortest_path()` semantics. Tests cover:

- deterministic POI and departure-grid derivation;
- equality of cold and warmed path results;
- directed connectivity through the existing graph;
- cache-required behavior;
- repeatable/idempotent warm-up results;
- evaluator-level explicit and eager APIs;
- rejection of incompatible legacy-provider eager mode.

## Operational guidance

Use eager mode when a workload has a stable set of routing POIs and repeated
queries over the same graph/cost/forecast snapshot. Disable it for small
one-shot evaluations where the preprocessing cost is unlikely to amortize.

Network-state changes should continue to create a new `RouteEvaluator` via
`with_network_state(...)`. This preserves the existing evaluator lifetime
boundary and prevents stale topology state from being reused.
