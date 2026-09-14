"""Process-pool parallel fitness evaluation for AdaptiveQPSO / PSO.

Why this exists
----------------
Within one QPSO iteration, all `population_size` candidate positions are
generated up front and are evaluated completely independently of each
other (see `AdaptiveQPSO.optimize()` — evaluation order only matters for
the *sequential* personal-best/global-best bookkeeping that happens
*after* each fitness value is known, not for computing the fitness
values themselves). That makes fitness evaluation embarrassingly
parallel across a CPU's cores, and it is the single largest remaining
lever after the caching/graph-reuse fixes in `route_evaluator.py`,
`cost_view.py` and `path_builder.py` — those fixes reduced *serial*
per-evaluation cost; this reduces *wall-clock* cost by spending multiple
cores at once.

Why a plain ProcessPoolExecutor.submit(oracle, position) does not work
-----------------------------------------------------------------------
`RouteFitnessOracle` closes over a `Step7RouteEngine` / `RouteEvaluator`
/ `Scenario`. In this environment, `Scenario` (a pydantic model) and
`DirectedPathBuilder` (which holds a cache with an internal lock) are
**not picklable** — attempting to pickle them directly raises
`TypeError: cannot pickle 'mappingproxy' object` / `'_thread.RLock'
object`. `ProcessPoolExecutor.submit` pickles its callable and arguments
to send them to a worker, so sending the live oracle object fails.

The fix used here is the standard one for this situation: never send
the live oracle across the process boundary. Instead, each worker
process reconstructs its own oracle **once**, from the scenario's plain
JSON representation (`Scenario.model_dump(mode="json")`, which is
ordinary picklable dict/list/str/float/bool data), via a
`ProcessPoolExecutor(initializer=...)`. After that one-time setup, only
lightweight position vectors go out to workers and lightweight
`FitnessResult` objects come back. Each worker also gets, and reuses,
its own `RouteEvaluator` and therefore its own `PathCache` — so the
cache-hit-rate wins measured in this repo's own benchmarks (~98% at 40
customers) are preserved *within* each worker; they are just not shared
*across* workers, which is a reasonable trade-off since each worker
still converges to a high hit rate quickly on its own share of the
work.

Usage
-----
    from src.optim.parallel_oracle import ParallelRouteFitnessOracle

    oracle = ParallelRouteFitnessOracle(
        scenario,
        commitments=None,
        planning_time_s=0.0,
        repair=True,
        repair_penalty=0.0,
        max_workers=os.cpu_count(),
    )

    qpso = AdaptiveQPSO(qpso_config, oracle)
    result = qpso.optimize()
    oracle.shutdown()

`ParallelRouteFitnessOracle` is a drop-in replacement for
`RouteFitnessOracle` wherever a single-position callable is expected
(so it does not break any existing call site), and additionally exposes
`evaluate_batch(positions) -> list[FitnessResult]`, which
`AdaptiveQPSO.optimize()` (see the small, backward-compatible change in
`qpso.py`) will use automatically via `getattr(oracle, "evaluate_batch",
None)` when present, falling back to the original per-particle serial
loop for any oracle that does not define it.

Always call `shutdown()` (or use as a context manager) when done, to
terminate worker processes cleanly.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

from src.optim.common import FitnessResult

# Populated once per worker process by `_init_worker`. Deliberately a
# module-level global: this is how ProcessPoolExecutor's `initializer`
# pattern shares one reconstructed oracle across every task a given
# worker process subsequently handles, instead of rebuilding it (and
# throwing away its warmed-up PathCache) on every single call.
_worker_oracle = None


def _init_worker(
    scenario_json: dict,
    oracle_kwargs: dict,
) -> None:
    """Runs exactly once per worker process, before it handles any task.

    Rebuilds the full evaluation stack (Scenario -> RouteEvaluator ->
    Step7RouteEngine -> RouteFitnessOracle) locally inside the worker
    from plain JSON data, since none of those live objects can cross a
    process boundary via pickle in this environment.
    """

    global _worker_oracle

    # Imported inside the worker, not at module import time, so that a
    # process which never spawns workers (e.g. running fully serially)
    # never pays for importing the full routing stack twice.
    from src.contracts.scenario import Scenario
    from src.routing.route_evaluator import RouteEvaluator
    from src.routing.route_encoding import Step7RouteEngine
    from src.optim.qpso import RouteFitnessOracle

    scenario = Scenario.model_validate(scenario_json)
    evaluator = RouteEvaluator(scenario)
    engine = Step7RouteEngine(scenario, evaluator)

    _worker_oracle = RouteFitnessOracle(engine, **oracle_kwargs)


def _evaluate_in_worker(
    position: Sequence[float],
) -> FitnessResult:
    """Task function actually submitted to worker processes."""

    if _worker_oracle is None:
        raise RuntimeError(
            "Worker oracle was not initialized. This function must "
            "only be executed inside a ProcessPoolExecutor created by "
            "ParallelRouteFitnessOracle (which sets the required "
            "initializer)."
        )

    return _worker_oracle(position)


class ParallelRouteFitnessOracle:
    """Drop-in, batch-capable replacement for RouteFitnessOracle.

    `__call__(position)` behaves exactly like `RouteFitnessOracle` for
    any existing single-evaluation caller. `evaluate_batch(positions)`
    additionally lets a caller (namely `AdaptiveQPSO.optimize()`)
    evaluate a whole iteration's candidates across multiple processes
    at once.
    """

    def __init__(
        self,
        scenario: Any,
        *,
        commitments: Any = None,
        planning_time_s: float = 0.0,
        repair: bool = True,
        repair_penalty: float = 0.0,
        max_workers: int | None = None,
    ) -> None:
        # Only plain JSON-safe data is kept for worker reconstruction —
        # never the live `scenario` object, which is not picklable here.
        self._scenario_json = scenario.model_dump(mode="json")

        self._oracle_kwargs = dict(
            commitments=commitments,
            planning_time_s=planning_time_s,
            repair=repair,
            repair_penalty=repair_penalty,
        )

        self._max_workers = max_workers
        self._executor: ProcessPoolExecutor | None = None

        # A local, single-process oracle. This backs `__call__` (so this
        # class works anywhere a plain callable oracle is expected,
        # without requiring a batch call first) and is also the
        # sequential fallback for tiny batches, where process-pool
        # overhead would exceed any benefit.
        from src.routing.route_evaluator import RouteEvaluator
        from src.routing.route_encoding import Step7RouteEngine
        from src.optim.qpso import RouteFitnessOracle

        evaluator = RouteEvaluator(scenario)
        engine = Step7RouteEngine(scenario, evaluator)

        self._local_oracle = RouteFitnessOracle(
            engine,
            **self._oracle_kwargs,
        )

    def __call__(
        self,
        position: Sequence[float],
    ) -> FitnessResult:
        return self._local_oracle(position)

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_init_worker,
                initargs=(
                    self._scenario_json,
                    self._oracle_kwargs,
                ),
            )

        return self._executor

    def evaluate_batch(
        self,
        positions: Sequence[Sequence[float]],
    ) -> list[FitnessResult]:
        """Evaluate many candidate positions, in input order.

        Falls back to local sequential evaluation for batches of size
        <= 1 or when `max_workers` was explicitly set to 1, to avoid
        paying process-pool overhead where it cannot help.
        """

        if len(positions) <= 1 or self._max_workers == 1:
            return [
                self._local_oracle(position)
                for position in positions
            ]

        executor = self._ensure_executor()

        # executor.map preserves input order, matching the ordering
        # AdaptiveQPSO.optimize() expects when it zips results back
        # against `candidate_positions`.
        return list(
            executor.map(
                _evaluate_in_worker,
                positions,
            )
        )

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "ParallelRouteFitnessOracle":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    def __del__(self) -> None:
        # Best-effort cleanup; do not raise from a destructor.
        try:
            self.shutdown()
        except Exception:
            pass