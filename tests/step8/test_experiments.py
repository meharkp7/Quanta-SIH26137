from src.optim.common import FitnessOracle, FitnessResult
from src.optim.experiments import run_ablation_suite
from src.optim.qpso import AdaptiveQPSOConfig


def test_ablation_suite_runs_paired_conditions():
    def factory(seed):
        return FitnessOracle(lambda x: FitnessResult(sum(v * v for v in x)))

    config = AdaptiveQPSOConfig(
        dimensions=4,
        population_size=4,
        max_evaluations=12,
        seed=0,
    )
    runs, summaries = run_ablation_suite(
        base_qpso_config=config,
        oracle_factory=factory,
        seeds=(3, 5),
        conditions=("canonical", "adaptive_alpha", "full_adaptive"),
        include_pso=True,
    )
    assert len(runs) == 2 * 4
    assert len(summaries) == 4
    assert all(run.evaluations == 12 for run in runs)


def test_ablation_suite_rejects_unknown_condition():
    def factory(seed):
        return FitnessOracle(lambda x: FitnessResult(sum(v * v for v in x)))

    config = AdaptiveQPSOConfig(dimensions=2, population_size=3, max_evaluations=6)
    try:
        run_ablation_suite(
            base_qpso_config=config,
            oracle_factory=factory,
            seeds=(1,),
            conditions=("does_not_exist",),
        )
    except ValueError as exc:
        assert "unknown QPSO conditions" in str(exc)
    else:
        raise AssertionError("unknown condition was accepted")
