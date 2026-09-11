from src.optim.common import FitnessOracle, FitnessResult
from src.optim.pso import MatchedPSO, PSOConfig


def test_pso_respects_exact_evaluation_budget():
    oracle = FitnessOracle(lambda x: FitnessResult(sum(v * v for v in x)))
    config = PSOConfig(dimensions=3, population_size=4, max_evaluations=13, seed=7)
    result = MatchedPSO(config, oracle).optimize()
    assert result.evaluations == 13
    assert oracle.calls == 13


def test_pso_is_deterministic_for_same_seed_and_population():
    initial = ((0.1, 0.2), (0.3, 0.4), (0.7, 0.8))

    def run():
        oracle = FitnessOracle(lambda x: FitnessResult(sum(v * v for v in x)))
        config = PSOConfig(dimensions=2, population_size=3, max_evaluations=9, seed=19)
        return MatchedPSO(config, oracle, initial_population=initial).optimize()

    a = run()
    b = run()
    assert a == b


def test_pso_history_lengths_are_consistent():
    oracle = FitnessOracle(lambda x: FitnessResult(sum(v * v for v in x)))
    config = PSOConfig(dimensions=2, population_size=3, max_evaluations=8, seed=2)
    result = MatchedPSO(config, oracle).optimize()
    assert len(result.history_best) == len(result.history_mean)
    assert len(result.history_best) == len(result.history_coordinate_diversity)
    assert len(result.history_best) == len(result.history_route_diversity)
