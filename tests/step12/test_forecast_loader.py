import json

from src.data.causal_episodes import generate_causal_pilot
from src.learning.baselines import PersistenceForecaster, TemporalOnlyForecaster
from src.learning.inspect import inspect_window, write_inspect_example
from src.learning.loader import load_pilot_windows
from src.learning.schema import FEATURE_COUNT, HISTORY_MINUTES, HORIZONS_MINUTES
from src.learning.windows import assert_window_causal
from src.platform.catalog import load_scenario


def _pilot(tmp_path):
    scenario = load_scenario("S3_BASE")
    root = tmp_path / "pilot"
    generate_causal_pilot(
        scenario,
        root,
        duration_s=2100,
        warmup_s=180,
        base_seed=21,
    )
    return scenario, load_pilot_windows(root, scenario)


def test_windows_have_declared_shapes_and_chronology(tmp_path):
    _scenario, datasets = _pilot(tmp_path)
    train = datasets["train"]
    assert len(train) > 0
    for window in train.windows:
        assert_window_causal(window)
        assert window.features.shape[0] == HISTORY_MINUTES
        assert window.features.shape[2] == FEATURE_COUNT
        assert window.speed_targets.shape[1] == len(HORIZONS_MINUTES)
        assert max(window.history_times_s) <= window.issue_time_s
        assert min(window.target_times_s) > window.issue_time_s
    batch = train.batch()
    b, l, e, f = batch.shape
    assert l == 12
    assert f == 6
    assert e == len(_scenario.edges)
    assert batch.speed_targets.shape == (b, e, 3)


def test_scaler_fits_train_only_and_splits_do_not_mix(tmp_path):
    _scenario, datasets = _pilot(tmp_path)
    train_ids = {window.episode_id for window in datasets["train"].windows}
    val_ids = {window.episode_id for window in datasets["validation"].windows}
    test_ids = {window.episode_id for window in datasets["test"].windows}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert test_ids == {"ep-006"}
    assert datasets["train"].scaler is not None
    assert datasets["validation"].scaler.mean == datasets["train"].scaler.mean


def test_baselines_and_inspect_example(tmp_path):
    _scenario, datasets = _pilot(tmp_path)
    batch = datasets["train"].batch()
    persistence = PersistenceForecaster().predict(batch)
    temporal = TemporalOnlyForecaster().fit(batch).predict(batch)
    assert persistence.shape == batch.speed_targets.shape
    assert temporal.shape == batch.speed_targets.shape
    example = inspect_window(datasets["train"].windows[0])
    assert example["chronology"]["window_ends_at_or_before_issue"]
    assert example["chronology"]["targets_strictly_after_issue"]
    assert any(window.speed_target_mask.any() for window in datasets["train"].windows)
    path = tmp_path / "inspect_example.json"
    write_inspect_example(datasets["train"], path)
    loaded = json.loads(path.read_text())
    assert loaded["issue_time_s"] == example["issue_time_s"]
