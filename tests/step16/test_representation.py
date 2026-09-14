from __future__ import annotations

import numpy as np
import pytest

from src.learning.representation import (
    RepresentationOutput,
    RepresentationSpec,
    validate_representation_provider,
)


def _make_output(
    spec: RepresentationSpec | None = None,
) -> RepresentationOutput:
    if spec is None:
        spec = RepresentationSpec()

    return RepresentationOutput(
        spec=spec,
        spatial=np.zeros(spec.spatial_dim, dtype=np.float32),
        temporal=np.ones(spec.temporal_dim, dtype=np.float32),
        forecast=np.full(spec.forecast_dim, 0.5, dtype=np.float32),
        uncertainty=np.full(
            spec.uncertainty_dim,
            0.1,
            dtype=np.float32,
        ),
        context=np.full(
            spec.context_dim,
            0.25,
            dtype=np.float32,
        ),
        observation_time_s=60.0,
        graph_version="graph-v1",
        representation_version=spec.version,
    )


def test_spec_total_dimension() -> None:
    spec = RepresentationSpec(
        spatial_dim=10,
        temporal_dim=20,
        forecast_dim=4,
        uncertainty_dim=2,
        context_dim=8,
    )

    assert spec.total_dim == 44


def test_output_shapes_match_contract() -> None:
    spec = RepresentationSpec(
        spatial_dim=5,
        temporal_dim=6,
        forecast_dim=3,
        uncertainty_dim=2,
        context_dim=4,
    )

    output = _make_output(spec)

    assert output.spatial.shape == (5,)
    assert output.temporal.shape == (6,)
    assert output.forecast.shape == (3,)
    assert output.uncertainty.shape == (2,)
    assert output.context.shape == (4,)
    assert output.flat.shape == (20,)


def test_flat_order_is_deterministic() -> None:
    spec = RepresentationSpec(
        spatial_dim=2,
        temporal_dim=2,
        forecast_dim=1,
        uncertainty_dim=1,
        context_dim=2,
    )

    output = RepresentationOutput(
        spec=spec,
        spatial=np.asarray([1.0, 2.0], dtype=np.float32),
        temporal=np.asarray([3.0, 4.0], dtype=np.float32),
        forecast=np.asarray([5.0], dtype=np.float32),
        uncertainty=np.asarray([6.0], dtype=np.float32),
        context=np.asarray([7.0, 8.0], dtype=np.float32),
        observation_time_s=0.0,
        graph_version="g1",
        representation_version=spec.version,
    )

    np.testing.assert_array_equal(
        output.flat,
        np.asarray(
            [1, 2, 3, 4, 5, 6, 7, 8],
            dtype=np.float32,
        ),
    )


def test_representation_arrays_are_immutable() -> None:
    output = _make_output()

    assert output.spatial.flags.writeable is False
    assert output.temporal.flags.writeable is False
    assert output.forecast.flags.writeable is False
    assert output.uncertainty.flags.writeable is False
    assert output.context.flags.writeable is False

    with pytest.raises(ValueError):
        output.spatial[0] = 99.0


def test_nan_and_inf_are_rejected() -> None:
    spec = RepresentationSpec()

    bad = np.zeros(spec.spatial_dim, dtype=np.float32)
    bad[0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinite"):
        RepresentationOutput(
            spec=spec,
            spatial=bad,
            temporal=np.zeros(spec.temporal_dim, dtype=np.float32),
            forecast=np.zeros(spec.forecast_dim, dtype=np.float32),
            uncertainty=np.zeros(
                spec.uncertainty_dim,
                dtype=np.float32,
            ),
            context=np.zeros(spec.context_dim, dtype=np.float32),
            observation_time_s=0.0,
            graph_version="g1",
            representation_version=spec.version,
        )


def test_version_mismatch_is_rejected() -> None:
    spec = RepresentationSpec(version="ppo-representation-v1")

    with pytest.raises(ValueError, match="representation_version"):
        RepresentationOutput(
            spec=spec,
            spatial=np.zeros(spec.spatial_dim, dtype=np.float32),
            temporal=np.zeros(spec.temporal_dim, dtype=np.float32),
            forecast=np.zeros(spec.forecast_dim, dtype=np.float32),
            uncertainty=np.zeros(
                spec.uncertainty_dim,
                dtype=np.float32,
            ),
            context=np.zeros(spec.context_dim, dtype=np.float32),
            observation_time_s=0.0,
            graph_version="g1",
            representation_version="wrong-version",
        )


def test_provider_contract_can_be_validated() -> None:
    class DummyProvider:
        spec = RepresentationSpec()

        def encode(self, state):
            return _make_output(self.spec)

    provider = DummyProvider()

    spec = validate_representation_provider(provider)

    assert spec.version == "ppo-representation-v1"
    assert spec.total_dim == 332


def test_provider_without_encode_is_rejected() -> None:
    class InvalidProvider:
        spec = RepresentationSpec()

    with pytest.raises(
        TypeError,
        match="callable encode",
    ):
        validate_representation_provider(InvalidProvider())