from __future__ import annotations

from pydantic import Field, model_validator

from ._base import Contract
from .core_types import EpisodeId, ForecastVersion, RoadEdgeId, ScenarioId, TargetKind, TimeS

PredictionRow = tuple[float | None, ...]


class Forecast(Contract):
    scenario_id: ScenarioId
    episode_id: EpisodeId

    forecast_version: ForecastVersion

    issued_at_s: TimeS = Field(ge=0)

    # Absolute future timestamps corresponding to the prediction rows.
    target_times_s: tuple[TimeS, ...]

    edge_ids: tuple[RoadEdgeId, ...]

    # Shape conceptually: [edge, horizon]
    prediction: tuple[PredictionRow, ...]

    # Optional interval bounds, e.g. q0.1 / q0.5 / q0.9.
    lower_prediction: tuple[PredictionRow, ...] | None = None
    median_prediction: tuple[PredictionRow, ...] | None = None
    upper_prediction: tuple[PredictionRow, ...] | None = None

    # True where a prediction is valid and usable.
    valid_mask: tuple[tuple[bool, ...], ...] = ()

    target_kind: TargetKind = TargetKind.SPEED_PROXY

    # What quantity the prediction represents.
    target_unit: str = "m/s"

    model_version: str = "unversioned"

    @model_validator(mode="after")
    def _horizons_are_causal_and_ordered(self) -> "Forecast":
        # A forecast issued at time T cannot target a time <= T -- that
        # would not be a forecast, it would be a (possibly stale) reading.
        for t in self.target_times_s:
            if t <= self.issued_at_s:
                raise ValueError(
                    f"target_times_s contains {t}, which is not strictly "
                    f"after issued_at_s ({self.issued_at_s})"
                )
        # Horizons should be strictly increasing -- duplicate or
        # out-of-order horizons almost always indicate a bug in whichever
        # window-construction code produced this record, not a real
        # multi-horizon forecast.
        if list(self.target_times_s) != sorted(self.target_times_s):
            raise ValueError("target_times_s must be strictly increasing")
        if len(set(self.target_times_s)) != len(self.target_times_s):
            raise ValueError("target_times_s contains duplicate horizons")
        return self

    @model_validator(mode="after")
    def _shapes_agree(self) -> "Forecast":
        n_edges = len(self.edge_ids)
        n_horizons = len(self.target_times_s)

        if len(set(self.edge_ids)) != n_edges:
            raise ValueError("edge_ids contains duplicate edge IDs")

        def _check(name: str, rows: tuple[PredictionRow, ...] | tuple[tuple[bool, ...], ...] | None) -> None:
            if rows is None or len(rows) == 0:
                return
            if len(rows) != n_edges:
                raise ValueError(
                    f"{name} has {len(rows)} rows but there are "
                    f"{n_edges} edge_ids -- these must match 1:1"
                )
            for i, row in enumerate(rows):
                if len(row) != n_horizons:
                    raise ValueError(
                        f"{name}[{i}] has {len(row)} columns but there are "
                        f"{n_horizons} target_times_s -- these must match"
                    )

        _check("prediction", self.prediction)
        _check("lower_prediction", self.lower_prediction)
        _check("median_prediction", self.median_prediction)
        _check("upper_prediction", self.upper_prediction)
        _check("valid_mask", self.valid_mask)

        # prediction itself is mandatory-shaped (not optional like the
        # interval bounds), so check it even when the tuple is empty only
        # because there happen to be zero edges/horizons.
        if len(self.prediction) != n_edges:
            raise ValueError(
                f"prediction has {len(self.prediction)} rows but there are "
                f"{n_edges} edge_ids"
            )
        return self

    @model_validator(mode="after")
    def _interval_ordering_where_present(self) -> "Forecast":
        if self.lower_prediction is None or self.upper_prediction is None:
            return self
        for e, (lo_row, hi_row) in enumerate(
            zip(self.lower_prediction, self.upper_prediction)
        ):
            for h, (lo, hi) in enumerate(zip(lo_row, hi_row)):
                if lo is not None and hi is not None and lo > hi:
                    raise ValueError(
                        f"lower_prediction[{e}][{h}]={lo} exceeds "
                        f"upper_prediction[{e}][{h}]={hi}"
                    )
        return self
