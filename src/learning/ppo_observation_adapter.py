from __future__ import annotations

from typing import Any, Mapping

from src.contracts.observation import (
    EdgeObservation,
    Observation,
    VehicleObservation,
    VisibleEvent,
    VisibleJob,
)


class PPOObservationAdapter:
    """
    Build a causal Observation from the scenario and simulator-visible state.

    This adapter intentionally does NOT infer unavailable simulator truth.

    Missing dynamic traffic remains missing.
    Static scenario information may be exposed when it is legitimately known
    at the current observation time.
    """

    def __init__(self, scenario):
        self.scenario = scenario

    def build(
        self,
        *,
        episode_id: str,
        observation_time_s: float,
        state_version: str,
        latest_state: Mapping[str, Any] | None = None,
    ) -> Observation:

        state = dict(latest_state or {})
        now = float(observation_time_s)

        if now < 0:
            raise ValueError(
                "observation_time_s must be non-negative"
            )

        # --------------------------------------------------------------
        # Road observations
        # --------------------------------------------------------------
        #
        # The current fixture does not expose edge-level measurements.
        # Therefore we explicitly mark every edge as missing instead of
        # manufacturing speed/travel-time values.

        edge_observations = tuple(
            EdgeObservation(
                edge_id=edge.edge_id,
                observed_speed_mps=None,
                observed_travel_time_s=None,
                observation_age_s=0.0,
                missing=True,
                known_closed=False,
                occupancy=None,
                halting_count=None,
            )
            for edge in self.scenario.edges
        )

        # --------------------------------------------------------------
        # Visible jobs
        # --------------------------------------------------------------

        visible_jobs = []

        for request in self.scenario.requests:
            # A request can only be visible once it is known.
            if float(request.known_at_s) > now:
                continue

            # Future unreleased requests are not currently actionable.
            # They are therefore not exposed as pending jobs.
            if float(request.release_s) > now:
                continue

            visible_jobs.append(
                VisibleJob(
                    request_id=request.request_id,
                    demand=float(request.demand),
                    release_s=float(request.release_s),
                    earliest_service_start_s=float(
                        request.earliest_service_start_s
                    ),
                    latest_service_start_s=float(
                        request.latest_service_start_s
                    ),
                    service_duration_s=float(
                        request.service_duration_s
                    ),
                    status=str(
                        getattr(
                            request.status,
                            "value",
                            request.status,
                        )
                    ),
                )
            )

        visible_jobs = tuple(
            sorted(
                visible_jobs,
                key=lambda job: str(job.request_id),
            )
        )

        pending_request_ids = tuple(
            job.request_id
            for job in visible_jobs
            if str(job.status).lower()
            in {
                "pending",
                "released",
                "waiting",
            }
        )

        # --------------------------------------------------------------
        # Fleet
        # --------------------------------------------------------------

        vehicle_loads = tuple(
            float(x)
            for x in state.get(
                "vehicle_loads",
                (),
            )
        )

        vehicle_current_edges = tuple(
            state.get(
                "vehicle_current_edge_ids",
                (),
            )
        )

        vehicle_current_nodes = tuple(
            state.get(
                "vehicle_current_node_ids",
                (),
            )
        )

        vehicle_distances = tuple(
            float(x)
            for x in state.get(
                "vehicle_distance_remaining_m",
                (),
            )
        )

        fleet = []

        for index, vehicle in enumerate(
            self.scenario.fleet
        ):
            remaining_load = (
                vehicle_loads[index]
                if index < len(vehicle_loads)
                else float(vehicle.remaining_load)
            )

            current_edge_id = (
                vehicle_current_edges[index]
                if index < len(vehicle_current_edges)
                else vehicle.current_edge_id
            )

            current_node_id = (
                vehicle_current_nodes[index]
                if index < len(vehicle_current_nodes)
                else vehicle.current_node_id
            )

            distance_remaining_m = (
                vehicle_distances[index]
                if index < len(vehicle_distances)
                else float(vehicle.distance_remaining_m)
            )

            fleet.append(
                VehicleObservation(
                    vehicle_id=vehicle.vehicle_id,
                    current_edge_id=current_edge_id,
                    current_node_id=current_node_id,
                    distance_remaining_m=max(
                        0.0,
                        distance_remaining_m,
                    ),
                    onboard_request_ids=tuple(
                        vehicle.onboard_request_ids
                    ),
                    remaining_load=max(
                        0.0,
                        remaining_load,
                    ),
                    executed_prefix_edge_ids=tuple(
                        vehicle.executed_prefix_edge_ids
                    ),
                    committed_request_ids=tuple(
                        vehicle.onboard_request_ids
                    ),
                )
            )

        fleet = tuple(
            sorted(
                fleet,
                key=lambda vehicle: str(
                    vehicle.vehicle_id
                ),
            )
        )

        # --------------------------------------------------------------
        # Events
        # --------------------------------------------------------------
        #
        # No event data is currently exposed by DeterministicSimulator.
        # Never reconstruct future incidents from Scenario.

        visible_events: tuple[VisibleEvent, ...] = ()

        # --------------------------------------------------------------
        # Version
        # --------------------------------------------------------------

        graph_version = str(
            self.scenario.graph_version
        )

        return Observation(
            scenario_id=str(
                self.scenario.scenario_id
            ),
            episode_id=str(
                episode_id
            ),
            observation_time_s=now,
            graph_version=graph_version,
            edge_observations=edge_observations,
            visible_jobs=visible_jobs,
            fleet=fleet,
            visible_events=visible_events,
            pending_request_ids=tuple(
                pending_request_ids
            ),
            state_version=str(
                state_version
            ),
        )