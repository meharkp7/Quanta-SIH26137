"""Runtime-enforcement tests for the V1.2 contract layer.

These are not exhaustive property tests -- they are the minimum needed to
prove the specific gaps identified in the V1 review are actually closed:
duplicate IDs, inconsistent identifier typing, visibility-rule leakage,
shape mismatches, timing inconsistencies, and dishonest override/validation
bookkeeping. Each `pytest.raises` block is a claim from CONTRACTS.md made
checkable.

Run with: pytest tests/test_contracts.py -v
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contracts import (  # noqa: E402
    CoordinateTransform,
    EdgeObservation,
    EdgeTruth,
    Forecast,
    HiddenEvent,
    ModelPair,
    ModelPromotionStatus,
    NodeKind,
    Observation,
    RandomSeeds,
    Request,
    RequestStatus,
    RoadClass,
    RoadEdge,
    RoadNode,
    RoutePlan,
    Scenario,
    ScopeAction,
    ScopeDecision,
    SolveResult,
    SolveStatus,
    StopLeg,
    Units,
    ValidationResult,
    Vehicle,
    VehicleObservation,
    VehicleRoute,
    VisibleEvent,
    VisibleJob,
)
from contracts._immutable import ImmutableStrMap


# ---------------------------------------------------------------------------
# Step 3 fixture: depot, five jobs, two vehicles, asymmetric legal roads.
# ---------------------------------------------------------------------------

def make_tiny_scenario() -> Scenario:
    nodes = (
        RoadNode(node_id="depot", x_m=0, y_m=0, kind=NodeKind.DEPOT, zone_id="z0"),
        RoadNode(node_id="j1", x_m=100, y_m=0, kind=NodeKind.JUNCTION, zone_id="z0"),
        RoadNode(node_id="j2", x_m=200, y_m=0, kind=NodeKind.JUNCTION, zone_id="z0"),
        RoadNode(node_id="c1", x_m=100, y_m=50, kind=NodeKind.CUSTOMER_ACCESS, zone_id="z0"),
        RoadNode(node_id="c2", x_m=200, y_m=50, kind=NodeKind.CUSTOMER_ACCESS, zone_id="z0"),
    )
    edges = (
        RoadEdge(
            edge_id="e_depot_j1", parent_road_id="r1", from_node="depot", to_node="j1",
            length_m=100, road_class=RoadClass.COLLECTOR, speed_limit_mps=10,
            lane_count=1, capacity_veh_per_hour=600,
        ),
        RoadEdge(
            edge_id="e_j1_j2", parent_road_id="r2", from_node="j1", to_node="j2",
            length_m=100, road_class=RoadClass.ARTERIAL, speed_limit_mps=14,
            # one-way: forward lane only -- this is the "asymmetric legal
            # road" the Step 3 fixture calls for.
            lane_count=2, capacity_veh_per_hour=1200,
        ),
        RoadEdge(
            edge_id="e_j1_c1", parent_road_id="r3", from_node="j1", to_node="c1",
            length_m=50, road_class=RoadClass.ACCESS_CONNECTOR, speed_limit_mps=3,
            lane_count=1, capacity_veh_per_hour=200,
        ),
        RoadEdge(
            edge_id="e_j2_c2", parent_road_id="r4", from_node="j2", to_node="c2",
            length_m=50, road_class=RoadClass.ACCESS_CONNECTOR, speed_limit_mps=3,
            lane_count=1, capacity_veh_per_hour=200,
        ),
    )
    requests = (
        Request(
            request_id="c1", original_customer_id="C1", original_x=100, original_y=50,
            access_node_id="c1", access_distance_m=0, demand=10,
            known_at_s=0, release_s=0, earliest_service_start_s=0,
            latest_service_start_s=3600, service_duration_s=60,
        ),
        Request(
            request_id="c2", original_customer_id="C2", original_x=200, original_y=50,
            access_node_id="c2", access_distance_m=0, demand=15,
            known_at_s=0, release_s=0, earliest_service_start_s=0,
            latest_service_start_s=3600, service_duration_s=60,
        ),
    )
    fleet = (
        Vehicle(vehicle_id="v1", capacity=50, start_node_id="depot", depot_node_id="depot"),
        Vehicle(vehicle_id="v2", capacity=50, start_node_id="depot", depot_node_id="depot"),
    )
    return Scenario(
        schema_version="v1",
        scenario_id="tiny-fixture",
        source_name="hand-built",
        source_checksum=None,
        units=Units(),
        coordinate_transform=CoordinateTransform(
            scale=1.0, translation_x_m=0, translation_y_m=0, description="identity"
        ),
        graph_version="g1",
        nodes=nodes,
        edges=edges,
        requests=requests,
        fleet=fleet,
        seeds=RandomSeeds(
            scenario_seed=0, road_seed=0, traffic_seed=0, incident_seed=0,
            window_seed=0, optimizer_seed=0, learning_seed=0,
        ),
        generator_version="v1",
        dataset_split="dev",
        configuration_version="v1",
    )


def test_tiny_scenario_is_valid():
    make_tiny_scenario()


def test_scenario_rejects_duplicate_node_ids():
    with pytest.raises(ValidationError, match="duplicate node_id"):
        Scenario(
            schema_version="v1", scenario_id="bad", source_name="x", source_checksum=None,
            units=Units(), coordinate_transform=None, graph_version="g1",
            nodes=(
                RoadNode(node_id="depot", x_m=0, y_m=0, kind=NodeKind.DEPOT, zone_id="z0"),
                RoadNode(node_id="depot", x_m=1, y_m=1, kind=NodeKind.JUNCTION, zone_id="z0"),
            ),
            edges=(), requests=(), fleet=(),
            seeds=RandomSeeds(scenario_seed=0, road_seed=0, traffic_seed=0,
                               incident_seed=0, window_seed=0, optimizer_seed=0,
                               learning_seed=0),
            generator_version="v1", dataset_split="dev", configuration_version="v1",
        )


def test_scenario_rejects_edge_referencing_unknown_node():
    with pytest.raises(ValidationError, match="unknown node"):
        Scenario(
            schema_version="v1", scenario_id="bad", source_name="x", source_checksum=None,
            units=Units(), coordinate_transform=None, graph_version="g1",
            nodes=(RoadNode(node_id="depot", x_m=0, y_m=0, kind=NodeKind.DEPOT, zone_id="z0"),),
            edges=(RoadEdge(
                edge_id="e1", parent_road_id="r1", from_node="depot", to_node="ghost",
                length_m=10, road_class=RoadClass.LOCAL, speed_limit_mps=5,
                lane_count=1, capacity_veh_per_hour=100,
            ),),
            requests=(), fleet=(),
            seeds=RandomSeeds(scenario_seed=0, road_seed=0, traffic_seed=0,
                               incident_seed=0, window_seed=0, optimizer_seed=0,
                               learning_seed=0),
            generator_version="v1", dataset_split="dev", configuration_version="v1",
        )


def test_request_rejects_missed_window_construction():
    with pytest.raises(ValidationError, match="latest_service_start_s"):
        Request(
            request_id="bad", original_customer_id="X", original_x=0, original_y=0,
            access_node_id="n1", access_distance_m=0, demand=1,
            known_at_s=0, release_s=0, earliest_service_start_s=100,
            latest_service_start_s=50,  # latest before earliest
            service_duration_s=10,
        )


def test_vehicle_rejects_load_exceeding_capacity():
    with pytest.raises(ValidationError, match="exceeds capacity"):
        Vehicle(
            vehicle_id="v1", capacity=10, start_node_id="depot", depot_node_id="depot",
            remaining_load=20,
        )


def test_road_edge_rejects_zero_lane_count():
    with pytest.raises(ValidationError, match="lane_count must be at least 1"):
        RoadEdge(
            edge_id="e1", parent_road_id="r1", from_node="a", to_node="b",
            length_m=10, road_class=RoadClass.LOCAL, speed_limit_mps=5,
            lane_count=0, capacity_veh_per_hour=100,
        )


# ---------------------------------------------------------------------------
# Observation / visibility rule -- the corrected invariant.
# ---------------------------------------------------------------------------

def _obs_kwargs(**overrides):
    base = dict(
        scenario_id="tiny-fixture",
        episode_id="ep1",
        observation_time_s=100.0,
        graph_version="g1",
        edge_observations=(),
        visible_jobs=(
            VisibleJob(
                request_id="c1", demand=10, release_s=0, earliest_service_start_s=0,
                latest_service_start_s=3600, service_duration_s=60, status="pending",
            ),
        ),
        fleet=(),
        visible_events=(),
        pending_request_ids=("c1",),
        state_version="sv1",
    )
    base.update(overrides)
    return base


def test_observation_valid_baseline():
    Observation(**_obs_kwargs())


def test_observation_rejects_event_revealed_in_the_future():
    with pytest.raises(ValidationError, match="Visibility Rule"):
        Observation(**_obs_kwargs(
            visible_events=(
                VisibleEvent(
                    event_id="ev1", revealed_at_s=200.0,  # after obs time 100.0
                    event_type="closure", affected_parent_road_ids=("r1",),
                ),
            ),
        ))


def test_observation_allows_future_effect_start_of_a_revealed_event():
    """The corrected invariant: an announced-in-advance closure is legal.

    revealed_at_s (90) <= observation_time_s (100), even though
    effect_start_s (500) is well in the future. This must NOT raise.
    """
    Observation(**_obs_kwargs(
        visible_events=(
            VisibleEvent(
                event_id="ev1", revealed_at_s=90.0, event_type="scheduled_closure",
                affected_parent_road_ids=("r1",), effect_start_s=500.0,
            ),
        ),
    ))


def test_observation_rejects_dangling_pending_request():
    with pytest.raises(ValidationError, match="dangling|not present"):
        Observation(**_obs_kwargs(pending_request_ids=("c1", "ghost")))


def test_edge_observation_rejects_missing_with_a_value_populated():
    with pytest.raises(ValidationError, match="missing=True"):
        EdgeObservation(
            edge_id="e1", observed_speed_mps=5.0, observed_travel_time_s=None,
            observation_age_s=1.0, missing=True, known_closed=False,
        )


def test_vehicle_observation_uses_typed_node_id_not_bare_str():
    # This is a type-level regression check: constructing with a value is
    # fine (Pydantic coerces compatible str), but the point is that the
    # *declared* type is RoadNodeId, matching scenario.Vehicle, not a bare
    # str as in V1.
    import contracts.observation as obs_mod
    from contracts.core_types import RoadNodeId
    annotation = obs_mod.VehicleObservation.model_fields["current_node_id"].annotation
    assert RoadNodeId in getattr(annotation, "__args__", (annotation,))


# ---------------------------------------------------------------------------
# Forecast shape / causality checks.
# ---------------------------------------------------------------------------

def test_forecast_valid_baseline():
    Forecast(
        scenario_id="tiny-fixture", episode_id="ep1", forecast_version="f1",
        issued_at_s=0.0, target_times_s=(300.0, 600.0, 900.0),
        edge_ids=("e1", "e2"),
        prediction=((5.0, 4.0, 3.0), (6.0, 6.0, 5.0)),
    )


def test_forecast_rejects_row_count_mismatch():
    with pytest.raises(ValidationError, match="edge_ids"):
        Forecast(
            scenario_id="s", episode_id="ep1", forecast_version="f1",
            issued_at_s=0.0, target_times_s=(300.0,),
            edge_ids=("e1", "e2"),
            prediction=((5.0,),),  # only one row for two edges
        )


def test_forecast_rejects_non_causal_horizon():
    with pytest.raises(ValidationError, match="not strictly after"):
        Forecast(
            scenario_id="s", episode_id="ep1", forecast_version="f1",
            issued_at_s=1000.0, target_times_s=(300.0,),  # before issued_at_s
            edge_ids=("e1",), prediction=((5.0,),),
        )


def test_forecast_rejects_inverted_interval():
    with pytest.raises(ValidationError, match="exceeds"):
        Forecast(
            scenario_id="s", episode_id="ep1", forecast_version="f1",
            issued_at_s=0.0, target_times_s=(300.0,),
            edge_ids=("e1",), prediction=((5.0,),),
            lower_prediction=((9.0,),), upper_prediction=((1.0,),),
        )


# ---------------------------------------------------------------------------
# Routing consistency.
# ---------------------------------------------------------------------------

def test_stop_leg_rejects_inconsistent_travel_time():
    with pytest.raises(ValidationError, match="does not match"):
        StopLeg(
            from_stop_id="depot", to_stop_id="c1", physical_edge_ids=("e1",),
            travel_time_s=999.0, distance_m=100.0, congestion_exposure=0.0,
            departure_time_s=0.0, arrival_time_s=60.0,
        )


def test_vehicle_route_rejects_broken_leg_chain():
    with pytest.raises(ValidationError, match="leg chain is broken"):
        VehicleRoute(
            vehicle_id="v1", customer_order=("c1", "c2"),
            legs=(
                StopLeg(from_stop_id="depot", to_stop_id="c1", physical_edge_ids=("e1",),
                        travel_time_s=60, distance_m=100, congestion_exposure=0,
                        departure_time_s=0, arrival_time_s=60),
                StopLeg(from_stop_id="WRONG", to_stop_id="c2", physical_edge_ids=("e2",),
                        travel_time_s=60, distance_m=100, congestion_exposure=0,
                        departure_time_s=60, arrival_time_s=120),
            ),
            expected_departure_s=0, expected_return_s=200,
            expected_driving_time_s=120, expected_waiting_time_s=0,
            expected_service_time_s=0, served_request_ids=("c1", "c2"),
            frozen_prefix_edge_ids=(),
        )


# ---------------------------------------------------------------------------
# ScopeDecision override honesty.
# ---------------------------------------------------------------------------

def test_scope_decision_rejects_silent_deviation_without_override_flag():
    with pytest.raises(ValidationError, match="must be recorded as an override"):
        ScopeDecision(
            scenario_id="s", state_version="sv1", decision_time_s=0.0,
            requested_action=ScopeAction.GLOBAL, executed_action=ScopeAction.KEEP,
            affected_vehicle_ids=(), mutable_request_ids=(),
            budget_seconds=0.0, overridden=False, override_reason=None,
            selection_reason="policy chose GLOBAL", selected_by="ppo_v1",
            decision_version="d1",
        )


def test_scope_decision_rejects_override_without_reason():
    with pytest.raises(ValidationError, match="requires a non-empty override_reason"):
        ScopeDecision(
            scenario_id="s", state_version="sv1", decision_time_s=0.0,
            requested_action=ScopeAction.GLOBAL, executed_action=ScopeAction.KEEP,
            affected_vehicle_ids=(), mutable_request_ids=(),
            budget_seconds=0.0, overridden=True, override_reason=None,
            selection_reason="policy chose GLOBAL", selected_by="ppo_v1",
            decision_version="d1",
        )


def test_scope_decision_valid_honest_override():
    ScopeDecision(
        scenario_id="s", state_version="sv1", decision_time_s=0.0,
        requested_action=ScopeAction.GLOBAL, executed_action=ScopeAction.KEEP,
        affected_vehicle_ids=(), mutable_request_ids=(),
        budget_seconds=0.0, overridden=True,
        override_reason="safety layer: GLOBAL exceeds latency budget",
        selection_reason="policy chose GLOBAL", selected_by="ppo_v1",
        decision_version="d1",
    )


# ---------------------------------------------------------------------------
# SolveResult / ValidationResult honesty.
# ---------------------------------------------------------------------------

def _feasible_validation():
    return ValidationResult(
        feasible=True, capacity_feasible=True, time_window_feasible=True,
        route_continuity_feasible=True, legal_road_feasible=True,
        commitment_feasible=True, depot_feasible=True, subtour_free=True,
    )


def test_validation_result_rejects_feasible_flag_inconsistent_with_components():
    with pytest.raises(ValidationError, match="inconsistent"):
        ValidationResult(
            feasible=True,  # but capacity check failed below
            capacity_feasible=False, time_window_feasible=True,
            route_continuity_feasible=True, legal_road_feasible=True,
            commitment_feasible=True, depot_feasible=True, subtour_free=True,
        )


def test_solve_result_rejects_feasible_status_without_route_plan():
    with pytest.raises(ValidationError, match="route_plan is None"):
        SolveResult(
            scenario_id="s", state_version="sv1", method="qpso", solver_version="v1",
            status=SolveStatus.FEASIBLE, route_plan=None, objective_value=10.0,
            elapsed_time_s=1.0, validation=_feasible_validation(), search_trace=None,
        )


def test_solve_result_rejects_solver_claiming_feasible_when_validator_disagrees():
    plan = RoutePlan(
        scenario_id="s", state_version="sv1", route_version="r1", generated_at_s=0.0,
        vehicle_routes=(), objective_value=10.0, objective_time=10.0,
        objective_distance=0.0, objective_congestion=0.0, route_change_penalty=0.0,
        changed_vehicle_ids=(), generated_by="qpso",
    )
    bad_validation = ValidationResult(
        feasible=False, capacity_feasible=False, time_window_feasible=True,
        route_continuity_feasible=True, legal_road_feasible=True,
        commitment_feasible=True, depot_feasible=True, subtour_free=True,
    )
    with pytest.raises(ValidationError, match="never overrides the independent validator"):
        SolveResult(
            scenario_id="s", state_version="sv1", method="qpso", solver_version="v1",
            status=SolveStatus.FEASIBLE, route_plan=plan, objective_value=10.0,
            elapsed_time_s=1.0, validation=bad_validation, search_trace=None,
        )


# ---------------------------------------------------------------------------
# ModelPair rollback bookkeeping.
# ---------------------------------------------------------------------------

def test_model_pair_rejects_accepted_pair_with_no_rollback_target():
    with pytest.raises(ValidationError, match="rollback-able"):
        ModelPair(
            pair_version="p2", forecast_version="f2", policy_version="pol2",
            schema_version="s1", scaler_version="sc1",
            action_semantics_version="a1", reward_version="r1",
            qpso_version="q1", sumo_version="sumo1",
            training_cutoff_time_s=0.0, parent_pair_version="p1",
            validation_status=ModelPromotionStatus.ACCEPTED,
            validation_summary="ok", created_at_s=0.0,
            rollback_pair_version=None,
        )


def test_model_pair_first_release_may_omit_rollback_target():
    ModelPair(
        pair_version="p1", forecast_version="f1", policy_version="pol1",
        schema_version="s1", scaler_version="sc1",
        action_semantics_version="a1", reward_version="r1",
        qpso_version="q1", sumo_version="sumo1",
        training_cutoff_time_s=0.0, parent_pair_version=None,
        validation_status=ModelPromotionStatus.ACCEPTED,
        validation_summary="first release", created_at_s=0.0,
        rollback_pair_version=None,
    )


# ---------------------------------------------------------------------------
# Environment truth.
# ---------------------------------------------------------------------------

def test_hidden_event_rejects_reveal_before_generation():
    with pytest.raises(ValidationError, match="precedes generation_time_s"):
        HiddenEvent(
            event_id="ev1", generation_time_s=100.0, reveal_time_s=50.0,
            effect_start_s=200.0, event_type="closure",
            affected_parent_road_ids=("r1",),
        )


def test_edge_truth_rejects_closed_edge_with_positive_speed():
    with pytest.raises(ValidationError, match="is_closed=True"):
        EdgeTruth(edge_id="e1", timestamp_s=0.0, true_speed_mps=10.0,
                  true_travel_time_s=None, is_closed=True)


# ---------------------------------------------------------------------------
# Immutability.
# ---------------------------------------------------------------------------

def test_frozen_model_rejects_attribute_reassignment():
    scenario = make_tiny_scenario()
    with pytest.raises(ValidationError):
        scenario.scenario_id = "changed"


def test_immutable_str_map_cannot_be_mutated_in_place():
    m = ImmutableStrMap({"a": "1"})
    with pytest.raises(TypeError):
        m["a"] = "2"  # Mapping (not MutableMapping) has no __setitem__


def test_model_pair_metadata_round_trips_through_json():
    pair = ModelPair(
        pair_version="p1", forecast_version="f1", policy_version="pol1",
        schema_version="s1", scaler_version="sc1",
        action_semantics_version="a1", reward_version="r1",
        qpso_version="q1", sumo_version="sumo1",
        training_cutoff_time_s=0.0, parent_pair_version=None,
        validation_status=ModelPromotionStatus.ACCEPTED,
        validation_summary="first release", created_at_s=0.0,
        rollback_pair_version=None, metadata={"note": "pilot"},
    )
    dumped = pair.model_dump_json()
    restored = ModelPair.model_validate_json(dumped)
    assert restored.metadata.to_dict() == {"note": "pilot"}
