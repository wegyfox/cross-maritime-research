"""Longitudinal same-route qualification track for R1 P4D.

This module exists to give Guo-style adjacent-ship headways a semantically compatible
measurement surface.  The native CROSS runtime remains independent: source FIFO,
ship-domain clearance and persistent same-route following determine admissions and
motion.  The Guo/M1 pair-headway matrix is applied only after the run as a labelled
post-hoc diagnostic against observed ``CONFLICT_CENTER_PASS`` intervals.

The track is qualification-only.  It is not a confirmatory scientific campaign and it
does not authorize interpreting the illustrative reference matrix as calibrated truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Sequence

from vessel_crossing.nextgen import (
    CrossTrackRecovery,
    EncounterRegistry,
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalRouteHandoffTarget,
    ManeuverLimits,
    Route2D,
    RouteFollower,
    RouteNavigator,
    ShipDomain,
    VesselClassProfile,
    VesselDimensions,
    VesselProfileCatalog,
    WaterRelativeVesselState,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingSupervisor
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.research.r1_p4b_behavior_adapter import P4BBehaviorBinding
from vessel_crossing.research.r1_p4d_command_response import (
    P4DBoundedCommandResponseEnvironmentalMotionBackend,
    P4D_COMMAND_RESPONSE_SEMANTICS_ID,
)
from vessel_crossing.testbed.behavior import BehaviorProfile
from vessel_crossing.research.r1_p4b_engineering_preflight import preflight_classification_config
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config
from vessel_crossing.research.r1_p4b_publication_reference import (
    GUO_2024_ILLUSTRATIVE,
    PairHeadwayMatrix,
    TypedDemandEvent,
)
from vessel_crossing.research.r1_robustness_stage2 import backend_for, first_order_profile

from .r1_p4d_conflict_observer import (
    P4DConflictGate,
    P4DObservedMetricEnvironmentalWorld,
)
from .r1_p4d_conflict_service import center_pass_service_summary

_EPS = 1e-10
P4D_LONGITUDINAL_TRACK_ID = "R1_P4D_LONGITUDINAL_GUO_SEMANTIC_QUALIFICATION_V1"
P4D_LONGITUDINAL_STREAM_ID = "longitudinal"
P4D_LONGITUDINAL_EVIDENCE_ROLE = "QUALIFICATION_ONLY_NOT_CONFIRMATORY"
PAIR_COVER_ROLE_SEQUENCE = ("AS", "AS", "HS", "HS", "AS")


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def longitudinal_gates() -> tuple[P4DConflictGate, ...]:
    """Common along-track gates for a northbound 3 NM qualification route."""
    return (
        P4DConflictGate("CONFLICT_REQUEST", 1, 0.5, +1),
        P4DConflictGate("CONFLICT_ENTRY", 1, 1.0, +1),
        P4DConflictGate("CONFLICT_CENTER_PASS", 1, 1.5, +1),
        P4DConflictGate("CONFLICT_EXIT", 1, 2.0, +1),
    )


def _limits() -> ManeuverLimits:
    return ManeuverLimits.from_knots_per_minute(
        max_acceleration_knots_per_min=2.0,
        max_deceleration_knots_per_min=2.0,
        max_turn_rate_deg_per_min=30.0,
    )


def _qualification_profile() -> VesselClassProfile:
    limits = _limits()
    return VesselClassProfile(
        class_id="p4d-longitudinal-neutral-120m",
        dimensions=VesselDimensions.from_meters(length_m=120.0, beam_m=20.0),
        maneuver_limits=limits,
        safety_domain=ShipDomain.circular(0.10),
    )


def _route_target(
    vessel_id: str,
    route: Route2D,
    *,
    target_speed_nm_per_min: float = 0.10,
) -> EnvironmentalRouteHandoffTarget:
    return EnvironmentalRouteHandoffTarget(
        vessel_id=vessel_id,
        navigator=RouteNavigator(
            RouteFollower(route, float(target_speed_nm_per_min)),
            CrossTrackRecovery(route),
        ),
    )


def _build_demands(
    demand_times_min: Sequence[float],
    *,
    profile: VesselClassProfile,
    route: Route2D,
) -> tuple[EnvironmentalNetworkSourceDemand, ...]:
    if not demand_times_min:
        raise ValueError("at least one demand is required")
    out = []
    previous = None
    for i, raw_time in enumerate(demand_times_min, start=1):
        t = float(raw_time)
        if not isfinite(t):
            raise ValueError("demand times must be finite")
        if previous is not None and t < previous - _EPS:
            raise ValueError("demand times must be non-decreasing")
        previous = t
        vessel_id = f"LONG-{i:02d}"
        out.append(EnvironmentalNetworkSourceDemand(
            stream_id=P4D_LONGITUDINAL_STREAM_ID,
            nominal_vessel_id=vessel_id,
            demand_time=t,
            vessel=WaterRelativeVesselState(vessel_id, (0.0, 0.0), 0.0, 0.10),
            route_target=_route_target(vessel_id, route),
            limits=profile.maneuver_limits,
            environmental_drift_velocity=(0.0, 0.0),
            profile=profile,
        ))
    return tuple(out)


def pair_cover_typed_events(
    demand_times_min: Sequence[float],
    *,
    role_sequence: Sequence[str] = PAIR_COVER_ROLE_SEQUENCE,
) -> tuple[TypedDemandEvent, ...]:
    if len(demand_times_min) != len(role_sequence):
        raise ValueError("demand_times_min and role_sequence must have the same length")
    if any(role not in {"AS", "HS"} for role in role_sequence):
        raise ValueError("roles must be AS or HS")
    return tuple(
        TypedDemandEvent(
            event_id=f"LONG-{i:02d}",
            arrival_min=float(t),
            stream_id=P4D_LONGITUDINAL_STREAM_ID,
            vessel_type=str(role),
            type_uniform=float(i) / (len(role_sequence) + 1.0),
        )
        for i, (t, role) in enumerate(zip(demand_times_min, role_sequence), start=1)
    )


@dataclass(frozen=True, slots=True)
class P4DLongitudinalQualificationResult:
    document: dict[str, object]
    canonical_bytes: bytes

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes


def run_longitudinal_pair_cover_qualification(
    *,
    demand_gap_min: float = 0.5,
    reference_matrix: PairHeadwayMatrix = GUO_2024_ILLUSTRATIVE,
    end_time_min: float = 40.0,
) -> P4DLongitudinalQualificationResult:
    """Run one native same-route fixture covering all four directed AS/HS pairs.

    AS/HS are labels only in this neutral semantic fixture.  They do not change the
    controller.  This isolates whether the observation boundary and ordered-pair
    orientation are correct before any behavioral hypothesis is introduced.
    """
    if not isfinite(demand_gap_min) or demand_gap_min <= 0.0:
        raise ValueError("demand_gap_min must be finite and positive")
    if not isfinite(end_time_min) or end_time_min <= 0.0:
        raise ValueError("end_time_min must be finite and positive")

    demand_times = tuple(i * float(demand_gap_min) for i in range(len(PAIR_COVER_ROLE_SEQUENCE)))
    typed = pair_cover_typed_events(demand_times)
    profile = _qualification_profile()
    route = Route2D(((0.0, 0.0), (0.0, 3.0)))
    demands = _build_demands(demand_times, profile=profile, route=route)
    vessel_ids = tuple(x.vessel_id for x in demands)
    catalog = VesselProfileCatalog({vessel_id: profile for vessel_id in vessel_ids})

    # Reuse only generic runtime/discovery constants from the publication-qualified
    # engine.  No publication headway or matched-source adapter enters this track.
    base = build_publication_config(room_nm=1.375, demand_per_hour=4.0, traffic_seed=1)
    scenario = base.scenario
    world = P4DObservedMetricEnvironmentalWorld(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=0.0,
        profile_catalog=catalog,
        safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=backend_for(first_order_profile()),
        retain_safety_intervals=False,
        stream_by_vessel={vessel_id: P4D_LONGITUDINAL_STREAM_ID for vessel_id in vessel_ids},
        gates_by_stream={P4D_LONGITUDINAL_STREAM_ID: longitudinal_gates()},
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=catalog,
    )
    following = SameRouteFollowingSupervisor()
    started = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        {},
        EnvironmentalNetworkSourceReservoir(demands),
        recapture_config=scenario.recapture,
        runtime_config=scenario.runtime,
        recheck_config=base.recheck,
        classification_config=preflight_classification_config(),
        profile_catalog=catalog,
        same_route_dynamic_ingress=following,
        persistent_same_route_following=following,
    )
    outcome = started.session.run_until_capturing_terminal_outcome(float(end_time_min))
    service = center_pass_service_summary(
        world.p4d_conflict_observations,
        typed,
        reference_matrix=reference_matrix,
    )
    admissions = tuple(outcome.reservoir.admissions)
    observed_roles = {x.role for x in world.p4d_conflict_observations}
    required_roles = {"CONFLICT_REQUEST", "CONFLICT_ENTRY", "CONFLICT_CENTER_PASS", "CONFLICT_EXIT"}
    transition_pairs = [
        f"{row['predecessor_type']}->{row['vessel_type']}"
        for row in service["records"]
        if row["predecessor_type"] is not None
    ]
    expected_pairs = ["AS->AS", "AS->HS", "HS->HS", "HS->AS"]
    qualification_pass = bool(
        outcome.status.value == "SUCCESS"
        and len(admissions) == len(demands)
        and service["passage_count"] == len(demands)
        and service["busy_transition_count"] == len(demands) - 1
        and transition_pairs == expected_pairs
        and required_roles.issubset(observed_roles)
        and service["arrival_order_inversion_count"] == 0
    )
    document = {
        "format": "cross-r1-p4d-longitudinal-qualification",
        "schema_version": 1,
        "track_id": P4D_LONGITUDINAL_TRACK_ID,
        "evidence_role": P4D_LONGITUDINAL_EVIDENCE_ROLE,
        "scientific_confirmatory_interpretation_authorized": False,
        "target_semantics": "ADJACENT_SAME_ROUTE_CENTER_PASSAGE_HEADWAY",
        "reference_use": "POST_HOC_DIAGNOSTIC_ONLY_NOT_RUNTIME_FORCING",
        "runtime_mechanism": "NATIVE_SOURCE_FIFO_PLUS_SHIP_DOMAIN_CLEARANCE_PLUS_SAME_ROUTE_FOLLOWING",
        "demand_gap_min": float(demand_gap_min),
        "role_sequence": list(PAIR_COVER_ROLE_SEQUENCE),
        "expected_directed_pairs": expected_pairs,
        "terminal": {
            "status": outcome.status.value,
            "terminal_time_min": outcome.terminal_time_min,
            "infeasibility_type": outcome.infeasibility_type,
            "resource_unavailability_type": outcome.resource_unavailability_type,
        },
        "source": {
            "demand_count": len(demands),
            "admission_count": len(admissions),
            "admissions": [
                {
                    "vessel_id": x.vessel_id,
                    "demand_time_min": x.demand_time,
                    "admission_time_min": x.admission_time,
                    "waiting_time_min": x.waiting_time_min,
                }
                for x in admissions
            ],
        },
        "observer": {
            "observation_count": len(world.p4d_conflict_observations),
            "required_roles_present": required_roles.issubset(observed_roles),
        },
        "center_service": service,
        "response": {
            "rebootstrap_count": len(started.runtime_start.session.persistent.rebootstrap_results),
            "fleet_observation_count": len(started.runtime_start.session.coordinator.fleet_observations),
        },
        "qualification_pass": qualification_pass,
    }
    raw = _canonical_json(document)
    return P4DLongitudinalQualificationResult(document, raw)



def _run_longitudinal_response_arm(
    *,
    follower_delay_min: float,
    leader_observation_delay_min: float = 0.0,
) -> dict[str, object]:
    """Two-vessel native following fixture that requires a follower speed response."""
    if not isfinite(follower_delay_min) or follower_delay_min < 0.0:
        raise ValueError("follower_delay_min must be finite and non-negative")
    if not isfinite(leader_observation_delay_min) or leader_observation_delay_min < 0.0:
        raise ValueError("leader_observation_delay_min must be finite and non-negative")
    base = build_publication_config(room_nm=1.375, demand_per_hour=4.0, traffic_seed=2)
    scenario = base.scenario
    profile = _qualification_profile()
    route = Route2D(((0.0, 0.0), (0.0, 3.0)))
    leader_id = "RESP-LEADER"
    follower_id = "RESP-FOLLOWER"
    demands = (
        EnvironmentalNetworkSourceDemand(
            stream_id=P4D_LONGITUDINAL_STREAM_ID,
            nominal_vessel_id=leader_id,
            demand_time=0.0,
            vessel=WaterRelativeVesselState(leader_id, (0.0, 0.0), 0.0, 0.10),
            route_target=_route_target(leader_id, route, target_speed_nm_per_min=0.05),
            limits=profile.maneuver_limits,
            environmental_drift_velocity=(0.0, 0.0),
            profile=profile,
        ),
        EnvironmentalNetworkSourceDemand(
            stream_id=P4D_LONGITUDINAL_STREAM_ID,
            nominal_vessel_id=follower_id,
            demand_time=6.0,
            vessel=WaterRelativeVesselState(follower_id, (0.0, 0.0), 0.0, 0.10),
            route_target=_route_target(follower_id, route, target_speed_nm_per_min=0.10),
            limits=profile.maneuver_limits,
            environmental_drift_velocity=(0.0, 0.0),
            profile=profile,
        ),
    )
    catalog = VesselProfileCatalog({leader_id: profile, follower_id: profile})
    bindings = {
        leader_id: P4BBehaviorBinding(
            BehaviorProfile.AUTONOMOUS_POLICY,
            "R1_P4D_LONGITUDINAL_RESPONSE/LEADER",
            0.0,
        ),
        follower_id: P4BBehaviorBinding(
            BehaviorProfile.COMPLIANT_MERCHANT,
            "R1_P4D_LONGITUDINAL_RESPONSE/FOLLOWER",
            float(follower_delay_min),
        ),
    }
    backend = P4DBoundedCommandResponseEnvironmentalMotionBackend(
        backend_for(first_order_profile()), bindings
    )
    world = P4DObservedMetricEnvironmentalWorld(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=0.0,
        profile_catalog=catalog,
        safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=backend,
        retain_safety_intervals=True,
        stream_by_vessel={leader_id: P4D_LONGITUDINAL_STREAM_ID, follower_id: P4D_LONGITUDINAL_STREAM_ID},
        gates_by_stream={P4D_LONGITUDINAL_STREAM_ID: longitudinal_gates()},
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=catalog,
    )
    following = SameRouteFollowingSupervisor(
        additional_reaction_interval_min_by_vessel={
            leader_id: 0.0,
            follower_id: float(follower_delay_min),
        },
        additional_leader_observation_interval_min_by_vessel={
            leader_id: float(leader_observation_delay_min),
        },
    )
    started = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        {},
        EnvironmentalNetworkSourceReservoir(demands),
        recapture_config=scenario.recapture,
        runtime_config=scenario.runtime,
        recheck_config=base.recheck,
        classification_config=preflight_classification_config(),
        profile_catalog=catalog,
        same_route_dynamic_ingress=following,
        persistent_same_route_following=following,
    )
    outcome = started.session.run_until_capturing_terminal_outcome(50.0)
    typed = (
        TypedDemandEvent(leader_id, 0.0, P4D_LONGITUDINAL_STREAM_ID, "AS", 0.1),
        TypedDemandEvent(follower_id, 6.0, P4D_LONGITUDINAL_STREAM_ID, "HS", 0.9),
    )
    service = center_pass_service_summary(
        world.p4d_conflict_observations,
        typed,
        reference_matrix=GUO_2024_ILLUSTRATIVE,
    )
    safety = None
    if int(getattr(world, "safety_interval_count", 0)):
        summary = world.finalize_safety_experiment_summary()
        safety = {
            "modeled_hull_overlap_episode_count": summary.modeled_hull_overlap_episode_count,
            "safety_domain_intrusion_episode_count": summary.safety_domain_intrusion_episode_count,
            "near_miss_episode_count": summary.near_miss_episode_count,
            "unresolved_episode_count": summary.unresolved_episode_count,
            "sampled_min_center_separation_nm": summary.interval_summary.sampled_min_center_separation_nm,
        }
    follower_request = next(
        (x for x in world.p4d_conflict_observations if x.vessel_id == follower_id and x.role == "CONFLICT_REQUEST"),
        None,
    )
    follower_center = next(
        (x for x in world.p4d_conflict_observations if x.vessel_id == follower_id and x.role == "CONFLICT_CENTER_PASS"),
        None,
    )
    same_route_supervisions = tuple(
        started.runtime_start.session.mixed_session.same_route_supervisions
    )
    return {
        "follower_delay_min": float(follower_delay_min),
        "leader_observation_delay_min": float(leader_observation_delay_min),
        "pair_reaction_budget_min": following.config_for_pair(
            follower_id, leader_id
        ).reaction_interval_min,
        "command_response_semantics_id": P4D_COMMAND_RESPONSE_SEMANTICS_ID,
        "terminal_status": outcome.status.value,
        "admissions": [
            {
                "vessel_id": x.vessel_id,
                "demand_time_min": x.demand_time,
                "admission_time_min": x.admission_time,
                "waiting_time_min": x.waiting_time_min,
            }
            for x in outcome.reservoir.admissions
        ],
        "center_service": service,
        "follower_request_speed_nm_per_min": None if follower_request is None else follower_request.through_water_speed_nm_per_min,
        "follower_center_speed_nm_per_min": None if follower_center is None else follower_center.through_water_speed_nm_per_min,
        "same_route_supervisions": [
            {
                "time_min": item.time_min,
                "follower_id": item.follower_id,
                "leader_id": item.leader_id,
                "previous_speed_nm_per_min": item.previous_command.target_through_water_speed,
                "issued_speed_nm_per_min": item.issued_command.target_through_water_speed,
                "max_follower_speed_nm_per_min": item.max_follower_speed_nm_per_min,
            }
            for item in same_route_supervisions
        ],
        "safety": safety,
        "rebootstrap_count": len(started.runtime_start.session.persistent.rebootstrap_results),
    }


def run_longitudinal_response_latency_qualification() -> P4DLongitudinalQualificationResult:
    """Compare neutral, bounded-small and 60 s stress response arms.

    This is an architecture qualification, not a parameter-selection result.  It proves
    that the corrected P4D latency semantics do not starve under changing following
    commands and that service order is preserved in the fixture.
    """
    arms = {
        "NEUTRAL_0S": _run_longitudinal_response_arm(follower_delay_min=0.0),
        "BOUNDED_6S": _run_longitudinal_response_arm(follower_delay_min=0.1),
        "STRESS_60S": _run_longitudinal_response_arm(follower_delay_min=1.0),
    }
    order_ok = all(
        arm["center_service"]["arrival_order_inversion_count"] == 0
        and [r["vessel_id"] for r in arm["center_service"]["records"]] == ["RESP-LEADER", "RESP-FOLLOWER"]
        for arm in arms.values()
    )
    success = all(arm["terminal_status"] == "SUCCESS" for arm in arms.values())
    no_hull_overlap = all(
        arm["safety"] is None or arm["safety"]["modeled_hull_overlap_episode_count"] == 0
        for arm in arms.values()
    )
    stress_wait = arms["STRESS_60S"]["admissions"][1]["waiting_time_min"]
    neutral_wait = arms["NEUTRAL_0S"]["admissions"][1]["waiting_time_min"]
    document = {
        "format": "cross-r1-p4d-longitudinal-response-latency-qualification",
        "schema_version": 1,
        "track_id": P4D_LONGITUDINAL_TRACK_ID,
        "evidence_role": P4D_LONGITUDINAL_EVIDENCE_ROLE,
        "scientific_confirmatory_interpretation_authorized": False,
        "purpose": "VERIFY_NON_STARVING_P4D_RESPONSE_LATENCY_WITH_SAME_ROUTE_SAFETY_COUPLING",
        "reference_use": "POST_HOC_DIAGNOSTIC_ONLY_NOT_RUNTIME_FORCING",
        "arms": arms,
        "qualification": {
            "all_terminal_success": success,
            "service_order_preserved": order_ok,
            "no_modeled_hull_overlap": no_hull_overlap,
            "stress_source_wait_not_less_than_neutral": stress_wait + _EPS >= neutral_wait,
            "pass": bool(success and order_ok and no_hull_overlap and stress_wait + _EPS >= neutral_wait),
        },
        "interpretation_guardrail": (
            "The 6 s and 60 s values are architecture qualification points only.  This artifact "
            "does not select a confirmatory human-response parameter or claim empirical calibration."
        ),
    }
    raw = _canonical_json(document)
    return P4DLongitudinalQualificationResult(document, raw)
