"""Authoritative R2 B4/B5 managed campaign and deterministic B6 envelope.

This module is a research-layer adapter over the qualified FS2/P4D runtime.  It does
not replace or retune physical dynamics, safety geometry, pair-context response, or
source admission physics.  B4/B5 only use the opt-in source-management controller
qualified by T0.  B6 changes demand supply only and explicitly stops new admissions
at the frozen release cutoff before draining the same physical runtime.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from typing import Mapping, Sequence

from vessel_crossing.nextgen import (
    EncounterRegistry,
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceFutureIngressConfig,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalRouteHandoffTarget,
    RouteFollower,
    RouteFollowerState,
    RouteNavigator,
    CrossTrackRecovery,
    VesselProfileCatalog,
    WaterRelativeVesselState,
    generate_poisson_network_traffic,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_scenario import build_environmental_network_scenario_ingress
from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import EnvironmentalNetworkSourceControllerContext
from vessel_crossing.nextgen.environmental_network_source_reservoir import (
    EnvironmentalNetworkSourceResolution,
    source_queue_metrics,
)
from vessel_crossing.nextgen.pair_context_following import PairContextFollowingSupervisor
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.research.r1_p4b_behavior_qualification import traffic_input_sha256
from vessel_crossing.research.r1_p4b_engineering_preflight import preflight_classification_config
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config
from vessel_crossing.research.r1_p4d_conflict_observer import P4DObservedMetricEnvironmentalWorld
from vessel_crossing.research.r1_p4d_lifecycle_audit import capture_terminal_lifecycle_audit
from vessel_crossing.research.r1_p4d_screening import (
    _conflict_summary,
    _route_completion_records,
    _stream_by_vessel,
)
from vessel_crossing.research.r1_robustness_stage2 import backend_for, first_order_profile
from vessel_crossing.research.r2_b0_b6_reference_controller import (
    R2B4B5SourceManagementController,
    R2ManagedReferenceConfig,
    controller_trace_as_dicts,
)
from vessel_crossing.research.r2_fs2_full_network_publication import (
    AS,
    HS,
    FS2_DEMAND_PER_HOUR,
    FS2_EVALUATION_HORIZON_MIN,
    FS2_ROOM_NM,
    FS2_TRAFFIC_HORIZON_MIN,
    FS2Point,
    FS2Plan,
    delta_arm_semantics,
    markov_typed_demand,
    ordering_exposure,
    pair_context_config_for_arm,
    typed_demand_sha256,
)

AUTHORITATIVE_POINT_FORMAT = "cross-r2-b0-b6-authoritative-point-v1"
AUTHORITATIVE_CAMPAIGN_ID = "R2_B0_B6_AUTHORITATIVE_REFERENCE_ENVELOPE_V1_2026-09-30"
B6_DEMANDS_PER_STREAM = 256
B6_WARMUP_MIN = 90.0
B6_MEASUREMENT_START_MIN = 90.0
B6_MEASUREMENT_END_MIN = 270.0
B6_RELEASE_CUTOFF_MIN = 270.0
B6_DRAIN_END_MIN = 330.0
_EPS = 1e-10


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _fixed_center_window(observations, *, start_min: float, end_min: float) -> dict[str, object]:
    rows = [
        row for row in observations
        if row.role == "CONFLICT_CENTER_PASS"
        and float(row.observation_time_min) >= start_min - _EPS
        and float(row.observation_time_min) <= end_min + _EPS
    ]
    width = end_min - start_min
    return {
        "service_event": "CONFLICT_CENTER_PASS",
        "window_min": [start_min, end_min],
        "passage_count": len(rows),
        "fixed_window_service_yield_per_hour": 60.0 * len(rows) / width,
        "passages": [
            {"vessel_id": row.vessel_id, "stream_id": row.stream_id, "time_min": row.observation_time_min}
            for row in rows
        ],
    }


def _safety_doc(world) -> dict[str, object] | None:
    interval_count = int(getattr(world, "safety_interval_count", 0))
    if not interval_count:
        return None
    safety = world.finalize_safety_experiment_summary()
    return {
        "modeled_hull_overlap_episode_count": safety.modeled_hull_overlap_episode_count,
        "safety_domain_intrusion_episode_count": safety.safety_domain_intrusion_episode_count,
        "near_miss_episode_count": safety.near_miss_episode_count,
        "unresolved_episode_count": safety.unresolved_episode_count,
        "sampled_min_center_separation_nm": safety.interval_summary.sampled_min_center_separation_nm,
        "sampled_min_hull_margin_nm": safety.interval_summary.sampled_min_hull_margin_nm,
        "sampled_min_safety_domain_margin_nm": safety.interval_summary.sampled_min_safety_domain_margin_nm,
        "worst_observed_evidence": safety.worst_observed_evidence.value,
    }


def _terminal_doc(outcome) -> dict[str, object]:
    return {
        "status": outcome.status.value,
        "terminal_time_min": outcome.terminal_time_min,
        "infeasibility_type": outcome.infeasibility_type,
        "resource_unavailability_type": outcome.resource_unavailability_type,
    }


def _source_doc(outcome) -> dict[str, object]:
    queue = source_queue_metrics(outcome.reservoir, at=outcome.terminal_time_min)
    return {
        "demand_count": queue.demand_count,
        "admission_count": queue.admission_count,
        "pending_count": queue.pending_count,
        "due_pending_count": queue.due_pending_count,
        "mean_admitted_waiting_time_min": queue.mean_admitted_waiting_time_min,
        "max_admitted_waiting_time_min": queue.max_admitted_waiting_time_min,
        "admissions": [
            {
                "vessel_id": x.vessel_id,
                "stream_id": x.stream_id,
                "demand_time_min": x.demand_time,
                "admission_time_min": x.admission_time,
                "waiting_time_min": x.waiting_time_min,
            }
            for x in outcome.reservoir.admissions
        ],
    }


def _build_world_runtime(*, cfg, reservoir, type_map: Mapping[str, str], delta_arm: str, level: str):
    response = pair_context_config_for_arm(type_map, delta_arm=delta_arm)
    following = PairContextFollowingSupervisor(response)
    profiles = VesselProfileCatalog()
    motion_backend = backend_for(first_order_profile())
    stream_by_vessel = {d.vessel_id: d.stream_id for d in reservoir.demands}
    world = P4DObservedMetricEnvironmentalWorld(
        (), limits={}, environmental_drift_velocity={}, start_time=cfg.scenario.start_time,
        profile_catalog=profiles, safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=motion_backend, retain_safety_intervals=False,
        stream_by_vessel=stream_by_vessel,
    )
    registry = EncounterRegistry(
        discovery=cfg.scenario.discovery,
        lifecycle=cfg.scenario.lifecycle,
        profile_catalog=profiles,
    )
    controller = R2B4B5SourceManagementController(
        config=R2ManagedReferenceConfig(level=level, delta_arm=delta_arm),
        type_by_vessel=type_map,
        center_observation_provider=lambda: world.p4d_conflict_observations,
    )
    pair = cfg.scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    started = start_environmental_network_source_fifo_runtime(
        registry, world, {}, reservoir,
        recapture_config=cfg.scenario.recapture,
        runtime_config=cfg.scenario.runtime,
        recheck_config=cfg.recheck,
        classification_config=preflight_classification_config(),
        profile_catalog=profiles,
        same_route_dynamic_ingress=following,
        future_ingress_clearance=EnvironmentalNetworkSourceFutureIngressConfig(pair_clearance=pair),
        persistent_same_route_following=following,
        source_admission_controller=controller,
    )
    return world, started, controller, following, motion_backend


def run_managed_fs2_point(
    *, parent_plan: FS2Plan, parent_point: FS2Point, level: str,
    authoritative_plan_sha256: str, authoritative_protocol_sha256: str,
) -> bytes:
    if level not in {"B4", "B5"}:
        raise ValueError("managed FS2 point level must be B4 or B5")
    if float(parent_point.rho) != 0.0:
        raise ValueError("B4/B5 authoritative campaign is frozen to FS2 rho=0 blocks")
    if parent_point not in parent_plan.points:
        raise ValueError("parent point not in frozen FS2 plan")

    cfg = build_publication_config(
        room_nm=FS2_ROOM_NM,
        demand_per_hour=FS2_DEMAND_PER_HOUR,
        traffic_seed=parent_point.traffic_seed,
    )
    scenario = cfg.scenario
    traffic = generate_poisson_network_traffic(
        cfg.network, cfg.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    traffic_sha = traffic_input_sha256(traffic)
    if traffic_sha != parent_point.expected_traffic_input_sha256 or traffic.count != parent_point.expected_traffic_count:
        raise RuntimeError("authoritative B4/B5 traffic identity drift")
    typed = markov_typed_demand(
        traffic, rho=parent_point.rho, type_assignment_seed=parent_point.type_assignment_seed,
    )
    typed_sha = typed_demand_sha256(typed)
    if typed_sha != parent_point.expected_typed_sha256:
        raise RuntimeError("authoritative B4/B5 typed assignment identity drift")
    type_map = {event.event_id: event.vessel_type for event in typed}
    response = pair_context_config_for_arm(type_map, delta_arm=parent_point.delta_arm)
    if PairContextFollowingSupervisor(response).config_identity_sha256 != parent_point.expected_mechanism_config_sha256:
        raise RuntimeError("authoritative B4/B5 mechanism identity drift")

    ingress = build_environmental_network_scenario_ingress(traffic, scenario)
    reservoir = EnvironmentalNetworkSourceReservoir.from_scenario_ingress(ingress)
    world, started, controller, following, motion_backend = _build_world_runtime(
        cfg=cfg, reservoir=reservoir, type_map=type_map,
        delta_arm=parent_point.delta_arm, level=level,
    )
    outcome = started.session.run_until_capturing_terminal_outcome(FS2_EVALUATION_HORIZON_MIN)
    runtime = started.runtime_start.session
    completions = _route_completion_records(runtime, outcome.reservoir, at=outcome.terminal_time_min)
    conflict = _conflict_summary(world.p4d_conflict_observations, outcome.reservoir.admissions, room_nm=FS2_ROOM_NM)
    lifecycle = capture_terminal_lifecycle_audit(runtime, outcome.reservoir, completions)
    nominal = float(conflict["nominal_route_duration_min"])
    route_extra = [float(x["route_duration_min"]) - nominal for x in completions]

    document = {
        "format": AUTHORITATIVE_POINT_FORMAT,
        "schema_version": 1,
        "scientific_role": "AUTHORITATIVE_MANAGED_REFERENCE_LADDER",
        "campaign_id": AUTHORITATIVE_CAMPAIGN_ID,
        "level": level,
        "delta_arm": parent_point.delta_arm,
        "parent_fs2": {
            "campaign_id": parent_plan.campaign_id,
            "plan_sha256": parent_plan.identity_sha256,
            "point_id": parent_point.point_id,
            "point_identity_sha256": parent_point.compute_identity_sha256(),
            "replication": parent_point.replication,
            "rho": parent_point.rho,
        },
        "authoritative_plan_sha256": authoritative_plan_sha256,
        "authoritative_protocol_sha256": authoritative_protocol_sha256,
        "runtime_semantics": {
            "qualified_parent": "R2_FS2_FULL_NETWORK_PUBLICATION_V1_2026-09-30",
            "canonical_physics_runtime_unchanged": True,
            "source_management_level": level,
            "reference_headway_forcing": False,
            "future_demand_lookahead": False,
            "canonical_source_clearance_final_gate": True,
            "as_hs_physical_profiles_identical": True,
            "motion_backend_id": getattr(motion_backend, "backend_id", type(motion_backend).__name__),
            "pair_context_mechanism_id": following.mechanism_id,
            "pair_context_config_sha256": following.config_identity_sha256,
        },
        "traffic_input_sha256": traffic_sha,
        "traffic_count": traffic.count,
        "typed_assignment_sha256": typed_sha,
        "typed_roles": [asdict(event) for event in typed],
        "ordering_exposure": ordering_exposure(typed),
        "delta_semantics": delta_arm_semantics(parent_point.delta_arm),
        "terminal": _terminal_doc(outcome),
        "source": _source_doc(outcome),
        "lifecycle": lifecycle,
        "route": {
            "completion_count": len(completions),
            "records": list(completions),
            "mean_route_extra_min": None if not route_extra else sum(route_extra) / len(route_extra),
            "max_route_extra_min": None if not route_extra else max(route_extra),
        },
        "conflict": conflict,
        "primary_kpi": _fixed_center_window(
            world.p4d_conflict_observations, start_min=0.0, end_min=FS2_TRAFFIC_HORIZON_MIN,
        ),
        "controller": {
            "trace": list(controller_trace_as_dicts(controller.trace)),
            "trace_count": len(controller.trace),
            "bypass_count": dict(controller.bypass_count),
        },
        "safety": _safety_doc(world),
    }
    return canonical_json(document)


def _saturated_demand(cfg, *, vessel_id: str, stream_id: str) -> EnvironmentalNetworkSourceDemand:
    stream = next(item for item in cfg.streams if item.stream_id == stream_id)
    path = cfg.network.resolve(stream.request)
    route = path.route
    profile = stream.vessel_class
    if profile is None:
        raise RuntimeError("qualified R2 stream lacks vessel profile")
    drift = cfg.scenario.drift_for_stream(stream_id)
    heading = route.segment_heading(0)
    vessel = WaterRelativeVesselState(
        vessel_id=vessel_id,
        position=route.waypoints[0],
        through_water_heading_deg=heading,
        through_water_speed=stream.cruise_speed,
    )
    target = EnvironmentalRouteHandoffTarget(
        vessel_id=vessel_id,
        navigator=RouteNavigator(
            RouteFollower(route, cruise_speed=stream.cruise_speed),
            CrossTrackRecovery(route, max_intercept_angle_deg=stream.recovery_max_intercept_angle_deg),
        ),
        follower_state=RouteFollowerState(),
        end_action=stream.end_action,
    )
    return EnvironmentalNetworkSourceDemand(
        stream_id=stream_id,
        nominal_vessel_id=vessel_id,
        demand_time=0.0,
        vessel=vessel,
        route_target=target,
        limits=profile.maneuver_limits,
        environmental_drift_velocity=drift,
        profile=profile,
    )


class _B6ReleaseCutoffController:
    """Composition wrapper that closes source release after the frozen 270-min cutoff."""

    def __init__(self, inner: R2B4B5SourceManagementController) -> None:
        self.inner = inner
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def prepare_reservoir(
        self, reservoir: EnvironmentalNetworkSourceReservoir, context: EnvironmentalNetworkSourceControllerContext,
    ) -> EnvironmentalNetworkSourceReservoir:
        managed = self.inner.prepare_reservoir(reservoir, context)
        if not self.closed:
            return managed
        pending = {d.vessel_id for d in managed.pending_demands}
        changed = tuple(
            replace(d, release_not_before_time=(1.0e9 if d.vessel_id in pending else d.release_not_before_time))
            for d in managed.demands
        )
        return EnvironmentalNetworkSourceReservoir(
            demands=changed,
            admissions=managed.admissions,
            service_order_by_stream=managed.service_order_by_stream,
        )

    def observe_resolution(
        self, resolution: EnvironmentalNetworkSourceResolution, context: EnvironmentalNetworkSourceControllerContext,
    ) -> None:
        self.inner.observe_resolution(resolution, context)


def run_b6_point(
    *, delta_arm: str, authoritative_plan_sha256: str, authoritative_protocol_sha256: str,
) -> bytes:
    if delta_arm not in {"NEG050", "ZERO", "POS050"}:
        raise ValueError(delta_arm)
    cfg = build_publication_config(room_nm=FS2_ROOM_NM, demand_per_hour=FS2_DEMAND_PER_HOUR, traffic_seed=0)
    cycle = (AS, AS, HS, HS)
    demands: list[EnvironmentalNetworkSourceDemand] = []
    type_map: dict[str, str] = {}
    for stream_id in ("north", "west"):
        for index in range(1, B6_DEMANDS_PER_STREAM + 1):
            vessel_id = f"{stream_id}-b6-{index:03d}"
            vessel_type = cycle[(index - 1) % len(cycle)]
            type_map[vessel_id] = vessel_type
            demands.append(_saturated_demand(cfg, vessel_id=vessel_id, stream_id=stream_id))
    demands.sort(key=lambda d: (d.demand_time, d.vessel_id))
    reservoir = EnvironmentalNetworkSourceReservoir(tuple(demands))
    level = "B4" if delta_arm == "ZERO" else "B5"

    response = pair_context_config_for_arm(type_map, delta_arm=delta_arm)
    following = PairContextFollowingSupervisor(response)
    profiles = VesselProfileCatalog()
    motion_backend = backend_for(first_order_profile())
    world = P4DObservedMetricEnvironmentalWorld(
        (), limits={}, environmental_drift_velocity={}, start_time=0.0,
        profile_catalog=profiles, safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=motion_backend, retain_safety_intervals=False,
        stream_by_vessel={d.vessel_id: d.stream_id for d in demands},
    )
    registry = EncounterRegistry(
        discovery=cfg.scenario.discovery,
        lifecycle=cfg.scenario.lifecycle,
        profile_catalog=profiles,
    )
    inner = R2B4B5SourceManagementController(
        config=R2ManagedReferenceConfig(level=level, delta_arm=delta_arm),
        type_by_vessel=type_map,
        center_observation_provider=lambda: world.p4d_conflict_observations,
    )
    controller = _B6ReleaseCutoffController(inner)
    pair = cfg.scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    started = start_environmental_network_source_fifo_runtime(
        registry, world, {}, reservoir,
        recapture_config=cfg.scenario.recapture,
        runtime_config=cfg.scenario.runtime,
        recheck_config=cfg.recheck,
        classification_config=preflight_classification_config(),
        profile_catalog=profiles,
        same_route_dynamic_ingress=following,
        future_ingress_clearance=EnvironmentalNetworkSourceFutureIngressConfig(pair_clearance=pair),
        persistent_same_route_following=following,
        source_admission_controller=controller,
    )
    pre_cutoff = started.session.run_until_capturing_terminal_outcome(B6_RELEASE_CUTOFF_MIN)
    if pre_cutoff.status.value == "SUCCESS" and pre_cutoff.terminal_time_min >= B6_RELEASE_CUTOFF_MIN - _EPS:
        controller.close()
        outcome = started.session.run_until_capturing_terminal_outcome(B6_DRAIN_END_MIN)
    else:
        outcome = pre_cutoff

    runtime = started.runtime_start.session
    completions = _route_completion_records(runtime, outcome.reservoir, at=outcome.terminal_time_min)
    conflict = _conflict_summary(world.p4d_conflict_observations, outcome.reservoir.admissions, room_nm=FS2_ROOM_NM)
    lifecycle = capture_terminal_lifecycle_audit(runtime, outcome.reservoir, completions)
    admissions_after_cutoff = [x for x in outcome.reservoir.admissions if x.admission_time > B6_RELEASE_CUTOFF_MIN + _EPS]
    if admissions_after_cutoff:
        raise RuntimeError("B6 release cutoff violation")

    document = {
        "format": AUTHORITATIVE_POINT_FORMAT,
        "schema_version": 1,
        "scientific_role": "DETERMINISTIC_MODEL_DERIVED_SATURATED_REFERENCE_ENVELOPE",
        "campaign_id": AUTHORITATIVE_CAMPAIGN_ID,
        "level": "B6",
        "delta_arm": delta_arm,
        "authoritative_plan_sha256": authoritative_plan_sha256,
        "authoritative_protocol_sha256": authoritative_protocol_sha256,
        "demand_contract": {
            "streams": ["north", "west"],
            "demands_per_stream": B6_DEMANDS_PER_STREAM,
            "all_demand_time_min": 0.0,
            "type_cycle_per_stream": [AS, AS, HS, HS],
            "controller_level": level,
            "warmup_min": B6_WARMUP_MIN,
            "measurement_window_min": [B6_MEASUREMENT_START_MIN, B6_MEASUREMENT_END_MIN],
            "release_cutoff_min": B6_RELEASE_CUTOFF_MIN,
            "drain_end_min": B6_DRAIN_END_MIN,
            "stochastic_arrival_starvation": False,
        },
        "runtime_semantics": {
            "canonical_physics_runtime_unchanged": True,
            "reference_headway_forcing": False,
            "canonical_source_clearance_final_gate": True,
            "as_hs_physical_profiles_identical": True,
            "pair_context_config_sha256": following.config_identity_sha256,
        },
        "terminal": _terminal_doc(outcome),
        "source": _source_doc(outcome),
        "lifecycle": lifecycle,
        "conflict": conflict,
        "primary_kpi": _fixed_center_window(
            world.p4d_conflict_observations,
            start_min=B6_MEASUREMENT_START_MIN,
            end_min=B6_MEASUREMENT_END_MIN,
        ),
        "controller": {
            "trace": list(controller_trace_as_dicts(inner.trace)),
            "trace_count": len(inner.trace),
            "bypass_count": dict(inner.bypass_count),
            "release_cutoff_applied": controller.closed,
        },
        "safety": _safety_doc(world),
        "claim_scope": {
            "field_calibrated_capacity": False,
            "sustainable_capacity_if_terminal_not_success": False,
            "stochastic_confidence_interval": False,
        },
    }
    return canonical_json(document)
