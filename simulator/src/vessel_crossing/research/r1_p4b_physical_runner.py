"""R1 P4B one-point physical runner.

This module owns only one-point runtime materialization, execution, measurement extraction,
and canonical result-document construction.  Campaign scheduling and durable writes live elsewhere.
"""
from __future__ import annotations

from dataclasses import asdict
from statistics import mean

from vessel_crossing import __version__
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.nextgen import (
    EncounterRegistry,
    EnvironmentalNetworkSourceFutureIngressConfig,
    EnvironmentalNetworkSourceReservoir,
    VesselProfileCatalog,
    generate_poisson_network_traffic,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_paired_experiment import _MetricEnvironmentalWorld
from vessel_crossing.nextgen.environmental_network_scenario import _environmental_ingress_records
from vessel_crossing.nextgen.environmental_network_source_reservoir import source_queue_metrics
from vessel_crossing.nextgen.mission_safety import summarize_mission_safety
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingSupervisor
from vessel_crossing.research.r1_p4b_behavior_adapter import P4BBehaviorEnvironmentalMotionBackend
from vessel_crossing.research.r1_p4b_behavior_grounding import T0, publication_bindings_for_endpoint
from vessel_crossing.research.r1_p4b_behavior_qualification import (
    bindings_for_penetration,
    endpoint_bindings,
    traffic_input_sha256,
    typed_demand_for_penetration,
)
from vessel_crossing.research.r1_p4b_engineering_preflight import preflight_classification_config
from vessel_crossing.research.r1_p4b_experiment_spec import (
    P4B_EVALUATION_END_MIN,
    P4B_POINT_FORMAT,
    P4B_POINT_SCHEMA_VERSION,
    P4B_PRIMARY_MEASUREMENT_END_MIN,
    _EPS,
    _canonical_json,
    _plant_profiles,
    _sha_bytes,
    _timing_endpoint,
    build_publication_config,
)
from vessel_crossing.research.r1_p4b_plan_materialization import (
    P4BPhysicalPoint,
    P4BPublicationExecutionPlan,
)
from vessel_crossing.research.r1_robustness_stage2 import backend_for

def _materialize_runtime_bindings(plan: P4BPublicationExecutionPlan, point: P4BPhysicalPoint, traffic):
    endpoint = _timing_endpoint(point.timing_endpoint_id)
    as_binding, hs_binding = publication_bindings_for_endpoint(endpoint)
    if endpoint is T0:
        bindings = endpoint_bindings(
            traffic,
            profile=as_binding.profile,
            adapter_id=as_binding.adapter_id + "/DEDUP_NEUTRAL",
            command_delay_min=0.0,
        )
        typed = None
    else:
        assert point.penetration_execution is not None
        typed, bindings = bindings_for_penetration(
            traffic,
            penetration=point.penetration_execution,
            campaign_id=plan.campaign_id,
            replication=point.replication,
            as_binding=as_binding,
            hs_binding=hs_binding,
        )
    return typed, bindings

def _source_window_metrics(reservoir: EnvironmentalNetworkSourceReservoir, *, at: float) -> dict[str, object]:
    """Compute immutable source metrics at an earlier/equal cutoff from final evidence.

    Unlike ``source_queue_metrics`` this helper is allowed to inspect a *final* reservoir
    at an earlier cutoff by filtering immutable demand/admission evidence.  It never
    rewinds or mutates runtime state.
    """
    due_demands = tuple(d for d in reservoir.demands if d.demand_time <= at + _EPS)
    admissions = tuple(sorted(
        (a for a in reservoir.admissions if a.admission_time <= at + _EPS),
        key=lambda a: (a.admission_time, a.vessel_id),
    ))
    admitted_ids = {a.vessel_id for a in admissions}
    due_unadmitted = tuple(d for d in due_demands if d.vessel_id not in admitted_ids)
    waits = tuple(float(a.waiting_time_min) for a in admissions)
    busy_headways: list[float] = []
    for previous, current in zip(admissions, admissions[1:]):
        # Same convention already used by R1 headway-screening: the next demand was
        # already waiting when the previous physical admission occurred.
        if current.demand_time <= previous.admission_time + _EPS:
            busy_headways.append(float(current.admission_time - previous.admission_time))
    return {
        "cutoff_min": float(at),
        "due_demand_count": len(due_demands),
        "admission_count": len(admissions),
        "backlog_count": len(due_unadmitted),
        "mean_admitted_source_wait_min": (sum(waits) / len(waits)) if waits else None,
        "max_admitted_source_wait_min": max(waits) if waits else None,
        "censored_due_demand_count": len(due_unadmitted),
        "busy_admission_headway_mean_min": (sum(busy_headways) / len(busy_headways)) if busy_headways else None,
        "busy_admission_headway_count": len(busy_headways),
    }

def _route_completion_metrics(runtime, reservoir: EnvironmentalNetworkSourceReservoir, *, at: float) -> dict[str, object]:
    """Summarize immutable route-completion evidence observed by ``at``."""
    admissions = {item.vessel_id: item for item in reservoir.admissions}
    records: list[tuple[str, float, float, float]] = []
    seen: set[str] = set()
    for disposition in runtime.persistent.disposition_results:
        completion_time = float(disposition.committed_observation_time)
        if completion_time > at + _EPS:
            continue
        for vessel in disposition.dispositions:
            if vessel.vessel_id in seen:
                raise RuntimeError(f"duplicate route-completion evidence for {vessel.vessel_id!r}")
            admission = admissions.get(vessel.vessel_id)
            if admission is None:
                raise RuntimeError(f"route completion lacks source admission for {vessel.vessel_id!r}")
            seen.add(vessel.vessel_id)
            records.append((
                vessel.vessel_id,
                completion_time,
                completion_time - float(admission.admission_time),
                completion_time - float(admission.demand_time),
            ))
    route_durations = tuple(x[2] for x in records)
    system_durations = tuple(x[3] for x in records)
    return {
        "cutoff_min": float(at),
        "route_completed_count": len(records),
        "mean_route_duration_min": (sum(route_durations) / len(route_durations)) if route_durations else None,
        "mean_system_duration_min": (sum(system_durations) / len(system_durations)) if system_durations else None,
    }

def run_publication_physical_point(
    plan: P4BPublicationExecutionPlan,
    point: P4BPhysicalPoint,
) -> bytes:
    """Run one physical point. Call only from the separately authorized executor."""
    if point.source_archive_sha256 != plan.source_archive_sha256:
        raise RuntimeError("point/source archive identity mismatch")
    config = build_publication_config(
        room_nm=point.room_nm,
        demand_per_hour=point.demand_per_hour,
        traffic_seed=point.traffic_seed,
    )
    if config.config_sha256 != point.config_sha256:
        raise RuntimeError("publication config identity drift")
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=config.scenario.traffic_horizon_min,
        seed=config.scenario.seed,
        start_time=config.scenario.start_time,
    )
    traffic_sha = traffic_input_sha256(traffic)
    if traffic_sha != point.expected_traffic_input_sha256:
        raise RuntimeError("publication traffic identity drift")

    typed, bindings = _materialize_runtime_bindings(plan, point, traffic)
    profile = _plant_profiles()[point.plant_id]
    base_backend = backend_for(profile)
    behavior_backend = P4BBehaviorEnvironmentalMotionBackend(base_backend, bindings)
    if behavior_backend.backend_id != point.expected_behavior_backend_id:
        raise RuntimeError("publication behavior backend identity drift")

    scenario = config.scenario
    reservoir = EnvironmentalNetworkSourceReservoir.from_scenario_ingress(
        _environmental_ingress_records(traffic, scenario)
    )
    profiles = VesselProfileCatalog()
    world = _MetricEnvironmentalWorld(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=scenario.start_time,
        profile_catalog=profiles,
        safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=behavior_backend,
        retain_safety_intervals=False,
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=profiles,
    )
    following = SameRouteFollowingSupervisor()
    pair = scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    started = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        {},
        reservoir,
        recapture_config=scenario.recapture,
        runtime_config=scenario.runtime,
        recheck_config=config.recheck,
        classification_config=preflight_classification_config(),
        profile_catalog=profiles,
        same_route_dynamic_ingress=following,
        future_ingress_clearance=EnvironmentalNetworkSourceFutureIngressConfig(pair_clearance=pair),
        persistent_same_route_following=following,
    )
    outcome = started.session.run_until_capturing_terminal_outcome(scenario.end_time)
    interval_count = int(getattr(started.session.world, "safety_interval_count", 0))
    safety = started.session.world.finalize_safety_experiment_summary() if interval_count else None
    queue = source_queue_metrics(outcome.reservoir, at=outcome.terminal_time_min)
    runtime = started.runtime_start.session
    success_full_horizon = outcome.status.value == "SUCCESS" and abs(outcome.terminal_time_min - scenario.end_time) <= _EPS
    performance = None
    if success_full_horizon:
        source_primary = _source_window_metrics(
            outcome.reservoir, at=P4B_PRIMARY_MEASUREMENT_END_MIN
        )
        source_evaluation = _source_window_metrics(
            outcome.reservoir, at=P4B_EVALUATION_END_MIN
        )
        completion_primary = _route_completion_metrics(
            runtime, outcome.reservoir, at=P4B_PRIMARY_MEASUREMENT_END_MIN
        )
        completion_evaluation = _route_completion_metrics(
            runtime, outcome.reservoir, at=P4B_EVALUATION_END_MIN
        )
        primary_duration_h = (P4B_PRIMARY_MEASUREMENT_END_MIN - scenario.start_time) / 60.0
        evaluation_duration_h = (P4B_EVALUATION_END_MIN - scenario.start_time) / 60.0
        cohort_complete_by_evaluation = (
            int(source_evaluation["admission_count"]) == int(source_evaluation["due_demand_count"])
        )
        performance = {
            # Primary confirmatory estimand: upstream physical admissible inflow.
            "primary_source_admission_throughput_per_hour_90": int(source_primary["admission_count"]) / primary_duration_h,
            "primary_source_admission_count_90": source_primary["admission_count"],
            "primary_source_backlog_count_90": source_primary["backlog_count"],
            "primary_mean_admitted_source_wait_min_90": source_primary["mean_admitted_source_wait_min"],
            "primary_max_admitted_source_wait_min_90": source_primary["max_admitted_source_wait_min"],
            "primary_censored_due_demand_count_90": source_primary["censored_due_demand_count"],
            "primary_busy_admission_headway_mean_min_90": source_primary["busy_admission_headway_mean_min"],
            "primary_busy_admission_headway_count_90": source_primary["busy_admission_headway_count"],
            "primary_route_completed_count_90": completion_primary["route_completed_count"],
            "primary_mean_route_duration_min_90": completion_primary["mean_route_duration_min"],
            "primary_mean_system_duration_min_90": completion_primary["mean_system_duration_min"],
            # Evaluation/clearance evidence retained through 150 min.
            "evaluation_source_admission_throughput_per_hour_150": int(source_evaluation["admission_count"]) / evaluation_duration_h,
            "evaluation_source_admission_count_150": source_evaluation["admission_count"],
            "evaluation_source_backlog_count_150": source_evaluation["backlog_count"],
            "evaluation_mean_admitted_source_wait_min_150": source_evaluation["mean_admitted_source_wait_min"],
            "evaluation_max_admitted_source_wait_min_150": source_evaluation["max_admitted_source_wait_min"],
            "evaluation_censored_due_demand_count_150": source_evaluation["censored_due_demand_count"],
            "evaluation_busy_admission_headway_mean_min_150": source_evaluation["busy_admission_headway_mean_min"],
            "evaluation_busy_admission_headway_count_150": source_evaluation["busy_admission_headway_count"],
            "evaluation_route_completed_count_150": completion_evaluation["route_completed_count"],
            "evaluation_mean_route_duration_min_150": completion_evaluation["mean_route_duration_min"],
            "evaluation_mean_system_duration_min_150": completion_evaluation["mean_system_duration_min"],
            "cohort_all_due_demands_admitted_by_150": cohort_complete_by_evaluation,
            "cohort_mean_source_wait_min_if_complete_by_150": (
                source_evaluation["mean_admitted_source_wait_min"] if cohort_complete_by_evaluation else None
            ),
        }
    behavior_manifest = [
        {
            "vessel_id": vessel_id,
            "profile": binding.profile.value,
            "adapter_id": binding.adapter_id,
            "command_delay_min": binding.command_delay_min,
            "binding_sha256": binding.identity_sha256,
        }
        for vessel_id, binding in sorted(bindings.items())
    ]
    document = {
        "format": P4B_POINT_FORMAT,
        "schema_version": P4B_POINT_SCHEMA_VERSION,
        "campaign_id": plan.campaign_id,
        "protocol_id": plan.protocol_id,
        "plan_sha256": plan.identity_sha256,
        "source_archive_sha256": plan.source_archive_sha256,
        "software_version": __version__,
        "point": asdict(point),
        "config_sha256": config.config_sha256,
        "traffic_input_sha256": traffic_sha,
        "behavior_backend_id": behavior_backend.backend_id,
        "behavior_manifest_sha256": _sha_bytes(_canonical_json(behavior_manifest)),
        "behavior_manifest": behavior_manifest,
        "typed_execution_sha256": None if typed is None else _sha_bytes(_canonical_json([asdict(x) for x in typed])),
        "terminal": {
            "status": outcome.status.value,
            "terminal_time_min": outcome.terminal_time_min,
            "infeasibility_type": outcome.infeasibility_type,
            "resource_unavailability_type": outcome.resource_unavailability_type,
        },
        "performance": performance,
        "source_diagnostics": {
            "demand_count": queue.demand_count,
            "admission_count": queue.admission_count,
            "pending_count": queue.pending_count,
            "due_pending_count": queue.due_pending_count,
            "mean_admitted_waiting_time_min": queue.mean_admitted_waiting_time_min,
            "max_admitted_waiting_time_min": queue.max_admitted_waiting_time_min,
        },
        "response_diagnostics": {
            "rebootstrap_count": len(runtime.persistent.rebootstrap_results),
            "fleet_observation_count": len(runtime.coordinator.fleet_observations),
        },
        "safety": None if safety is None else {
            "modeled_hull_overlap_episode_count": safety.modeled_hull_overlap_episode_count,
            "safety_domain_intrusion_episode_count": safety.safety_domain_intrusion_episode_count,
            "near_miss_episode_count": safety.near_miss_episode_count,
            "unresolved_episode_count": safety.unresolved_episode_count,
            "sampled_min_center_separation_nm": safety.interval_summary.sampled_min_center_separation_nm,
            "sampled_min_hull_margin_nm": safety.interval_summary.sampled_min_hull_margin_nm,
            "sampled_min_safety_domain_margin_nm": safety.interval_summary.sampled_min_safety_domain_margin_nm,
            "worst_observed_evidence": safety.worst_observed_evidence.value,
        },
    }
    return _canonical_json(document)
