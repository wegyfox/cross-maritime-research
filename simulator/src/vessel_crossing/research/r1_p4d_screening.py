"""Qualification/screening runtime for R1 P4D interaction-active validity work.

This is deliberately not a confirmatory runner.  It keeps ordinary independent CROSS
source FIFO semantics, applies only the declared microscopic AS/HS behavior contract,
and attaches the passive conflict-boundary observer.  No macro/reference pair headway
is inserted anywhere in the target runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from statistics import mean
from typing import Mapping

from vessel_crossing.nextgen import (
    EncounterRegistry,
    EnvironmentalNetworkSourceFutureIngressConfig,
    EnvironmentalNetworkSourceReservoir,
    VesselProfileCatalog,
    generate_poisson_network_traffic,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_scenario import _environmental_ingress_records
from vessel_crossing.nextgen.environmental_network_source_reservoir import source_queue_metrics
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingSupervisor
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.research.r1_p4b_publication_reference import GUO_2024_ILLUSTRATIVE
from vessel_crossing.research.r1_p4d_command_response import (
    P4DBoundedCommandResponseEnvironmentalMotionBackend,
    P4D_COMMAND_RESPONSE_SEMANTICS_ID,
)
from vessel_crossing.research.r1_p4b_behavior_qualification import (
    traffic_input_sha256,
    typed_demand_for_penetration,
)
from vessel_crossing.research.r1_p4b_engineering_preflight import preflight_classification_config
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config
from vessel_crossing.research.r1_robustness_stage2 import backend_for, first_order_profile
from vessel_crossing.testbed.behavior import BehaviorProfile

from .r1_p4d_behavior_contract import (
    P4DBehaviorContract,
    P4DMicroscopicRoleSpec,
    control_only_provenance,
    delay_binding_for_p4d_role,
    profile_for_p4d_role,
    validate_role_map,
)
from .r1_p4d_conflict_observer import P4DObservedMetricEnvironmentalWorld
from .r1_p4d_conflict_service import center_pass_service_summary
from .r1_p4d_lifecycle_audit import capture_terminal_lifecycle_audit

_EPS = 1e-10
P4D_SCREENING_CAMPAIGN_ID = "R1_P4D_SCREENING_QUALIFICATION_V1"
P4D_SCREENING_EVIDENCE_ROLE = "QUALIFICATION_SCREENING_ONLY_NOT_CONFIRMATORY"


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _angular_abs_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _stream_by_vessel(traffic) -> dict[str, str]:
    out: dict[str, str] = {}
    for stream in traffic.streams:
        for vessel_id in stream.vessel_ids:
            if vessel_id in out:
                raise ValueError(f"duplicate vessel id {vessel_id!r}")
            out[vessel_id] = stream.stream_id
    return out


def _role_map(traffic, *, penetration: float, replication: int, campaign_id: str) -> tuple[dict[str, str], tuple]:
    typed = typed_demand_for_penetration(
        traffic,
        penetration=penetration,
        campaign_id=campaign_id,
        replication=replication,
    )
    role_by_vessel = {event.event_id: event.vessel_type for event in typed}
    expected = {arrival.vessel.vessel_id for arrival in traffic.arrivals}
    validate_role_map(role_by_vessel, expected)
    return role_by_vessel, typed


def _following_supervisor_for_role_map(
    role_by_vessel: Mapping[str, str],
    contract: P4DBehaviorContract,
) -> SameRouteFollowingSupervisor:
    """Bind microscopic command latency into the physical following reaction envelope.

    This is not a macro headway.  It makes the same-route safety kernel aware of the
    additional behavior-layer time before a requested speed command can take effect.
    Omitting this coupling can let a delayed follower violate the braking envelope.
    """
    additional = {
        vessel_id: contract.spec_for_role(role).command_delay_min
        for vessel_id, role in role_by_vessel.items()
    }
    return SameRouteFollowingSupervisor(
        additional_reaction_interval_min_by_vessel=additional
    )


def _role_profiles_and_bindings(traffic, role_by_vessel: Mapping[str, str], contract: P4DBehaviorContract):
    bindings = {}
    profiles = {}
    role_binding_cache = {
        role: delay_binding_for_p4d_role(contract.spec_for_role(role), contract_id=contract.contract_id)
        for role in ("AS", "HS")
    }
    for arrival in traffic.arrivals:
        vessel_id = arrival.vessel.vessel_id
        role = role_by_vessel[vessel_id]
        spec = contract.spec_for_role(role)
        if arrival.vessel_class is None:
            raise ValueError(f"traffic vessel {vessel_id!r} has no vessel class")
        profiles[vessel_id] = profile_for_p4d_role(arrival.vessel_class, spec)
        bindings[vessel_id] = role_binding_cache[role]
    return profiles, bindings


def _custom_ingress(traffic, scenario, profile_by_vessel):
    base = _environmental_ingress_records(traffic, scenario)
    out = []
    for item in base:
        profile = profile_by_vessel[item.vessel_id]
        record = replace(item.record, profile=profile, limits=profile.maneuver_limits)
        out.append(replace(item, record=record))
    return tuple(out)


def _route_completion_records(runtime, reservoir, *, at: float):
    admissions = {item.vessel_id: item for item in reservoir.admissions}
    records = []
    seen = set()
    for disposition in runtime.persistent.disposition_results:
        completion_time = float(disposition.committed_observation_time)
        if completion_time > at + _EPS:
            continue
        for vessel in disposition.dispositions:
            if vessel.vessel_id in seen:
                raise RuntimeError(f"duplicate route completion for {vessel.vessel_id!r}")
            admission = admissions.get(vessel.vessel_id)
            if admission is None:
                raise RuntimeError(f"route completion lacks source admission for {vessel.vessel_id!r}")
            seen.add(vessel.vessel_id)
            records.append({
                "vessel_id": vessel.vessel_id,
                "completion_time_min": completion_time,
                "route_duration_min": completion_time - float(admission.admission_time),
                "system_duration_min": completion_time - float(admission.demand_time),
            })
    return tuple(sorted(records, key=lambda x: (x["completion_time_min"], x["vessel_id"])))


def _conflict_summary(observations, admissions, *, room_nm: float):
    by_vessel: dict[str, dict[str, object]] = {}
    admission_by_vessel = {x.vessel_id: x for x in admissions}
    nominal_route_duration = 2.0 * room_nm / 0.1
    for obs in observations:
        item = by_vessel.setdefault(obs.vessel_id, {"stream_id": obs.stream_id, "gates": {}})
        item["gates"][obs.role] = obs

    complete = 0
    request_entry_excess = []
    max_cross_track = 0.0
    max_heading_deviation = 0.0
    max_speed_deviation = 0.0
    center_events = []
    gate_rows = []
    for vessel_id, item in sorted(by_vessel.items()):
        stream_id = item["stream_id"]
        gates = item["gates"]
        if all(role in gates for role in ("CONFLICT_REQUEST", "CONFLICT_ENTRY", "CONFLICT_CENTER_PASS", "CONFLICT_EXIT")):
            complete += 1
        if "CONFLICT_REQUEST" in gates and "CONFLICT_ENTRY" in gates:
            actual = gates["CONFLICT_ENTRY"].observation_time_min - gates["CONFLICT_REQUEST"].observation_time_min
            request_entry_excess.append(actual - 3.0)
        if "CONFLICT_CENTER_PASS" in gates:
            center = gates["CONFLICT_CENTER_PASS"]
            cross = abs(center.position_nm[1]) if stream_id == "west" else abs(center.position_nm[0])
            max_cross_track = max(max_cross_track, cross)
            nominal_heading = 270.0 if stream_id == "west" else 0.0
            max_heading_deviation = max(max_heading_deviation, _angular_abs_delta(center.through_water_heading_deg, nominal_heading))
            max_speed_deviation = max(max_speed_deviation, abs(center.through_water_speed_nm_per_min - 0.1))
            center_events.append(center)
        admission = admission_by_vessel.get(vessel_id)
        for role, obs in sorted(gates.items(), key=lambda x: x[1].observation_time_min):
            gate_rows.append({
                "vessel_id": vessel_id,
                "stream_id": stream_id,
                "role": role,
                "observation_time_min": obs.observation_time_min,
                "elapsed_since_admission_min": None if admission is None else obs.observation_time_min - admission.admission_time,
                "position_nm": list(obs.position_nm),
                "through_water_heading_deg": obs.through_water_heading_deg,
                "through_water_speed_nm_per_min": obs.through_water_speed_nm_per_min,
            })
    center_events.sort(key=lambda x: (x.observation_time_min, x.vessel_id))
    center_headways = [b.observation_time_min - a.observation_time_min for a, b in zip(center_events, center_events[1:])]
    return {
        "nominal_route_duration_min": nominal_route_duration,
        "observation_count": len(observations),
        "vessels_with_complete_gate_sequence": complete,
        "max_positive_request_to_entry_excess_min": max([x for x in request_entry_excess if x > 0.0], default=0.0),
        "mean_request_to_entry_excess_min": mean(request_entry_excess) if request_entry_excess else None,
        "max_center_cross_track_nm": max_cross_track,
        "max_center_heading_deviation_deg": max_heading_deviation,
        "max_center_speed_deviation_nm_per_min": max_speed_deviation,
        "center_passage_headway_min": center_headways,
        "gate_rows": gate_rows,
    }


@dataclass(frozen=True, slots=True)
class P4DScreeningCase:
    case_id: str
    room_nm: float
    demand_per_hour: float
    traffic_seed: int
    replication: int
    penetration: float
    contract: P4DBehaviorContract
    campaign_id: str = P4D_SCREENING_CAMPAIGN_ID

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.penetration < 0.0 or self.penetration > 1.0:
            raise ValueError("penetration must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class P4DScreeningResult:
    document: dict[str, object]
    canonical_bytes: bytes

    @property
    def result_sha256(self) -> str:
        return sha256(self.canonical_bytes).hexdigest()

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes


def run_p4d_screening_case(case: P4DScreeningCase) -> P4DScreeningResult:
    config = build_publication_config(
        room_nm=case.room_nm,
        demand_per_hour=case.demand_per_hour,
        traffic_seed=case.traffic_seed,
    )
    scenario = config.scenario
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    traffic_sha = traffic_input_sha256(traffic)
    stream_by_vessel = _stream_by_vessel(traffic)
    role_by_vessel, typed = _role_map(
        traffic,
        penetration=case.penetration,
        replication=case.replication,
        campaign_id=case.campaign_id,
    )
    profile_by_vessel, bindings = _role_profiles_and_bindings(traffic, role_by_vessel, case.contract)
    ingress = _custom_ingress(traffic, scenario, profile_by_vessel)
    reservoir = EnvironmentalNetworkSourceReservoir.from_scenario_ingress(ingress)

    profiles = VesselProfileCatalog()
    behavior_backend = P4DBoundedCommandResponseEnvironmentalMotionBackend(backend_for(first_order_profile()), bindings)
    world = P4DObservedMetricEnvironmentalWorld(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=scenario.start_time,
        profile_catalog=profiles,
        safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=behavior_backend,
        retain_safety_intervals=False,
        stream_by_vessel=stream_by_vessel,
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=profiles,
    )
    following = _following_supervisor_for_role_map(role_by_vessel, case.contract)
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
    runtime = started.runtime_start.session
    queue = source_queue_metrics(outcome.reservoir, at=outcome.terminal_time_min)
    interval_count = int(getattr(world, "safety_interval_count", 0))
    safety = world.finalize_safety_experiment_summary() if interval_count else None
    completions = _route_completion_records(runtime, outcome.reservoir, at=outcome.terminal_time_min)
    conflict = _conflict_summary(
        world.p4d_conflict_observations,
        outcome.reservoir.admissions,
        room_nm=case.room_nm,
    )
    center_service = center_pass_service_summary(
        world.p4d_conflict_observations,
        typed,
        reference_matrix=GUO_2024_ILLUSTRATIVE,
    )
    lifecycle = capture_terminal_lifecycle_audit(runtime, outcome.reservoir, completions)
    route_extra = [x["route_duration_min"] - conflict["nominal_route_duration_min"] for x in completions]
    interaction_active = bool(
        len(runtime.persistent.rebootstrap_results)
        or any(x > 1e-6 for x in route_extra)
        or conflict["max_positive_request_to_entry_excess_min"] > 1e-3
        or conflict["max_center_cross_track_nm"] > 1e-3
        or conflict["max_center_heading_deviation_deg"] > 1e-3
    )
    behavior_manifest = []
    for vessel_id in sorted(role_by_vessel):
        role = role_by_vessel[vessel_id]
        spec = case.contract.spec_for_role(role)
        behavior_manifest.append({
            "vessel_id": vessel_id,
            "role": role,
            "command_delay_min": spec.command_delay_min,
            "operational_domain_scale": spec.operational_domain_scale,
            "profile_id": profile_by_vessel[vessel_id].class_id,
            "delay_binding_sha256": bindings[vessel_id].identity_sha256,
            "following_reaction_interval_min": following.config_for_follower(vessel_id).reaction_interval_min,
        })

    document = {
        "format": "cross-r1-p4d-screening-result",
        "schema_version": 1,
        "evidence_role": P4D_SCREENING_EVIDENCE_ROLE,
        "scientific_confirmatory_interpretation_authorized": False,
        "case": {
            "case_id": case.case_id,
            "room_nm": case.room_nm,
            "demand_per_hour": case.demand_per_hour,
            "traffic_seed": case.traffic_seed,
            "replication": case.replication,
            "penetration": case.penetration,
            "campaign_id": case.campaign_id,
            "contract_id": case.contract.contract_id,
            "contract_sha256": case.contract.identity_sha256,
        },
        "runtime_semantics": {
            "command_response_semantics_id": P4D_COMMAND_RESPONSE_SEMANTICS_ID,
            "motion_backend_id": behavior_backend.backend_id,
            "base_same_route_reaction_interval_min": following.config.reaction_interval_min,
            "microscopic_command_delay_added_to_following_reaction_envelope": True,
            "reference_headway_forcing": False,
        },
        "traffic_input_sha256": traffic_sha,
        "traffic_count": traffic.count,
        "typed_roles": [asdict(x) for x in typed],
        "behavior_manifest": behavior_manifest,
        "terminal": {
            "status": outcome.status.value,
            "terminal_time_min": outcome.terminal_time_min,
            "infeasibility_type": outcome.infeasibility_type,
            "resource_unavailability_type": outcome.resource_unavailability_type,
        },
        "source": {
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
        },
        "lifecycle": lifecycle,
        "route": {
            "completion_count": len(completions),
            "records": list(completions),
            "mean_route_duration_min": mean([x["route_duration_min"] for x in completions]) if completions else None,
            "mean_route_extra_min": mean(route_extra) if route_extra else None,
            "max_route_extra_min": max(route_extra) if route_extra else None,
        },
        "conflict": conflict,
        "center_service": center_service,
        "response": {
            "rebootstrap_count": len(runtime.persistent.rebootstrap_results),
            "fleet_observation_count": len(runtime.coordinator.fleet_observations),
            "interaction_active": interaction_active,
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
    raw = _canonical_json(document)
    return P4DScreeningResult(document, raw)


def qualification_contracts() -> dict[str, P4DBehaviorContract]:
    """Control-only axis activation contracts.  Values are not confirmatory parameters."""
    p = control_only_provenance("P4D axis-activation qualification only; not a scientific parameter choice")
    neutral_as = P4DMicroscopicRoleSpec("AS", BehaviorProfile.AUTONOMOUS_POLICY, 0.0, 1.0, p, p)
    neutral_hs = P4DMicroscopicRoleSpec("HS", BehaviorProfile.COMPLIANT_MERCHANT, 0.0, 1.0, p, p)
    delay_hs = P4DMicroscopicRoleSpec("HS", BehaviorProfile.COMPLIANT_MERCHANT, 1.0, 1.0, p, p)
    domain_hs = P4DMicroscopicRoleSpec("HS", BehaviorProfile.COMPLIANT_MERCHANT, 0.0, 1.5, p, p)
    combined_hs = P4DMicroscopicRoleSpec("HS", BehaviorProfile.COMPLIANT_MERCHANT, 1.0, 1.5, p, p)
    return {
        "NEUTRAL": P4DBehaviorContract("P4D-Q-NEUTRAL", neutral_as, neutral_hs),
        "DELAY_ONLY": P4DBehaviorContract("P4D-Q-DELAY_ONLY", neutral_as, delay_hs),
        "DOMAIN_ONLY": P4DBehaviorContract("P4D-Q-DOMAIN_ONLY", neutral_as, domain_hs),
        "COMBINED": P4DBehaviorContract("P4D-Q-COMBINED", neutral_as, combined_hs),
    }
