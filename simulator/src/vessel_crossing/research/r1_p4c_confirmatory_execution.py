"""R1 P4C matched-mechanism confirmatory plan and guarded point runner.

DESIGN/PREFLIGHT BY DEFAULT.  Building or validating a plan never executes a scientific
point.  The module preserves the independent R1/P4B CROSS target and materializes only
the explicitly labelled matched-mechanism diagnostic control described by P4C.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
import os
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Mapping

from vessel_crossing import __version__
from vessel_crossing.nextgen import (
    EncounterRegistry,
    EnvironmentalNetworkSourceFutureIngressConfig,
    EnvironmentalNetworkSourceReservoir,
    VesselProfileCatalog,
    generate_poisson_network_traffic,
)
from vessel_crossing.nextgen.environmental_network_paired_experiment import _MetricEnvironmentalWorld
from vessel_crossing.nextgen.environmental_network_scenario import _environmental_ingress_records
from vessel_crossing.nextgen.environmental_network_source_reservoir import source_queue_metrics
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingSupervisor
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.research.r1_p4b_behavior_qualification import (
    traffic_input_sha256,
    typed_demand_for_penetration,
)
from vessel_crossing.research.r1_p4b_engineering_preflight import preflight_classification_config
from vessel_crossing.research.r1_p4b_publication_execution import (
    P4B_DEMANDS_PER_HOUR,
    P4B_EVALUATION_END_MIN,
    P4B_PENETRATIONS,
    P4B_PRIMARY_MEASUREMENT_END_MIN,
    P4B_ROOMS_NM,
    _plant_profiles,
    _route_completion_metrics,
    _source_window_metrics,
    build_publication_config,
)
from vessel_crossing.research.r1_p4b_publication_reference import (
    GUO_2024_ILLUSTRATIVE,
    publication_seed,
    run_pair_headway_reference,
)
from vessel_crossing.research.r1_p4c_matched_source_runtime import start_r1_p4c_matched_source_runtime
from vessel_crossing.research.r1_p4c_matched_source_service import (
    matched_source_service_throughput_per_hour,
    run_matched_micro_lite,
)
from vessel_crossing.research.r1_robustness_stage2 import backend_for

P4C_CAMPAIGN_ID = "R1_P4C_MATCHED_MECHANISM_CONFIRMATORY_V1_2026-09-20"
P4C_PROTOCOL_ID = "R1_P4C_MATCHED_MECHANISM_PROTOCOL_V1"
P4C_PLAN_FORMAT = "cross-r1-p4c-confirmatory-execution-plan"
P4C_PLAN_SCHEMA_VERSION = 1
P4C_POINT_FORMAT = "cross-r1-p4c-matched-full-cross-point"
P4C_POINT_SCHEMA_VERSION = 1
P4C_REPLICATION_COUNT = 24
P4C_MAX_PARALLEL_SCIENTIFIC_WORKERS = 2
P4C_P2_PENETRATIONS = (0.0, 0.50, 1.0)
P4C_P2_SENTINEL_ROOM_DEMAND = (
    (1.300, 8.0),
    (1.375, 6.0),
    (1.375, 8.0),
    (1.600, 6.0),
    (1.600, 8.0),
)
P4C_BEHAVIOR_TIMING = "NEUTRAL_BASE_PLANT_NO_P4B_BEHAVIOR_WRAPPER"
P4C_MATCHED_ARM_ROLE = "DIAGNOSTIC_MECHANISM_TRANSFER_CONTROL_NOT_INDEPENDENT_R1_TARGET"
P4C_EXECUTION_AUTHORIZATION_STATE = "EXTERNAL_AUTHORIZATION_FILE_REQUIRED_NOT_INCLUDED"
P4C_CHECKPOINT_FORMAT = "cross-r1-p4c-confirmatory-checkpoint"
P4C_AUTH_FORMAT = "cross-r1-p4c-scientific-execution-authorization"
P4C_AUTH_SCHEMA_VERSION = 1
_EPS = 1e-10


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _valid_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _room_tag(room: float) -> str:
    return f"R{int(round(room * 1000)):04d}"


def _demand_tag(demand: float) -> str:
    return f"D{int(round(demand)):02d}"


def _penetration_tag(p: float) -> str:
    return f"P{int(round(p * 100)):03d}"


@dataclass(frozen=True, slots=True)
class P4CSeedRecord:
    replication: int
    traffic_arrivals_seed: int
    type_assignment_seed: int
    behavior_stochasticity_seed: int


@dataclass(frozen=True, slots=True)
class P4CInputBasis:
    replication: int
    demand_per_hour: float
    traffic_input_sha256: str
    traffic_seed: int
    type_assignment_seed: int
    behavior_stochasticity_seed: int
    event_count: int
    events: tuple[dict[str, object], ...]
    layer1_by_penetration: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class P4CPhysicalPoint:
    execution_order: int
    point_id: str
    replication: int
    plant_id: str
    plant_sha256: str
    room_nm: float
    demand_per_hour: float
    penetration: float
    traffic_seed: int
    type_assignment_seed: int
    behavior_stochasticity_seed: int
    expected_traffic_input_sha256: str
    config_sha256: str
    expected_motion_backend_id: str
    source_archive_sha256: str
    identity_sha256: str = field(default="")

    def identity_payload(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("identity_sha256", None)
        return value

    def compute_identity_sha256(self) -> str:
        return _sha_bytes(_canonical_json(self.identity_payload()))

    def __post_init__(self) -> None:
        if self.execution_order < 1:
            raise ValueError("execution_order must be positive")
        if self.room_nm not in P4B_ROOMS_NM or self.demand_per_hour not in P4B_DEMANDS_PER_HOUR:
            raise ValueError("point is outside frozen P4C room/demand surface")
        if self.penetration not in P4B_PENETRATIONS:
            raise ValueError("point penetration outside frozen P4C surface")
        if not _valid_sha256(self.source_archive_sha256):
            raise ValueError("source_archive_sha256 must be lowercase SHA-256")
        if self.identity_sha256 and self.identity_sha256 != self.compute_identity_sha256():
            raise ValueError("physical point identity mismatch")


@dataclass(frozen=True, slots=True)
class P4CConceptualRow:
    row_id: str
    replication: int
    plant_id: str
    room_nm: float
    demand_per_hour: float
    penetration: float
    physical_point_id: str
    input_basis_key: str


@dataclass(frozen=True, slots=True)
class P4CExecutionPlan:
    campaign_id: str
    protocol_id: str
    source_archive_sha256: str
    confirmatory_protocol_sha256: str
    statistical_plan_sha256: str
    seed_schedule_sha256: str
    m1_reference_sha256: str
    matched_mechanism_sha256: str
    replication_count: int
    max_parallel_scientific_workers: int
    scientific_execution_authorization: str
    behavior_timing: str
    matched_arm_role: str
    seed_schedule: tuple[P4CSeedRecord, ...]
    input_bases: tuple[P4CInputBasis, ...]
    physical_points: tuple[P4CPhysicalPoint, ...]
    conceptual_rows: tuple[P4CConceptualRow, ...]
    format: str = P4C_PLAN_FORMAT
    schema_version: int = P4C_PLAN_SCHEMA_VERSION

    def to_json_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def identity_sha256(self) -> str:
        return _sha_bytes(self.to_json_bytes())


def _seed_record(replication: int, campaign_id: str) -> P4CSeedRecord:
    return P4CSeedRecord(
        replication=replication,
        traffic_arrivals_seed=publication_seed(campaign_id=campaign_id, replication=replication, namespace="TRAFFIC_ARRIVALS"),
        type_assignment_seed=publication_seed(campaign_id=campaign_id, replication=replication, namespace="TYPE_ASSIGNMENT"),
        behavior_stochasticity_seed=publication_seed(campaign_id=campaign_id, replication=replication, namespace="BEHAVIOR_STOCHASTICITY"),
    )


def _basis_key(replication: int, demand: float) -> str:
    return f"R{replication:02d}-{_demand_tag(demand)}"


def _layer1_parity_rows(typed, *, horizon_end_min: float) -> tuple[object, object]:
    macro = run_pair_headway_reference(typed, matrix=GUO_2024_ILLUSTRATIVE, horizon_end_min=horizon_end_min)
    micro = run_matched_micro_lite(typed, matrix=GUO_2024_ILLUSTRATIVE, horizon_end_min=horizon_end_min)
    if len(macro.records) != len(micro.records):
        raise RuntimeError("M1/Micro-lite record-count parity failure")
    for m, q in zip(macro.records, micro.records):
        if (
            m.event_id != q.event_id
            or m.stream_id != q.stream_id
            or m.vessel_type != q.vessel_type
            or m.arrival_min != q.arrival_min
            or m.departure_min != q.release_min
            or m.waiting_min != q.waiting_min
            or m.predecessor_type != q.predecessor_type
            or m.required_headway_min != q.required_headway_min
        ):
            raise RuntimeError("M1/Micro-lite record-level parity failure")
    return macro, micro


def _materialize_input_basis(*, campaign_id: str, replication: int, demand_per_hour: float) -> P4CInputBasis:
    seeds = _seed_record(replication, campaign_id)
    config = build_publication_config(
        room_nm=P4B_ROOMS_NM[0], demand_per_hour=demand_per_hour, traffic_seed=seeds.traffic_arrivals_seed
    )
    traffic = generate_poisson_network_traffic(
        config.network, config.streams, horizon=config.scenario.traffic_horizon_min,
        seed=config.scenario.seed, start_time=config.scenario.start_time,
    )
    typed0 = typed_demand_for_penetration(
        traffic, penetration=0.0, campaign_id=campaign_id, replication=replication
    )
    events = tuple({
        "event_id": e.event_id, "arrival_min": e.arrival_min, "stream_id": e.stream_id,
        "type_uniform": e.type_uniform,
    } for e in typed0)
    rows=[]
    for penetration in P4B_PENETRATIONS:
        typed = typed_demand_for_penetration(
            traffic, penetration=penetration, campaign_id=campaign_id, replication=replication
        )
        macro90, micro90 = _layer1_parity_rows(typed, horizon_end_min=P4B_PRIMARY_MEASUREMENT_END_MIN)
        macro150, micro150 = _layer1_parity_rows(typed, horizon_end_min=P4B_EVALUATION_END_MIN)
        micro_t90 = matched_source_service_throughput_per_hour(
            micro90.records, measurement_start_min=0.0, horizon_end_min=P4B_PRIMARY_MEASUREMENT_END_MIN
        )
        if micro_t90 != macro90.throughput_per_hour:
            raise RuntimeError("M1/Micro-lite 90-min throughput parity failure")
        rows.append({
            "penetration": penetration,
            "typed_event_sha256": _sha_bytes(_canonical_json([asdict(x) for x in typed])),
            "primary_completed_count": macro90.completed_count,
            "primary_backlog_count": macro90.backlog_count,
            "primary_throughput_per_hour": macro90.throughput_per_hour,
            "primary_mean_waiting_min": macro90.mean_waiting_min,
            "evaluation_completed_count": macro150.completed_count,
            "evaluation_backlog_count": macro150.backlog_count,
            "evaluation_throughput_per_hour": macro150.throughput_per_hour,
            "evaluation_mean_waiting_min": macro150.mean_waiting_min,
            "m1_reference_sha256": macro90.reference_sha256,
            "micro_lite_record_sha256": _sha_bytes(_canonical_json([asdict(x) for x in micro90.records])),
            "m1_micro_lite_exact_parity": True,
        })
    return P4CInputBasis(
        replication=replication,
        demand_per_hour=demand_per_hour,
        traffic_input_sha256=traffic_input_sha256(traffic),
        traffic_seed=seeds.traffic_arrivals_seed,
        type_assignment_seed=seeds.type_assignment_seed,
        behavior_stochasticity_seed=seeds.behavior_stochasticity_seed,
        event_count=traffic.count,
        events=events,
        layer1_by_penetration=tuple(rows),
    )


def _physical_point(*, execution_order: int, source_archive_sha256: str, replication: int,
                    plant_id: str, room_nm: float, demand_per_hour: float, penetration: float,
                    basis: P4CInputBasis) -> P4CPhysicalPoint:
    profile=_plant_profiles()[plant_id]
    config=build_publication_config(room_nm=room_nm,demand_per_hour=demand_per_hour,traffic_seed=basis.traffic_seed)
    traffic=generate_poisson_network_traffic(
        config.network,config.streams,horizon=config.scenario.traffic_horizon_min,
        seed=config.scenario.seed,start_time=config.scenario.start_time,
    )
    if traffic_input_sha256(traffic) != basis.traffic_input_sha256:
        raise RuntimeError("room changed exogenous traffic identity")
    backend=backend_for(profile)
    plant_tag="P1" if plant_id.startswith("P1_") else "P2"
    point_id=(f"R1P4C-R{replication:02d}-{plant_tag}-{_room_tag(room_nm)}-"
              f"{_demand_tag(demand_per_hour)}-{_penetration_tag(penetration)}")
    raw=P4CPhysicalPoint(
        execution_order=execution_order,point_id=point_id,replication=replication,
        plant_id=plant_id,plant_sha256=profile.identity_sha256,room_nm=room_nm,
        demand_per_hour=demand_per_hour,penetration=penetration,
        traffic_seed=basis.traffic_seed,type_assignment_seed=basis.type_assignment_seed,
        behavior_stochasticity_seed=basis.behavior_stochasticity_seed,
        expected_traffic_input_sha256=basis.traffic_input_sha256,config_sha256=config.config_sha256,
        expected_motion_backend_id=backend.backend_id,source_archive_sha256=source_archive_sha256,
    )
    return replace(raw,identity_sha256=raw.compute_identity_sha256())


def build_p4c_execution_plan(*, source_archive_sha256: str, confirmatory_protocol_sha256: str,
                             statistical_plan_sha256: str, seed_schedule_sha256: str,
                             matched_mechanism_sha256: str,
                             campaign_id: str=P4C_CAMPAIGN_ID) -> P4CExecutionPlan:
    for name,value in (
        ("source_archive_sha256",source_archive_sha256),
        ("confirmatory_protocol_sha256",confirmatory_protocol_sha256),
        ("statistical_plan_sha256",statistical_plan_sha256),
        ("seed_schedule_sha256",seed_schedule_sha256),
        ("matched_mechanism_sha256",matched_mechanism_sha256),
    ):
        if not _valid_sha256(value): raise ValueError(f"{name} must be lowercase SHA-256")
    seeds=tuple(_seed_record(r,campaign_id) for r in range(1,P4C_REPLICATION_COUNT+1))
    bases=tuple(_materialize_input_basis(campaign_id=campaign_id,replication=r,demand_per_hour=d)
                for r in range(1,P4C_REPLICATION_COUNT+1) for d in P4B_DEMANDS_PER_HOUR)
    bm={(b.replication,b.demand_per_hour):b for b in bases}
    p1=_plant_profiles()["P1_FIRST_ORDER_RESPONSE_SYNTHETIC"].plant_id
    p2=_plant_profiles()["P2_TABULATED_STEERAGE_SYNTHETIC"].plant_id
    physical=[]; conceptual=[]; order=1
    for r in range(1,P4C_REPLICATION_COUNT+1):
        for room in P4B_ROOMS_NM:
            for demand in P4B_DEMANDS_PER_HOUR:
                basis=bm[(r,demand)]
                for p in P4B_PENETRATIONS:
                    point=_physical_point(execution_order=order,source_archive_sha256=source_archive_sha256,
                        replication=r,plant_id=p1,room_nm=room,demand_per_hour=demand,penetration=p,basis=basis)
                    physical.append(point); order+=1
                    conceptual.append(P4CConceptualRow(
                        row_id=f"C-R{r:02d}-P1-{_room_tag(room)}-{_demand_tag(demand)}-{_penetration_tag(p)}",
                        replication=r,plant_id=p1,room_nm=room,demand_per_hour=demand,penetration=p,
                        physical_point_id=point.point_id,input_basis_key=_basis_key(r,demand)))
        for room,demand in P4C_P2_SENTINEL_ROOM_DEMAND:
            basis=bm[(r,demand)]
            for p in P4C_P2_PENETRATIONS:
                point=_physical_point(execution_order=order,source_archive_sha256=source_archive_sha256,
                    replication=r,plant_id=p2,room_nm=room,demand_per_hour=demand,penetration=p,basis=basis)
                physical.append(point); order+=1
                conceptual.append(P4CConceptualRow(
                    row_id=f"C-R{r:02d}-P2-{_room_tag(room)}-{_demand_tag(demand)}-{_penetration_tag(p)}",
                    replication=r,plant_id=p2,room_nm=room,demand_per_hour=demand,penetration=p,
                    physical_point_id=point.point_id,input_basis_key=_basis_key(r,demand)))
    return P4CExecutionPlan(
        campaign_id=campaign_id,protocol_id=P4C_PROTOCOL_ID,source_archive_sha256=source_archive_sha256,
        confirmatory_protocol_sha256=confirmatory_protocol_sha256,statistical_plan_sha256=statistical_plan_sha256,
        seed_schedule_sha256=seed_schedule_sha256,m1_reference_sha256=GUO_2024_ILLUSTRATIVE.identity_sha256,
        matched_mechanism_sha256=matched_mechanism_sha256,replication_count=P4C_REPLICATION_COUNT,
        max_parallel_scientific_workers=P4C_MAX_PARALLEL_SCIENTIFIC_WORKERS,
        scientific_execution_authorization=P4C_EXECUTION_AUTHORIZATION_STATE,
        behavior_timing=P4C_BEHAVIOR_TIMING,matched_arm_role=P4C_MATCHED_ARM_ROLE,
        seed_schedule=seeds,input_bases=bases,physical_points=tuple(physical),conceptual_rows=tuple(conceptual))


def validate_p4c_execution_plan(plan: P4CExecutionPlan) -> dict[str, object]:
    if plan.campaign_id != P4C_CAMPAIGN_ID: raise ValueError("unexpected P4C campaign id")
    if plan.replication_count != 24 or plan.max_parallel_scientific_workers != 2:
        raise ValueError("P4C replication/worker freeze drift")
    if len(plan.seed_schedule)!=24 or len(plan.input_bases)!=72:
        raise ValueError("P4C seed/input basis count drift")
    if len(plan.physical_points)!=1440 or len(plan.conceptual_rows)!=1440:
        raise ValueError("P4C point count drift")
    ids=[p.point_id for p in plan.physical_points]
    if len(set(ids))!=len(ids): raise ValueError("duplicate P4C point ids")
    if [p.execution_order for p in plan.physical_points] != list(range(1,1441)):
        raise ValueError("P4C execution order is not contiguous")
    if any(p.identity_sha256 != p.compute_identity_sha256() for p in plan.physical_points):
        raise ValueError("P4C point identity drift")
    p1=sum(p.plant_id.startswith("P1_") for p in plan.physical_points)
    p2=sum(p.plant_id.startswith("P2_") for p in plan.physical_points)
    if (p1,p2)!=(1080,360): raise ValueError("P4C P1/P2 count drift")
    return {
        "campaign_id":plan.campaign_id,"plan_sha256":plan.identity_sha256,
        "seed_records":len(plan.seed_schedule),"input_bases":len(plan.input_bases),
        "physical_points":len(plan.physical_points),"p1_points":p1,"p2_points":p2,
        "conceptual_rows":len(plan.conceptual_rows),"max_parallel_scientific_workers":2,
        "m1_micro_lite_parity_all_bases":all(
            row["m1_micro_lite_exact_parity"] for b in plan.input_bases for row in b.layer1_by_penetration
        ),
    }


def _basis_for(plan: P4CExecutionPlan, point: P4CPhysicalPoint) -> P4CInputBasis:
    matches=[b for b in plan.input_bases if b.replication==point.replication and b.demand_per_hour==point.demand_per_hour]
    if len(matches)!=1: raise RuntimeError("point input basis is not unique")
    return matches[0]


def run_p4c_physical_point(plan: P4CExecutionPlan, point: P4CPhysicalPoint) -> bytes:
    """Execute exactly one already-frozen matched Full CROSS point.

    Calling this function is scientific execution.  Higher-level wrappers must verify an
    external authorization document before invoking it for a confirmatory campaign.
    """
    if point.source_archive_sha256 != plan.source_archive_sha256:
        raise RuntimeError("point/source archive identity mismatch")
    basis=_basis_for(plan,point)
    config=build_publication_config(room_nm=point.room_nm,demand_per_hour=point.demand_per_hour,traffic_seed=point.traffic_seed)
    if config.config_sha256 != point.config_sha256: raise RuntimeError("P4C config identity drift")
    traffic=generate_poisson_network_traffic(config.network,config.streams,horizon=config.scenario.traffic_horizon_min,
                                              seed=config.scenario.seed,start_time=config.scenario.start_time)
    traffic_sha=traffic_input_sha256(traffic)
    if traffic_sha != point.expected_traffic_input_sha256: raise RuntimeError("P4C traffic identity drift")
    typed=typed_demand_for_penetration(traffic,penetration=point.penetration,campaign_id=plan.campaign_id,replication=point.replication)
    profile=_plant_profiles()[point.plant_id]; backend=backend_for(profile)
    if backend.backend_id != point.expected_motion_backend_id: raise RuntimeError("P4C plant backend identity drift")
    scenario=config.scenario
    reservoir=EnvironmentalNetworkSourceReservoir.from_scenario_ingress(_environmental_ingress_records(traffic,scenario))
    profiles=VesselProfileCatalog()
    world=_MetricEnvironmentalWorld((),limits={},environmental_drift_velocity={},start_time=scenario.start_time,
        profile_catalog=profiles,safety_config=stage1.stage1_safety_metric_config(),motion_backend=backend,retain_safety_intervals=False)
    registry=EncounterRegistry(discovery=scenario.discovery,lifecycle=scenario.lifecycle,profile_catalog=profiles)
    following=SameRouteFollowingSupervisor()
    pair=scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    started=start_r1_p4c_matched_source_runtime(registry,world,{},reservoir,typed,
        recapture_config=scenario.recapture,runtime_config=scenario.runtime,recheck_config=config.recheck,
        classification_config=preflight_classification_config(),profile_catalog=profiles,
        same_route_dynamic_ingress=following,
        future_ingress_clearance=EnvironmentalNetworkSourceFutureIngressConfig(pair_clearance=pair),
        persistent_same_route_following=following)
    outcome=started.session.run_until_capturing_terminal_outcome(scenario.end_time)
    runtime=started.runtime_start.session
    gate=started.session._extension.gate
    interval_count=int(getattr(started.session.world,"safety_interval_count",0))
    safety=started.session.world.finalize_safety_experiment_summary() if interval_count else None
    queue=source_queue_metrics(outcome.reservoir,at=outcome.terminal_time_min)
    success=outcome.status.value=="SUCCESS" and abs(outcome.terminal_time_min-scenario.end_time)<=_EPS
    performance=None
    if success:
        s90=_source_window_metrics(outcome.reservoir,at=P4B_PRIMARY_MEASUREMENT_END_MIN)
        s150=_source_window_metrics(outcome.reservoir,at=P4B_EVALUATION_END_MIN)
        c90=_route_completion_metrics(runtime,outcome.reservoir,at=P4B_PRIMARY_MEASUREMENT_END_MIN)
        c150=_route_completion_metrics(runtime,outcome.reservoir,at=P4B_EVALUATION_END_MIN)
        delays=[float(r.release_min-r.headway_ready_min) for r in gate.records]
        performance={
            "primary_source_service_throughput_per_hour_90":int(s90["admission_count"])/1.5,
            "primary_source_service_release_count_90":s90["admission_count"],
            "primary_source_backlog_count_90":s90["backlog_count"],
            "primary_mean_admitted_source_wait_min_90":s90["mean_admitted_source_wait_min"],
            "primary_route_completed_count_90":c90["route_completed_count"],
            "evaluation_source_service_release_count_150":s150["admission_count"],
            "evaluation_source_backlog_count_150":s150["backlog_count"],
            "evaluation_route_completed_count_150":c150["route_completed_count"],
            "evaluation_mean_route_duration_min_150":c150["mean_route_duration_min"],
            "evaluation_mean_system_duration_min_150":c150["mean_system_duration_min"],
            "matched_physical_extra_delay_positive_count":sum(x>_EPS for x in delays),
            "matched_physical_extra_delay_mean_min":mean(delays) if delays else 0.0,
            "matched_physical_extra_delay_max_min":max(delays,default=0.0),
        }
    document={
        "format":P4C_POINT_FORMAT,"schema_version":P4C_POINT_SCHEMA_VERSION,
        "campaign_id":plan.campaign_id,"protocol_id":plan.protocol_id,"plan_sha256":plan.identity_sha256,
        "source_archive_sha256":plan.source_archive_sha256,"software_version":__version__,
        "matched_arm_role":P4C_MATCHED_ARM_ROLE,"behavior_timing":P4C_BEHAVIOR_TIMING,
        "point":asdict(point),"config_sha256":config.config_sha256,"traffic_input_sha256":traffic_sha,
        "typed_execution_sha256":_sha_bytes(_canonical_json([asdict(x) for x in typed])),
        "motion_backend_id":backend.backend_id,
        "matched_gate_records_sha256":_sha_bytes(_canonical_json([asdict(x) for x in gate.records])),
        "matched_gate_records":[asdict(x) for x in gate.records],
        "terminal":{"status":outcome.status.value,"terminal_time_min":outcome.terminal_time_min,
                    "infeasibility_type":outcome.infeasibility_type,
                    "resource_unavailability_type":outcome.resource_unavailability_type},
        "performance":performance,
        "source_diagnostics":{"demand_count":queue.demand_count,"admission_count":queue.admission_count,
                              "pending_count":queue.pending_count,"due_pending_count":queue.due_pending_count,
                              "mean_admitted_waiting_time_min":queue.mean_admitted_waiting_time_min,
                              "max_admitted_waiting_time_min":queue.max_admitted_waiting_time_min},
        "response_diagnostics":{"rebootstrap_count":len(runtime.persistent.rebootstrap_results),
                                "fleet_observation_count":len(runtime.coordinator.fleet_observations)},
        "safety":None if safety is None else {
            "modeled_hull_overlap_episode_count":safety.modeled_hull_overlap_episode_count,
            "safety_domain_intrusion_episode_count":safety.safety_domain_intrusion_episode_count,
            "near_miss_episode_count":safety.near_miss_episode_count,
            "unresolved_episode_count":safety.unresolved_episode_count,
            "sampled_min_center_separation_nm":safety.interval_summary.sampled_min_center_separation_nm,
            "sampled_min_hull_margin_nm":safety.interval_summary.sampled_min_hull_margin_nm,
            "sampled_min_safety_domain_margin_nm":safety.interval_summary.sampled_min_safety_domain_margin_nm,
            "worst_observed_evidence":safety.worst_observed_evidence.value,
        },
    }
    return _canonical_json(document)



def result_file_name(point: P4CPhysicalPoint) -> str:
    return f"{point.execution_order:04d}_{point.point_id}.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_result_atomically(results_dir: Path, point: P4CPhysicalPoint, payload: bytes) -> tuple[Path, str]:
    path = results_dir / result_file_name(point)
    digest = _sha_bytes(payload)
    _atomic_write(path, payload)
    _atomic_write(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode())
    return path, digest


def initial_checkpoint(plan: P4CExecutionPlan) -> dict[str, object]:
    return {
        "format": P4C_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.identity_sha256,
        "source_archive_sha256": plan.source_archive_sha256,
        "completed": {},
        "last_durable_execution_order": 0,
        "status": "NOT_STARTED",
    }


def write_checkpoint_atomic(path: Path, checkpoint: Mapping[str, object]) -> str:
    payload = _canonical_json(checkpoint)
    _atomic_write(path, payload)
    digest = _sha_bytes(payload)
    _atomic_write(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode())
    return digest


def load_checkpoint(path: Path, plan: P4CExecutionPlan) -> dict[str, object]:
    if not path.exists():
        return initial_checkpoint(plan)
    payload = path.read_bytes()
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.exists():
        raise RuntimeError("P4C checkpoint sidecar missing")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if _sha_bytes(payload) != expected:
        raise RuntimeError("P4C checkpoint SHA mismatch")
    obj = json.loads(payload)
    if obj.get("format") != P4C_CHECKPOINT_FORMAT:
        raise RuntimeError("unexpected P4C checkpoint format")
    if obj.get("campaign_id") != plan.campaign_id or obj.get("plan_sha256") != plan.identity_sha256:
        raise RuntimeError("P4C checkpoint plan identity mismatch")
    if obj.get("source_archive_sha256") != plan.source_archive_sha256:
        raise RuntimeError("P4C checkpoint source identity mismatch")
    completed = obj.get("completed")
    if not isinstance(completed, dict):
        raise RuntimeError("P4C checkpoint completed map malformed")
    known = {p.point_id for p in plan.physical_points}
    if not set(completed).issubset(known):
        raise RuntimeError("P4C checkpoint contains unknown point ids")
    return obj


def verify_execution_authorization(authorization: Mapping[str, object], plan: P4CExecutionPlan) -> None:
    if authorization.get("format") != P4C_AUTH_FORMAT:
        raise RuntimeError("unexpected P4C authorization format")
    if authorization.get("schema_version") != P4C_AUTH_SCHEMA_VERSION:
        raise RuntimeError("unexpected P4C authorization schema")
    if authorization.get("campaign_id") != plan.campaign_id:
        raise RuntimeError("P4C authorization campaign mismatch")
    if authorization.get("plan_sha256") != plan.identity_sha256:
        raise RuntimeError("P4C authorization plan mismatch")
    if authorization.get("source_archive_sha256") != plan.source_archive_sha256:
        raise RuntimeError("P4C authorization source mismatch")
    if authorization.get("authorization") != "AUTHORIZED_TO_EXECUTE_FROZEN_P4C_CONFIRMATORY_PLAN":
        raise RuntimeError("P4C scientific execution is not explicitly authorized")
    if authorization.get("max_parallel_scientific_workers") != P4C_MAX_PARALLEL_SCIENTIFIC_WORKERS:
        raise RuntimeError("P4C authorization worker ceiling mismatch")


def plan_from_json_bytes(payload: bytes) -> P4CExecutionPlan:
    obj = json.loads(payload)
    if obj.get("format") != P4C_PLAN_FORMAT or obj.get("schema_version") != P4C_PLAN_SCHEMA_VERSION:
        raise ValueError("unexpected P4C plan format/schema")
    plan = P4CExecutionPlan(
        campaign_id=obj["campaign_id"],
        protocol_id=obj["protocol_id"],
        source_archive_sha256=obj["source_archive_sha256"],
        confirmatory_protocol_sha256=obj["confirmatory_protocol_sha256"],
        statistical_plan_sha256=obj["statistical_plan_sha256"],
        seed_schedule_sha256=obj["seed_schedule_sha256"],
        m1_reference_sha256=obj["m1_reference_sha256"],
        matched_mechanism_sha256=obj["matched_mechanism_sha256"],
        replication_count=int(obj["replication_count"]),
        max_parallel_scientific_workers=int(obj["max_parallel_scientific_workers"]),
        scientific_execution_authorization=obj["scientific_execution_authorization"],
        behavior_timing=obj["behavior_timing"],
        matched_arm_role=obj["matched_arm_role"],
        seed_schedule=tuple(P4CSeedRecord(**x) for x in obj["seed_schedule"]),
        input_bases=tuple(P4CInputBasis(
            replication=x["replication"], demand_per_hour=x["demand_per_hour"],
            traffic_input_sha256=x["traffic_input_sha256"], traffic_seed=x["traffic_seed"],
            type_assignment_seed=x["type_assignment_seed"], behavior_stochasticity_seed=x["behavior_stochasticity_seed"],
            event_count=x["event_count"], events=tuple(x["events"]), layer1_by_penetration=tuple(x["layer1_by_penetration"]),
        ) for x in obj["input_bases"]),
        physical_points=tuple(P4CPhysicalPoint(
            execution_order=x["execution_order"], point_id=x["point_id"], replication=x["replication"],
            plant_id=x["plant_id"], plant_sha256=x["plant_sha256"], room_nm=x["room_nm"],
            demand_per_hour=x["demand_per_hour"], penetration=x["penetration"], traffic_seed=x["traffic_seed"],
            type_assignment_seed=x["type_assignment_seed"], behavior_stochasticity_seed=x["behavior_stochasticity_seed"],
            expected_traffic_input_sha256=x["expected_traffic_input_sha256"], config_sha256=x["config_sha256"],
            expected_motion_backend_id=x["expected_motion_backend_id"], source_archive_sha256=x["source_archive_sha256"],
            identity_sha256=x["identity_sha256"],
        ) for x in obj["physical_points"]),
        conceptual_rows=tuple(P4CConceptualRow(**x) for x in obj["conceptual_rows"]),
        format=obj["format"], schema_version=int(obj["schema_version"]),
    )
    validate_p4c_execution_plan(plan)
    return plan


def summarize_plan(plan: P4CExecutionPlan) -> dict[str, object]:
    validated = validate_p4c_execution_plan(plan)
    return {
        **validated,
        "protocol_id": plan.protocol_id,
        "source_archive_sha256": plan.source_archive_sha256,
        "behavior_timing": plan.behavior_timing,
        "matched_arm_role": plan.matched_arm_role,
        "scientific_execution_authorization": plan.scientific_execution_authorization,
    }
