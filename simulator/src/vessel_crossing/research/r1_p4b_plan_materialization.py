"""R1 P4B immutable plan/input materialization.

Refactor-only extraction from the historical monolithic publication execution module.
Building and validating a plan never executes a scientific CROSS point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json

from vessel_crossing.nextgen import generate_poisson_network_traffic
from vessel_crossing.research.r1_p4b_behavior_adapter import P4BBehaviorEnvironmentalMotionBackend
from vessel_crossing.research.r1_p4b_behavior_grounding import (
    P4B_BEHAVIOR_TIMING_ENDPOINTS,
    T0,
    T60,
    publication_bindings_for_endpoint,
)
from vessel_crossing.research.r1_p4b_behavior_qualification import (
    bindings_for_penetration,
    endpoint_bindings,
    traffic_input_sha256,
    typed_demand_for_penetration,
)
from vessel_crossing.research.r1_p4b_experiment_spec import (
    P4B_CAMPAIGN_ID,
    P4B_DEMANDS_PER_HOUR,
    P4B_EVALUATION_END_MIN,
    P4B_EXECUTION_AUTHORIZATION_STATE,
    P4B_MAX_PARALLEL_SCIENTIFIC_WORKERS,
    P4B_P2_PENETRATIONS,
    P4B_P2_SENTINEL_ROOM_DEMAND,
    P4B_PENETRATIONS,
    P4B_PLAN_FORMAT,
    P4B_PLAN_SCHEMA_VERSION,
    P4B_PRIMARY_MEASUREMENT_END_MIN,
    P4B_PROTOCOL_ID,
    P4B_REPLICATION_COUNT,
    P4B_ROOMS_NM,
    _canonical_json,
    _demand_tag,
    _penetration_tag,
    _plant_profiles,
    _room_tag,
    _sha_bytes,
    _timing_endpoint,
    _valid_sha256,
    build_publication_config,
)
from vessel_crossing.research.r1_p4b_publication_reference import (
    GUO_2024_ILLUSTRATIVE,
    run_pair_headway_reference,
)
from vessel_crossing.research.r1_p4b_seed_schedule import P4BSeedRecord, _seed_record
from vessel_crossing.research.r1_robustness_stage2 import (
    backend_for,
    first_order_profile,
    tabulated_steerage_profile,
)

@dataclass(frozen=True, slots=True)
class P4BInputBasis:
    replication: int
    demand_per_hour: float
    traffic_input_sha256: str
    traffic_seed: int
    type_assignment_seed: int
    behavior_stochasticity_seed: int
    event_count: int
    events: tuple[dict[str, object], ...]
    macro_by_penetration: tuple[dict[str, object], ...]

@dataclass(frozen=True, slots=True)
class P4BPhysicalPoint:
    execution_order: int
    point_id: str
    replication: int
    plant_id: str
    plant_sha256: str
    room_nm: float
    demand_per_hour: float
    timing_endpoint_id: str
    penetration_execution: float | None
    deduplicated_penetrations: tuple[float, ...]
    traffic_seed: int
    type_assignment_seed: int
    behavior_stochasticity_seed: int
    expected_traffic_input_sha256: str
    config_sha256: str
    expected_behavior_backend_id: str
    source_archive_sha256: str
    identity_sha256: str = field(default="")

    def __post_init__(self) -> None:
        if self.execution_order < 1:
            raise ValueError("execution_order must be positive")
        if not self.point_id:
            raise ValueError("point_id must be non-empty")
        if not _valid_sha256(self.source_archive_sha256):
            raise ValueError("source_archive_sha256 must be lowercase SHA-256")
        if self.identity_sha256:
            expected = self.compute_identity_sha256()
            if self.identity_sha256 != expected:
                raise ValueError("physical point identity_sha256 mismatch")

    def identity_payload(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("identity_sha256", None)
        return value

    def compute_identity_sha256(self) -> str:
        return _sha_bytes(_canonical_json(self.identity_payload()))

@dataclass(frozen=True, slots=True)
class P4BConceptualRow:
    row_id: str
    replication: int
    plant_id: str
    room_nm: float
    demand_per_hour: float
    timing_endpoint_id: str
    penetration: float
    physical_point_id: str
    input_basis_key: str

@dataclass(frozen=True, slots=True)
class P4BPublicationExecutionPlan:
    campaign_id: str
    protocol_id: str
    source_archive_sha256: str
    b5_plan_sha256: str
    b5_measurement_addendum_sha256: str
    b6_audit_sha256: str
    m1_reference_sha256: str
    replication_count: int
    max_parallel_scientific_workers: int
    scientific_execution_authorization: str
    seed_schedule: tuple[P4BSeedRecord, ...]
    input_bases: tuple[P4BInputBasis, ...]
    physical_points: tuple[P4BPhysicalPoint, ...]
    conceptual_rows: tuple[P4BConceptualRow, ...]
    format: str = P4B_PLAN_FORMAT
    schema_version: int = P4B_PLAN_SCHEMA_VERSION

    def to_json_object(self) -> dict[str, object]:
        return asdict(self)

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def identity_sha256(self) -> str:
        return _sha_bytes(self.to_json_bytes())

def _basis_key(replication: int, demand: float) -> str:
    return f"R{replication:02d}-{_demand_tag(demand)}"

def _materialize_input_basis(*, campaign_id: str, replication: int, demand_per_hour: float) -> P4BInputBasis:
    seeds = _seed_record(replication, campaign_id)
    config = build_publication_config(
        room_nm=P4B_ROOMS_NM[0],
        demand_per_hour=demand_per_hour,
        traffic_seed=seeds.traffic_arrivals_seed,
    )
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=config.scenario.traffic_horizon_min,
        seed=config.scenario.seed,
        start_time=config.scenario.start_time,
    )
    typed0 = typed_demand_for_penetration(
        traffic, penetration=0.0, campaign_id=campaign_id, replication=replication
    )
    events = tuple(
        {
            "event_id": event.event_id,
            "arrival_min": event.arrival_min,
            "stream_id": event.stream_id,
            "type_uniform": event.type_uniform,
        }
        for event in typed0
    )
    macro_rows: list[dict[str, object]] = []
    for penetration in P4B_PENETRATIONS:
        typed = typed_demand_for_penetration(
            traffic,
            penetration=penetration,
            campaign_id=campaign_id,
            replication=replication,
        )
        macro = run_pair_headway_reference(
            typed,
            matrix=GUO_2024_ILLUSTRATIVE,
            horizon_end_min=P4B_PRIMARY_MEASUREMENT_END_MIN,
        )
        macro_evaluation = run_pair_headway_reference(
            typed,
            matrix=GUO_2024_ILLUSTRATIVE,
            horizon_end_min=P4B_EVALUATION_END_MIN,
        )
        macro_rows.append(
            {
                "penetration": penetration,
                "primary_horizon_end_min": P4B_PRIMARY_MEASUREMENT_END_MIN,
                "primary_completed_count": macro.completed_count,
                "primary_backlog_count": macro.backlog_count,
                "primary_throughput_per_hour": macro.throughput_per_hour,
                "primary_mean_waiting_min": macro.mean_waiting_min,
                "evaluation_horizon_end_min": P4B_EVALUATION_END_MIN,
                "evaluation_completed_count": macro_evaluation.completed_count,
                "evaluation_backlog_count": macro_evaluation.backlog_count,
                "evaluation_throughput_per_hour": macro_evaluation.throughput_per_hour,
                "evaluation_mean_waiting_min": macro_evaluation.mean_waiting_min,
                "typed_event_sha256": _sha_bytes(_canonical_json([asdict(x) for x in typed])),
                "macro_summary_sha256": _sha_bytes(_canonical_json({
                    "reference_sha256": macro.reference_sha256,
                    "primary": {
                        "horizon_end_min": macro.horizon_end_min,
                        "event_count": macro.event_count,
                        "completed_count": macro.completed_count,
                        "backlog_count": macro.backlog_count,
                        "throughput_per_hour": macro.throughput_per_hour,
                        "mean_waiting_min": macro.mean_waiting_min,
                    },
                    "evaluation": {
                        "horizon_end_min": macro_evaluation.horizon_end_min,
                        "event_count": macro_evaluation.event_count,
                        "completed_count": macro_evaluation.completed_count,
                        "backlog_count": macro_evaluation.backlog_count,
                        "throughput_per_hour": macro_evaluation.throughput_per_hour,
                        "mean_waiting_min": macro_evaluation.mean_waiting_min,
                    },
                })),
            }
        )
    return P4BInputBasis(
        replication=replication,
        demand_per_hour=demand_per_hour,
        traffic_input_sha256=traffic_input_sha256(traffic),
        traffic_seed=seeds.traffic_arrivals_seed,
        type_assignment_seed=seeds.type_assignment_seed,
        behavior_stochasticity_seed=seeds.behavior_stochasticity_seed,
        event_count=traffic.count,
        events=events,
        macro_by_penetration=tuple(macro_rows),
    )

def _bindings_and_backend_id(
    *,
    campaign_id: str,
    replication: int,
    plant_id: str,
    timing_endpoint_id: str,
    penetration_execution: float | None,
    traffic,
) -> str:
    profiles = _plant_profiles()
    profile = profiles[plant_id]
    endpoint = _timing_endpoint(timing_endpoint_id)
    as_binding, hs_binding = publication_bindings_for_endpoint(endpoint)
    if endpoint is T0:
        bindings = endpoint_bindings(
            traffic,
            profile=as_binding.profile,
            adapter_id=as_binding.adapter_id + "/DEDUP_NEUTRAL",
            command_delay_min=0.0,
        )
    else:
        if penetration_execution is None:
            raise ValueError("non-T0 point requires penetration_execution")
        _typed, bindings = bindings_for_penetration(
            traffic,
            penetration=penetration_execution,
            campaign_id=campaign_id,
            replication=replication,
            as_binding=as_binding,
            hs_binding=hs_binding,
        )
    wrapped = P4BBehaviorEnvironmentalMotionBackend(backend_for(profile), bindings)
    return wrapped.backend_id

def _physical_point(
    *,
    execution_order: int,
    campaign_id: str,
    source_archive_sha256: str,
    replication: int,
    plant_id: str,
    room_nm: float,
    demand_per_hour: float,
    timing_endpoint_id: str,
    penetration_execution: float | None,
    deduplicated_penetrations: tuple[float, ...],
    basis: P4BInputBasis,
) -> P4BPhysicalPoint:
    profiles = _plant_profiles()
    profile = profiles[plant_id]
    config = build_publication_config(
        room_nm=room_nm,
        demand_per_hour=demand_per_hour,
        traffic_seed=basis.traffic_seed,
    )
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=config.scenario.traffic_horizon_min,
        seed=config.scenario.seed,
        start_time=config.scenario.start_time,
    )
    observed_traffic_sha = traffic_input_sha256(traffic)
    if observed_traffic_sha != basis.traffic_input_sha256:
        raise RuntimeError("room changed exogenous arrival identity")
    backend_id = _bindings_and_backend_id(
        campaign_id=campaign_id,
        replication=replication,
        plant_id=plant_id,
        timing_endpoint_id=timing_endpoint_id,
        penetration_execution=penetration_execution,
        traffic=traffic,
    )
    point_id = (
        f"R1P4BV2-R{replication:02d}-{('P1' if plant_id.startswith('P1_') else 'P2')}-"
        f"{('T0' if timing_endpoint_id == T0.endpoint_id else 'T60')}-"
        f"{_room_tag(room_nm)}-{_demand_tag(demand_per_hour)}-{_penetration_tag(penetration_execution)}"
    )
    raw = P4BPhysicalPoint(
        execution_order=execution_order,
        point_id=point_id,
        replication=replication,
        plant_id=plant_id,
        plant_sha256=profile.identity_sha256,
        room_nm=room_nm,
        demand_per_hour=demand_per_hour,
        timing_endpoint_id=timing_endpoint_id,
        penetration_execution=penetration_execution,
        deduplicated_penetrations=deduplicated_penetrations,
        traffic_seed=basis.traffic_seed,
        type_assignment_seed=basis.type_assignment_seed,
        behavior_stochasticity_seed=basis.behavior_stochasticity_seed,
        expected_traffic_input_sha256=basis.traffic_input_sha256,
        config_sha256=config.config_sha256,
        expected_behavior_backend_id=backend_id,
        source_archive_sha256=source_archive_sha256,
    )
    return replace(raw, identity_sha256=raw.compute_identity_sha256())

def build_publication_execution_plan(
    *,
    source_archive_sha256: str,
    b5_plan_sha256: str,
    b5_measurement_addendum_sha256: str,
    b6_audit_sha256: str,
    campaign_id: str = P4B_CAMPAIGN_ID,
) -> P4BPublicationExecutionPlan:
    """Materialize the complete pre-execution plan without running a CROSS point."""
    for label, value in (
        ("source_archive_sha256", source_archive_sha256),
        ("b5_plan_sha256", b5_plan_sha256),
        ("b5_measurement_addendum_sha256", b5_measurement_addendum_sha256),
        ("b6_audit_sha256", b6_audit_sha256),
    ):
        if not _valid_sha256(value):
            raise ValueError(f"{label} must be lowercase SHA-256")
    if not campaign_id:
        raise ValueError("campaign_id must be non-empty")

    seed_schedule = tuple(_seed_record(r, campaign_id) for r in range(1, P4B_REPLICATION_COUNT + 1))
    bases = tuple(
        _materialize_input_basis(campaign_id=campaign_id, replication=r, demand_per_hour=d)
        for r in range(1, P4B_REPLICATION_COUNT + 1)
        for d in P4B_DEMANDS_PER_HOUR
    )
    basis_map = {(b.replication, b.demand_per_hour): b for b in bases}

    physical: list[P4BPhysicalPoint] = []
    order = 1
    p1_id = first_order_profile().plant_id
    p2_id = tabulated_steerage_profile().plant_id
    for replication in range(1, P4B_REPLICATION_COUNT + 1):
        # P1 T0: 9 unique microscopic points, each reused across all five penetration rows.
        for room in P4B_ROOMS_NM:
            for demand in P4B_DEMANDS_PER_HOUR:
                physical.append(_physical_point(
                    execution_order=order,
                    campaign_id=campaign_id,
                    source_archive_sha256=source_archive_sha256,
                    replication=replication,
                    plant_id=p1_id,
                    room_nm=room,
                    demand_per_hour=demand,
                    timing_endpoint_id=T0.endpoint_id,
                    penetration_execution=None,
                    deduplicated_penetrations=P4B_PENETRATIONS,
                    basis=basis_map[(replication, demand)],
                ))
                order += 1
        # P1 T60: penetration remains a real microscopic treatment.
        for room in P4B_ROOMS_NM:
            for demand in P4B_DEMANDS_PER_HOUR:
                for penetration in P4B_PENETRATIONS:
                    physical.append(_physical_point(
                        execution_order=order,
                        campaign_id=campaign_id,
                        source_archive_sha256=source_archive_sha256,
                        replication=replication,
                        plant_id=p1_id,
                        room_nm=room,
                        demand_per_hour=demand,
                        timing_endpoint_id=T60.endpoint_id,
                        penetration_execution=penetration,
                        deduplicated_penetrations=(penetration,),
                        basis=basis_map[(replication, demand)],
                    ))
                    order += 1
        # P2 T0 sparse sentinel: five room-demand points, each reused across P00/P50/P100.
        for room, demand in P4B_P2_SENTINEL_ROOM_DEMAND:
            physical.append(_physical_point(
                execution_order=order,
                campaign_id=campaign_id,
                source_archive_sha256=source_archive_sha256,
                replication=replication,
                plant_id=p2_id,
                room_nm=room,
                demand_per_hour=demand,
                timing_endpoint_id=T0.endpoint_id,
                penetration_execution=None,
                deduplicated_penetrations=P4B_P2_PENETRATIONS,
                basis=basis_map[(replication, demand)],
            ))
            order += 1
        # P2 T60 sentinel: 15 real microscopic conditions.
        for room, demand in P4B_P2_SENTINEL_ROOM_DEMAND:
            for penetration in P4B_P2_PENETRATIONS:
                physical.append(_physical_point(
                    execution_order=order,
                    campaign_id=campaign_id,
                    source_archive_sha256=source_archive_sha256,
                    replication=replication,
                    plant_id=p2_id,
                    room_nm=room,
                    demand_per_hour=demand,
                    timing_endpoint_id=T60.endpoint_id,
                    penetration_execution=penetration,
                    deduplicated_penetrations=(penetration,),
                    basis=basis_map[(replication, demand)],
                ))
                order += 1

    physical_lookup: dict[tuple[object, ...], str] = {}
    for point in physical:
        if point.timing_endpoint_id == T0.endpoint_id:
            for penetration in point.deduplicated_penetrations:
                physical_lookup[(point.replication, point.plant_id, point.room_nm, point.demand_per_hour, point.timing_endpoint_id, penetration)] = point.point_id
        else:
            physical_lookup[(point.replication, point.plant_id, point.room_nm, point.demand_per_hour, point.timing_endpoint_id, point.penetration_execution)] = point.point_id

    conceptual: list[P4BConceptualRow] = []
    for replication in range(1, P4B_REPLICATION_COUNT + 1):
        for endpoint in P4B_BEHAVIOR_TIMING_ENDPOINTS:
            for room in P4B_ROOMS_NM:
                for demand in P4B_DEMANDS_PER_HOUR:
                    for penetration in P4B_PENETRATIONS:
                        key = (replication, p1_id, room, demand, endpoint.endpoint_id, penetration)
                        conceptual.append(P4BConceptualRow(
                            row_id=f"C-R{replication:02d}-P1-{endpoint.endpoint_id}-{_room_tag(room)}-{_demand_tag(demand)}-{_penetration_tag(penetration)}",
                            replication=replication,
                            plant_id=p1_id,
                            room_nm=room,
                            demand_per_hour=demand,
                            timing_endpoint_id=endpoint.endpoint_id,
                            penetration=penetration,
                            physical_point_id=physical_lookup[key],
                            input_basis_key=_basis_key(replication, demand),
                        ))
            for room, demand in P4B_P2_SENTINEL_ROOM_DEMAND:
                for penetration in P4B_P2_PENETRATIONS:
                    key = (replication, p2_id, room, demand, endpoint.endpoint_id, penetration)
                    conceptual.append(P4BConceptualRow(
                        row_id=f"C-R{replication:02d}-P2-{endpoint.endpoint_id}-{_room_tag(room)}-{_demand_tag(demand)}-{_penetration_tag(penetration)}",
                        replication=replication,
                        plant_id=p2_id,
                        room_nm=room,
                        demand_per_hour=demand,
                        timing_endpoint_id=endpoint.endpoint_id,
                        penetration=penetration,
                        physical_point_id=physical_lookup[key],
                        input_basis_key=_basis_key(replication, demand),
                    ))

    plan = P4BPublicationExecutionPlan(
        campaign_id=campaign_id,
        protocol_id=P4B_PROTOCOL_ID,
        source_archive_sha256=source_archive_sha256,
        b5_plan_sha256=b5_plan_sha256,
        b5_measurement_addendum_sha256=b5_measurement_addendum_sha256,
        b6_audit_sha256=b6_audit_sha256,
        m1_reference_sha256=GUO_2024_ILLUSTRATIVE.identity_sha256,
        replication_count=P4B_REPLICATION_COUNT,
        max_parallel_scientific_workers=P4B_MAX_PARALLEL_SCIENTIFIC_WORKERS,
        scientific_execution_authorization=P4B_EXECUTION_AUTHORIZATION_STATE,
        seed_schedule=seed_schedule,
        input_bases=bases,
        physical_points=tuple(physical),
        conceptual_rows=tuple(conceptual),
    )
    validate_publication_execution_plan(plan)
    return plan

def validate_publication_execution_plan(plan: P4BPublicationExecutionPlan) -> dict[str, object]:
    """Static preflight only; never executes a microscopic point."""
    if plan.campaign_id != P4B_CAMPAIGN_ID:
        raise ValueError("unexpected final P4B campaign id")
    if plan.replication_count != P4B_REPLICATION_COUNT:
        raise ValueError("replication_count drift")
    if plan.max_parallel_scientific_workers != 2:
        raise ValueError("publication executor must use max two scientific workers")
    if plan.scientific_execution_authorization != P4B_EXECUTION_AUTHORIZATION_STATE:
        raise ValueError("authorization state drift")
    if len(plan.seed_schedule) != 24 or len(plan.input_bases) != 72:
        raise ValueError("seed/input-basis count drift")
    if len(plan.physical_points) != 1776:
        raise ValueError("physical point count must be 1776")
    if len(plan.conceptual_rows) != 2880:
        raise ValueError("conceptual row count must be 2880")
    if [p.execution_order for p in plan.physical_points] != list(range(1, 1777)):
        raise ValueError("physical execution order must be exactly 1..1776")
    if len({p.point_id for p in plan.physical_points}) != 1776:
        raise ValueError("duplicate physical point_id")
    if len({r.row_id for r in plan.conceptual_rows}) != 2880:
        raise ValueError("duplicate conceptual row_id")
    point_ids = {p.point_id for p in plan.physical_points}
    if any(row.physical_point_id not in point_ids for row in plan.conceptual_rows):
        raise ValueError("conceptual row references unknown physical point")
    if any(p.identity_sha256 != p.compute_identity_sha256() for p in plan.physical_points):
        raise ValueError("physical point identity drift")

    by_rep: dict[int, list[P4BPhysicalPoint]] = {}
    for point in plan.physical_points:
        by_rep.setdefault(point.replication, []).append(point)
    if any(len(items) != 74 for items in by_rep.values()) or set(by_rep) != set(range(1, 25)):
        raise ValueError("each replication must contain 74 physical CROSS points")

    concept_by_rep: dict[int, int] = {}
    for row in plan.conceptual_rows:
        concept_by_rep[row.replication] = concept_by_rep.get(row.replication, 0) + 1
    if any(value != 120 for value in concept_by_rep.values()):
        raise ValueError("each replication must contain 120 conceptual rows")

    # Explicitly prove T0 reuse rather than inferring it from counts.
    p1_id = first_order_profile().plant_id
    p2_id = tabulated_steerage_profile().plant_id
    rows = plan.conceptual_rows
    for rep in range(1, 25):
        for room in P4B_ROOMS_NM:
            for demand in P4B_DEMANDS_PER_HOUR:
                ids = {
                    r.physical_point_id for r in rows
                    if r.replication == rep and r.plant_id == p1_id and r.room_nm == room
                    and r.demand_per_hour == demand and r.timing_endpoint_id == T0.endpoint_id
                }
                if len(ids) != 1:
                    raise ValueError("P1 T0 penetration rows must deduplicate to one physical result")
        for room, demand in P4B_P2_SENTINEL_ROOM_DEMAND:
            ids = {
                r.physical_point_id for r in rows
                if r.replication == rep and r.plant_id == p2_id and r.room_nm == room
                and r.demand_per_hour == demand and r.timing_endpoint_id == T0.endpoint_id
            }
            if len(ids) != 1:
                raise ValueError("P2 T0 sentinel rows must deduplicate to one physical result")

    return {
        "format": "cross-r1-p4b-static-preflight",
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.identity_sha256,
        "source_archive_sha256": plan.source_archive_sha256,
        "seed_records": len(plan.seed_schedule),
        "input_bases": len(plan.input_bases),
        "physical_points": len(plan.physical_points),
        "conceptual_rows": len(plan.conceptual_rows),
        "replication_count": plan.replication_count,
        "primary_measurement_end_min": P4B_PRIMARY_MEASUREMENT_END_MIN,
        "evaluation_end_min": P4B_EVALUATION_END_MIN,
        "physical_points_per_replication": 74,
        "conceptual_rows_per_replication": 120,
        "max_parallel_scientific_workers": 2,
        "scientific_execution_authorized": False,
        "authorization_requirement": P4B_EXECUTION_AUTHORIZATION_STATE,
        "status": "PASS_STATIC_PREFLIGHT_NO_SCIENTIFIC_POINTS_RUN",
    }

def plan_from_json_bytes(payload: bytes) -> P4BPublicationExecutionPlan:
    raw = json.loads(payload)
    return P4BPublicationExecutionPlan(
        campaign_id=raw["campaign_id"],
        protocol_id=raw["protocol_id"],
        source_archive_sha256=raw["source_archive_sha256"],
        b5_plan_sha256=raw["b5_plan_sha256"],
        b5_measurement_addendum_sha256=raw["b5_measurement_addendum_sha256"],
        b6_audit_sha256=raw["b6_audit_sha256"],
        m1_reference_sha256=raw["m1_reference_sha256"],
        replication_count=raw["replication_count"],
        max_parallel_scientific_workers=raw["max_parallel_scientific_workers"],
        scientific_execution_authorization=raw["scientific_execution_authorization"],
        seed_schedule=tuple(P4BSeedRecord(**x) for x in raw["seed_schedule"]),
        input_bases=tuple(P4BInputBasis(
            replication=x["replication"], demand_per_hour=x["demand_per_hour"],
            traffic_input_sha256=x["traffic_input_sha256"], traffic_seed=x["traffic_seed"],
            type_assignment_seed=x["type_assignment_seed"], behavior_stochasticity_seed=x["behavior_stochasticity_seed"],
            event_count=x["event_count"], events=tuple(x["events"]), macro_by_penetration=tuple(x["macro_by_penetration"])
        ) for x in raw["input_bases"]),
        physical_points=tuple(P4BPhysicalPoint(**x) for x in raw["physical_points"]),
        conceptual_rows=tuple(P4BConceptualRow(**x) for x in raw["conceptual_rows"]),
        format=raw.get("format", P4B_PLAN_FORMAT),
        schema_version=raw.get("schema_version", P4B_PLAN_SCHEMA_VERSION),
    )

def summarize_plan(plan: P4BPublicationExecutionPlan) -> dict[str, object]:
    preflight = validate_publication_execution_plan(plan)
    return {
        **preflight,
        "b5_plan_sha256": plan.b5_plan_sha256,
        "b5_measurement_addendum_sha256": plan.b5_measurement_addendum_sha256,
        "b6_audit_sha256": plan.b6_audit_sha256,
        "m1_reference_sha256": plan.m1_reference_sha256,
        "publication_replications": plan.replication_count,
        "physical_run_count": len(plan.physical_points),
        "conceptual_row_count": len(plan.conceptual_rows),
        "t0_deduplication": "QUALIFIED_AND_PREDECLARED",
        "behavior_stochasticity_seed_usage": "RESERVED_UNUSED_BECAUSE_T0_T60_BEHAVIOR_IS_DETERMINISTIC",
    }
