"""R2-FS2 publication-scale transfer of the accepted pair-context mechanism into full P4D runtime.

FS2 is the frozen publication-scale full-network validation campaign.  The physical runtime
is the qualified R1/P4D interaction-active FIFO path.  AS/HS labels do not alter vessel
profiles, motion backends, geometry, arrivals, or collision-avoidance policy; they only
parameterize the qualified pair-context same-route response supervisor.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

from vessel_crossing.nextgen import (
    EncounterRegistry,
    EnvironmentalNetworkSourceFutureIngressConfig,
    EnvironmentalNetworkSourceReservoir,
    VesselProfileCatalog,
    generate_poisson_network_traffic,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_fifo_causal_diagnostics import causal_telemetry_summary
from vessel_crossing.nextgen.environmental_network_fifo_runtime_causal_diagnostics import (
    causal_telemetry_from_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_scenario import build_environmental_network_scenario_ingress
from vessel_crossing.nextgen.environmental_network_source_reservoir import source_queue_metrics
from vessel_crossing.nextgen.pair_context_following import (
    PairContextFollowingSupervisor,
    PairContextResponseConfig,
)
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingConfig
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.research.r1_p4b_behavior_qualification import traffic_input_sha256
from vessel_crossing.research.r1_p4b_engineering_preflight import preflight_classification_config
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config
from vessel_crossing.research.r1_p4b_publication_reference import (
    AS,
    HS,
    ExogenousDemandEvent,
    TypedDemandEvent,
    publication_seed,
)
from vessel_crossing.research.r1_p4d_conflict_observer import P4DObservedMetricEnvironmentalWorld
from vessel_crossing.research.r1_p4d_conflict_service import center_pass_service_summary
from vessel_crossing.research.r1_p4d_lifecycle_audit import capture_terminal_lifecycle_audit
from vessel_crossing.research.r1_p4d_screening import (
    _conflict_summary,
    _route_completion_records,
    _stream_by_vessel,
)
from vessel_crossing.research.r1_robustness_stage2 import backend_for, first_order_profile
from vessel_crossing.research.r2_s4_markov_clustering import stationary_markov_parameterization

FS2_CAMPAIGN_ID = "R2_FS2_FULL_NETWORK_PUBLICATION_V1_2026-09-30"
FS2_PLAN_FORMAT = "cross-r2-fs2-full-network-publication-plan-v1"
FS2_RESULT_FORMAT = "cross-r2-fs2-full-network-publication-point-v1"
FS2_REPLICATIONS = 48
FS2_MAX_WORKERS = 2
FS2_ROOM_NM = 1.375
FS2_DEMAND_PER_HOUR = 6.0
FS2_P_AS = 0.50
FS2_RHOS = (-0.50, 0.0, 0.50)
FS2_DELTA_ARMS = ("NEG050", "ZERO", "POS050")
FS2_TRAFFIC_HORIZON_MIN = 90.0
FS2_EVALUATION_HORIZON_MIN = 150.0
FS2_PRIMARY_SERVICE_EVENT = "CONFLICT_CENTER_PASS"
FS2_BASE_REACTION_INTERVAL_MIN = 0.25
FS2_HS_FOLLOWER_EXTRA_MIN = 0.50
FS2_HS_LEADER_OBSERVATION_EXTRA_MIN = 0.25
FS2_PAIR_CONTEXT_EXTRA_MIN = 0.50
FS2_P4D_H2_SOURCE_SHA256 = "935a8b85185176aad242fd96d8edba6692b48a2e2d5340b39259250bc44f944d"
FS2_T1_PACKAGE_SHA256 = "9bf5032cb96d1bb66b88e56bb05d17cc28b9e0d9ba3ad3758ab3cf850ee7379c"
_EPS = 1e-10


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _rho_tag(rho: float) -> str:
    return { -0.5: "RN050", 0.0: "RZERO", 0.5: "RP050" }[float(rho)]


def _type_uniform(*, type_seed: int, stream_id: str, ordinal: int, event: ExogenousDemandEvent) -> float:
    payload = {
        "namespace": "cross-r2-fs2-stationary-markov-type-draw-v1",
        "campaign_id": FS2_CAMPAIGN_ID,
        "type_assignment_seed": int(type_seed),
        "stream_id": stream_id,
        "ordinal": int(ordinal),
        "event_id": event.event_id,
        "arrival_min": format(event.arrival_min, ".17g"),
    }
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) / float(2**64)


def markov_typed_demand(
    traffic,
    *,
    rho: float,
    type_assignment_seed: int,
) -> tuple[TypedDemandEvent, ...]:
    """Assign stationary S4 AS/HS chains independently within each physical stream.

    Stream-local chains are the full-network transfer of S4 because the accepted
    pair-context mechanism acts on same-route predecessor/follower pairs.  Arrival times,
    stream membership, routes and vessel physical classes are read-only inputs.
    """
    param = stationary_markov_parameterization(p_as=FS2_P_AS, rho=float(rho))
    events_by_stream: dict[str, list[ExogenousDemandEvent]] = {}
    stream_by_vessel = _stream_by_vessel(traffic)
    for arrival in traffic.arrivals:
        vessel_id = arrival.vessel.vessel_id
        stream_id = stream_by_vessel[vessel_id]
        events_by_stream.setdefault(stream_id, []).append(
            ExogenousDemandEvent(vessel_id, float(arrival.at), stream_id)
        )

    typed: list[TypedDemandEvent] = []
    for stream_id, events in sorted(events_by_stream.items()):
        ordered = sorted(events, key=lambda event: (event.arrival_min, event.event_id))
        previous: str | None = None
        for ordinal, event in enumerate(ordered):
            u = _type_uniform(
                type_seed=type_assignment_seed,
                stream_id=stream_id,
                ordinal=ordinal,
                event=event,
            )
            if previous is None:
                vessel_type = AS if u < param.model.initial_as_probability else HS
            elif previous == AS:
                vessel_type = AS if u < param.model.p_as_to_as else HS
            else:
                vessel_type = AS if u < param.model.p_hs_to_as else HS
            typed.append(
                TypedDemandEvent(
                    event_id=event.event_id,
                    arrival_min=event.arrival_min,
                    stream_id=event.stream_id,
                    vessel_type=vessel_type,
                    type_uniform=u,
                )
            )
            previous = vessel_type

    typed.sort(key=lambda event: (event.arrival_min, event.event_id))
    expected = {arrival.vessel.vessel_id for arrival in traffic.arrivals}
    if {event.event_id for event in typed} != expected:
        raise RuntimeError("FS2 type assignment does not cover the realized traffic exactly")
    return tuple(typed)


def typed_demand_sha256(typed: Sequence[TypedDemandEvent] | Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for event in typed:
        if isinstance(event, Mapping):
            row = dict(event)
        else:
            row = asdict(event)
        rows.append(row)
    return _sha(_canonical_json(rows))


def ordering_exposure(typed: Sequence[TypedDemandEvent]) -> dict[str, object]:
    by_stream: dict[str, list[TypedDemandEvent]] = {}
    for event in typed:
        by_stream.setdefault(event.stream_id, []).append(event)
    transitions: Counter[str] = Counter()
    type_counts: Counter[str] = Counter(event.vessel_type for event in typed)
    stream_rows = []
    indicator_pairs: list[tuple[int, int]] = []
    for stream_id, rows in sorted(by_stream.items()):
        ordered = sorted(rows, key=lambda event: (event.arrival_min, event.event_id))
        local = Counter()
        for leader, follower in zip(ordered, ordered[1:]):
            key = f"{leader.vessel_type}->{follower.vessel_type}"
            transitions[key] += 1
            local[key] += 1
            indicator_pairs.append((1 if leader.vessel_type == AS else 0, 1 if follower.vessel_type == AS else 0))
        stream_rows.append({
            "stream_id": stream_id,
            "vessel_count": len(ordered),
            "type_counts": dict(sorted(Counter(x.vessel_type for x in ordered).items())),
            "transition_counts": dict(sorted(local.items())),
        })
    realized_p = type_counts[AS] / len(typed) if typed else None
    realized_rho = None
    if indicator_pairs:
        xs = [x for x, _ in indicator_pairs]
        ys = [y for _, y in indicator_pairs]
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
        if vx > _EPS and vy > _EPS:
            realized_rho = sum((x-mx)*(y-my) for x, y in indicator_pairs) / sqrt(vx*vy)
    return {
        "vessel_count": len(typed),
        "type_counts": dict(sorted(type_counts.items())),
        "realized_p_as": realized_p,
        "adjacent_transition_count": sum(transitions.values()),
        "transition_counts": dict(sorted(transitions.items())),
        "realized_stream_local_lag1_indicator_correlation": realized_rho,
        "streams": stream_rows,
    }


def pair_context_config_for_arm(type_by_vessel: Mapping[str, str], *, delta_arm: str) -> PairContextResponseConfig:
    if delta_arm not in FS2_DELTA_ARMS:
        raise ValueError(f"unknown FS2 delta arm {delta_arm!r}")
    pair_extra: dict[tuple[str, str], float] = {}
    if delta_arm == "NEG050":
        pair_extra[(HS, HS)] = FS2_PAIR_CONTEXT_EXTRA_MIN
    elif delta_arm == "POS050":
        pair_extra[(HS, AS)] = FS2_PAIR_CONTEXT_EXTRA_MIN
    return PairContextResponseConfig(
        base_following=SameRouteFollowingConfig(reaction_interval_min=FS2_BASE_REACTION_INTERVAL_MIN),
        vessel_type_by_id=dict(type_by_vessel),
        follower_extra_min_by_type={HS: FS2_HS_FOLLOWER_EXTRA_MIN},
        leader_extra_min_by_type={HS: FS2_HS_LEADER_OBSERVATION_EXTRA_MIN},
        pair_context_extra_min_by_type_pair=pair_extra,
    )


def delta_arm_semantics(delta_arm: str) -> dict[str, object]:
    cfg = pair_context_config_for_arm({"A": AS, "H": HS}, delta_arm=delta_arm)
    sup = PairContextFollowingSupervisor(cfg)
    a = sup.evaluate_pair("A", "A").total_reaction_interval_min
    b = sup.evaluate_pair("A", "H").total_reaction_interval_min
    c = sup.evaluate_pair("H", "A").total_reaction_interval_min
    d = sup.evaluate_pair("H", "H").total_reaction_interval_min
    delta = (b + c) - (a + d)
    return {"a_AA_min": a, "b_AH_min": b, "c_HA_min": c, "d_HH_min": d, "delta_min": delta}


@dataclass(frozen=True, slots=True)
class FS2Point:
    execution_order: int
    point_id: str
    replication: int
    rho: float
    delta_arm: str
    traffic_seed: int
    type_assignment_seed: int
    expected_traffic_count: int
    expected_traffic_input_sha256: str
    expected_typed_sha256: str
    expected_mechanism_config_sha256: str
    identity_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "execution_order": self.execution_order,
            "point_id": self.point_id,
            "replication": self.replication,
            "rho": self.rho,
            "delta_arm": self.delta_arm,
            "traffic_seed": self.traffic_seed,
            "type_assignment_seed": self.type_assignment_seed,
            "expected_traffic_count": self.expected_traffic_count,
            "expected_traffic_input_sha256": self.expected_traffic_input_sha256,
            "expected_typed_sha256": self.expected_typed_sha256,
            "expected_mechanism_config_sha256": self.expected_mechanism_config_sha256,
        }

    def compute_identity_sha256(self) -> str:
        return _sha(_canonical_json(self.identity_payload()))

    def __post_init__(self) -> None:
        if not 1 <= self.replication <= FS2_REPLICATIONS:
            raise ValueError("replication outside frozen FS2 publication range")
        if float(self.rho) not in FS2_RHOS:
            raise ValueError("rho outside frozen FS2 grid")
        if self.delta_arm not in FS2_DELTA_ARMS:
            raise ValueError("delta arm outside frozen FS2 grid")
        if self.execution_order < 1:
            raise ValueError("execution_order must be positive")
        if self.identity_sha256 and self.identity_sha256 != self.compute_identity_sha256():
            raise ValueError("FS2 point identity mismatch")


@dataclass(frozen=True, slots=True)
class FS2Plan:
    protocol_sha256: str
    points: tuple[FS2Point, ...]
    campaign_id: str = FS2_CAMPAIGN_ID
    scientific_role: str = "PUBLICATION_SCALE_FULL_NETWORK_VALIDATION"
    authoritative_execution_location: str = "USER_PC_ONLY"
    max_parallel_workers: int = FS2_MAX_WORKERS
    excluded_future_seed_namespace: str = "FS1_SEEDS_EXCLUDED_FS2_FROZEN_NEW_CAMPAIGN"
    parent_p4d_h2_source_sha256: str = FS2_P4D_H2_SOURCE_SHA256
    parent_t1_package_sha256: str = FS2_T1_PACKAGE_SHA256

    @property
    def document(self) -> dict[str, object]:
        return {
            "format": FS2_PLAN_FORMAT,
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "scientific_role": self.scientific_role,
            "authoritative_execution_location": self.authoritative_execution_location,
            "protocol_sha256": self.protocol_sha256,
            "parent_p4d_h2_source_sha256": self.parent_p4d_h2_source_sha256,
            "parent_t1_package_sha256": self.parent_t1_package_sha256,
            "max_parallel_workers": self.max_parallel_workers,
            "replications": FS2_REPLICATIONS,
            "room_nm": FS2_ROOM_NM,
            "demand_per_hour": FS2_DEMAND_PER_HOUR,
            "stream_demand_per_hour": FS2_DEMAND_PER_HOUR / 2.0,
            "traffic_horizon_min": FS2_TRAFFIC_HORIZON_MIN,
            "primary_measurement_window_min": [0.0, FS2_TRAFFIC_HORIZON_MIN],
            "evaluation_horizon_min": FS2_EVALUATION_HORIZON_MIN,
            "primary_service_event": FS2_PRIMARY_SERVICE_EVENT,
            "p_as": FS2_P_AS,
            "rho_grid": list(FS2_RHOS),
            "delta_arms": {arm: delta_arm_semantics(arm) for arm in FS2_DELTA_ARMS},
            "seed_exclusion_rule": self.excluded_future_seed_namespace,
            "crn_contract": {
                "same_replication_all_9_cells_share_traffic": True,
                "same_replication_and_rho_all_delta_arms_share_exact_type_assignment": True,
                "rho_cells_share_hash_draws_but_apply_distinct_S4_transition_probabilities": True,
                "arrival_generation_independent_of_type_assignment": True,
            },
            "points": [dict(p.identity_payload(), identity_sha256=p.compute_identity_sha256()) for p in self.points],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.document)

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_json_bytes())


def build_fs2_plan(*, protocol_sha256: str) -> FS2Plan:
    points: list[FS2Point] = []
    execution_order = 0
    for rep in range(1, FS2_REPLICATIONS + 1):
        traffic_seed = publication_seed(campaign_id=FS2_CAMPAIGN_ID, replication=rep, namespace="TRAFFIC_ARRIVALS")
        type_seed = publication_seed(campaign_id=FS2_CAMPAIGN_ID, replication=rep, namespace="TYPE_ASSIGNMENT")
        cfg = build_publication_config(room_nm=FS2_ROOM_NM, demand_per_hour=FS2_DEMAND_PER_HOUR, traffic_seed=traffic_seed)
        traffic = generate_poisson_network_traffic(
            cfg.network, cfg.streams,
            horizon=cfg.scenario.traffic_horizon_min,
            seed=cfg.scenario.seed,
            start_time=cfg.scenario.start_time,
        )
        traffic_sha = traffic_input_sha256(traffic)
        for rho in FS2_RHOS:
            typed = markov_typed_demand(traffic, rho=rho, type_assignment_seed=type_seed)
            typed_sha = typed_demand_sha256(typed)
            type_map = {event.event_id: event.vessel_type for event in typed}
            for delta_arm in FS2_DELTA_ARMS:
                execution_order += 1
                supervisor = PairContextFollowingSupervisor(pair_context_config_for_arm(type_map, delta_arm=delta_arm))
                point = FS2Point(
                    execution_order=execution_order,
                    point_id=f"FS2-R{rep:03d}-{_rho_tag(rho)}-{delta_arm}",
                    replication=rep,
                    rho=rho,
                    delta_arm=delta_arm,
                    traffic_seed=traffic_seed,
                    type_assignment_seed=type_seed,
                    expected_traffic_count=traffic.count,
                    expected_traffic_input_sha256=traffic_sha,
                    expected_typed_sha256=typed_sha,
                    expected_mechanism_config_sha256=supervisor.config_identity_sha256,
                )
                points.append(FS2Point(**dict(point.identity_payload(), identity_sha256=point.compute_identity_sha256())))
    if len(points) != FS2_REPLICATIONS * len(FS2_RHOS) * len(FS2_DELTA_ARMS):
        raise RuntimeError(f"FS2 frozen plan must contain {FS2_REPLICATIONS * len(FS2_RHOS) * len(FS2_DELTA_ARMS)} points, got {len(points)}")
    return FS2Plan(protocol_sha256=protocol_sha256, points=tuple(points))


def plan_from_json_bytes(raw: bytes) -> FS2Plan:
    value = json.loads(raw)
    if value.get("format") != FS2_PLAN_FORMAT:
        raise ValueError("unexpected FS2 plan format")
    points = tuple(FS2Point(**row) for row in value["points"])
    plan = FS2Plan(
        protocol_sha256=str(value["protocol_sha256"]),
        points=points,
        campaign_id=str(value["campaign_id"]),
        scientific_role=str(value["scientific_role"]),
        authoritative_execution_location=str(value["authoritative_execution_location"]),
        max_parallel_workers=int(value["max_parallel_workers"]),
        excluded_future_seed_namespace=str(value["seed_exclusion_rule"]),
        parent_p4d_h2_source_sha256=str(value["parent_p4d_h2_source_sha256"]),
        parent_t1_package_sha256=str(value["parent_t1_package_sha256"]),
    )
    if plan.to_json_bytes() != raw:
        raise ValueError("FS2 plan canonical/semantic identity drift")
    return plan


def _run_physical_point(point: FS2Point) -> tuple[dict[str, object], bytes]:
    cfg = build_publication_config(room_nm=FS2_ROOM_NM, demand_per_hour=FS2_DEMAND_PER_HOUR, traffic_seed=point.traffic_seed)
    scenario = cfg.scenario
    if abs(scenario.traffic_horizon_min - FS2_TRAFFIC_HORIZON_MIN) > _EPS or abs(scenario.end_time - FS2_EVALUATION_HORIZON_MIN) > _EPS:
        raise RuntimeError("qualified P4D horizon contract drift")
    traffic = generate_poisson_network_traffic(
        cfg.network, cfg.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
    traffic_sha = traffic_input_sha256(traffic)
    if traffic_sha != point.expected_traffic_input_sha256 or traffic.count != point.expected_traffic_count:
        raise RuntimeError("FS2 traffic identity drift")
    typed = markov_typed_demand(traffic, rho=point.rho, type_assignment_seed=point.type_assignment_seed)
    typed_sha = typed_demand_sha256(typed)
    if typed_sha != point.expected_typed_sha256:
        raise RuntimeError("FS2 type assignment identity drift")
    type_map = {event.event_id: event.vessel_type for event in typed}
    response = pair_context_config_for_arm(type_map, delta_arm=point.delta_arm)
    following = PairContextFollowingSupervisor(response)
    if following.config_identity_sha256 != point.expected_mechanism_config_sha256:
        raise RuntimeError("FS2 mechanism configuration identity drift")

    stream_by_vessel = _stream_by_vessel(traffic)
    ingress = build_environmental_network_scenario_ingress(traffic, scenario)
    reservoir = EnvironmentalNetworkSourceReservoir.from_scenario_ingress(ingress)
    profiles = VesselProfileCatalog()
    motion_backend = backend_for(first_order_profile())
    world = P4DObservedMetricEnvironmentalWorld(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=scenario.start_time,
        profile_catalog=profiles,
        safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=motion_backend,
        retain_safety_intervals=False,
        stream_by_vessel=stream_by_vessel,
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=profiles,
    )
    pair = scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    started = start_environmental_network_source_fifo_runtime(
        registry,
        world,
        {},
        reservoir,
        recapture_config=scenario.recapture,
        runtime_config=scenario.runtime,
        recheck_config=cfg.recheck,
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
    conflict = _conflict_summary(world.p4d_conflict_observations, outcome.reservoir.admissions, room_nm=FS2_ROOM_NM)
    center_service = center_pass_service_summary(world.p4d_conflict_observations, typed, reference_matrix=None)
    lifecycle = capture_terminal_lifecycle_audit(runtime, outcome.reservoir, completions)
    nominal = float(conflict["nominal_route_duration_min"])
    route_extra = [float(x["route_duration_min"]) - nominal for x in completions]
    interaction_active = bool(
        len(runtime.persistent.rebootstrap_results)
        or any(x > 1e-6 for x in route_extra)
        or float(conflict["max_positive_request_to_entry_excess_min"]) > 1e-3
        or float(conflict["max_center_cross_track_nm"]) > 1e-3
        or float(conflict["max_center_heading_deviation_deg"]) > 1e-3
    )
    causal = causal_telemetry_from_fifo_runtime(started, outcome, pair_context_supervisor=following)
    causal_summary = causal_telemetry_summary(causal)

    document: dict[str, object] = {
        "format": FS2_RESULT_FORMAT,
        "schema_version": 1,
        "scientific_role": "PUBLICATION_SCALE_FULL_NETWORK_VALIDATION",
        "campaign_id": FS2_CAMPAIGN_ID,
        "point": dict(point.identity_payload(), identity_sha256=point.compute_identity_sha256()),
        "runtime_semantics": {
            "qualified_parent": "R1_P4D_INTERACTION_ACTIVE_FULL_NETWORK_H2",
            "same_runtime_path_as_p4d": True,
            "future_ingress_clearance_enabled": True,
            "passive_conflict_observer_enabled": True,
            "reference_headway_forcing": False,
            "as_hs_physical_profiles_identical": True,
            "type_assignment_changes_exogenous_arrivals": False,
            "motion_backend_id": getattr(motion_backend, "backend_id", type(motion_backend).__name__),
            "pair_context_mechanism_id": following.mechanism_id,
            "pair_context_config_sha256": following.config_identity_sha256,
            "base_same_route_reaction_interval_min": response.base_following.reaction_interval_min,
        },
        "traffic_input_sha256": traffic_sha,
        "traffic_count": traffic.count,
        "typed_assignment_sha256": typed_sha,
        "typed_roles": [asdict(event) for event in typed],
        "ordering_exposure": ordering_exposure(typed),
        "delta_semantics": delta_arm_semantics(point.delta_arm),
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
            "mean_route_extra_min": None if not route_extra else sum(route_extra) / len(route_extra),
            "max_route_extra_min": None if not route_extra else max(route_extra),
        },
        "conflict": conflict,
        "center_service": center_service,
        "response": {
            "rebootstrap_count": len(runtime.persistent.rebootstrap_results),
            "fleet_observation_count": len(runtime.coordinator.fleet_observations),
            "same_route_supervision_count": len(runtime.persistent.mixed_session.same_route_supervisions),
            "interaction_active": interaction_active,
        },
        "causal_telemetry": causal.to_json_object(),
        "causal_summary": causal_summary,
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
    return document, raw


def run_fs2_point(plan: FS2Plan, point: FS2Point) -> bytes:
    if point not in plan.points:
        raise ValueError("point is not part of the frozen FS2 plan")
    document, _ = _run_physical_point(point)
    document["plan_sha256"] = plan.identity_sha256
    document["protocol_sha256"] = plan.protocol_sha256
    return _canonical_json(document)
