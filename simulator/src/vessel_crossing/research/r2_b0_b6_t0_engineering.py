"""Engineering-only T0 audit for the frozen R2 B0--B6 reference-envelope controller.

T0 is wiring/semantic evidence only.  It must never be used as a capacity estimate or as
scientific evidence for B4/B5/B6 effects.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping, Sequence

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
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.research import r1_robustness_matrix as stage1
from vessel_crossing.research.r1_p4b_engineering_preflight import preflight_classification_config
from vessel_crossing.research.r1_p4b_publication_execution import build_publication_config
from vessel_crossing.research.r1_p4d_conflict_observer import P4DObservedMetricEnvironmentalWorld
from vessel_crossing.research.r1_robustness_stage2 import backend_for, first_order_profile
from vessel_crossing.research.r2_b0_b6_reference_controller import (
    R2B4B5SourceManagementController,
    R2ManagedReferenceConfig,
    controller_trace_as_dicts,
)
from vessel_crossing.research.r2_fs2_full_network_publication import pair_context_config_for_arm
from vessel_crossing.nextgen.pair_context_following import PairContextFollowingSupervisor

T0_FORMAT = "cross-r2-b0-b6-t0-engineering-result-v1"
T0_ROOM_NM = 1.375
T0_DEMAND_PER_HOUR = 6.0
_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class T0DemandSpec:
    vessel_id: str
    stream_id: str
    demand_time_min: float
    vessel_type: str


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _fixture_demand(cfg, spec: T0DemandSpec) -> EnvironmentalNetworkSourceDemand:
    stream = next(item for item in cfg.streams if item.stream_id == spec.stream_id)
    path = cfg.network.resolve(stream.request)
    route = path.route
    profile = stream.vessel_class
    if profile is None:
        raise RuntimeError("qualified R2 stream lacks a vessel profile")
    if stream.limits is not None and stream.limits != profile.maneuver_limits:
        raise RuntimeError("qualified R2 stream limit/profile drift")
    drift = cfg.scenario.drift_for_stream(spec.stream_id)
    if abs(drift[0]) > _EPS or abs(drift[1]) > _EPS:
        raise RuntimeError("T0 fixture assumes the qualified zero-drift R2 publication config")
    heading = route.segment_heading(0)
    vessel = WaterRelativeVesselState(
        vessel_id=spec.vessel_id,
        position=route.waypoints[0],
        through_water_heading_deg=heading,
        through_water_speed=stream.cruise_speed,
    )
    target = EnvironmentalRouteHandoffTarget(
        vessel_id=spec.vessel_id,
        navigator=RouteNavigator(
            RouteFollower(route, cruise_speed=stream.cruise_speed),
            CrossTrackRecovery(route, max_intercept_angle_deg=stream.recovery_max_intercept_angle_deg),
        ),
        follower_state=RouteFollowerState(),
        end_action=stream.end_action,
    )
    return EnvironmentalNetworkSourceDemand(
        stream_id=spec.stream_id,
        nominal_vessel_id=spec.vessel_id,
        demand_time=float(spec.demand_time_min),
        vessel=vessel,
        route_target=target,
        limits=profile.maneuver_limits,
        environmental_drift_velocity=drift,
        profile=profile,
    )


def _physical_contract(cfg) -> dict[str, object]:
    streams = []
    for stream in sorted(cfg.streams, key=lambda item: item.stream_id):
        profile = stream.vessel_class
        assert profile is not None and profile.safety_domain is not None
        streams.append(
            {
                "stream_id": stream.stream_id,
                "cruise_speed": stream.cruise_speed,
                "route_waypoints": [list(x) for x in cfg.network.resolve(stream.request).route.waypoints],
                "dimensions": asdict(profile.dimensions),
                "safety_domain": asdict(profile.safety_domain),
                "maneuver_limits": asdict(profile.maneuver_limits),
                "speed_envelope": None if profile.speed_envelope is None else asdict(profile.speed_envelope),
            }
        )
    pair = cfg.scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    return {
        "room_nm": T0_ROOM_NM,
        "streams": streams,
        "pair_clearance_repr": repr(pair),
        "safety_metric_config_repr": repr(stage1.stage1_safety_metric_config()),
        "motion_backend_profile": asdict(first_order_profile()),
    }


def _max_same_time_boundary_count(times: Sequence[float]) -> int:
    if not times:
        return 0
    counts: Counter[float] = Counter(round(float(t), 10) for t in times)
    return max(counts.values())


def run_managed_t0_fixture(
    *,
    case_id: str,
    level: str,
    delta_arm: str,
    specs: Sequence[T0DemandSpec],
    end_time_min: float,
) -> dict[str, object]:
    cfg = build_publication_config(
        room_nm=T0_ROOM_NM,
        demand_per_hour=T0_DEMAND_PER_HOUR,
        traffic_seed=1,
    )
    demands = tuple(_fixture_demand(cfg, spec) for spec in specs)
    type_map = {spec.vessel_id: spec.vessel_type for spec in specs}
    stream_map = {spec.vessel_id: spec.stream_id for spec in specs}
    response = pair_context_config_for_arm(type_map, delta_arm=delta_arm)
    following = PairContextFollowingSupervisor(response)
    profiles = VesselProfileCatalog()
    backend = backend_for(first_order_profile())
    world = P4DObservedMetricEnvironmentalWorld(
        (),
        limits={},
        environmental_drift_velocity={},
        start_time=0.0,
        profile_catalog=profiles,
        safety_config=stage1.stage1_safety_metric_config(),
        motion_backend=backend,
        retain_safety_intervals=False,
        stream_by_vessel=stream_map,
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
        registry,
        world,
        {},
        EnvironmentalNetworkSourceReservoir(demands),
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
    outcome = started.session.run_until_capturing_terminal_outcome(end_time_min)
    safety_interval_count = int(getattr(world, "safety_interval_count", 0))
    safety = world.finalize_safety_experiment_summary() if safety_interval_count else None
    boundary_times = [row.time_min for row in started.session.source_boundaries]
    center_passes = [
        {
            "vessel_id": row.vessel_id,
            "stream_id": row.stream_id,
            "time_min": row.observation_time_min,
        }
        for row in world.p4d_conflict_observations
        if row.role == "CONFLICT_CENTER_PASS"
    ]
    result = {
        "format": T0_FORMAT,
        "scientific_role": "ENGINEERING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "case_id": case_id,
        "level": level,
        "delta_arm": delta_arm,
        "end_time_min": end_time_min,
        "physics_contract": _physical_contract(cfg),
        "physics_contract_sha256": _sha(_canonical_json(_physical_contract(cfg))),
        "pair_context_config_sha256": following.config_identity_sha256,
        "terminal": {
            "status": outcome.status.value,
            "terminal_time_min": outcome.terminal_time_min,
            "infeasibility_type": outcome.infeasibility_type,
            "resource_unavailability_type": outcome.resource_unavailability_type,
        },
        "demands": [asdict(spec) for spec in specs],
        "admissions": [
            {
                "vessel_id": row.vessel_id,
                "stream_id": row.stream_id,
                "demand_time_min": row.demand_time,
                "admission_time_min": row.admission_time,
                "waiting_time_min": row.waiting_time_min,
            }
            for row in outcome.reservoir.admissions
        ],
        "pending_ids": [row.vessel_id for row in outcome.reservoir.pending_demands_in_service_order],
        "service_order_by_stream": [
            [stream_id, list(order)] for stream_id, order in outcome.reservoir.service_order_by_stream
        ],
        "controller_trace": list(controller_trace_as_dicts(controller.trace)),
        "bypass_count": dict(controller.bypass_count),
        "center_passes": center_passes,
        "source_boundary_count": len(boundary_times),
        "max_same_time_source_boundary_count": _max_same_time_boundary_count(boundary_times),
        "safety_outputs_retained": safety is not None,
        "safety": None if safety is None else {
            "modeled_hull_overlap_episode_count": safety.modeled_hull_overlap_episode_count,
            "safety_domain_intrusion_episode_count": safety.safety_domain_intrusion_episode_count,
            "near_miss_episode_count": safety.near_miss_episode_count,
            "sampled_min_hull_margin_nm": safety.interval_summary.sampled_min_hull_margin_nm,
            "sampled_min_safety_domain_margin_nm": safety.interval_summary.sampled_min_safety_domain_margin_nm,
        },
        "external_or_guo_headway_input": False,
    }
    return result


def _b4_active_specs() -> tuple[T0DemandSpec, ...]:
    return (
        T0DemandSpec("north-t0-001", "north", 0.0, "AS"),
        T0DemandSpec("west-t0-001", "west", 0.0, "HS"),
    )


def _b5_specs(delta_arm: str) -> tuple[T0DemandSpec, ...]:
    if delta_arm == "NEG050":
        types = ("AS", "AS", "HS")
    elif delta_arm == "POS050":
        types = ("AS", "HS", "AS")
    elif delta_arm == "ZERO":
        types = ("AS", "AS", "HS")
    else:
        raise ValueError(delta_arm)
    return (
        T0DemandSpec("north-t0-001", "north", 0.0, types[0]),
        T0DemandSpec("north-t0-002", "north", 5.0, types[1]),
        T0DemandSpec("north-t0-003", "north", 5.0, types[2]),
    )


def _b6_smoke_specs() -> tuple[T0DemandSpec, ...]:
    rows: list[T0DemandSpec] = []
    cycle = ("AS", "AS", "HS", "HS")
    # T0 is deliberately tiny: two immediately waiting demands per stream are
    # sufficient to exercise saturated-queue wiring without turning this engineering
    # smoke into an accidental capacity campaign.
    for stream_id in ("north", "west"):
        for index in range(1, 3):
            rows.append(
                T0DemandSpec(
                    f"{stream_id}-sat-{index:03d}",
                    stream_id,
                    0.0,
                    cycle[(index - 1) % len(cycle)],
                )
            )
    rows.sort(key=lambda row: (row.demand_time_min, row.vessel_id))
    return tuple(rows)


def run_t0_managed_suite() -> dict[str, object]:
    b4 = run_managed_t0_fixture(
        case_id="T0-B4-ACTIVE",
        level="B4",
        delta_arm="ZERO",
        specs=_b4_active_specs(),
        end_time_min=25.0,
    )
    b5_neg = run_managed_t0_fixture(
        case_id="T0-B5-NEG-ACTIVE",
        level="B5",
        delta_arm="NEG050",
        specs=_b5_specs("NEG050"),
        end_time_min=30.0,
    )
    b5_pos = run_managed_t0_fixture(
        case_id="T0-B5-POS-ACTIVE",
        level="B5",
        delta_arm="POS050",
        specs=_b5_specs("POS050"),
        end_time_min=30.0,
    )
    zero_specs = _b5_specs("ZERO")
    zero_b4 = run_managed_t0_fixture(
        case_id="T0-B5-ZERO-PARITY-B4-REFERENCE",
        level="B4",
        delta_arm="ZERO",
        specs=zero_specs,
        end_time_min=30.0,
    )
    zero_b5 = run_managed_t0_fixture(
        case_id="T0-B5-ZERO-PARITY",
        level="B5",
        delta_arm="ZERO",
        specs=zero_specs,
        end_time_min=30.0,
    )
    b6 = run_managed_t0_fixture(
        case_id="T0-B6-SMOKE",
        level="B5",
        delta_arm="NEG050",
        specs=_b6_smoke_specs(),
        end_time_min=15.0,
    )

    def trace_events(case: Mapping[str, object]) -> list[str]:
        return [str(row["event"]) for row in case["controller_trace"]]  # type: ignore[index]

    zero_keys = (
        "terminal",
        "admissions",
        "pending_ids",
        "service_order_by_stream",
        "center_passes",
        "safety",
        "source_boundary_count",
        "max_same_time_source_boundary_count",
        "physics_contract_sha256",
        "pair_context_config_sha256",
    )
    zero_parity = all(zero_b4[key] == zero_b5[key] for key in zero_keys)
    physics_hashes = {
        case["physics_contract_sha256"] for case in (b4, b5_neg, b5_pos, zero_b4, zero_b5, b6)
    }
    gates = {
        "B4_CONTROLLER_ACTIVE": "B4_SLOT_RESERVED" in trace_events(b4),
        "B4_CANONICAL_GATE_ADMISSION_OBSERVED": "B4_SLOT_ADMITTED_BY_CANONICAL_GATE" in trace_events(b4),
        "B5_NEG_SWAP_OBSERVED": "B5_ADJACENT_SWAP" in trace_events(b5_neg),
        "B5_POS_SWAP_OBSERVED": "B5_ADJACENT_SWAP" in trace_events(b5_pos),
        "B5_MAX_ONE_BYPASS_NEG": max([0, *b5_neg["bypass_count"].values()]) <= 1,  # type: ignore[union-attr]
        "B5_MAX_ONE_BYPASS_POS": max([0, *b5_pos["bypass_count"].values()]) <= 1,  # type: ignore[union-attr]
        "ZERO_B5_EQUALS_B4": zero_parity,
        "NO_PHYSICS_CONFIG_HASH_DRIFT": len(physics_hashes) == 1,
        "NO_EXTERNAL_OR_GUO_HEADWAY": all(
            case["external_or_guo_headway_input"] is False
            for case in (b4, b5_neg, b5_pos, zero_b4, zero_b5, b6)
        ),
        "SAFETY_OUTPUTS_RETAINED": all(
            bool(case["safety_outputs_retained"])
            for case in (b4, b5_neg, b5_pos, zero_b4, zero_b5, b6)
        ),
        "NO_ZERO_TIME_EVENT_LOOP": all(
            int(case["max_same_time_source_boundary_count"]) <= 4
            for case in (b4, b5_neg, b5_pos, zero_b4, zero_b5, b6)
        ),
        "NO_T0_RESOURCE_PATHOLOGY": all(
            case["terminal"]["status"] != "RESOURCE_UNAVAILABLE"  # type: ignore[index]
            for case in (b4, b5_neg, b5_pos, zero_b4, zero_b5, b6)
        ),
    }
    return {
        "format": "cross-r2-b0-b6-t0-managed-suite-v1",
        "scientific_role": "ENGINEERING_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "cases": [b4, b5_neg, b5_pos, zero_b4, zero_b5, b6],
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def write_t0_managed_suite(path: str) -> str:
    document = run_t0_managed_suite()
    raw = _canonical_json(document)
    with open(path, "wb") as handle:
        handle.write(raw)
    return _sha(raw)
