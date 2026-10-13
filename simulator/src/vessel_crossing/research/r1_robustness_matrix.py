"""Staged R1 engineering robustness matrix for vessel geometry and safety-domain policy.

Stage 1 varies only horizontal vessel dimensions and operational-domain geometry while
holding traffic, maneuver limits, candidate policy and the recovered P4B runtime fixed.
It is engineering sensitivity evidence, not a regulatory CPA study and not a statement
that the synthetic dimensions correspond to named real vessel classes.

The design deliberately uses common random numbers (the same traffic seed) across all
points so differences are attributable to geometry/domain choices rather than a new
Poisson realization.  Plant fidelity is a separate Stage-2 factor and must not be mixed
into Stage 1 before sensitive points are selected.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from typing import Any

from vessel_crossing.nextgen import (
    EncounterRegistry,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalNetworkSourceFutureIngressConfig,
    EnvironmentalWorld2D,
    ShipDomain,
    VesselClassProfile,
    VesselDimensions,
    VesselProfileCatalog,
    VesselSpeedEnvelope,
    generate_poisson_network_traffic,
    start_environmental_network_source_fifo_runtime,
)
from vessel_crossing.nextgen.environmental_network_paired_experiment import _MetricEnvironmentalWorld
from vessel_crossing.nextgen.environmental_network_scenario import _environmental_ingress_records
from vessel_crossing.nextgen.mission_safety import summarize_mission_safety
from vessel_crossing.nextgen.multi_intent import MultiIntentResolutionPolicy
from vessel_crossing.nextgen.safety_metrics import PairSafetyMetricConfig
from vessel_crossing.nextgen.same_route_following import SameRouteFollowingSupervisor
from vessel_crossing.nextgen.environmental_network_source_reservoir import source_queue_metrics

from .r1_p4b_engineering_preflight import (
    P4B_SHARED_REPLICATION_SEED,
    build_bounded_resource_preflight_config,
    preflight_classification_config,
)

_METERS_PER_NM = 1852.0


@dataclass(frozen=True, slots=True)
class HullClearanceMargins:
    """Additional operational clearance outside the modeled hull ellipse."""

    ahead_m: float
    astern_m: float
    port_m: float
    starboard_m: float

    def __post_init__(self) -> None:
        for name in ("ahead_m", "astern_m", "port_m", "starboard_m"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def uniform(cls, clearance_m: float) -> "HullClearanceMargins":
        return cls(clearance_m, clearance_m, clearance_m, clearance_m)

    def to_domain(self, dimensions: VesselDimensions) -> ShipDomain:
        half_length_m = 0.5 * dimensions.length_m
        half_beam_m = 0.5 * dimensions.beam_m
        return ShipDomain.from_meters(
            ahead_m=half_length_m + self.ahead_m,
            astern_m=half_length_m + self.astern_m,
            port_m=half_beam_m + self.port_m,
            starboard_m=half_beam_m + self.starboard_m,
        )


@dataclass(frozen=True, slots=True)
class DomainHullAudit:
    contains_modeled_hull: bool
    ahead_clearance_m: float
    astern_clearance_m: float
    port_clearance_m: float
    starboard_clearance_m: float

    @property
    def minimum_clearance_m(self) -> float:
        return min(
            self.ahead_clearance_m,
            self.astern_clearance_m,
            self.port_clearance_m,
            self.starboard_clearance_m,
        )


def audit_domain_against_hull(
    dimensions: VesselDimensions,
    domain: ShipDomain,
    *,
    tolerance_m: float = 1e-9,
) -> DomainHullAudit:
    """Verify that an operational ellipse contains the modeled hull ellipse."""

    half_length = 0.5 * dimensions.length_m
    half_beam = 0.5 * dimensions.beam_m
    clearances = (
        domain.ahead_nm * _METERS_PER_NM - half_length,
        domain.astern_nm * _METERS_PER_NM - half_length,
        domain.port_nm * _METERS_PER_NM - half_beam,
        domain.starboard_nm * _METERS_PER_NM - half_beam,
    )
    return DomainHullAudit(
        contains_modeled_hull=all(value >= -tolerance_m for value in clearances),
        ahead_clearance_m=clearances[0],
        astern_clearance_m=clearances[1],
        port_clearance_m=clearances[2],
        starboard_clearance_m=clearances[3],
    )


@dataclass(frozen=True, slots=True)
class RobustnessStage1Point:
    point_id: str
    geometry_id: str
    domain_policy_id: str
    dimensions: VesselDimensions
    safety_domain: ShipDomain
    traffic_seed: int = P4B_SHARED_REPLICATION_SEED
    evidence_role: str = "ENGINEERING_ROBUSTNESS_DIAGNOSTIC_NOT_RESEARCH_RESULT"

    def __post_init__(self) -> None:
        audit = audit_domain_against_hull(self.dimensions, self.safety_domain)
        if not audit.contains_modeled_hull:
            raise ValueError(f"operational domain does not contain modeled hull for {self.point_id}")

    @property
    def identity_sha256(self) -> str:
        payload = {
            "point_id": self.point_id,
            "geometry_id": self.geometry_id,
            "domain_policy_id": self.domain_policy_id,
            "dimensions": asdict(self.dimensions),
            "safety_domain": asdict(self.safety_domain),
            "traffic_seed": self.traffic_seed,
            "evidence_role": self.evidence_role,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


_GEOMETRIES_M = (
    ("G090x015", 90.0, 15.0),
    ("G120x020", 120.0, 20.0),
    ("G180x030", 180.0, 30.0),
    ("G240x040", 240.0, 40.0),
)


def _domain_variants(dimensions: VesselDimensions) -> tuple[tuple[str, ShipDomain], ...]:
    return (
        ("D_LEGACY_CIRCULAR_0P10NM", ShipDomain.circular(0.10)),
        ("D_HULL_PLUS_025M", HullClearanceMargins.uniform(25.0).to_domain(dimensions)),
        ("D_HULL_PLUS_075M", HullClearanceMargins.uniform(75.0).to_domain(dimensions)),
        (
            "D_DIRECTIONAL_A150_AST75_P50_S50M",
            HullClearanceMargins(150.0, 75.0, 50.0, 50.0).to_domain(dimensions),
        ),
    )


def stage1_design() -> tuple[RobustnessStage1Point, ...]:
    points: list[RobustnessStage1Point] = []
    for geometry_id, length_m, beam_m in _GEOMETRIES_M:
        dims = VesselDimensions.from_meters(length_m=length_m, beam_m=beam_m)
        for domain_id, domain in _domain_variants(dims):
            points.append(
                RobustnessStage1Point(
                    point_id=f"R1ROB-S1-{geometry_id}-{domain_id}",
                    geometry_id=geometry_id,
                    domain_policy_id=domain_id,
                    dimensions=dims,
                    safety_domain=domain,
                )
            )
    return tuple(points)


def stage1_design_sha256() -> str:
    return sha256(
        json.dumps(
            [
                {
                    "point_id": p.point_id,
                    "point_sha256": p.identity_sha256,
                }
                for p in stage1_design()
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def baseline_point() -> RobustnessStage1Point:
    return next(
        p
        for p in stage1_design()
        if p.geometry_id == "G120x020" and p.domain_policy_id == "D_LEGACY_CIRCULAR_0P10NM"
    )


def build_stage1_config(point: RobustnessStage1Point):
    """Materialize one point on the recovered D02-derived engineering runtime."""

    base = build_bounded_resource_preflight_config(
        rule13_progress_speed_fractions=(1.1,),
        rule13_progress_course_offsets_from_counterpart_deg=(-30.0, -15.0, 15.0, 30.0),
        rule13_progress_include_speed_envelope_ceiling=True,
        maximum_speed_nm_per_min=0.12,
        multi_intent_resolution=MultiIntentResolutionPolicy.RELAX_STAND_ON_FOR_REQUIRED_AVOIDANCE,
        dynamic_same_route_ingress=True,
    )
    streams = []
    for stream in base.streams:
        assert stream.vessel_class is not None
        profile = VesselClassProfile(
            class_id=f"r1-robustness-stage1-{point.geometry_id}-{point.domain_policy_id}",
            dimensions=point.dimensions,
            maneuver_limits=stream.vessel_class.maneuver_limits,
            safety_domain=point.safety_domain,
            speed_envelope=VesselSpeedEnvelope(0.12),
        )
        streams.append(replace(stream, vessel_class=profile))
    scenario = replace(base.scenario, seed=point.traffic_seed)
    return replace(base, streams=tuple(streams), scenario=scenario, label=point.point_id)


def stage1_safety_metric_config() -> PairSafetyMetricConfig:
    return PairSafetyMetricConfig(
        near_miss_margin_nm=0.03,
        time_tolerance_min=0.0001,
        margin_tolerance_nm=0.00001,
        max_subdivisions=20,
    )


def run_stage1_point(
    point: RobustnessStage1Point,
    *,
    future_ingress_clearance: EnvironmentalNetworkSourceFutureIngressConfig | None = None,
    motion_backend=None,
    materialized_config=None,
    bounded_safety_telemetry: bool = False,
) -> dict[str, Any]:
    """Run one deterministic Stage-1 point and return compact engineering evidence.

    ``materialized_config`` is a research-only escape hatch for later staged campaigns
    that hold the Stage-1 geometry/domain point fixed while changing an independently
    frozen execution method (for example V4 coordination or Stage-2 plant fidelity).
    Omitting it preserves the historical Stage-1 builder path exactly.
    """

    config = build_stage1_config(point) if materialized_config is None else materialized_config
    scenario = config.scenario
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=scenario.traffic_horizon_min,
        seed=scenario.seed,
        start_time=scenario.start_time,
    )
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
        safety_config=stage1_safety_metric_config(),
        motion_backend=motion_backend,
        retain_safety_intervals=not bounded_safety_telemetry,
    )
    registry = EncounterRegistry(
        discovery=scenario.discovery,
        lifecycle=scenario.lifecycle,
        profile_catalog=profiles,
    )
    following = SameRouteFollowingSupervisor()
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
        future_ingress_clearance=future_ingress_clearance,
        persistent_same_route_following=following,
    )
    outcome = started.session.run_until_capturing_terminal_outcome(scenario.end_time)
    if bounded_safety_telemetry:
        interval_count = int(getattr(started.session.world, "safety_interval_count", 0))
        safety = (
            started.session.world.finalize_safety_experiment_summary()
            if interval_count
            else None
        )
    else:
        intervals = tuple(getattr(started.session.world, "safety_intervals", ()))
        safety = summarize_mission_safety(intervals) if intervals else None
    queue = source_queue_metrics(outcome.reservoir, at=outcome.terminal_time_min)
    runtime = started.runtime_start.session
    domain_audit = audit_domain_against_hull(point.dimensions, point.safety_domain)
    return {
        "evidence_role": point.evidence_role,
        "point_id": point.point_id,
        "point_sha256": point.identity_sha256,
        "design_sha256": stage1_design_sha256(),
        "config_sha256": config.config_sha256,
        "traffic_seed": point.traffic_seed,
        "traffic_count": traffic.count,
        "motion_backend_id": None if motion_backend is None else motion_backend.backend_id,
        "geometry": {
            "length_m": point.dimensions.length_m,
            "beam_m": point.dimensions.beam_m,
        },
        "domain_policy_id": point.domain_policy_id,
        "domain": asdict(point.safety_domain),
        "domain_hull_audit": asdict(domain_audit),
        "terminal_status": outcome.status.value,
        "terminal_time_min": outcome.terminal_time_min,
        "infeasibility_type": outcome.infeasibility_type,
        "resource_unavailability_type": outcome.resource_unavailability_type,
        "rebootstrap_count": len(runtime.persistent.rebootstrap_results),
        "fleet_observation_count": len(runtime.coordinator.fleet_observations),
        "source": {
            "demand_count": queue.demand_count,
            "admission_count": queue.admission_count,
            "pending_count": queue.pending_count,
            "due_pending_count": queue.due_pending_count,
            "mean_admitted_waiting_time_min": queue.mean_admitted_waiting_time_min,
            "max_admitted_waiting_time_min": queue.max_admitted_waiting_time_min,
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
