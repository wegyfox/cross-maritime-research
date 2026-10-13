"""R1 P4A headway-validity assumption-control screening.

This module is deliberately an orchestration/measurement layer.  It does not alter
CROSS physics and it never feeds the macro reference headway into microscopic source
admission, guidance, collision avoidance, or vessel behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from vessel_crossing.nextgen import (
    EnvironmentalNetworkFIFOExperimentConfig,
    EnvironmentalNetworkScenarioConfig,
    ManeuverLimits,
    PoissonNetworkStreamSpec,
    Route2D,
    RouteEndAction,
    RouteNetwork,
    RouteNetworkEdge,
    RouteNetworkNode,
    RouteRequest,
    ShipDomain,
    VesselClassProfile,
    VesselDimensions,
    run_environmental_network_fifo_experiment,
)
from vessel_crossing.nextgen.avoidance import AvoidanceCandidateConfig
from vessel_crossing.nextgen.candidate_generation import IntentDrivenCandidatePolicy
from vessel_crossing.nextgen.encounter_registry import EncounterDiscoveryConfig
from vessel_crossing.nextgen.environmental_fleet_guarded_joint_control import EnvironmentalFleetGuardedJointConfig
from vessel_crossing.nextgen.environmental_fleet_guarded_pair_control import EnvironmentalFleetGuardedPairConfig
from vessel_crossing.nextgen.environmental_fleet_multi_owner_recapture import EnvironmentalFleetMultiOwnerRecaptureConfig
from vessel_crossing.nextgen.environmental_joint_control_cycle import EnvironmentalJointControlCycleConfig
from vessel_crossing.nextgen.environmental_joint_execution_session import EnvironmentalJointExecutionSessionConfig
from vessel_crossing.nextgen.environmental_joint_physics import EnvironmentalJointPhysicsConfig
from vessel_crossing.nextgen.environmental_joint_replanning import EnvironmentalJointReplanningConfig
from vessel_crossing.nextgen.environmental_joint_semantics import EnvironmentalJointSemanticConfig
from vessel_crossing.nextgen.environmental_nominal_route_session import EnvironmentalNominalRouteSessionConfig
from vessel_crossing.nextgen.environmental_pair_candidate import EnvironmentalPairCandidateConfig
from vessel_crossing.nextgen.environmental_pair_clearance import EnvironmentalPairTransitionConfig
from vessel_crossing.nextgen.environmental_pair_control_cycle import EnvironmentalPairControlCycleConfig
from vessel_crossing.nextgen.environmental_pair_execution_session import EnvironmentalPairExecutionSessionConfig
from vessel_crossing.nextgen.environmental_pair_replanning import EnvironmentalPairReplanningConfig
from vessel_crossing.nextgen.environmental_route_handoff_session import EnvironmentalRouteHandoffSessionConfig
from vessel_crossing.nextgen.joint_arbiter import JointArbitrationConfig
from vessel_crossing.nextgen.lifecycle import EncounterLifecycleConfig
from vessel_crossing.nextgen.network_traffic import generate_poisson_network_traffic
from vessel_crossing.nextgen.observation import BearingTrendConfig
from vessel_crossing.nextgen.pair_arbiter import PairArbitrationConfig
from vessel_crossing.nextgen.risk import RiskGateConfig
from vessel_crossing.testbed.factor_composition import (
    FactorAxisKind,
    ScenarioFactorAxis,
    ScenarioFactorDesign,
    ScenarioFactorLevel,
    ScenarioFactorPoint,
)
from vessel_crossing.testbed.macro_reference import AnalyticalMacroReferenceModel, MacroReferenceSpec

_EPS = 1e-10
_PROTOCOL_ID = "cross-r1-headway-validity-screening-v1"
_PROTOCOL_VERSION = "1.0.1"
_DESIGN_ID = "cross-r1/screening-v1"
_BASE_SEED = 20260909
_MACRO_HEADWAY_MIN = 2.0
_TRAFFIC_HORIZON_MIN = 90.0
_WARMUP_MIN = 30.0
_MEASUREMENT_END_MIN = 90.0
_EVALUATION_END_MIN = 150.0
_REPLICATIONS = 16


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def build_design() -> ScenarioFactorDesign:
    demand = ScenarioFactorAxis.from_unsorted(
        axis_id="demand_load",
        kind=FactorAxisKind.DEMAND_TRAFFIC_DENSITY,
        levels=(
            ScenarioFactorLevel.from_value("rho030", {"demand_per_hour": 9.0, "rho": 0.30}),
            ScenarioFactorLevel.from_value("rho060", {"demand_per_hour": 18.0, "rho": 0.60}),
            ScenarioFactorLevel.from_value("rho080", {"demand_per_hour": 24.0, "rho": 0.80}),
            ScenarioFactorLevel.from_value("rho110", {"demand_per_hour": 33.0, "rho": 1.10}),
        ),
    )
    geometry = ScenarioFactorAxis.from_unsorted(
        axis_id="geometry",
        kind=FactorAxisKind.SEA_ROOM_CHANNEL_GEOMETRY,
        levels=(
            ScenarioFactorLevel.from_value("straight", {"geometry": "straight"}),
            ScenarioFactorLevel.from_value("turn90", {"geometry": "turn90"}),
        ),
    )
    return ScenarioFactorDesign.full_factorial(
        design_id=_DESIGN_ID,
        design_version="1",
        axes=(demand, geometry),
        base_seed=_BASE_SEED,
        design_note="",
    )


def point_values(design: ScenarioFactorDesign, point: ScenarioFactorPoint) -> tuple[float, float, str]:
    values = design.resolved_values(point)
    demand = values["demand_load"]
    geometry = values["geometry"]
    return float(demand["demand_per_hour"]), float(demand["rho"]), str(geometry["geometry"])


def _limits() -> ManeuverLimits:
    return ManeuverLimits.from_knots_per_minute(
        max_acceleration_knots_per_min=2.0,
        max_deceleration_knots_per_min=2.0,
        max_turn_rate_deg_per_min=30.0,
    )


def _recapture_config() -> EnvironmentalFleetMultiOwnerRecaptureConfig:
    # Generic frozen controller configuration.  No R1 macro headway appears here.
    pair_transition = EnvironmentalPairTransitionConfig(10.0, 0.0)
    pair_cycle = EnvironmentalPairControlCycleConfig(
        execution=EnvironmentalPairExecutionSessionConfig(review_interval_min=0.75),
        replanning=EnvironmentalPairReplanningConfig(
            candidate_policy=IntentDrivenCandidatePolicy(
                AvoidanceCandidateConfig((30.0, 60.0), (0.8, 1.0), 10.0, 0.0)
            ),
            candidate_config=EnvironmentalPairCandidateConfig(pair_transition),
            arbitration_config=PairArbitrationConfig.least_intervention(),
        ),
        route_handoff=EnvironmentalRouteHandoffSessionConfig(
            review_interval_min=0.1,
            recovery_lookahead_nm=0.15,
            cross_track_tolerance_nm=0.015,
            ground_velocity_tolerance=1e-5,
        ),
        nominal_route=EnvironmentalNominalRouteSessionConfig(
            review_interval_min=0.1,
            route_lookahead_nm=0.15,
            cross_track_tolerance_nm=0.015,
        ),
        max_replans=16,
        max_route_events_per_run=4096,
    )
    joint_transition = EnvironmentalPairTransitionConfig(12.0, 0.0)
    joint_cycle = EnvironmentalJointControlCycleConfig(
        execution=EnvironmentalJointExecutionSessionConfig(review_interval_min=1.5),
        replanning=EnvironmentalJointReplanningConfig(
            semantic_config=EnvironmentalJointSemanticConfig(
                candidate_policy=IntentDrivenCandidatePolicy(
                    AvoidanceCandidateConfig(
                        (30.0, 60.0),
                        (1.0,),
                        12.0,
                        0.0,
                        substantial_course_change_deg=10.0,
                        substantial_speed_change_fraction=0.1,
                    )
                ),
                physics=EnvironmentalJointPhysicsConfig(joint_transition, max_combinations=100),
            ),
            arbitration_config=JointArbitrationConfig.least_intervention(),
        ),
        max_replans=16,
    )
    return EnvironmentalFleetMultiOwnerRecaptureConfig(
        pair_cycle=pair_cycle,
        pair_fleet_guard=EnvironmentalFleetGuardedPairConfig(0.75),
        joint_cycle=joint_cycle,
        joint_fleet_guard=EnvironmentalFleetGuardedJointConfig(0.75),
    )


def _network(geometry: str) -> tuple[RouteNetwork, RouteRequest]:
    if geometry == "straight":
        return (
            RouteNetwork(
                (RouteNetworkNode("A", (0.0, 0.0)), RouteNetworkNode("B", (0.0, 0.30))),
                (RouteNetworkEdge("ab", "A", "B", Route2D(((0.0, 0.0), (0.0, 0.30)))),),
            ),
            RouteRequest("A", "B"),
        )
    if geometry == "turn90":
        return (
            RouteNetwork(
                (
                    RouteNetworkNode("A", (0.0, 0.0)),
                    RouteNetworkNode("B", (0.0, 0.20)),
                    RouteNetworkNode("C", (0.20, 0.20)),
                ),
                (
                    RouteNetworkEdge("ab", "A", "B", Route2D(((0.0, 0.0), (0.0, 0.20)))),
                    RouteNetworkEdge("bc", "B", "C", Route2D(((0.0, 0.20), (0.20, 0.20)))),
                ),
            ),
            RouteRequest("A", "C"),
        )
    raise ValueError(f"unsupported R1 P4A geometry {geometry!r}")


def build_micro_config(*, seed: int, demand_per_hour: float, geometry: str) -> EnvironmentalNetworkFIFOExperimentConfig:
    """Build the target micro model from exogenous factors only.

    Deliberately no macro/reference headway parameter is accepted.
    """
    network, request = _network(geometry)
    limits = _limits()
    profile = VesselClassProfile(
        class_id="r1-homogeneous-control",
        dimensions=VesselDimensions.from_meters(length_m=120.0, beam_m=20.0),
        maneuver_limits=limits,
        safety_domain=ShipDomain.circular(0.10),
    )
    stream = PoissonNetworkStreamSpec(
        stream_id="main",
        request=request,
        rate_per_minute=float(demand_per_hour) / 60.0,
        cruise_speed=0.10,
        end_action=RouteEndAction.REMOVE,
        limits=limits,
        vessel_class=profile,
    )
    scenario = EnvironmentalNetworkScenarioConfig(
        seed=int(seed),
        traffic_horizon_min=_TRAFFIC_HORIZON_MIN,
        start_time=0.0,
        end_time=_EVALUATION_END_MIN,
        discovery=EncounterDiscoveryConfig(5.0, 1.0, 60.0),
        lifecycle=EncounterLifecycleConfig(
            risk_gate=RiskGateConfig(0.5, 30.0),
            bearing_trend=BearingTrendConfig(min_duration_min=2.0, appreciable_change_deg=3.0),
            clear_observations_required=2,
            overtaking_clear_distance_nm=0.5,
            max_observations=8,
        ),
        recapture=_recapture_config(),
        default_environmental_drift_velocity=(0.0, 0.0),
    )
    return EnvironmentalNetworkFIFOExperimentConfig(
        network=network,
        streams=(stream,),
        scenario=scenario,
        label=f"r1-p4a-{geometry}-{demand_per_hour:g}ph",
    )


def build_macro_prediction(demand_per_hour: float):
    spec = MacroReferenceSpec.fixed_headway_md1(
        model_id="r1-p4a-independent-fixed-headway",
        model_version="1",
        demand_rate_per_hour=float(demand_per_hour),
        headway_min=_MACRO_HEADWAY_MIN,
        assumptions=(
            "independent declared fixed service headway",
            "stationary M/D/1 queue reference",
            "not calibrated from CROSS microscopic output",
        ),
    )
    return spec, AnalyticalMacroReferenceModel().predict(spec)


def demand_arrival_times(config: EnvironmentalNetworkFIFOExperimentConfig) -> tuple[float, ...]:
    traffic = generate_poisson_network_traffic(
        config.network,
        config.streams,
        horizon=config.scenario.traffic_horizon_min,
        seed=config.scenario.seed,
        start_time=config.scenario.start_time,
    )
    return tuple(traffic.stream("main").arrival_times)


def _backlog_at(demand_times: Iterable[float], admission_times: Iterable[float], at: float) -> int:
    due = sum(1 for item in demand_times if item <= at + _EPS)
    admitted = sum(1 for item in admission_times if item <= at + _EPS)
    return due - admitted


def _measurement_metrics(result: Any) -> dict[str, Any]:
    stream = result.traffic.stream("main")
    demand_times = tuple(float(value) for value in stream.arrival_times)
    admissions = tuple(sorted(result.run.reservoir.admissions, key=lambda item: (item.admission_time, item.vessel_id)))
    admission_by_id = {item.vessel_id: item for item in admissions}
    measurement_admissions = tuple(
        item for item in admissions
        if _WARMUP_MIN - _EPS <= item.admission_time < _MEASUREMENT_END_MIN - _EPS
    )
    measurement_hours = (_MEASUREMENT_END_MIN - _WARMUP_MIN) / 60.0
    throughput = len(measurement_admissions) / measurement_hours

    measurement_demand_ids = tuple(
        vessel_id
        for arrival, vessel_id in zip(stream.arrival_times, stream.vessel_ids, strict=True)
        if _WARMUP_MIN - _EPS <= arrival < _MEASUREMENT_END_MIN - _EPS
    )
    waits: list[float] = []
    censored: list[str] = []
    for vessel_id in measurement_demand_ids:
        admission = admission_by_id.get(vessel_id)
        if admission is None:
            censored.append(vessel_id)
        else:
            waits.append(float(admission.waiting_time_min))
    mean_wait = mean(waits) if waits and not censored else (0.0 if not measurement_demand_ids else None)

    busy_headways: list[float] = []
    for previous, current in zip(admissions, admissions[1:]):
        if not (_WARMUP_MIN - _EPS <= current.admission_time < _MEASUREMENT_END_MIN - _EPS):
            continue
        if current.demand_time <= previous.admission_time + _EPS:
            busy_headways.append(float(current.admission_time - previous.admission_time))
    mean_busy_headway = mean(busy_headways) if busy_headways else None

    admission_times = tuple(float(item.admission_time) for item in admissions)
    backlog_start = _backlog_at(demand_times, admission_times, _WARMUP_MIN)
    backlog_end = _backlog_at(demand_times, admission_times, _MEASUREMENT_END_MIN)
    backlog_growth_per_hour = (backlog_end - backlog_start) / measurement_hours

    return {
        "throughput_per_hour": throughput,
        "mean_source_wait_min": mean_wait,
        "busy_headway_min": mean_busy_headway,
        "busy_headway_count": len(busy_headways),
        "backlog_at_warmup_end": backlog_start,
        "backlog_at_measurement_end": backlog_end,
        "backlog_growth_per_hour": backlog_growth_per_hour,
        "measurement_demand_count": len(measurement_demand_ids),
        "measurement_admission_count": len(measurement_admissions),
        "censored_measurement_demand_count": len(censored),
        "censored_measurement_vessel_ids": list(censored),
    }


def run_replication(*, point: ScenarioFactorPoint, replication_index: int, design: ScenarioFactorDesign | None = None) -> dict[str, Any]:
    design = build_design() if design is None else design
    demand_per_hour, rho, geometry = point_values(design, point)
    seed = design.shared_replication_seed(replication_index)
    micro_config = build_micro_config(seed=seed, demand_per_hour=demand_per_hour, geometry=geometry)
    macro_spec, macro = build_macro_prediction(demand_per_hour)
    result = run_environmental_network_fifo_experiment(micro_config)
    micro = _measurement_metrics(result)

    throughput_rel = (micro["throughput_per_hour"] - macro.predicted_throughput_per_hour) / macro.predicted_throughput_per_hour
    queue_diff = None if micro["mean_source_wait_min"] is None or macro.mean_queue_wait_min is None else micro["mean_source_wait_min"] - macro.mean_queue_wait_min
    headway_diff = None if micro["busy_headway_min"] is None else micro["busy_headway_min"] - macro.mean_headway_min
    backlog_diff = micro["backlog_growth_per_hour"] - macro.backlog_growth_per_hour
    return {
        "format": "cross-r1-p4a-replication",
        "schema_version": 1,
        "protocol_id": _PROTOCOL_ID,
        "protocol_version": _PROTOCOL_VERSION,
        "design_configuration_sha256": design.configuration_sha256(),
        "point_id": point.point_id,
        "point_seed": point.point_seed,
        "replication_index": replication_index,
        "shared_replication_seed": seed,
        "factors": {"demand_per_hour": demand_per_hour, "rho": rho, "geometry": geometry},
        "micro_result_sha256": result.result_sha256,
        "micro_config_sha256": result.summary.config_sha256,
        "macro_spec_sha256": macro_spec.identity_sha256(),
        "macro_prediction_sha256": macro.identity_sha256(),
        "macro": {
            "mean_headway_min": macro.mean_headway_min,
            "capacity_per_hour": macro.capacity_per_hour,
            "utilization": macro.utilization,
            "stable": macro.stable,
            "mean_queue_wait_min": macro.mean_queue_wait_min,
            "predicted_throughput_per_hour": macro.predicted_throughput_per_hour,
            "backlog_growth_per_hour": macro.backlog_growth_per_hour,
        },
        "micro": micro,
        "discrepancy": {
            "throughput_relative_error": throughput_rel,
            "queue_wait_difference_min": queue_diff,
            "busy_headway_difference_min": headway_diff,
            "backlog_growth_difference_per_hour": backlog_diff,
        },
    }


def write_replication(document: Mapping[str, Any], path: str | Path) -> tuple[Path, str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(document)
    target.write_bytes(raw)
    digest = sha256(raw).hexdigest()
    target.with_name(target.name + ".sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return target, digest


_T_CRITICAL_975 = {
    1: 12.7062047364, 2: 4.30265272975, 3: 3.18244630528, 4: 2.7764451052,
    5: 2.57058183564, 6: 2.44691184879, 7: 2.36462425101, 8: 2.3060041352,
    9: 2.26215716285, 10: 2.22813885196, 11: 2.20098516009, 12: 2.17881282966,
    13: 2.16036865646, 14: 2.14478668792, 15: 2.13144954556, 16: 2.11990529922,
    17: 2.10981557783, 18: 2.10092204024, 19: 2.09302405441, 20: 2.08596344727,
    21: 2.07961384473, 22: 2.0738730679, 23: 2.06865761042, 24: 2.06389856163,
    25: 2.05953855275, 26: 2.05552943864, 27: 2.05183051648, 28: 2.0484071418,
    29: 2.04522964213, 30: 2.0422724563,
}


def _summary(values: Iterable[float | None]) -> dict[str, Any]:
    raw = tuple(values)
    valid = tuple(float(item) for item in raw if item is not None)
    missing = len(raw) - len(valid)
    if not valid:
        return {"n": 0, "n_missing": missing, "mean": None, "sd": None, "ci95": None}
    center = mean(valid)
    if len(valid) == 1:
        return {"n": 1, "n_missing": missing, "mean": center, "sd": None, "ci95": None}
    sd = stdev(valid)
    df = len(valid) - 1
    critical = _T_CRITICAL_975.get(df, 1.95996398454)
    half = critical * sd / sqrt(len(valid))
    return {"n": len(valid), "n_missing": missing, "mean": center, "sd": sd, "ci95": [center - half, center + half]}


def aggregate_replications(documents: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    docs = tuple(documents)
    if not docs:
        raise ValueError("at least one replication is required")
    point_ids = {str(item["point_id"]) for item in docs}
    if len(point_ids) != 1:
        raise ValueError("aggregate_replications requires one point_id")
    indices = sorted(int(item["replication_index"]) for item in docs)
    if len(indices) != len(set(indices)):
        raise ValueError("replication_index values must be unique")
    first = docs[0]
    metrics = {
        "micro_throughput_per_hour": _summary(item["micro"]["throughput_per_hour"] for item in docs),
        "micro_mean_source_wait_min": _summary(item["micro"]["mean_source_wait_min"] for item in docs),
        "micro_busy_headway_min": _summary(item["micro"]["busy_headway_min"] for item in docs),
        "micro_backlog_growth_per_hour": _summary(item["micro"]["backlog_growth_per_hour"] for item in docs),
        "throughput_relative_error": _summary(item["discrepancy"]["throughput_relative_error"] for item in docs),
        "queue_wait_difference_min": _summary(item["discrepancy"]["queue_wait_difference_min"] for item in docs),
        "busy_headway_difference_min": _summary(item["discrepancy"]["busy_headway_difference_min"] for item in docs),
        "backlog_growth_difference_per_hour": _summary(item["discrepancy"]["backlog_growth_difference_per_hour"] for item in docs),
    }
    return {
        "format": "cross-r1-p4a-point-aggregate",
        "schema_version": 1,
        "protocol_id": first["protocol_id"],
        "protocol_version": first["protocol_version"],
        "design_configuration_sha256": first["design_configuration_sha256"],
        "point_id": first["point_id"],
        "factors": first["factors"],
        "replication_indices": indices,
        "replication_count": len(docs),
        "macro": first["macro"],
        "metrics": metrics,
        "micro_result_sha256s": [item["micro_result_sha256"] for item in sorted(docs, key=lambda x: int(x["replication_index"]))],
    }


def protocol_constants() -> dict[str, Any]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "protocol_version": _PROTOCOL_VERSION,
        "replications": _REPLICATIONS,
        "macro_headway_min": _MACRO_HEADWAY_MIN,
        "traffic_horizon_min": _TRAFFIC_HORIZON_MIN,
        "warmup_min": _WARMUP_MIN,
        "measurement_end_min": _MEASUREMENT_END_MIN,
        "evaluation_end_min": _EVALUATION_END_MIN,
    }
