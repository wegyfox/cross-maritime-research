"""C17 real-data calibration/replay evidence with explicit data separation.

C17 does not prescribe an optimizer.  It freezes which empirical scenarios may be
used for fitting versus validation/holdout, records fitted parameter provenance, and
provides deterministic replay-fit metrics against recorded trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import atan2, degrees, hypot, isfinite, sqrt
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from vessel_crossing.kinematics import nm_per_minute_to_knots
from vessel_crossing.nextgen.dynamics import normalize_heading, signed_heading_delta
from vessel_crossing.nextgen.recorded_track import RecordedVesselTrack
from vessel_crossing.nextgen.recorded_track_kinematics import (
    RecordedTrackTimeSide,
    recorded_ground_motion_at,
)

from .corpus_lineage import CorpusManifest
from .scope import TestbedFeature


class CalibrationDataRole(str, Enum):
    CALIBRATION = "CALIBRATION"
    VALIDATION = "VALIDATION"
    HOLDOUT = "HOLDOUT"
    REPLAY_ONLY = "REPLAY_ONLY"


class SourceSeparationPolicy(str, Enum):
    SCENARIO_DISJOINT = "SCENARIO_DISJOINT"
    DIRECT_SOURCE_DISJOINT = "DIRECT_SOURCE_DISJOINT"
    ANCESTRY_DISJOINT = "ANCESTRY_DISJOINT"


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("calibration evidence cannot contain NaN/inf")
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    raise TypeError(f"unsupported calibration evidence type: {type(value).__name__}")


def _sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationPartition:
    role: CalibrationDataRole
    scenario_ids: tuple[str, ...]

    @classmethod
    def create(cls, role: CalibrationDataRole, scenario_ids: Iterable[str]) -> "CalibrationPartition":
        ids = tuple(sorted(set(scenario_ids)))
        if not ids:
            raise ValueError("calibration partition must contain at least one scenario")
        if any(not item for item in ids):
            raise ValueError("scenario ids must be non-empty")
        return cls(role=role, scenario_ids=ids)

    def identity_sha256(self) -> str:
        return _sha({"role": self.role.value, "scenario_ids": list(self.scenario_ids)})


@dataclass(frozen=True, slots=True)
class CalibrationProtocol:
    protocol_id: str
    protocol_version: str
    corpus_sha256: str
    partitions: tuple[CalibrationPartition, ...]
    separation_policy: SourceSeparationPolicy
    method_id: str
    method_version: str
    objective_ids: tuple[str, ...]
    random_seed: int | None = None
    component_id: TestbedFeature = TestbedFeature.C17_REAL_DATA_CALIBRATION

    @classmethod
    def create(
        cls,
        *,
        manifest: CorpusManifest,
        protocol_id: str,
        protocol_version: str,
        partitions: Iterable[CalibrationPartition],
        separation_policy: SourceSeparationPolicy = SourceSeparationPolicy.ANCESTRY_DISJOINT,
        method_id: str,
        method_version: str,
        objective_ids: Iterable[str],
        random_seed: int | None = None,
    ) -> "CalibrationProtocol":
        if not protocol_id or not protocol_version or not method_id or not method_version:
            raise ValueError("protocol/method identity fields must be non-empty")
        if random_seed is not None and random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        parts = tuple(sorted(tuple(partitions), key=lambda p: p.role.value))
        roles = [p.role for p in parts]
        if len(roles) != len(set(roles)):
            raise ValueError("each calibration data role may appear at most once")
        if CalibrationDataRole.CALIBRATION not in roles:
            raise ValueError("a calibration partition is required")
        known = {s.scenario_id for s in manifest.scenarios}
        referenced: set[str] = set()
        for part in parts:
            missing = set(part.scenario_ids) - known
            if missing:
                raise ValueError(f"partition references unknown scenarios: {sorted(missing)}")
            overlap = referenced.intersection(part.scenario_ids)
            if overlap:
                raise ValueError(f"scenario appears in more than one partition: {sorted(overlap)}")
            referenced.update(part.scenario_ids)
        _validate_partition_separation(manifest, parts, separation_policy)
        objectives = tuple(sorted(set(objective_ids)))
        if not objectives:
            raise ValueError("objective_ids must be non-empty")
        return cls(
            protocol_id=protocol_id,
            protocol_version=protocol_version,
            corpus_sha256=manifest.identity_sha256(),
            partitions=parts,
            separation_policy=separation_policy,
            method_id=method_id,
            method_version=method_version,
            objective_ids=objectives,
            random_seed=random_seed,
        )

    def partition(self, role: CalibrationDataRole) -> CalibrationPartition:
        for part in self.partitions:
            if part.role is role:
                return part
        raise KeyError(role)

    def identity_sha256(self) -> str:
        return _sha({
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "corpus_sha256": self.corpus_sha256,
            "partitions": [p.identity_sha256() for p in self.partitions],
            "separation_policy": self.separation_policy.value,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "objective_ids": list(self.objective_ids),
            "random_seed": self.random_seed,
        })


def _source_closure(manifest: CorpusManifest, scenario_ids: Iterable[str], *, ancestors: bool) -> set[str]:
    direct: set[str] = set()
    for scenario_id in scenario_ids:
        direct.update(manifest.scenario(scenario_id).source_artifact_ids)
    if not ancestors:
        return direct
    closure = set(direct)
    stack = list(direct)
    while stack:
        source_id = stack.pop()
        for parent in manifest.source(source_id).parent_source_ids:
            if parent not in closure:
                closure.add(parent)
                stack.append(parent)
    return closure


def _validate_partition_separation(
    manifest: CorpusManifest,
    partitions: tuple[CalibrationPartition, ...],
    policy: SourceSeparationPolicy,
) -> None:
    if policy is SourceSeparationPolicy.SCENARIO_DISJOINT:
        return
    closures: dict[CalibrationDataRole, set[str]] = {}
    for part in partitions:
        closures[part.role] = _source_closure(
            manifest, part.scenario_ids,
            ancestors=policy is SourceSeparationPolicy.ANCESTRY_DISJOINT,
        )
    calibration = closures[CalibrationDataRole.CALIBRATION]
    for role, sources in closures.items():
        if role is CalibrationDataRole.CALIBRATION:
            continue
        overlap = calibration.intersection(sources)
        if overlap:
            raise ValueError(
                f"{policy.value} violated between CALIBRATION and {role.value}: shared sources {sorted(overlap)}"
            )


@dataclass(frozen=True, slots=True, order=True)
class ParameterEstimate:
    parameter_name: str
    value: float
    unit: str
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.parameter_name or not self.unit:
            raise ValueError("parameter_name and unit must be non-empty")
        if not isfinite(self.value):
            raise ValueError("parameter value must be finite")
        if self.lower_bound is not None and not isfinite(self.lower_bound):
            raise ValueError("lower_bound must be finite")
        if self.upper_bound is not None and not isfinite(self.upper_bound):
            raise ValueError("upper_bound must be finite")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound cannot exceed upper_bound")
        if self.lower_bound is not None and self.value < self.lower_bound:
            raise ValueError("parameter value below lower_bound")
        if self.upper_bound is not None and self.value > self.upper_bound:
            raise ValueError("parameter value above upper_bound")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    protocol_sha256: str
    model_id: str
    model_version: str
    estimates: tuple[ParameterEstimate, ...]
    objective_values: tuple[tuple[str, float], ...]
    fit_evidence_ids: tuple[str, ...]
    component_id: TestbedFeature = TestbedFeature.C17_REAL_DATA_CALIBRATION

    @classmethod
    def create(
        cls,
        *,
        protocol: CalibrationProtocol,
        model_id: str,
        model_version: str,
        estimates: Iterable[ParameterEstimate],
        objective_values: Mapping[str, float],
        fit_evidence_ids: Iterable[str],
    ) -> "CalibrationResult":
        if not model_id or not model_version:
            raise ValueError("model_id and model_version must be non-empty")
        est = tuple(sorted(tuple(estimates), key=lambda x: x.parameter_name))
        if not est:
            raise ValueError("at least one parameter estimate is required")
        if len({x.parameter_name for x in est}) != len(est):
            raise ValueError("parameter names must be unique")
        objectives = tuple(sorted((str(k), float(v)) for k, v in objective_values.items()))
        if {k for k, _ in objectives} != set(protocol.objective_ids):
            raise ValueError("objective_values must match protocol objective_ids exactly")
        if any(not isfinite(v) for _, v in objectives):
            raise ValueError("objective values must be finite")
        evidence = tuple(sorted(set(fit_evidence_ids)))
        if not evidence:
            raise ValueError("fit_evidence_ids must be non-empty")
        return cls(protocol.identity_sha256(), model_id, model_version, est, objectives, evidence)

    def identity_sha256(self) -> str:
        return _sha({
            "protocol_sha256": self.protocol_sha256,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "estimates": [
                {"name": e.parameter_name, "value": e.value, "unit": e.unit,
                 "lower": e.lower_bound, "upper": e.upper_bound}
                for e in self.estimates
            ],
            "objective_values": [list(v) for v in self.objective_values],
            "fit_evidence_ids": list(self.fit_evidence_ids),
        })


@dataclass(frozen=True, slots=True)
class SimulatedTrackPoint:
    time_min: float
    x_east_nm: float
    y_north_nm: float
    heading_deg: float
    speed_knots: float

    def __post_init__(self) -> None:
        for name in ("time_min", "x_east_nm", "y_north_nm", "heading_deg", "speed_knots"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.speed_knots < 0:
            raise ValueError("speed_knots must be non-negative")
        object.__setattr__(self, "heading_deg", normalize_heading(self.heading_deg))


@dataclass(frozen=True, slots=True)
class ReplayFitMetrics:
    vessel_id: str
    sample_count: int
    position_rmse_nm: float
    position_max_error_nm: float
    heading_rmse_deg: float
    speed_rmse_knots: float
    recorded_track_sha256: str
    simulated_track_sha256: str
    component_id: TestbedFeature = TestbedFeature.C17_REAL_DATA_CALIBRATION

    def identity_sha256(self) -> str:
        return _sha({
            "vessel_id": self.vessel_id,
            "sample_count": self.sample_count,
            "position_rmse_nm": self.position_rmse_nm,
            "position_max_error_nm": self.position_max_error_nm,
            "heading_rmse_deg": self.heading_rmse_deg,
            "speed_rmse_knots": self.speed_rmse_knots,
            "recorded_track_sha256": self.recorded_track_sha256,
            "simulated_track_sha256": self.simulated_track_sha256,
        })


def _recorded_track_identity(track: RecordedVesselTrack) -> str:
    return _sha({
        "vessel_id": track.vessel_id,
        "points": [
            {
                "time": p.time,
                "position": list(p.position),
                "reported_cog_deg": p.reported_course_over_ground_deg,
                "reported_sog_nm_per_min": p.reported_speed_over_ground,
            }
            for p in track.points
        ],
    })


def compare_simulated_track_to_recorded(
    track: RecordedVesselTrack,
    simulated_points: Iterable[SimulatedTrackPoint],
) -> ReplayFitMetrics:
    points = tuple(sorted(tuple(simulated_points), key=lambda p: p.time_min))
    if not points:
        raise ValueError("simulated track must contain at least one point")
    if len({p.time_min for p in points}) != len(points):
        raise ValueError("simulated track times must be unique")
    position_sq: list[float] = []
    position_abs: list[float] = []
    heading_sq: list[float] = []
    speed_sq: list[float] = []
    for point in points:
        recorded_position = track.sample(point.time_min).position
        pos_err = hypot(point.x_east_nm - recorded_position[0], point.y_north_nm - recorded_position[1])
        position_sq.append(pos_err * pos_err)
        position_abs.append(pos_err)
        side = RecordedTrackTimeSide.LEFT if abs(point.time_min - track.end_time) <= 1e-12 else RecordedTrackTimeSide.RIGHT
        motion = recorded_ground_motion_at(track, point.time_min, side=side)
        recorded_heading = motion.course_over_ground_deg
        heading_error = 0.0 if recorded_heading is None else signed_heading_delta(recorded_heading, point.heading_deg)
        heading_sq.append(heading_error * heading_error)
        speed_error = point.speed_knots - nm_per_minute_to_knots(motion.speed_over_ground)
        speed_sq.append(speed_error * speed_error)
    n = len(points)
    sim_sha = _sha([
        [p.time_min, p.x_east_nm, p.y_north_nm, p.heading_deg, p.speed_knots] for p in points
    ])
    return ReplayFitMetrics(
        vessel_id=track.vessel_id,
        sample_count=n,
        position_rmse_nm=sqrt(sum(position_sq) / n),
        position_max_error_nm=max(position_abs),
        heading_rmse_deg=sqrt(sum(heading_sq) / n),
        speed_rmse_knots=sqrt(sum(speed_sq) / n),
        recorded_track_sha256=_recorded_track_identity(track),
        simulated_track_sha256=sim_sha,
    )


class NoOpCalibrationEvaluator:
    component_id = TestbedFeature.C17_REAL_DATA_CALIBRATION

    def compare(self, track: RecordedVesselTrack, simulated_points: Iterable[SimulatedTrackPoint]) -> None:
        return None
