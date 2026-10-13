"""Execution layer for frozen replication protocols.

The executor is intentionally separate from the simulation and experiment
harness.  It resolves named model components, executes only the replications
allowed by the frozen precision rule, and emits a deterministic summary package
plus an immutable run manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping

from .archive import environment_manifest, verify_run_package, write_package_index, write_sha256sums
from .control import PolicyController
from .experiments import (
    DemandSweepPointResult,
    PolicyDifferenceSummary,
    PolicyReplicationSummary,
    demand_point_seed_map,
    replication_seeds,
    run_paired_replications,
    summarize_all_paired_policy_differences,
    summarize_replications,
)
from .kinematics import SpeedLimits, knots_per_minute_to_nm_per_minute_squared
from .policies import (
    EncounterObserver,
    FCFSSpeedPolicy,
    SimplifiedRule15SpeedPolicy,
    diploma_rule15_responsibilities,
)
from .policies.base import ControlPolicy
from .protocol import ExecutableModelParameters, ExecutableReplicationProtocol
from .reference_scenarios import build_diploma_reference_geometry_nm
from .safety import CircularSafetyDomain
from .saturation import (
    DemandPointSaturationAssessment,
    SaturationLevel,
    assess_demand_sweep_saturation,
)
from .simulation import SimulationConfig
from .version import __version__
from .zones import AxisAlignedConflictZone


PolicyFactory = Callable[[ExecutableModelParameters], ControlPolicy]
GeometryFactory = Callable[[], object]


class RunStoppingReason(str, Enum):
    PRECISION_TARGET_MET = "precision_target_met"
    MAX_REPLICATIONS_REACHED = "max_replications_reached"


@dataclass(frozen=True, slots=True)
class PrecisionStatus:
    replication_count: int
    target_half_width: float
    half_width_by_point: tuple[tuple[str, float | None], ...]
    all_points_meet_target: bool


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: str
    run_id: str
    run_timestamp_utc: str
    software_version: str
    execution_protocol_id: str
    execution_protocol_fingerprint_sha256: str
    study_protocol_id: str
    study_protocol_fingerprint_sha256: str
    base_seed: int
    demand_point_seeds: tuple[tuple[str, int], ...]
    replication_seeds_by_point: tuple[tuple[str, tuple[int, ...]], ...]
    initial_replications: int
    replication_batch_size: int
    max_replications: int
    target_primary_ci_half_width: float
    final_replications: int
    batch_end_counts: tuple[int, ...]
    stopping_reason: RunStoppingReason
    final_primary_ci_half_width_by_point: tuple[tuple[str, float | None], ...]


@dataclass(frozen=True, slots=True)
class SaturationProfileResult:
    profile_name: str
    policy_name: str
    assessments: tuple[DemandPointSaturationAssessment, ...]


@dataclass(frozen=True, slots=True)
class ProtocolRunResult:
    protocol: ExecutableReplicationProtocol
    manifest: RunManifest
    demand_sweep: tuple[DemandSweepPointResult, ...]
    saturation_profiles: tuple[SaturationProfileResult, ...]
    capacity_proxy_by_policy: tuple[tuple[str, float | None], ...]


def default_policy_registry() -> dict[str, PolicyFactory]:
    return {
        "simplified_rule15_speed": lambda model: SimplifiedRule15SpeedPolicy(
            diploma_rule15_responsibilities(),
            reduced_speed_fraction=model.policy_reduced_speed_fraction,
        ),
        "fcfs_speed": lambda model: FCFSSpeedPolicy(
            reduced_speed_fraction=model.policy_reduced_speed_fraction,
        ),
    }


def default_geometry_registry() -> dict[str, GeometryFactory]:
    return {"diploma_reference_nm": build_diploma_reference_geometry_nm}


def build_execution_components(
    protocol: ExecutableReplicationProtocol,
    *,
    policy_registry: Mapping[str, PolicyFactory] | None = None,
    geometry_registry: Mapping[str, GeometryFactory] | None = None,
):
    """Resolve all named executable components from a frozen protocol."""

    policies = default_policy_registry() if policy_registry is None else dict(policy_registry)
    geometries = default_geometry_registry() if geometry_registry is None else dict(geometry_registry)

    model = protocol.model
    try:
        geometry = geometries[model.geometry_name]()
    except KeyError as exc:
        raise ValueError(f"unknown geometry_name: {model.geometry_name!r}") from exc

    domain = CircularSafetyDomain(
        min_dcpa=model.safety_min_dcpa_nm,
        tcpa_horizon=model.safety_tcpa_horizon_minutes,
    )
    observer = EncounterObserver(domain)

    controllers: list[PolicyController] = []
    for policy_name in (
        protocol.study.baseline_policy,
        protocol.study.comparison_policy,
    ):
        try:
            policy = policies[policy_name](model)
        except KeyError as exc:
            raise ValueError(f"unknown policy name: {policy_name!r}") from exc
        if policy.name != policy_name:
            raise ValueError(
                f"policy registry key {policy_name!r} produced policy named {policy.name!r}"
            )
        controllers.append(PolicyController(observer, policy))

    speed_limits = SpeedLimits(
        knots_per_minute_to_nm_per_minute_squared(
            model.max_acceleration_knots_per_minute
        ),
        knots_per_minute_to_nm_per_minute_squared(
            model.max_deceleration_knots_per_minute
        ),
    )
    zone = AxisAlignedConflictZone(
        model.conflict_zone_x_min_nm,
        model.conflict_zone_x_max_nm,
        model.conflict_zone_y_min_nm,
        model.conflict_zone_y_max_nm,
    )
    simulation_config = SimulationConfig(
        speed_limits=speed_limits,
        min_entry_spacing=model.same_stream_min_spacing_nm,
        conflict_zone=zone,
    )
    return geometry, simulation_config, tuple(controllers)


def _primary_difference_summary(
    point: DemandSweepPointResult,
    protocol: ExecutableReplicationProtocol,
) -> PolicyDifferenceSummary:
    study = protocol.study
    matches = [
        item
        for item in point.paired_differences
        if item.baseline_policy == study.baseline_policy
        and item.comparison_policy == study.comparison_policy
        and item.metric_name == study.primary_metric
    ]
    if len(matches) != 1:
        raise ValueError(
            f"primary paired metric {study.primary_metric!r} must exist exactly once "
            f"at demand point {point.point.name!r}"
        )
    return matches[0]


def primary_ci_half_widths(
    results: tuple[DemandSweepPointResult, ...] | list[DemandSweepPointResult],
    protocol: ExecutableReplicationProtocol,
) -> tuple[tuple[str, float | None], ...]:
    output: list[tuple[str, float | None]] = []
    for result in results:
        summary = _primary_difference_summary(result, protocol).difference
        half_width = None
        if summary.ci_low is not None and summary.ci_high is not None:
            half_width = (summary.ci_high - summary.ci_low) / 2.0
        output.append((result.point.name, half_width))
    return tuple(output)


def precision_status(
    results: tuple[DemandSweepPointResult, ...] | list[DemandSweepPointResult],
    protocol: ExecutableReplicationProtocol,
    *,
    replication_count: int,
) -> PrecisionStatus:
    widths = primary_ci_half_widths(results, protocol)
    target = protocol.study.target_primary_ci_half_width
    all_met = bool(widths) and all(
        width is not None and width <= target for _, width in widths
    )
    return PrecisionStatus(replication_count, target, widths, all_met)


def _point_result(point, point_seed: int, replications) -> DemandSweepPointResult:
    replications = tuple(sorted(replications, key=lambda item: item.replication_index))
    return DemandSweepPointResult(
        point=point,
        point_seed=point_seed,
        replications=replications,
        policy_summaries=summarize_replications(replications),
        paired_differences=summarize_all_paired_policy_differences(replications),
    )


def _capacity_proxy(
    assessments: tuple[DemandPointSaturationAssessment, ...],
) -> float | None:
    eligible = [
        item.point.total_demand_rate_per_minute
        for item in assessments
        if item.level in (SaturationLevel.STABLE, SaturationLevel.STRAINED)
    ]
    return max(eligible) if eligible else None


def _deterministic_run_id(
    protocol_fingerprint: str,
    final_replications: int,
    point_seeds: tuple[tuple[str, int], ...],
) -> str:
    payload = json.dumps(
        {
            "protocol_fingerprint": protocol_fingerprint,
            "final_replications": final_replications,
            "point_seeds": point_seeds,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def finalize_protocol_run(
    protocol: ExecutableReplicationProtocol,
    *,
    latest_results: tuple[DemandSweepPointResult, ...],
    point_seed_map: Mapping[str, int],
    current_count: int,
    batch_end_counts: tuple[int, ...] | list[int],
    stopping_reason: RunStoppingReason,
    run_timestamp_utc: str | None = None,
) -> ProtocolRunResult:
    """Build the common statistical/archive result after replications exist.

    Serial and parallel executors intentionally share this function so
    execution strategy cannot silently change saturation, capacity, seed or
    manifest semantics.
    """

    if current_count <= 0 or not latest_results:
        raise ValueError("cannot finalize an empty protocol run")

    study = protocol.study
    points = study.demand_points()
    by_name = {item.point.name: item for item in latest_results}
    expected_names = tuple(point.name for point in points)
    if tuple(by_name) != expected_names:
        # Reorder is allowed at the call boundary, missing/extra points are not.
        if set(by_name) != set(expected_names):
            raise ValueError("latest_results do not match frozen demand points")
        latest_results = tuple(by_name[name] for name in expected_names)

    for item in latest_results:
        if len(item.replications) != current_count:
            raise ValueError(
                f"demand point {item.point.name!r} has {len(item.replications)} "
                f"replications, expected {current_count}"
            )

    saturation_results: list[SaturationProfileResult] = []
    capacity: list[tuple[str, float | None]] = []
    primary_profile = study.primary_saturation_profile
    for profile in study.saturation_profiles:
        for policy_name in (study.baseline_policy, study.comparison_policy):
            assessments = assess_demand_sweep_saturation(
                latest_results,
                policy_name=policy_name,
                thresholds=profile.thresholds,
            )
            saturation_results.append(
                SaturationProfileResult(profile.name, policy_name, assessments)
            )
            if profile.name == primary_profile:
                capacity.append((policy_name, _capacity_proxy(assessments)))

    ordered_point_seeds = tuple(
        (point.name, int(point_seed_map[point.name])) for point in points
    )
    seed_schedule = tuple(
        (
            point.name,
            replication_seeds(
                base_seed=int(point_seed_map[point.name]),
                count=current_count,
            ),
        )
        for point in points
    )
    widths = primary_ci_half_widths(latest_results, protocol)
    fingerprint = protocol.fingerprint_sha256()
    timestamp = run_timestamp_utc or datetime.now(timezone.utc).isoformat()
    manifest = RunManifest(
        schema_version="1",
        run_id=_deterministic_run_id(fingerprint, current_count, ordered_point_seeds),
        run_timestamp_utc=timestamp,
        software_version=__version__,
        execution_protocol_id=protocol.execution_protocol_id,
        execution_protocol_fingerprint_sha256=fingerprint,
        study_protocol_id=study.protocol_id,
        study_protocol_fingerprint_sha256=study.fingerprint_sha256(),
        base_seed=study.base_seed,
        demand_point_seeds=ordered_point_seeds,
        replication_seeds_by_point=seed_schedule,
        initial_replications=study.initial_replications,
        replication_batch_size=study.replication_batch_size,
        max_replications=study.max_replications,
        target_primary_ci_half_width=study.target_primary_ci_half_width,
        final_replications=current_count,
        batch_end_counts=tuple(batch_end_counts),
        stopping_reason=stopping_reason,
        final_primary_ci_half_width_by_point=widths,
    )
    return ProtocolRunResult(
        protocol=protocol,
        manifest=manifest,
        demand_sweep=latest_results,
        saturation_profiles=tuple(saturation_results),
        capacity_proxy_by_policy=tuple(capacity),
    )


def execute_replication_protocol(
    protocol: ExecutableReplicationProtocol,
    *,
    policy_registry: Mapping[str, PolicyFactory] | None = None,
    geometry_registry: Mapping[str, GeometryFactory] | None = None,
    run_timestamp_utc: str | None = None,
) -> ProtocolRunResult:
    """Execute the frozen global precision rule without result-driven choices."""

    study = protocol.study
    geometry, simulation_config, controllers = build_execution_components(
        protocol,
        policy_registry=policy_registry,
        geometry_registry=geometry_registry,
    )
    points = study.demand_points()
    point_seed_map = demand_point_seed_map(points, base_seed=study.base_seed)
    by_point: dict[str, list] = {point.name: [] for point in points}

    current_count = 0
    batch_end_counts: list[int] = []
    latest_results: tuple[DemandSweepPointResult, ...] = ()
    stopping_reason: RunStoppingReason | None = None

    while stopping_reason is None:
        if current_count == 0:
            target_count = study.initial_replications
        else:
            target_count = min(
                study.max_replications,
                current_count + study.replication_batch_size,
            )
        batch_size = target_count - current_count
        if batch_size <= 0:
            stopping_reason = RunStoppingReason.MAX_REPLICATIONS_REACHED
            break

        for point in points:
            new_reps = run_paired_replications(
                geometry=geometry,
                scenario_config=point.scenario_config,
                base_config=simulation_config,
                controllers=controllers,
                scenario_name=f"{study.protocol_id}:{point.name}",
                base_seed=point_seed_map[point.name],
                replications=batch_size,
                replication_start_index=current_count,
            )
            by_point[point.name].extend(new_reps)

        current_count = target_count
        batch_end_counts.append(current_count)
        latest_results = tuple(
            _point_result(point, point_seed_map[point.name], by_point[point.name])
            for point in points
        )
        status = precision_status(
            latest_results,
            protocol,
            replication_count=current_count,
        )
        if status.all_points_meet_target:
            stopping_reason = RunStoppingReason.PRECISION_TARGET_MET
        elif current_count >= study.max_replications:
            stopping_reason = RunStoppingReason.MAX_REPLICATIONS_REACHED

    if stopping_reason is None or not latest_results:
        raise RuntimeError("executor ended without a completed demand sweep")

    return finalize_protocol_run(
        protocol,
        latest_results=latest_results,
        point_seed_map=point_seed_map,
        current_count=current_count,
        batch_end_counts=batch_end_counts,
        stopping_reason=stopping_reason,
        run_timestamp_utc=run_timestamp_utc,
    )

def _summary_to_dict(summary: PolicyReplicationSummary) -> dict:
    return asdict(summary)


def _difference_to_dict(summary: PolicyDifferenceSummary) -> dict:
    return asdict(summary)


def _saturation_assessment_to_dict(item: DemandPointSaturationAssessment) -> dict:
    return {
        "point": asdict(item.point),
        "level": item.level.value,
        "reasons": [reason.value for reason in item.reasons],
        "congestion_reasons": [reason.value for reason in item.congestion_reasons],
        "relative_throughput_gain_from_previous": (
            item.relative_throughput_gain_from_previous
        ),
    }


def protocol_run_to_dict(result: ProtocolRunResult) -> dict:
    """Deterministic archive representation without bulky vessel trajectories."""

    points = []
    for item in result.demand_sweep:
        points.append(
            {
                "name": item.point.name,
                "point_seed": item.point_seed,
                "total_demand_rate_per_minute": sum(
                    d.rate_per_minute for d in item.point.scenario_config.route_demands
                ),
                "replication_count": len(item.replications),
                "replication_seeds": [rep.seed for rep in item.replications],
                "policy_summaries": [
                    _summary_to_dict(summary) for summary in item.policy_summaries
                ],
                "paired_differences": [
                    _difference_to_dict(summary) for summary in item.paired_differences
                ],
            }
        )

    saturation = [
        {
            "profile_name": item.profile_name,
            "policy_name": item.policy_name,
            "assessments": [
                _saturation_assessment_to_dict(assessment)
                for assessment in item.assessments
            ],
        }
        for item in result.saturation_profiles
    ]
    manifest = asdict(result.manifest)
    manifest["stopping_reason"] = result.manifest.stopping_reason.value
    return {
        "manifest": manifest,
        "capacity_proxy_by_policy": dict(result.capacity_proxy_by_policy),
        "demand_points": points,
        "saturation_profiles": saturation,
    }


def protocol_run_json(result: ProtocolRunResult) -> str:
    return json.dumps(
        protocol_run_to_dict(result),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def write_protocol_run_package(
    result: ProtocolRunResult,
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write a self-contained, checksummed run package to a new/empty directory.

    The two-path return value is retained for compatibility.  The directory also
    receives frozen protocol copies, environment provenance, a package index,
    and SHA256SUMS.
    """

    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("run package directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)

    manifest_path = directory / "run_manifest.json"
    results_path = directory / "summary_results.json"
    study_protocol_path = directory / "study_protocol.json"
    execution_protocol_path = directory / "execution_protocol.json"
    environment_path = directory / "environment.json"

    manifest = asdict(result.manifest)
    manifest["stopping_reason"] = result.manifest.stopping_reason.value
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    results_path.write_text(protocol_run_json(result), encoding="utf-8")
    study_protocol_path.write_text(result.protocol.study.pretty_json(), encoding="utf-8")
    execution_protocol_path.write_text(result.protocol.pretty_json(), encoding="utf-8")
    environment_path.write_text(
        json.dumps(environment_manifest(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )

    content_files = (
        "run_manifest.json",
        "summary_results.json",
        "study_protocol.json",
        "execution_protocol.json",
        "environment.json",
    )
    index_path = write_package_index(
        directory,
        content_files,
        run_id=result.manifest.run_id,
    )
    write_sha256sums(directory, (*content_files, index_path.name))

    verification = verify_run_package(directory)
    if not verification.passed:
        raise RuntimeError(
            "newly written run package failed integrity verification: "
            + "; ".join(verification.errors)
        )
    return manifest_path, results_path
