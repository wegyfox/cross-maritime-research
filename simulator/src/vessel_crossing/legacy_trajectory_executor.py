"""Parallel/checkpoint executor for the frozen legacy trajectory completion study."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import sqrt
from multiprocessing import get_context
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from .archive import environment_manifest, file_integrity, sha256_file, write_package_index, write_sha256sums
from .control import PolicyController
from .experiments import compare_policies, replication_seeds
from .kinematics import SpeedLimits, knots_per_minute_to_nm_per_minute_squared, knots_to_nm_per_minute
from .legacy_trajectory import legacy_trajectory_metrics
from .legacy_trajectory_protocol import LegacyTrajectoryProtocol, load_legacy_trajectory_protocol
from .policies import EncounterObserver, SimplifiedRule15SpeedPolicy, SternPasserYieldSpeedPolicy, diploma_rule15_responsibilities
from .reference_scenarios import build_diploma_reference_geometry_nm
from .safety import CircularSafetyDomain
from .scenarios import PoissonScenarioConfig, RouteDemand, generate_poisson_scenario
from .simulation import SimulationConfig
from .version import __version__
from .zones import AxisAlignedConflictZone


CHECKPOINT_SCHEMA = "legacy-trajectory-checkpoint-v1"
PACKAGE_SCHEMA = "legacy-trajectory-package-v1"


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryPolicyObservation:
    policy_name: str
    crossing_throughput_per_hour: float | None
    maneuvers_per_generated: float | None
    stops_per_generated: float | None
    mean_route_delay_minutes: float | None
    mean_source_waiting_delay_minutes: float | None


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryCompactReplication:
    point_name: str
    total_demand_rate_per_minute: float
    replication_index: int
    seed: int
    generated: int
    policies: tuple[LegacyTrajectoryPolicyObservation, ...]


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryScalarSummary:
    n: int
    mean: float | None
    sample_std: float | None
    standard_error: float | None
    confidence_level: float | None
    ci_low: float | None
    ci_high: float | None


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryExecutionProgress:
    complete: bool
    completed_tasks: int
    total_tasks: int
    checkpoint_path: str
    package_path: str | None = None


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryPackageVerification:
    passed: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryPreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryPreflightReport:
    passed: bool
    protocol_fingerprint: str
    checks: tuple[LegacyTrajectoryPreflightCheck, ...]


def preflight_legacy_trajectory_protocol_file(path: str | Path) -> LegacyTrajectoryPreflightReport:
    path = Path(path)
    protocol = load_legacy_trajectory_protocol(path)
    fingerprint = protocol.fingerprint_sha256()
    checks: list[LegacyTrajectoryPreflightCheck] = []
    sidecar = Path(str(path) + ".fingerprint.sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").strip()
        checks.append(LegacyTrajectoryPreflightCheck("semantic_protocol_fingerprint", expected == fingerprint, f"expected={expected} actual={fingerprint}"))
    else:
        checks.append(LegacyTrajectoryPreflightCheck("semantic_protocol_fingerprint", False, "fingerprint sidecar missing"))

    from .legacy_topology import build_diploma_route_crossing_visits_nm, build_diploma_stop_slots_nm
    visits = build_diploma_route_crossing_visits_nm()
    topology_ok = {name: tuple(item.crossing_id for item in items) for name, items in visits.items()} == {
        "A": ("AD", "AC"), "B": ("BC", "BD"), "C": ("AC", "BC"), "D": ("BD", "AD")
    }
    checks.append(LegacyTrajectoryPreflightCheck("four_crossing_topology", topology_ok, "A/B/C/D ordered crossing sequence"))
    slots = build_diploma_stop_slots_nm(minimum_spacing_nm=protocol.same_stream_min_spacing_nm)
    checks.append(LegacyTrajectoryPreflightCheck("automatic_stop_slots", all(len(value) >= 1 for value in slots.values()), "generated for all four routes"))
    schedule_count = len(protocol.total_demand_rates_per_minute) * protocol.replications
    checks.append(LegacyTrajectoryPreflightCheck("replication_schedule", schedule_count > 0, f"tasks={schedule_count}"))
    try:
        _build_components(protocol)
        component_ok, detail = True, "geometry/controllers/config resolved"
    except Exception as exc:
        component_ok, detail = False, str(exc)
    checks.append(LegacyTrajectoryPreflightCheck("execution_components", component_ok, detail))
    return LegacyTrajectoryPreflightReport(all(item.passed for item in checks), fingerprint, tuple(checks))


def _point_name(rate: float) -> str:
    return f"total_{rate:.3f}_per_min"


def legacy_trajectory_point_seed_map(protocol: LegacyTrajectoryProtocol) -> dict[str, int]:
    names = sorted(_point_name(rate) for rate in protocol.total_demand_rates_per_minute)
    children = np.random.SeedSequence(protocol.base_seed).spawn(len(names))
    return {
        name: int(child.generate_state(1, dtype=np.uint64)[0])
        for name, child in zip(names, children)
    }


def _scalar_summary(values: list[float], confidence_level: float) -> LegacyTrajectoryScalarSummary:
    n = len(values)
    if n == 0:
        return LegacyTrajectoryScalarSummary(0, None, None, None, None, None, None)
    mean = sum(values) / n
    if n < 2:
        return LegacyTrajectoryScalarSummary(n, mean, None, None, None, None, None)
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    sample_std = sqrt(variance)
    standard_error = sample_std / sqrt(n)
    critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, df=n - 1))
    half = critical * standard_error
    return LegacyTrajectoryScalarSummary(
        n, mean, sample_std, standard_error, confidence_level, mean - half, mean + half
    )


def _build_components(protocol: LegacyTrajectoryProtocol):
    geometry = build_diploma_reference_geometry_nm()
    domain = CircularSafetyDomain(
        min_dcpa=protocol.safety_min_dcpa_nm,
        tcpa_horizon=protocol.safety_tcpa_horizon_minutes,
    )
    observer = EncounterObserver(domain)
    controllers = (
        PolicyController(
            observer,
            SimplifiedRule15SpeedPolicy(
                diploma_rule15_responsibilities(),
                reduced_speed_fraction=protocol.reduced_speed_fraction,
            ),
        ),
        PolicyController(
            observer,
            SternPasserYieldSpeedPolicy(
                reduced_speed_fraction=protocol.reduced_speed_fraction,
            ),
        ),
    )
    speed_limits = SpeedLimits(
        knots_per_minute_to_nm_per_minute_squared(protocol.max_acceleration_knots_per_minute),
        knots_per_minute_to_nm_per_minute_squared(protocol.max_deceleration_knots_per_minute),
    )
    config = SimulationConfig(
        speed_limits=speed_limits,
        min_entry_spacing=protocol.same_stream_min_spacing_nm,
        conflict_zone=AxisAlignedConflictZone(
            protocol.conflict_zone_x_min_nm,
            protocol.conflict_zone_x_max_nm,
            protocol.conflict_zone_y_min_nm,
            protocol.conflict_zone_y_max_nm,
        ),
    )
    return geometry, config, controllers


def _run_one_task(
    protocol_dict: dict[str, Any],
    rate: float,
    point_seed: int,
    replication_index: int,
) -> LegacyTrajectoryCompactReplication:
    protocol = LegacyTrajectoryProtocol.from_dict(protocol_dict)
    geometry, base_config, controllers = _build_components(protocol)
    seed = replication_seeds(
        base_seed=point_seed,
        count=1,
        start_index=replication_index,
    )[0]
    scenario_config = PoissonScenarioConfig(
        horizon=protocol.horizon_minutes,
        desired_speed=knots_to_nm_per_minute(protocol.desired_speed_knots),
        route_demands=tuple(
            RouteDemand(route_name, rate / len(protocol.route_names))
            for route_name in protocol.route_names
        ),
    )
    vessels = generate_poisson_scenario(geometry, scenario_config, seed=seed)
    experiments = compare_policies(
        vessels,
        base_config=base_config,
        controllers=controllers,
        scenario_name=f"legacy-trajectory:{_point_name(rate)}",
        seed=seed,
        observation_end=protocol.horizon_minutes,
    )
    observations: list[LegacyTrajectoryPolicyObservation] = []
    generated = len(vessels)
    for experiment in experiments:
        metrics = legacy_trajectory_metrics(experiment)
        observations.append(
            LegacyTrajectoryPolicyObservation(
                policy_name=experiment.policy_name,
                crossing_throughput_per_hour=experiment.aggregate.crossing_throughput_per_hour,
                maneuvers_per_generated=(metrics.maneuver_count / generated if generated else None),
                stops_per_generated=(metrics.stop_command_count / generated if generated else None),
                mean_route_delay_minutes=metrics.mean_route_delay,
                mean_source_waiting_delay_minutes=metrics.mean_source_waiting_delay,
            )
        )
    return LegacyTrajectoryCompactReplication(
        point_name=_point_name(rate),
        total_demand_rate_per_minute=rate,
        replication_index=replication_index,
        seed=seed,
        generated=generated,
        policies=tuple(observations),
    )


def _record_to_dict(record: LegacyTrajectoryCompactReplication) -> dict[str, Any]:
    return {
        "point_name": record.point_name,
        "total_demand_rate_per_minute": record.total_demand_rate_per_minute,
        "replication_index": record.replication_index,
        "seed": record.seed,
        "generated": record.generated,
        "policies": [asdict(item) for item in record.policies],
    }


def _record_from_dict(data: dict[str, Any]) -> LegacyTrajectoryCompactReplication:
    return LegacyTrajectoryCompactReplication(
        point_name=str(data["point_name"]),
        total_demand_rate_per_minute=float(data["total_demand_rate_per_minute"]),
        replication_index=int(data["replication_index"]),
        seed=int(data["seed"]),
        generated=int(data["generated"]),
        policies=tuple(LegacyTrajectoryPolicyObservation(**item) for item in data["policies"]),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _checkpoint_payload(
    protocol: LegacyTrajectoryProtocol,
    records: tuple[LegacyTrajectoryCompactReplication, ...],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint_sha256(),
        "software_version": __version__,
        "records": [_record_to_dict(item) for item in sorted(records, key=lambda r: (r.point_name, r.replication_index))],
    }


def load_legacy_trajectory_checkpoint(
    protocol: LegacyTrajectoryProtocol,
    checkpoint_path: str | Path,
) -> tuple[LegacyTrajectoryCompactReplication, ...]:
    path = Path(checkpoint_path)
    if not path.exists():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported legacy trajectory checkpoint schema")
    if data.get("protocol_id") != protocol.protocol_id or data.get("protocol_fingerprint") != protocol.fingerprint_sha256():
        raise ValueError("checkpoint protocol identity mismatch")
    if data.get("software_version") != __version__:
        raise ValueError("checkpoint software version mismatch")
    records = tuple(_record_from_dict(item) for item in data.get("records", []))
    _validate_record_schedule(protocol, records, allow_partial=True)
    return records


def save_legacy_trajectory_checkpoint(
    protocol: LegacyTrajectoryProtocol,
    checkpoint_path: str | Path,
    records: tuple[LegacyTrajectoryCompactReplication, ...] | list[LegacyTrajectoryCompactReplication],
) -> Path:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = tuple(records)
    _validate_record_schedule(protocol, records, allow_partial=True)
    _atomic_write_json(path, _checkpoint_payload(protocol, records))
    return path


def _validate_record_schedule(
    protocol: LegacyTrajectoryProtocol,
    records: tuple[LegacyTrajectoryCompactReplication, ...],
    *,
    allow_partial: bool,
) -> None:
    rate_by_name = {_point_name(rate): rate for rate in protocol.total_demand_rates_per_minute}
    point_seeds = legacy_trajectory_point_seed_map(protocol)
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (record.point_name, record.replication_index)
        if key in seen:
            raise ValueError("duplicate checkpoint/result replication key")
        seen.add(key)
        if record.point_name not in rate_by_name:
            raise ValueError("record contains unknown demand point")
        if abs(record.total_demand_rate_per_minute - rate_by_name[record.point_name]) > 1e-12:
            raise ValueError("record demand rate mismatch")
        if not 0 <= record.replication_index < protocol.replications:
            raise ValueError("record replication index outside frozen schedule")
        expected_seed = replication_seeds(
            base_seed=point_seeds[record.point_name],
            count=1,
            start_index=record.replication_index,
        )[0]
        if record.seed != expected_seed:
            raise ValueError("record replication seed mismatch")
        if tuple(item.policy_name for item in record.policies) != (protocol.baseline_policy, protocol.comparison_policy):
            raise ValueError("record policy order mismatch")
    expected_count = len(protocol.total_demand_rates_per_minute) * protocol.replications
    if not allow_partial and len(records) != expected_count:
        raise ValueError("completed result does not contain the full frozen replication schedule")


def _missing_tasks(
    protocol: LegacyTrajectoryProtocol,
    records: tuple[LegacyTrajectoryCompactReplication, ...],
) -> list[tuple[float, int, int]]:
    point_seeds = legacy_trajectory_point_seed_map(protocol)
    existing = {(item.point_name, item.replication_index) for item in records}
    tasks: list[tuple[float, int, int]] = []
    # Round-robin by replication index prevents one demand point from being
    # fully exposed/completed long before the others during chunked execution.
    for replication_index in range(protocol.replications):
        for rate in protocol.total_demand_rates_per_minute:
            name = _point_name(rate)
            if (name, replication_index) not in existing:
                tasks.append((rate, point_seeds[name], replication_index))
    return tasks


def summarize_legacy_trajectory_records(
    protocol: LegacyTrajectoryProtocol,
    records: tuple[LegacyTrajectoryCompactReplication, ...] | list[LegacyTrajectoryCompactReplication],
) -> dict[str, Any]:
    records = tuple(records)
    _validate_record_schedule(protocol, records, allow_partial=False)
    output_points: list[dict[str, Any]] = []
    metric_names = protocol.metrics
    for rate in protocol.total_demand_rates_per_minute:
        name = _point_name(rate)
        point_records = sorted((item for item in records if item.point_name == name), key=lambda r: r.replication_index)
        policy_summaries: dict[str, dict[str, Any]] = {}
        for policy_index, policy_name in enumerate((protocol.baseline_policy, protocol.comparison_policy)):
            metric_summaries: dict[str, Any] = {}
            for metric_name in metric_names:
                values = [getattr(item.policies[policy_index], metric_name) for item in point_records]
                numeric = [float(value) for value in values if value is not None]
                metric_summaries[metric_name] = asdict(_scalar_summary(numeric, protocol.confidence_level))
            policy_summaries[policy_name] = metric_summaries

        paired: dict[str, Any] = {}
        for metric_name in metric_names:
            diffs: list[float] = []
            for item in point_records:
                base = getattr(item.policies[0], metric_name)
                comp = getattr(item.policies[1], metric_name)
                if base is not None and comp is not None:
                    diffs.append(float(comp - base))
            paired[metric_name] = asdict(_scalar_summary(diffs, protocol.confidence_level))

        output_points.append(
            {
                "point_name": name,
                "total_demand_rate_per_minute": rate,
                "total_demand_rate_per_hour": rate * 60.0,
                "replication_count": len(point_records),
                "policy_summaries": policy_summaries,
                "paired_difference_comparison_minus_baseline": paired,
            }
        )
    return {"protocol_id": protocol.protocol_id, "protocol_fingerprint": protocol.fingerprint_sha256(), "demand_points": output_points}


def _run_id(protocol: LegacyTrajectoryProtocol, software_version: str | None = None) -> str:
    version = __version__ if software_version is None else software_version
    identity = f"{PACKAGE_SCHEMA}|{protocol.protocol_id}|{protocol.fingerprint_sha256()}|{version}"
    return sha256(identity.encode("utf-8")).hexdigest()[:24]


def execute_legacy_trajectory_protocol_parallel(
    protocol: LegacyTrajectoryProtocol,
    *,
    checkpoint_path: str | Path,
    output_directory: str | Path,
    max_workers: int = 5,
    max_wall_seconds: float | None = None,
) -> LegacyTrajectoryExecutionProgress:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    checkpoint_path = Path(checkpoint_path)
    records = list(load_legacy_trajectory_checkpoint(protocol, checkpoint_path))
    missing = _missing_tasks(protocol, tuple(records))
    total_tasks = len(protocol.total_demand_rates_per_minute) * protocol.replications
    start = time.monotonic()

    if missing:
        protocol_dict = protocol.to_dict()
        # Spawn avoids Python 3.13's fork-from-multithreaded-process warning and
        # gives each replication a clean interpreter state.
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as pool:
            cursor = 0
            while cursor < len(missing):
                batch = missing[cursor : cursor + max_workers]
                futures = [
                    pool.submit(_run_one_task, protocol_dict, rate, point_seed, replication_index)
                    for rate, point_seed, replication_index in batch
                ]
                for future in futures:
                    records.append(future.result())
                records.sort(key=lambda r: (r.point_name, r.replication_index))
                save_legacy_trajectory_checkpoint(protocol, checkpoint_path, records)
                cursor += len(batch)
                if max_wall_seconds is not None and time.monotonic() - start >= max_wall_seconds:
                    break

    complete = len(records) == total_tasks
    package_path = None
    if complete:
        package_path = str(write_legacy_trajectory_package(protocol, tuple(records), output_directory))
    return LegacyTrajectoryExecutionProgress(
        complete=complete,
        completed_tasks=len(records),
        total_tasks=total_tasks,
        checkpoint_path=str(checkpoint_path),
        package_path=package_path,
    )


def write_legacy_trajectory_package(
    protocol: LegacyTrajectoryProtocol,
    records: tuple[LegacyTrajectoryCompactReplication, ...],
    directory: str | Path,
) -> Path:
    _validate_record_schedule(protocol, records, allow_partial=False)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("run package directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)

    run_id = _run_id(protocol)
    point_seeds = legacy_trajectory_point_seed_map(protocol)
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "run_id": run_id,
        "protocol_id": protocol.protocol_id,
        "protocol_fingerprint": protocol.fingerprint_sha256(),
        "software_version": __version__,
        "base_seed": protocol.base_seed,
        "point_seeds": point_seeds,
        "replications_per_point": protocol.replications,
        "total_replications": len(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (directory / "protocol.json").write_text(protocol.pretty_json(), encoding="utf-8")
    (directory / "replication_results.json").write_text(
        json.dumps([_record_to_dict(item) for item in records], sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "summary_results.json").write_text(
        json.dumps(summarize_legacy_trajectory_records(protocol, records), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "environment.json").write_text(
        json.dumps(environment_manifest(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    content = (
        "run_manifest.json",
        "protocol.json",
        "replication_results.json",
        "summary_results.json",
        "environment.json",
    )
    index = write_package_index(directory, content, run_id=run_id, schema_version=PACKAGE_SCHEMA)
    write_sha256sums(directory, (*content, index.name))
    verification = verify_legacy_trajectory_package(directory)
    if not verification.passed:
        raise RuntimeError("new trajectory package failed verification: " + "; ".join(verification.errors))
    return directory


def verify_legacy_trajectory_package(directory: str | Path) -> LegacyTrajectoryPackageVerification:
    directory = Path(directory)
    errors: list[str] = []
    required = (
        "run_manifest.json",
        "protocol.json",
        "replication_results.json",
        "summary_results.json",
        "environment.json",
        "package_index.json",
        "SHA256SUMS",
    )
    for name in required:
        if not (directory / name).exists():
            errors.append(f"missing file: {name}")
    if errors:
        return LegacyTrajectoryPackageVerification(False, tuple(errors))

    try:
        index = json.loads((directory / "package_index.json").read_text())
        for name, expected in index["files"].items():
            integrity = file_integrity(directory / name, relative_to=directory)
            if integrity.sha256 != expected["sha256"] or integrity.size_bytes != expected["size_bytes"]:
                errors.append(f"package index mismatch: {name}")
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            if sha256_file(directory / name.strip()) != digest:
                errors.append(f"SHA256SUMS mismatch: {name.strip()}")

        protocol = LegacyTrajectoryProtocol.from_json((directory / "protocol.json").read_text())
        manifest = json.loads((directory / "run_manifest.json").read_text())
        if manifest.get("protocol_id") != protocol.protocol_id or manifest.get("protocol_fingerprint") != protocol.fingerprint_sha256():
            errors.append("manifest protocol identity mismatch")
        manifest_software_version = manifest.get("software_version")
        if not isinstance(manifest_software_version, str) or not manifest_software_version:
            errors.append("manifest software version missing")
        elif manifest.get("run_id") != _run_id(protocol, manifest_software_version):
            errors.append("deterministic run ID mismatch")
        records = tuple(_record_from_dict(item) for item in json.loads((directory / "replication_results.json").read_text()))
        _validate_record_schedule(protocol, records, allow_partial=False)
        expected_summary = summarize_legacy_trajectory_records(protocol, records)
        actual_summary = json.loads((directory / "summary_results.json").read_text())
        if actual_summary != expected_summary:
            errors.append("summary_results.json does not reconstruct from replication records")
    except Exception as exc:
        errors.append(f"semantic verification failed: {exc}")
    return LegacyTrajectoryPackageVerification(not errors, tuple(errors))


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--protocol", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--workers", type=int, default=5)
    run.add_argument("--max-wall-seconds", type=float, default=None)
    verify = sub.add_parser("verify")
    verify.add_argument("--package", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_legacy_trajectory_package(args.package)
        print(json.dumps(asdict(result), sort_keys=True))
        return 0 if result.passed else 2
    preflight = preflight_legacy_trajectory_protocol_file(args.protocol)
    if not preflight.passed:
        print(json.dumps(asdict(preflight), sort_keys=True))
        return 2
    protocol = load_legacy_trajectory_protocol(args.protocol)
    progress = execute_legacy_trajectory_protocol_parallel(
        protocol,
        checkpoint_path=args.checkpoint,
        output_directory=args.output,
        max_workers=args.workers,
        max_wall_seconds=args.max_wall_seconds,
    )
    print(json.dumps(asdict(progress), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
