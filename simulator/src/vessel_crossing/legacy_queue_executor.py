"""Executor and archive package for the frozen legacy single-server study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .archive import (
    PackageVerification,
    environment_manifest,
    file_integrity,
    sha256_file,
    write_package_index,
    write_sha256sums,
)
from .experiments import replication_seeds
from .legacy_capacity import thesis_priority_table_capacity_per_hour
from .legacy_queue_experiments import (
    QueueObservationMetrics,
    QueueScalarSummary,
    run_legacy_two_stream_pair,
    summarize_queue_scalars,
)
from .legacy_queue_protocol import LegacyQueueReplicationProtocol, load_legacy_queue_protocol
from .version import __version__


LEGACY_QUEUE_PACKAGE_SCHEMA = "1"
KNOWN_FCFS_SCHEDULER = "legacy_single_server_fcfs"
KNOWN_PRIORITY_SCHEDULER = "legacy_single_server_priority_reservation"
KNOWN_PRIMARY_METRIC = "throughput_difference_fcfs_minus_priority_per_hour"
KNOWN_SECONDARY_METRICS = {
    "fcfs_throughput_per_hour",
    "priority_throughput_per_hour",
    "fcfs_mean_waiting_time",
    "priority_mean_waiting_time",
    "fcfs_post_window_clearance_time",
    "priority_post_window_clearance_time",
    "fcfs_completion_fraction_in_window",
    "priority_completion_fraction_in_window",
}


@dataclass(frozen=True, slots=True)
class LegacyQueuePreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class LegacyQueuePreflightReport:
    protocol_id: str
    protocol_fingerprint_sha256: str
    passed: bool
    checks: tuple[LegacyQueuePreflightCheck, ...]


@dataclass(frozen=True, slots=True)
class LegacyQueueCompactReplication:
    demand_rate_per_hour: float
    replication_index: int
    seed: int
    generated: int
    fcfs: QueueObservationMetrics
    priority: QueueObservationMetrics


@dataclass(frozen=True, slots=True)
class LegacyQueuePointSummary:
    demand_rate_per_hour: float
    point_seed: int
    replications: int
    fcfs_throughput_per_hour: QueueScalarSummary
    priority_throughput_per_hour: QueueScalarSummary
    throughput_difference_fcfs_minus_priority_per_hour: QueueScalarSummary
    fcfs_mean_waiting_time: QueueScalarSummary
    priority_mean_waiting_time: QueueScalarSummary
    fcfs_post_window_clearance_time: QueueScalarSummary
    priority_post_window_clearance_time: QueueScalarSummary
    fcfs_completion_fraction_in_window: QueueScalarSummary
    priority_completion_fraction_in_window: QueueScalarSummary
    throughput_difference_fraction_of_priority_mean: float | None
    throughput_difference_fraction_of_fcfs_mean: float | None


@dataclass(frozen=True, slots=True)
class LegacyQueueRunManifest:
    schema_version: str
    run_id: str
    run_timestamp_utc: str
    software_version: str
    protocol_id: str
    protocol_fingerprint_sha256: str
    base_seed: int
    replications_per_demand_point: int
    demand_point_seeds: tuple[tuple[str, int], ...]
    total_replications: int


@dataclass(frozen=True, slots=True)
class LegacyQueueProtocolRunResult:
    protocol: LegacyQueueReplicationProtocol
    manifest: LegacyQueueRunManifest
    deterministic_reference: dict
    summaries: tuple[LegacyQueuePointSummary, ...]
    replications: tuple[LegacyQueueCompactReplication, ...]


def _check(name: str, condition: bool, detail: str) -> LegacyQueuePreflightCheck:
    return LegacyQueuePreflightCheck(name=name, passed=bool(condition), detail=detail)


def preflight_legacy_queue_protocol(
    protocol: LegacyQueueReplicationProtocol,
    *,
    sidecar_path: str | Path | None = None,
) -> LegacyQueuePreflightReport:
    """Fail-fast semantic and fingerprint validation for the frozen queue study."""

    fingerprint = protocol.fingerprint_sha256()
    checks: list[LegacyQueuePreflightCheck] = []
    if sidecar_path is not None:
        path = Path(sidecar_path)
        expected = path.read_text(encoding="utf-8").split()[0] if path.exists() else ""
        checks.append(
            _check(
                "fingerprint_sidecar",
                expected == fingerprint,
                f"expected={expected or '<missing>'}; actual={fingerprint}",
            )
        )
    checks.extend(
        [
            _check(
                "fcfs_scheduler_identity",
                protocol.fcfs_scheduler == KNOWN_FCFS_SCHEDULER,
                protocol.fcfs_scheduler,
            ),
            _check(
                "priority_scheduler_identity",
                protocol.priority_scheduler == KNOWN_PRIORITY_SCHEDULER,
                protocol.priority_scheduler,
            ),
            _check(
                "primary_metric",
                protocol.primary_metric == KNOWN_PRIMARY_METRIC,
                protocol.primary_metric,
            ),
            _check(
                "secondary_metrics",
                set(protocol.secondary_metrics) == KNOWN_SECONDARY_METRICS,
                ",".join(protocol.secondary_metrics),
            ),
            _check(
                "fixed_replication_design",
                protocol.replications >= 2 and protocol.report_all_demand_points,
                f"replications={protocol.replications}; points={len(protocol.demand_rates_per_hour)}",
            ),
            _check(
                "paired_common_random_numbers",
                protocol.paired_common_random_numbers,
                str(protocol.paired_common_random_numbers),
            ),
            _check(
                "both_percentage_denominators",
                protocol.report_both_percentage_denominators,
                str(protocol.report_both_percentage_denominators),
            ),
        ]
    )
    reference = protocol.capacity_reference()
    source_hand_check = (
        abs(reference.fcfs_capacity_per_hour - 2.0) < 1e-12
        and abs(reference.priority_nominal_capacity_per_hour - 1.6) < 1e-12
        and abs(reference.priority_shortfall_fraction_of_fcfs - 0.20) < 1e-12
        and abs(reference.fcfs_advantage_fraction_of_priority - 0.25) < 1e-12
    )
    checks.append(
        _check(
            "source_capacity_hand_check",
            source_hand_check,
            (
                f"fcfs={reference.fcfs_capacity_per_hour}; "
                f"priority={reference.priority_nominal_capacity_per_hour}; "
                f"shortfall={reference.priority_shortfall_fraction_of_fcfs}; "
                f"advantage={reference.fcfs_advantage_fraction_of_priority}"
            ),
        )
    )
    monotone = all(
        right > left
        for left, right in zip(protocol.demand_rates_per_hour, protocol.demand_rates_per_hour[1:])
    )
    checks.append(
        _check(
            "demand_grid",
            bool(protocol.demand_rates_per_hour) and monotone,
            repr(protocol.demand_rates_per_hour),
        )
    )
    return LegacyQueuePreflightReport(
        protocol_id=protocol.protocol_id,
        protocol_fingerprint_sha256=fingerprint,
        passed=all(item.passed for item in checks),
        checks=tuple(checks),
    )


def _point_seed_schedule(protocol: LegacyQueueReplicationProtocol) -> tuple[tuple[str, int], ...]:
    seeds = replication_seeds(base_seed=protocol.base_seed, count=len(protocol.demand_rates_per_hour))
    return tuple((f"{rate:.6g}_per_hour", seed) for rate, seed in zip(protocol.demand_rates_per_hour, seeds))


def _deterministic_run_id(
    protocol: LegacyQueueReplicationProtocol,
    software_version: str | None = None,
) -> str:
    version = __version__ if software_version is None else software_version
    payload = json.dumps(
        {
            "package_schema": LEGACY_QUEUE_PACKAGE_SCHEMA,
            "protocol_fingerprint": protocol.fingerprint_sha256(),
            "software_version": version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:24]


def deterministic_reference_report(protocol: LegacyQueueReplicationProtocol) -> dict:
    reference = protocol.capacity_reference()
    table = [
        {
            "cpa_limit_nm": cpa,
            "speed_knots": protocol.deterministic_table_reference_speed_knots,
            "priority_capacity_per_hour": thesis_priority_table_capacity_per_hour(
                cpa_limit_nm=cpa,
                speed_knots=protocol.deterministic_table_reference_speed_knots,
            ),
        }
        for cpa in protocol.deterministic_table_reference_cpa_limits_nm
    ]
    return {
        "estimand": "deterministic_service_time_capacity_reference",
        "model_reference": asdict(reference),
        "historical_percentage_conventions": {
            "priority_shortfall_fraction_of_fcfs": reference.priority_shortfall_fraction_of_fcfs,
            "fcfs_advantage_fraction_of_priority": reference.fcfs_advantage_fraction_of_priority,
        },
        "source_table_2_reference": table,
    }


def _metric_value(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _summarize_point(
    demand_rate_per_hour: float,
    point_seed: int,
    reps: list[LegacyQueueCompactReplication],
    confidence_level: float,
) -> LegacyQueuePointSummary:
    fcfs_throughput = [item.fcfs.throughput_per_hour for item in reps]
    priority_throughput = [item.priority.throughput_per_hour for item in reps]
    differences = [left - right for left, right in zip(fcfs_throughput, priority_throughput)]
    fcfs_summary = summarize_queue_scalars(fcfs_throughput, confidence_level=confidence_level)
    priority_summary = summarize_queue_scalars(priority_throughput, confidence_level=confidence_level)
    difference_summary = summarize_queue_scalars(differences, confidence_level=confidence_level)
    diff_mean = difference_summary.mean
    priority_mean = priority_summary.mean
    fcfs_mean = fcfs_summary.mean
    return LegacyQueuePointSummary(
        demand_rate_per_hour=demand_rate_per_hour,
        point_seed=point_seed,
        replications=len(reps),
        fcfs_throughput_per_hour=fcfs_summary,
        priority_throughput_per_hour=priority_summary,
        throughput_difference_fcfs_minus_priority_per_hour=difference_summary,
        fcfs_mean_waiting_time=summarize_queue_scalars(
            [_metric_value(item.fcfs.mean_waiting_time) for item in reps],
            confidence_level=confidence_level,
        ),
        priority_mean_waiting_time=summarize_queue_scalars(
            [_metric_value(item.priority.mean_waiting_time) for item in reps],
            confidence_level=confidence_level,
        ),
        fcfs_post_window_clearance_time=summarize_queue_scalars(
            [item.fcfs.post_window_clearance_time for item in reps],
            confidence_level=confidence_level,
        ),
        priority_post_window_clearance_time=summarize_queue_scalars(
            [item.priority.post_window_clearance_time for item in reps],
            confidence_level=confidence_level,
        ),
        fcfs_completion_fraction_in_window=summarize_queue_scalars(
            [_metric_value(item.fcfs.completion_fraction_in_window) for item in reps],
            confidence_level=confidence_level,
        ),
        priority_completion_fraction_in_window=summarize_queue_scalars(
            [_metric_value(item.priority.completion_fraction_in_window) for item in reps],
            confidence_level=confidence_level,
        ),
        throughput_difference_fraction_of_priority_mean=(
            None if diff_mean is None or not priority_mean else diff_mean / priority_mean
        ),
        throughput_difference_fraction_of_fcfs_mean=(
            None if diff_mean is None or not fcfs_mean else diff_mean / fcfs_mean
        ),
    )


def execute_legacy_queue_protocol(
    protocol: LegacyQueueReplicationProtocol,
    *,
    run_timestamp_utc: str | None = None,
    sidecar_path: str | Path | None = None,
) -> LegacyQueueProtocolRunResult:
    """Execute every pre-specified demand point and fixed paired replication."""

    preflight = preflight_legacy_queue_protocol(protocol, sidecar_path=sidecar_path)
    if not preflight.passed:
        failed = [item.name for item in preflight.checks if not item.passed]
        raise ValueError(f"legacy queue protocol preflight failed: {failed}")

    point_schedule = _point_seed_schedule(protocol)
    all_reps: list[LegacyQueueCompactReplication] = []
    summaries: list[LegacyQueuePointSummary] = []
    for demand_rate_per_hour, (_, point_seed) in zip(protocol.demand_rates_per_hour, point_schedule):
        seeds = replication_seeds(base_seed=point_seed, count=protocol.replications)
        point_reps: list[LegacyQueueCompactReplication] = []
        for index, seed in enumerate(seeds):
            pair = run_legacy_two_stream_pair(
                total_rate_per_minute=demand_rate_per_hour / 60.0,
                horizon=protocol.horizon_minutes,
                seed=seed,
                cpa_limit_nm=protocol.cpa_limit_nm,
                speed_knots=protocol.speed_knots,
                privileged_fraction=protocol.privileged_fraction,
                ordinary_multiplier=protocol.ordinary_multiplier,
                privileged_multiplier=protocol.privileged_multiplier,
            )
            record = LegacyQueueCompactReplication(
                demand_rate_per_hour=demand_rate_per_hour,
                replication_index=index,
                seed=seed,
                generated=pair.generated,
                fcfs=pair.fcfs,
                priority=pair.priority,
            )
            point_reps.append(record)
            all_reps.append(record)
        summaries.append(
            _summarize_point(
                demand_rate_per_hour,
                point_seed,
                point_reps,
                protocol.confidence_level,
            )
        )

    timestamp = run_timestamp_utc or datetime.now(timezone.utc).isoformat()
    manifest = LegacyQueueRunManifest(
        schema_version=LEGACY_QUEUE_PACKAGE_SCHEMA,
        run_id=_deterministic_run_id(protocol),
        run_timestamp_utc=timestamp,
        software_version=__version__,
        protocol_id=protocol.protocol_id,
        protocol_fingerprint_sha256=protocol.fingerprint_sha256(),
        base_seed=protocol.base_seed,
        replications_per_demand_point=protocol.replications,
        demand_point_seeds=point_schedule,
        total_replications=len(all_reps),
    )
    return LegacyQueueProtocolRunResult(
        protocol=protocol,
        manifest=manifest,
        deterministic_reference=deterministic_reference_report(protocol),
        summaries=tuple(summaries),
        replications=tuple(all_reps),
    )


def legacy_queue_run_to_dict(result: LegacyQueueProtocolRunResult) -> dict:
    return {
        "manifest": asdict(result.manifest),
        "deterministic_reference": result.deterministic_reference,
        "demand_points": [asdict(item) for item in result.summaries],
    }


def legacy_queue_replications_to_dict(result: LegacyQueueProtocolRunResult) -> dict:
    return {
        "schema_version": LEGACY_QUEUE_PACKAGE_SCHEMA,
        "run_id": result.manifest.run_id,
        "replications": [asdict(item) for item in result.replications],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_legacy_queue_run_package(
    result: LegacyQueueProtocolRunResult,
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write and immediately verify a self-contained queue-study package."""

    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("legacy queue run package directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)

    _write_json(directory / "run_manifest.json", asdict(result.manifest))
    (directory / "protocol.json").write_text(result.protocol.pretty_json(), encoding="utf-8")
    _write_json(directory / "deterministic_reference.json", result.deterministic_reference)
    _write_json(directory / "summary_results.json", legacy_queue_run_to_dict(result))
    _write_json(directory / "replication_results.json", legacy_queue_replications_to_dict(result))
    _write_json(directory / "environment.json", environment_manifest())

    content_files = (
        "run_manifest.json",
        "protocol.json",
        "deterministic_reference.json",
        "summary_results.json",
        "replication_results.json",
        "environment.json",
    )
    index = write_package_index(
        directory,
        content_files,
        run_id=result.manifest.run_id,
        schema_version=LEGACY_QUEUE_PACKAGE_SCHEMA,
    )
    write_sha256sums(directory, (*content_files, index.name))
    verification = verify_legacy_queue_package(directory)
    if not verification.passed:
        raise RuntimeError(
            "newly written legacy queue package failed verification: "
            + "; ".join(verification.errors)
        )
    return directory / "run_manifest.json", directory / "summary_results.json"


def verify_legacy_queue_package(directory: str | Path) -> PackageVerification:
    """Verify queue-study checksums plus protocol/run identity invariants."""

    directory = Path(directory)
    index_path = directory / "package_index.json"
    sums_path = directory / "SHA256SUMS"
    if not index_path.exists():
        return PackageVerification(False, (), ("package_index.json is missing",))
    if not sums_path.exists():
        return PackageVerification(False, (), ("SHA256SUMS is missing",))

    errors: list[str] = []
    checked = []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        indexed_files: Mapping[str, Mapping[str, object]] = index["files"]
    except Exception as exc:
        return PackageVerification(False, (), (f"invalid package_index.json: {exc}",))

    required = {
        "run_manifest.json",
        "protocol.json",
        "deterministic_reference.json",
        "summary_results.json",
        "replication_results.json",
        "environment.json",
    }
    missing = required - set(indexed_files)
    if missing:
        errors.append(f"required files missing from package index: {sorted(missing)}")

    for name, expected in sorted(indexed_files.items()):
        path = directory / name
        if not path.exists():
            errors.append(f"missing indexed file: {name}")
            continue
        integrity = file_integrity(path, relative_to=directory)
        checked.append(integrity)
        if integrity.sha256 != expected.get("sha256"):
            errors.append(f"SHA-256 mismatch: {name}")
        if integrity.size_bytes != expected.get("size_bytes"):
            errors.append(f"size mismatch: {name}")

    try:
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.strip()
            path = directory / name
            if not path.exists():
                errors.append(f"SHA256SUMS references missing file: {name}")
            elif sha256_file(path) != digest:
                errors.append(f"SHA256SUMS mismatch: {name}")
    except Exception as exc:
        errors.append(f"invalid SHA256SUMS: {exc}")

    try:
        protocol = load_legacy_queue_protocol(directory / "protocol.json")
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary_results.json").read_text(encoding="utf-8"))
        raw = json.loads((directory / "replication_results.json").read_text(encoding="utf-8"))
        fingerprint = protocol.fingerprint_sha256()
        if manifest.get("protocol_fingerprint_sha256") != fingerprint:
            errors.append("manifest protocol fingerprint mismatch")
        if manifest.get("protocol_id") != protocol.protocol_id:
            errors.append("manifest protocol id mismatch")
        run_id = manifest.get("run_id")
        manifest_software_version = manifest.get("software_version")
        if not isinstance(manifest_software_version, str) or not manifest_software_version:
            errors.append("manifest software version missing")
        elif run_id != _deterministic_run_id(protocol, manifest_software_version):
            errors.append("deterministic run_id mismatch")
        if index.get("run_id") != run_id:
            errors.append("package index run_id mismatch")
        if summary.get("manifest", {}).get("run_id") != run_id:
            errors.append("summary run_id mismatch")
        if raw.get("run_id") != run_id:
            errors.append("replication results run_id mismatch")
        expected_total = protocol.replications * len(protocol.demand_rates_per_hour)
        if manifest.get("total_replications") != expected_total:
            errors.append("manifest total_replications mismatch")
        records = raw.get("replications", [])
        if len(records) != expected_total:
            errors.append("replication record count mismatch")
        summary_rates = [item.get("demand_rate_per_hour") for item in summary.get("demand_points", [])]
        if summary_rates != list(protocol.demand_rates_per_hour):
            errors.append("summary demand grid mismatch")
        expected_point_seeds = dict(_point_seed_schedule(protocol))
        for rate in protocol.demand_rates_per_hour:
            key = f"{rate:.6g}_per_hour"
            point_seed = expected_point_seeds[key]
            expected_seeds = replication_seeds(base_seed=point_seed, count=protocol.replications)
            point_records = [item for item in records if item.get("demand_rate_per_hour") == rate]
            point_records.sort(key=lambda item: item.get("replication_index", -1))
            if [item.get("replication_index") for item in point_records] != list(range(protocol.replications)):
                errors.append(f"replication index schedule mismatch at demand {rate}")
            if [item.get("seed") for item in point_records] != list(expected_seeds):
                errors.append(f"replication seed schedule mismatch at demand {rate}")
    except Exception as exc:
        errors.append(f"invalid semantic package content: {exc}")

    return PackageVerification(not errors, tuple(checked), tuple(errors))
