"""Non-executable archive verification for A152/A153 FIFO result evidence.

A154 is intentionally separate from the frozen A144 environmental-network archive
surface.  It understands only the A152 FIFO single-run and A153 FIFO paired result
formats.  The archive stores and verifies canonical JSON evidence; it has no JSON-to-
runtime, restore, resume, controller, registry or world reconstruction path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from .environmental_network_fifo_experiment import (
    EnvironmentalNetworkFIFOExperimentResult,
    _FIFO_CONFIG_FORMAT,
    _FIFO_CONFIG_SCHEMA_VERSION,
    _FIFO_RESULT_FORMAT,
    _FIFO_RESULT_SCHEMA_VERSION,
    _FIFO_RUN_DOMAIN,
)
from .environmental_network_fifo_paired_experiment import (
    EnvironmentalNetworkFIFOPairedExperimentResult,
    _FIFO_PAIR_CONFIG_FORMAT,
    _FIFO_PAIR_CONFIG_SCHEMA_VERSION,
    _FIFO_PAIR_DOMAIN,
    _FIFO_PAIR_RESULT_FORMAT,
    _FIFO_PAIR_RESULT_SCHEMA_VERSION,
)
from .environmental_world_builder import (
    EnvironmentalWorldBuilderError,
    validate_environmental_world_builder_identity_document,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EnvironmentalNetworkFIFOResultArchiveError(RuntimeError):
    """Base class for non-executable FIFO result-archive failures."""


class EnvironmentalNetworkFIFOResultArchiveIntegrityError(EnvironmentalNetworkFIFOResultArchiveError):
    """Raised when FIFO result bytes or embedded evidence fail verification."""


class EnvironmentalNetworkFIFOResultArchiveUnsupported(EnvironmentalNetworkFIFOResultArchiveError):
    """Raised for formats/schemas outside the separate A154 verifier surface."""


class EnvironmentalNetworkFIFOResultArchiveKind(str, Enum):
    SINGLE_RUN = "single_run"
    PAIRED = "paired"


_SUPPORTED: Mapping[str, tuple[int, EnvironmentalNetworkFIFOResultArchiveKind]] = MappingProxyType(
    {
        _FIFO_RESULT_FORMAT: (
            _FIFO_RESULT_SCHEMA_VERSION,
            EnvironmentalNetworkFIFOResultArchiveKind.SINGLE_RUN,
        ),
        _FIFO_PAIR_RESULT_FORMAT: (
            _FIFO_PAIR_RESULT_SCHEMA_VERSION,
            EnvironmentalNetworkFIFOResultArchiveKind.PAIRED,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkFIFOResultArchive:
    """Verified immutable A152/A153 JSON evidence, never executable state."""

    kind: EnvironmentalNetworkFIFOResultArchiveKind
    format: str
    schema_version: int
    result_sha256: str
    size_bytes: int
    canonical_bytes: bytes = field(repr=False)
    document: Mapping[str, Any] = field(repr=False)
    source_path: str | None = None
    checksum_verified: bool = False

    def to_json_bytes(self) -> bytes:
        return self.canonical_bytes

    def to_json_object(self) -> dict[str, Any]:
        return _thaw_json(self.document)


_SUPPORTED_RESULT_TYPES = (
    EnvironmentalNetworkFIFOExperimentResult,
    EnvironmentalNetworkFIFOPairedExperimentResult,
)


def inspect_environmental_network_fifo_result_bytes(
    data: bytes | bytearray | memoryview,
    *,
    expected_sha256: str | None = None,
) -> EnvironmentalNetworkFIFOResultArchive:
    """Fail-closed inspect canonical FIFO result bytes without runtime reconstruction."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("archive data must be bytes-like")
    raw = bytes(data)
    digest = sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, field_name="expected_sha256")
        if digest != expected_sha256:
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
                "result SHA-256 does not match expected_sha256"
            )
    document = _load_json_object(raw)
    if _canonical_json(document) != raw:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO result archive is valid JSON but is not the exact canonical encoding"
        )
    kind = _validate_document(document)
    return EnvironmentalNetworkFIFOResultArchive(
        kind=kind,
        format=str(document["format"]),
        schema_version=int(document["schema_version"]),
        result_sha256=digest,
        size_bytes=len(raw),
        canonical_bytes=raw,
        document=_freeze_json(document),
    )


def write_environmental_network_fifo_result_archive(
    path: str | Path,
    result: EnvironmentalNetworkFIFOExperimentResult | EnvironmentalNetworkFIFOPairedExperimentResult,
) -> EnvironmentalNetworkFIFOResultArchive:
    """Atomically write canonical FIFO evidence plus a strict ``.sha256`` sidecar."""

    if not isinstance(result, _SUPPORTED_RESULT_TYPES):
        raise TypeError(
            "result must be EnvironmentalNetworkFIFOExperimentResult or "
            "EnvironmentalNetworkFIFOPairedExperimentResult"
        )
    raw = result.to_json_bytes()
    expected = result.result_sha256
    inspect_environmental_network_fifo_result_bytes(raw, expected_sha256=expected)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(destination, raw)
    _atomic_write_bytes(
        _checksum_path(destination),
        f"{expected}  {destination.name}\n".encode("ascii"),
    )
    return read_environmental_network_fifo_result_archive(destination, require_checksum=True)


def read_environmental_network_fifo_result_archive(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_checksum: bool = True,
) -> EnvironmentalNetworkFIFOResultArchive:
    """Read and verify one A152/A153 non-executable result archive."""

    source = Path(path)
    raw = source.read_bytes()
    checksum_verified = False
    sidecar = _checksum_path(source)
    sidecar_digest: str | None = None
    if sidecar.exists():
        sidecar_digest = _read_checksum_sidecar(sidecar, expected_name=source.name)
        checksum_verified = True
    elif require_checksum:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            f"checksum sidecar is missing: {sidecar.name}"
        )
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, field_name="expected_sha256")
    if expected_sha256 is not None and sidecar_digest is not None and expected_sha256 != sidecar_digest:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "caller expected_sha256 disagrees with checksum sidecar"
        )
    expected = expected_sha256 if expected_sha256 is not None else sidecar_digest
    archive = inspect_environmental_network_fifo_result_bytes(raw, expected_sha256=expected)
    return EnvironmentalNetworkFIFOResultArchive(
        kind=archive.kind,
        format=archive.format,
        schema_version=archive.schema_version,
        result_sha256=archive.result_sha256,
        size_bytes=archive.size_bytes,
        canonical_bytes=archive.canonical_bytes,
        document=archive.document,
        source_path=str(source),
        checksum_verified=checksum_verified,
    )


def verify_environmental_network_fifo_result_archive(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_checksum: bool = True,
) -> EnvironmentalNetworkFIFOResultArchive:
    return read_environmental_network_fifo_result_archive(
        path,
        expected_sha256=expected_sha256,
        require_checksum=require_checksum,
    )


def _validate_document(document: Mapping[str, Any]) -> EnvironmentalNetworkFIFOResultArchiveKind:
    format_name = document.get("format")
    schema = document.get("schema_version")
    if not isinstance(format_name, str):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("result format must be a string")
    if not isinstance(schema, int) or isinstance(schema, bool):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("schema_version must be an integer")
    supported = _SUPPORTED.get(format_name)
    if supported is None:
        raise EnvironmentalNetworkFIFOResultArchiveUnsupported(
            f"unsupported FIFO environmental-network result format {format_name!r}"
        )
    expected_schema, kind = supported
    if schema != expected_schema:
        raise EnvironmentalNetworkFIFOResultArchiveUnsupported(
            f"unsupported schema_version {schema} for {format_name!r}; expected {expected_schema}"
        )
    if kind is EnvironmentalNetworkFIFOResultArchiveKind.SINGLE_RUN:
        _validate_single(document)
    else:
        _validate_pair(document)
    return kind


def _validate_fifo_config(config: Mapping[str, Any]) -> str:
    _require_format(config, _FIFO_CONFIG_FORMAT, _FIFO_CONFIG_SCHEMA_VERSION, context="FIFO config")
    if config.get("admission_policy") != "ARM_LOCAL_FIFO_SHIP_DOMAIN":
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("unsupported FIFO admission_policy")
    if not isinstance(config.get("network"), Mapping):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO config network is missing")
    streams = config.get("streams")
    scenario = config.get("scenario_parameters")
    recheck = config.get("source_recheck")
    if not isinstance(streams, list) or not isinstance(scenario, Mapping) or not isinstance(recheck, Mapping):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO config streams/scenario/source_recheck evidence is invalid"
        )
    if not streams:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO config must contain a stream")
    stream_ids: list[str] = []
    for stream in streams:
        if not isinstance(stream, Mapping) or not isinstance(stream.get("stream_id"), str):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO config stream is invalid")
        stream_ids.append(str(stream["stream_id"]))
        vessel_class = stream.get("vessel_class")
        if not isinstance(vessel_class, Mapping) or not isinstance(vessel_class.get("safety_domain"), Mapping):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
                "FIFO config stream is missing explicit safety-domain evidence"
            )
    if stream_ids != sorted(stream_ids) or len(set(stream_ids)) != len(stream_ids):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO config stream IDs must be unique canonical sorted order"
        )
    return _sha256_json(config)


def _validate_single(document: Mapping[str, Any]) -> None:
    config = _require_mapping(document, "config")
    config_sha = _validate_fifo_config(config)
    if document.get("config_sha256") != config_sha:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO single config_sha256 mismatch")
    summary = _require_mapping(document, "summary")
    if summary.get("config_sha256") != config_sha:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO summary config_sha256 mismatch")
    software_version = summary.get("software_version")
    if not isinstance(software_version, str) or not software_version:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO summary software_version is invalid")
    expected_run = sha256(
        _FIFO_RUN_DOMAIN + software_version.encode("utf-8") + b"\x00" + config_sha.encode("ascii")
    ).hexdigest()
    if document.get("run_id") != expected_run or summary.get("run_id") != expected_run:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO run_id mismatch")
    environment = _require_mapping(document, "execution_environment")
    if environment.get("package_version") != software_version:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO execution environment package version mismatch"
        )

    world_builder = document.get("world_builder")
    if world_builder is not None:
        try:
            validate_environmental_world_builder_identity_document(world_builder)
        except EnvironmentalWorldBuilderError as exc:
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
                f"FIFO world-builder provenance is invalid: {exc}"
            ) from exc

    scenario = _require_mapping(config, "scenario_parameters")
    traffic = _require_mapping(document, "traffic")
    source = _require_mapping(document, "source")
    final_state = _require_mapping(document, "final_state")
    arrivals = traffic.get("arrivals")
    traffic_streams = traffic.get("streams")
    demands = source.get("demands")
    admissions = source.get("admissions")
    audits = source.get("ingress_audits")
    plans = source.get("plans")
    boundaries = source.get("boundaries")
    metrics = _require_mapping(source, "final_queue_metrics")
    if not all(isinstance(value, list) for value in (arrivals, traffic_streams, demands, admissions, audits, plans, boundaries)):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO list evidence is malformed")

    _require_equal(summary, "seed", traffic.get("seed"), message="FIFO traffic seed mismatch")
    _require_equal(summary, "start_time", traffic.get("start_time"), message="FIFO traffic start_time mismatch")
    _require_equal(summary, "traffic_horizon_min", traffic.get("horizon"), message="FIFO traffic horizon mismatch")
    _require_equal(summary, "seed", scenario.get("seed"), message="FIFO config/summary seed mismatch")
    _require_equal(summary, "start_time", scenario.get("start_time"), message="FIFO config/summary start_time mismatch")
    _require_equal(summary, "traffic_horizon_min", scenario.get("traffic_horizon_min"), message="FIFO config/summary horizon mismatch")
    _require_equal(summary, "end_time", scenario.get("end_time"), message="FIFO config/summary end_time mismatch")

    demand_count = len(demands)
    admission_count = len(admissions)
    if len(arrivals) != demand_count or summary.get("realized_demand_count") != demand_count:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO realized demand count mismatch")
    if metrics.get("demand_count") != demand_count:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO queue demand_count mismatch")
    if metrics.get("admission_count") != admission_count or summary.get("physical_admission_count") != admission_count:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO admission count mismatch")
    pending = demand_count - admission_count
    if metrics.get("pending_count") != pending or summary.get("pending_source_count") != pending:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO pending queue count mismatch")
    due_pending = metrics.get("due_pending_count")
    if not isinstance(due_pending, int) or isinstance(due_pending, bool) or not (0 <= due_pending <= pending):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO due-pending count is invalid")
    if summary.get("due_pending_source_count") != due_pending:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO summary due-pending count mismatch")

    demand_keys: list[list[Any]] = []
    demand_by_id: dict[str, Mapping[str, Any]] = {}
    for item in demands:
        if not isinstance(item, Mapping):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO demand entry must be an object")
        stream_id = item.get("stream_id")
        vessel_id = item.get("nominal_vessel_id")
        demand_time = item.get("demand_time")
        if not isinstance(stream_id, str) or not isinstance(vessel_id, str) or not _is_number(demand_time):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO demand identity is invalid")
        if vessel_id in demand_by_id:
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO duplicate demand vessel_id")
        demand_by_id[vessel_id] = item
        demand_keys.append([stream_id, vessel_id, demand_time])
    if metrics.get("demand_keys") != demand_keys:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO queue demand_keys mismatch")
    _validate_traffic_demand_alignment(traffic_streams, arrivals, demand_keys)

    admission_ids: list[str] = []
    waits: list[float] = []
    for item in admissions:
        if not isinstance(item, Mapping):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO admission entry must be an object")
        vessel_id = item.get("vessel_id")
        stream_id = item.get("stream_id")
        demand_time = item.get("demand_time")
        admission_time = item.get("admission_time")
        if not isinstance(vessel_id, str) or vessel_id not in demand_by_id:
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO admission references unknown demand")
        demand = demand_by_id[vessel_id]
        if stream_id != demand.get("stream_id") or demand_time != demand.get("demand_time"):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO admission demand identity mismatch")
        if not _is_number(admission_time) or admission_time < demand_time:
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO admission time precedes demand")
        admission_ids.append(vessel_id)
        waits.append(float(admission_time) - float(demand_time))
    if len(set(admission_ids)) != len(admission_ids):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO duplicate physical admission")

    audit_ids: list[str] = []
    for audit in audits:
        if not isinstance(audit, Mapping):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO ingress audit must be an object")
        record = audit.get("record")
        if not isinstance(record, Mapping):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO ingress audit record is missing")
        vessel = record.get("vessel")
        if not isinstance(vessel, Mapping) or not isinstance(vessel.get("vessel_id"), str):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO ingress audit vessel identity is invalid")
        audit_ids.append(str(vessel["vessel_id"]))
    if audit_ids != admission_ids:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO ingress audit ledger does not match physical admissions"
        )

    mean_wait = None if not waits else sum(waits) / len(waits)
    max_wait = None if not waits else max(waits)
    _require_float_optional(metrics.get("mean_admitted_waiting_time_min"), mean_wait, "FIFO mean waiting mismatch")
    _require_float_optional(metrics.get("max_admitted_waiting_time_min"), max_wait, "FIFO max waiting mismatch")
    _require_float_optional(summary.get("mean_admitted_waiting_time_min"), mean_wait, "FIFO summary mean waiting mismatch")
    _require_float_optional(summary.get("max_admitted_waiting_time_min"), max_wait, "FIFO summary max waiting mismatch")

    final_generation = source.get("final_generation")
    if final_generation != summary.get("final_source_generation"):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO final source generation mismatch")
    if summary.get("source_plan_count") != len(plans) or summary.get("source_boundary_count") != len(boundaries):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO plan/boundary count mismatch")

    snapshot = _require_mapping(final_state, "snapshot")
    commands = final_state.get("commands")
    vessels = snapshot.get("vessels")
    if not isinstance(commands, list) or not isinstance(vessels, list):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO terminal snapshot/commands malformed")
    if _sha256_json(final_state) != summary.get("final_snapshot_sha256"):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO final-state SHA-256 mismatch")
    if snapshot.get("time") != summary.get("final_time") or metrics.get("evaluated_at") != summary.get("final_time"):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO terminal/evaluation time mismatch")
    vessel_ids = [v.get("vessel_id") for v in vessels if isinstance(v, Mapping)]
    command_ids = [c[0] for c in commands if isinstance(c, list) and len(c) == 2]
    if len(vessel_ids) != len(vessels) or len(command_ids) != len(commands) or command_ids != vessel_ids:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO final commands do not match final snapshot vessel order"
        )
    if summary.get("final_vessel_count") != len(vessel_ids) or summary.get("final_vessel_ids") != vessel_ids:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO final vessel summary mismatch")


def _validate_traffic_demand_alignment(
    traffic_streams: list[Any], arrivals: list[Any], demand_keys: list[list[Any]]
) -> None:
    vessel_to_stream: dict[str, tuple[str, float]] = {}
    for stream in traffic_streams:
        if not isinstance(stream, Mapping):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO traffic stream must be an object")
        stream_id = stream.get("stream_id")
        ids = stream.get("vessel_ids")
        times = stream.get("arrival_times")
        if not isinstance(stream_id, str) or not isinstance(ids, list) or not isinstance(times, list) or len(ids) != len(times):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO traffic stream identity evidence is invalid")
        for vessel_id, at in zip(ids, times):
            if not isinstance(vessel_id, str) or not _is_number(at) or vessel_id in vessel_to_stream:
                raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO traffic stream vessel evidence is invalid")
            vessel_to_stream[vessel_id] = (stream_id, float(at))
    realized_keys: list[list[Any]] = []
    for arrival in arrivals:
        if not isinstance(arrival, Mapping) or not _is_number(arrival.get("at")):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO traffic arrival is invalid")
        vessel = arrival.get("vessel")
        if not isinstance(vessel, Mapping) or not isinstance(vessel.get("vessel_id"), str):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO traffic arrival vessel identity is invalid")
        vessel_id = str(vessel["vessel_id"])
        evidence = vessel_to_stream.get(vessel_id)
        if evidence is None or evidence[1] != float(arrival["at"]):
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO traffic stream/arrival alignment mismatch")
        realized_keys.append([evidence[0], vessel_id, arrival["at"]])
    if realized_keys != demand_keys:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO immutable traffic realization differs from source demand ordering"
        )


def _shared_basis_from_fifo_config(config: Mapping[str, Any]) -> dict[str, Any]:
    scenario = _require_mapping(config, "scenario_parameters")
    return {
        "network": config.get("network"),
        "streams": config.get("streams"),
        "seed": scenario.get("seed"),
        "start_time": scenario.get("start_time"),
        "traffic_horizon_min": scenario.get("traffic_horizon_min"),
        "end_time": scenario.get("end_time"),
        "default_environmental_drift_velocity": scenario.get("default_environmental_drift_velocity"),
        "stream_environmental_drift_velocity": scenario.get("stream_environmental_drift_velocity"),
        "admission_policy": config.get("admission_policy"),
        "source_recheck": config.get("source_recheck"),
    }


def _validate_pair(document: Mapping[str, Any]) -> None:
    config = _require_mapping(document, "config")
    _require_format(
        config,
        _FIFO_PAIR_CONFIG_FORMAT,
        _FIFO_PAIR_CONFIG_SCHEMA_VERSION,
        context="FIFO paired config",
    )
    pair_config_sha = _sha256_json(config)
    if document.get("config_sha256") != pair_config_sha:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO pair config_sha256 mismatch")
    baseline_config = _require_mapping(config, "baseline")
    control_config = _require_mapping(config, "control")
    _validate_fifo_config(baseline_config)
    _validate_fifo_config(control_config)
    shared_basis = _require_mapping(config, "shared_exogenous_basis")
    shared_sha = _sha256_json(shared_basis)
    if config.get("shared_basis_sha256") != shared_sha or document.get("shared_basis_sha256") != shared_sha:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO shared-basis SHA-256 mismatch")
    if _shared_basis_from_fifo_config(baseline_config) != dict(shared_basis):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO baseline does not match shared exogenous basis")
    if _shared_basis_from_fifo_config(control_config) != dict(shared_basis):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO control does not match shared exogenous basis")

    baseline = _require_mapping(document, "baseline")
    control = _require_mapping(document, "control")
    _require_format(baseline, _FIFO_RESULT_FORMAT, _FIFO_RESULT_SCHEMA_VERSION, context="FIFO baseline result")
    _require_format(control, _FIFO_RESULT_FORMAT, _FIFO_RESULT_SCHEMA_VERSION, context="FIFO control result")
    _validate_single(baseline)
    _validate_single(control)
    if baseline.get("config") != baseline_config or control.get("config") != control_config:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO paired nested result config mismatch")
    shared_traffic = document.get("shared_traffic")
    if not isinstance(shared_traffic, Mapping):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO paired shared traffic is missing")
    if baseline.get("traffic") != shared_traffic or control.get("traffic") != shared_traffic:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO paired arms do not contain the exact archived shared traffic realization"
        )
    baseline_summary = _require_mapping(baseline, "summary")
    control_summary = _require_mapping(control, "summary")
    version = baseline_summary.get("software_version")
    if not isinstance(version, str) or control_summary.get("software_version") != version:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO paired arm software versions mismatch")
    environment = _require_mapping(document, "execution_environment")
    if environment.get("package_version") != version:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO paired execution environment package version mismatch"
        )
    pair_id = sha256(
        _FIFO_PAIR_DOMAIN + version.encode("utf-8") + b"\x00" + pair_config_sha.encode("ascii")
    ).hexdigest()
    if document.get("pair_id") != pair_id:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO pair_id mismatch")
    _validate_pair_difference(
        _require_mapping(document, "difference"), baseline, control
    )


def _validate_pair_difference(
    difference: Mapping[str, Any], baseline: Mapping[str, Any], control: Mapping[str, Any]
) -> None:
    b_summary = _require_mapping(baseline, "summary")
    c_summary = _require_mapping(control, "summary")
    b_source = _require_mapping(baseline, "source")
    c_source = _require_mapping(control, "source")
    b_metrics = _require_mapping(b_source, "final_queue_metrics")
    c_metrics = _require_mapping(c_source, "final_queue_metrics")
    source_diff = _require_mapping(difference, "source_queue")
    for name in ("admission_count", "pending_count", "due_pending_count"):
        expected = c_metrics.get(name) - b_metrics.get(name)
        if source_diff.get(name + "_delta") != expected:
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
                f"FIFO paired source_queue {name}_delta mismatch"
            )
    for name in (
        "mean_admitted_waiting_time_min",
        "max_admitted_waiting_time_min",
        "mean_due_waiting_time_min",
        "max_due_waiting_time_min",
    ):
        expected = _optional_delta(b_metrics.get(name), c_metrics.get(name))
        _require_float_optional(
            source_diff.get(name + "_delta"), expected, f"FIFO paired {name}_delta mismatch"
        )
    scalar_pairs = {
        "final_vessel_count_delta": "final_vessel_count",
        "fleet_observation_count_delta": "fleet_observation_count",
        "source_plan_count_delta": "source_plan_count",
        "source_boundary_count_delta": "source_boundary_count",
        "final_source_generation_delta": "final_source_generation",
    }
    for delta_name, field_name in scalar_pairs.items():
        expected = c_summary.get(field_name) - b_summary.get(field_name)
        if difference.get(delta_name) != expected:
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
                f"FIFO paired {delta_name} mismatch"
            )


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    if not _is_number(left) or not _is_number(right):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO paired metric is not numeric")
    return float(right) - float(left)


def _require_format(document: Mapping[str, Any], expected_format: str, expected_schema: int, *, context: str) -> None:
    if document.get("format") != expected_format or document.get("schema_version") != expected_schema:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(f"{context} format/schema mismatch")


def _require_mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(f"{key} must be a JSON object")
    return value


def _require_equal(mapping: Mapping[str, Any], key: str, expected: Any, *, message: str) -> None:
    if mapping.get(key) != expected:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_float_optional(actual: Any, expected: float | None, message: str) -> None:
    if expected is None:
        if actual is not None:
            raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(message)
        return
    if not _is_number(actual) or abs(float(actual) - expected) > 1e-12:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(message)


def _load_json_object(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO archive is not valid UTF-8") from exc

    def object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
                    f"duplicate JSON object key {key!r}"
                )
            result[key] = value
        return result

    def parse_constant(value: str) -> Any:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            f"non-finite JSON numeric token {value!r} is forbidden"
        )

    try:
        value = json.loads(text, object_pairs_hook=object_pairs_hook, parse_constant=parse_constant)
    except EnvironmentalNetworkFIFOResultArchiveIntegrityError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError("FIFO archive is not valid JSON") from exc
    if not isinstance(value, dict):
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO archive top level must be a JSON object"
        )
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "FIFO archive contains non-canonical JSON values"
        ) from exc


def _sha256_json(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 hex digest")


def _checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _read_checksum_sidecar(path: Path, *, expected_name: str) -> str:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "checksum sidecar is not readable ASCII"
        ) from exc
    lines = text.splitlines()
    if len(lines) != 1:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "checksum sidecar must contain exactly one line"
        )
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", lines[0])
    if match is None:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "checksum sidecar must use '<sha256>  <filename>' format"
        )
    digest, name = match.groups()
    if name != expected_name:
        raise EnvironmentalNetworkFIFOResultArchiveIntegrityError(
            "checksum sidecar filename does not match archive filename"
        )
    return digest


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(descriptor)
