"""Non-executable archive I/O for canonical environmental-network research results.

A144 deliberately stops at immutable evidence.  It can write, read and verify the
canonical JSON produced by the A142 single-run and A143 comparative result surfaces,
but it cannot reconstruct an ``EnvironmentalWorld2D``, runtime session, controller,
registry or command authority from JSON.
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

from .environmental_network_experiment import (
    EnvironmentalNetworkExperimentResult,
    _EXPERIMENT_FORMAT,
    _EXPERIMENT_SCHEMA_VERSION,
    _RESULT_FORMAT,
    _RESULT_SCHEMA_VERSION,
    _RUN_DOMAIN,
)
from .environmental_network_paired_experiment import (
    EnvironmentalNetworkPairedExperimentResult,
    _PAIR_CONFIG_FORMAT,
    _PAIR_CONFIG_SCHEMA_VERSION,
    _PAIR_DOMAIN,
    _PAIR_RESULT_FORMAT,
    _PAIR_RESULT_SCHEMA_VERSION,
)
from .environmental_network_stress_campaign import (
    EnvironmentalNetworkStressCampaignResult,
    _CAMPAIGN_DOMAIN,
    _CAMPAIGN_FORMAT,
    _CAMPAIGN_RESULT_FORMAT,
    _CAMPAIGN_RESULT_SCHEMA_VERSION,
    _CAMPAIGN_SCHEMA_VERSION,
    _CASE_DOMAIN,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EnvironmentalNetworkResultArchiveError(RuntimeError):
    """Base class for non-executable result-archive failures."""


class EnvironmentalNetworkResultArchiveIntegrityError(EnvironmentalNetworkResultArchiveError):
    """Raised when bytes, hashes or embedded provenance do not verify."""


class EnvironmentalNetworkResultArchiveUnsupported(EnvironmentalNetworkResultArchiveError):
    """Raised when an archive format/schema is not part of the frozen A144 surface."""


class EnvironmentalNetworkResultArchiveKind(str, Enum):
    SINGLE_RUN = "single_run"
    PAIRED = "paired"
    STRESS_CAMPAIGN = "stress_campaign"


_SUPPORTED_RESULT_SCHEMAS: Mapping[str, tuple[int, EnvironmentalNetworkResultArchiveKind]] = MappingProxyType(
    {
        _RESULT_FORMAT: (_RESULT_SCHEMA_VERSION, EnvironmentalNetworkResultArchiveKind.SINGLE_RUN),
        _PAIR_RESULT_FORMAT: (
            _PAIR_RESULT_SCHEMA_VERSION,
            EnvironmentalNetworkResultArchiveKind.PAIRED,
        ),
        _CAMPAIGN_RESULT_FORMAT: (
            _CAMPAIGN_RESULT_SCHEMA_VERSION,
            EnvironmentalNetworkResultArchiveKind.STRESS_CAMPAIGN,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class EnvironmentalNetworkResultArchive:
    """Verified immutable JSON evidence; never an executable simulator state."""

    kind: EnvironmentalNetworkResultArchiveKind
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
    EnvironmentalNetworkExperimentResult,
    EnvironmentalNetworkPairedExperimentResult,
    EnvironmentalNetworkStressCampaignResult,
)


def write_environmental_network_result_archive(
    path: str | Path,
    result: EnvironmentalNetworkExperimentResult
    | EnvironmentalNetworkPairedExperimentResult
    | EnvironmentalNetworkStressCampaignResult,
) -> EnvironmentalNetworkResultArchive:
    """Atomically publish canonical result JSON plus a strict ``.sha256`` sidecar."""

    if not isinstance(result, _SUPPORTED_RESULT_TYPES):
        raise TypeError(
            "result must be EnvironmentalNetworkExperimentResult, "
            "EnvironmentalNetworkPairedExperimentResult, or EnvironmentalNetworkStressCampaignResult"
        )
    raw = result.to_json_bytes()
    expected = result.result_sha256
    verified = inspect_environmental_network_result_bytes(raw, expected_sha256=expected)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(destination, raw)
    checksum = f"{expected}  {destination.name}\n".encode("ascii")
    _atomic_write_bytes(_checksum_path(destination), checksum)
    return read_environmental_network_result_archive(destination, require_checksum=True)


def inspect_environmental_network_result_bytes(
    data: bytes | bytearray | memoryview,
    *,
    expected_sha256: str | None = None,
) -> EnvironmentalNetworkResultArchive:
    """Validate canonical result bytes without reading or constructing runtime state."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("archive data must be bytes-like")
    raw = bytes(data)
    digest = sha256(raw).hexdigest()
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, field_name="expected_sha256")
        if digest != expected_sha256:
            raise EnvironmentalNetworkResultArchiveIntegrityError(
                "result SHA-256 does not match expected_sha256"
            )

    document = _load_json_object(raw)
    canonical = _canonical_json(document)
    if canonical != raw:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "result archive is valid JSON but is not the exact canonical encoding"
        )
    kind = _validate_result_document(document)
    return EnvironmentalNetworkResultArchive(
        kind=kind,
        format=str(document["format"]),
        schema_version=int(document["schema_version"]),
        result_sha256=digest,
        size_bytes=len(raw),
        canonical_bytes=raw,
        document=_freeze_json(document),
    )


def read_environmental_network_result_archive(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_checksum: bool = True,
) -> EnvironmentalNetworkResultArchive:
    """Read and fail-closed verify one non-executable result archive."""

    source = Path(path)
    raw = source.read_bytes()
    checksum_verified = False
    sidecar = _checksum_path(source)
    sidecar_digest: str | None = None
    if sidecar.exists():
        sidecar_digest = _read_checksum_sidecar(sidecar, expected_name=source.name)
        checksum_verified = True
    elif require_checksum:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            f"checksum sidecar is missing: {sidecar.name}"
        )

    if expected_sha256 is not None and sidecar_digest is not None and expected_sha256 != sidecar_digest:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "caller expected_sha256 disagrees with checksum sidecar"
        )
    expected = expected_sha256 if expected_sha256 is not None else sidecar_digest
    archive = inspect_environmental_network_result_bytes(raw, expected_sha256=expected)
    return EnvironmentalNetworkResultArchive(
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


def verify_environmental_network_result_archive(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_checksum: bool = True,
) -> EnvironmentalNetworkResultArchive:
    """Fail closed on any byte, canonicalization, schema or embedded-evidence mismatch."""

    return read_environmental_network_result_archive(
        path,
        expected_sha256=expected_sha256,
        require_checksum=require_checksum,
    )


def _validate_result_document(document: dict[str, Any]) -> EnvironmentalNetworkResultArchiveKind:
    format_name = document.get("format")
    schema_version = document.get("schema_version")
    if not isinstance(format_name, str):
        raise EnvironmentalNetworkResultArchiveIntegrityError("result format must be a string")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise EnvironmentalNetworkResultArchiveIntegrityError("result schema_version must be an integer")
    supported = _SUPPORTED_RESULT_SCHEMAS.get(format_name)
    if supported is None:
        raise EnvironmentalNetworkResultArchiveUnsupported(
            f"unsupported environmental-network result format {format_name!r}"
        )
    expected_schema, kind = supported
    if schema_version != expected_schema:
        raise EnvironmentalNetworkResultArchiveUnsupported(
            f"unsupported schema_version {schema_version} for {format_name!r}; expected {expected_schema}"
        )

    if kind is EnvironmentalNetworkResultArchiveKind.SINGLE_RUN:
        _validate_single_run_document(document)
    elif kind is EnvironmentalNetworkResultArchiveKind.PAIRED:
        _validate_paired_document(document)
    else:
        _validate_stress_document(document)
    return kind


def _validate_single_run_document(document: Mapping[str, Any]) -> None:
    config = _require_mapping(document, "config")
    _require_format(config, _EXPERIMENT_FORMAT, _EXPERIMENT_SCHEMA_VERSION, context="single-run config")
    config_sha = _sha256_json(config)
    if document.get("config_sha256") != config_sha:
        raise EnvironmentalNetworkResultArchiveIntegrityError("single-run config_sha256 mismatch")

    summary = _require_mapping(document, "summary")
    if summary.get("config_sha256") != config_sha:
        raise EnvironmentalNetworkResultArchiveIntegrityError("single-run summary config_sha256 mismatch")
    software_version = summary.get("software_version")
    if not isinstance(software_version, str) or not software_version:
        raise EnvironmentalNetworkResultArchiveIntegrityError("single-run summary software_version is invalid")
    expected_run_id = sha256(
        _RUN_DOMAIN
        + software_version.encode("utf-8")
        + b"\x00"
        + config_sha.encode("ascii")
    ).hexdigest()
    if document.get("run_id") != expected_run_id or summary.get("run_id") != expected_run_id:
        raise EnvironmentalNetworkResultArchiveIntegrityError("single-run run_id mismatch")

    environment = _require_mapping(document, "execution_environment")
    if environment.get("package_version") != software_version:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "single-run execution environment package version mismatch"
        )
    final_state = _require_mapping(document, "final_state")
    final_hash = summary.get("final_snapshot_sha256")
    if not isinstance(final_hash, str) or _sha256_json(final_state) != final_hash:
        raise EnvironmentalNetworkResultArchiveIntegrityError("single-run final-state SHA-256 mismatch")
    if not isinstance(document.get("traffic"), Mapping):
        raise EnvironmentalNetworkResultArchiveIntegrityError("single-run traffic evidence is missing")
    if not isinstance(document.get("ingress"), list):
        raise EnvironmentalNetworkResultArchiveIntegrityError("single-run ingress evidence is missing")


def _validate_pair_config(config: Mapping[str, Any]) -> tuple[str, str]:
    _require_format(config, _PAIR_CONFIG_FORMAT, _PAIR_CONFIG_SCHEMA_VERSION, context="paired config")
    software_version = config.get("software_version")
    if not isinstance(software_version, str) or not software_version:
        raise EnvironmentalNetworkResultArchiveIntegrityError("paired config software_version is invalid")
    control = _require_mapping(config, "control")
    _require_format(control, _EXPERIMENT_FORMAT, _EXPERIMENT_SCHEMA_VERSION, context="paired control config")
    control_sha = _sha256_json(control)
    if config.get("control_config_sha256") != control_sha:
        raise EnvironmentalNetworkResultArchiveIntegrityError("paired control_config_sha256 mismatch")
    expected_control_run_id = sha256(
        _RUN_DOMAIN
        + software_version.encode("utf-8")
        + b"\x00"
        + control_sha.encode("ascii")
    ).hexdigest()
    if config.get("control_run_id") != expected_control_run_id:
        raise EnvironmentalNetworkResultArchiveIntegrityError("paired control_run_id mismatch")
    safety = config.get("safety")
    baseline_policy = config.get("baseline_policy")
    if not isinstance(safety, Mapping) or not isinstance(baseline_policy, str):
        raise EnvironmentalNetworkResultArchiveIntegrityError("paired safety/baseline policy evidence is invalid")
    pair_material = {
        "control_config_sha256": control_sha,
        "control_run_id": expected_control_run_id,
        "safety": safety,
        "baseline_policy": baseline_policy,
    }
    pair_id = sha256(_PAIR_DOMAIN + _canonical_json(pair_material)).hexdigest()
    return _sha256_json(config), pair_id


def _validate_paired_document(document: Mapping[str, Any]) -> None:
    config = _require_mapping(document, "config")
    config_sha, pair_id = _validate_pair_config(config)
    if document.get("config_sha256") != config_sha:
        raise EnvironmentalNetworkResultArchiveIntegrityError("paired config_sha256 mismatch")
    if document.get("pair_id") != pair_id:
        raise EnvironmentalNetworkResultArchiveIntegrityError("paired pair_id mismatch")
    environment = _require_mapping(document, "execution_environment")
    if environment.get("package_version") != config.get("software_version"):
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "paired execution environment package version mismatch"
        )

    traffic = document.get("traffic")
    if not isinstance(traffic, Mapping):
        raise EnvironmentalNetworkResultArchiveIntegrityError("paired shared traffic evidence is missing")
    control = _require_mapping(document, "control")
    _require_format(control, _RESULT_FORMAT, _RESULT_SCHEMA_VERSION, context="paired control result")
    _validate_single_run_document(control)
    if control.get("traffic") != traffic:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "paired control traffic differs from the archived shared CRN realization"
        )
    if control.get("config") != config.get("control"):
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "paired control result config differs from the paired config"
        )

    baseline = _require_mapping(document, "baseline")
    final_snapshot = _require_mapping(baseline, "final_snapshot")
    commands = baseline.get("final_commands")
    vessels = final_snapshot.get("vessels")
    if not isinstance(commands, list) or not isinstance(vessels, list):
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "paired baseline final snapshot/commands evidence is invalid"
        )
    command_ids = [item[0] for item in commands if isinstance(item, list) and len(item) == 2]
    vessel_ids = [item.get("vessel_id") for item in vessels if isinstance(item, Mapping)]
    if len(command_ids) != len(commands) or command_ids != vessel_ids:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "paired baseline final commands do not match final snapshot vessel order"
        )


def _validate_stress_document(document: Mapping[str, Any]) -> None:
    config = _require_mapping(document, "config")
    _require_format(config, _CAMPAIGN_FORMAT, _CAMPAIGN_SCHEMA_VERSION, context="stress config")
    config_sha = _sha256_json(config)
    if document.get("config_sha256") != config_sha:
        raise EnvironmentalNetworkResultArchiveIntegrityError("stress config_sha256 mismatch")
    campaign_id = sha256(_CAMPAIGN_DOMAIN + _canonical_json(config)).hexdigest()
    if document.get("campaign_id") != campaign_id:
        raise EnvironmentalNetworkResultArchiveIntegrityError("stress campaign_id mismatch")
    environment = _require_mapping(document, "execution_environment")
    if environment.get("package_version") != config.get("software_version"):
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "stress execution environment package version mismatch"
        )

    cases = document.get("cases")
    aggregate = _require_mapping(document, "aggregate")
    if not isinstance(cases, list):
        raise EnvironmentalNetworkResultArchiveIntegrityError("stress cases must be a list")
    success_count = 0
    infeasible_count = 0
    for case in cases:
        if not isinstance(case, Mapping):
            raise EnvironmentalNetworkResultArchiveIntegrityError("stress case must be an object")
        paired_config = _require_mapping(case, "paired_config")
        _, pair_id = _validate_pair_config(paired_config)
        environment_id = case.get("environment_id")
        if not isinstance(environment_id, str):
            raise EnvironmentalNetworkResultArchiveIntegrityError("stress environment_id is invalid")
        expected_case_id = sha256(
            _CASE_DOMAIN
            + _canonical_json(
                {
                    "campaign_id": campaign_id,
                    "environment_id": environment_id,
                    "pair_id": pair_id,
                }
            )
        ).hexdigest()
        if case.get("case_id") != expected_case_id:
            raise EnvironmentalNetworkResultArchiveIntegrityError("stress case_id mismatch")
        status = case.get("status")
        result = case.get("result")
        if status == "SUCCESS":
            success_count += 1
            if not isinstance(result, Mapping):
                raise EnvironmentalNetworkResultArchiveIntegrityError(
                    "successful stress case is missing paired result evidence"
                )
            _require_format(result, _PAIR_RESULT_FORMAT, _PAIR_RESULT_SCHEMA_VERSION, context="stress paired result")
            _validate_paired_document(result)
            if result.get("pair_id") != pair_id or result.get("config") != paired_config:
                raise EnvironmentalNetworkResultArchiveIntegrityError(
                    "successful stress case paired result does not match paired_config"
                )
            if case.get("infeasibility_type") is not None or case.get("infeasibility_message") is not None:
                raise EnvironmentalNetworkResultArchiveIntegrityError(
                    "successful stress case carries infeasibility evidence"
                )
        elif status == "MODELED_INFEASIBLE":
            infeasible_count += 1
            if result is not None:
                raise EnvironmentalNetworkResultArchiveIntegrityError(
                    "modeled-infeasible stress case must not contain a successful result"
                )
            if not isinstance(case.get("infeasibility_type"), str) or not isinstance(
                case.get("infeasibility_message"), str
            ):
                raise EnvironmentalNetworkResultArchiveIntegrityError(
                    "modeled-infeasible stress case is missing type/message evidence"
                )
        else:
            raise EnvironmentalNetworkResultArchiveIntegrityError(
                f"unsupported stress case status {status!r}"
            )

    if aggregate.get("total_case_count") != len(cases):
        raise EnvironmentalNetworkResultArchiveIntegrityError("stress aggregate total_case_count mismatch")
    if aggregate.get("successful_case_count") != success_count:
        raise EnvironmentalNetworkResultArchiveIntegrityError("stress aggregate successful_case_count mismatch")
    if aggregate.get("modeled_infeasible_case_count") != infeasible_count:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "stress aggregate modeled_infeasible_case_count mismatch"
        )


def _require_format(
    document: Mapping[str, Any],
    expected_format: str,
    expected_schema: int,
    *,
    context: str,
) -> None:
    if document.get("format") != expected_format or document.get("schema_version") != expected_schema:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            f"{context} format/schema mismatch"
        )


def _require_mapping(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise EnvironmentalNetworkResultArchiveIntegrityError(f"{key} must be a JSON object")
    return value


def _load_json_object(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "result archive is not valid UTF-8"
        ) from exc

    def object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EnvironmentalNetworkResultArchiveIntegrityError(
                    f"duplicate JSON object key {key!r}"
                )
            result[key] = value
        return result

    def parse_constant(value: str) -> Any:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            f"non-finite JSON numeric token {value!r} is forbidden"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs_hook,
            parse_constant=parse_constant,
        )
    except EnvironmentalNetworkResultArchiveIntegrityError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "result archive is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "result archive top level must be a JSON object"
        )
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "result archive contains non-canonical JSON values"
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
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "checksum sidecar is not readable ASCII"
        ) from exc
    lines = text.splitlines()
    if len(lines) != 1:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "checksum sidecar must contain exactly one line"
        )
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", lines[0])
    if match is None:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
            "checksum sidecar must use '<sha256>  <filename>' format"
        )
    digest, name = match.groups()
    if name != expected_name:
        raise EnvironmentalNetworkResultArchiveIntegrityError(
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
    except OSError:  # pragma: no cover - platform/filesystem specific
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform/filesystem specific
        pass
    finally:
        os.close(descriptor)
