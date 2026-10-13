"""Portable, non-executable committed-state evidence for the environmental fleet runtime.

A129/A130 checkpoints intentionally contain trusted cloudpickle payloads and are exact same-version
restart artifacts. A131 introduced a separate JSON evidence format for inspection and archival
comparison. A132 evolves that format to schema 2, separating the envelope schema from the committed
state audit schema and carrying explicit migration provenance.

Reading this format never deserializes Python objects and never grants command authority. Unknown
portable schemas still fail before nested state is interpreted; the A132 migration layer is the only
supported path from historical schema 1 evidence to schema 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from ..version import __version__
from .environmental_idle_fleet_control import EnvironmentalPersistentFleetControlSession
from .environmental_runtime_checkpoint import _audit_runtime, _validate_runtime_consistency

_FORMAT = "vessel-crossing-environmental-runtime-state"
_SCHEMA_VERSION = 2
_AUDIT_SCHEMA_VERSION = 1


class EnvironmentalFleetRuntimeStateExportError(RuntimeError):
    """Base error for portable runtime-state evidence."""


class EnvironmentalFleetRuntimeStateExportIntegrityError(EnvironmentalFleetRuntimeStateExportError):
    """Portable JSON evidence is malformed or a recorded digest does not match."""


class EnvironmentalFleetRuntimeStateExportMigrationRequired(EnvironmentalFleetRuntimeStateExportError):
    """The export schema is not understood directly and requires explicit migration."""


class EnvironmentalFleetRuntimeStateExportMismatch(EnvironmentalFleetRuntimeStateExportError):
    """A live runtime no longer matches the portable committed-state evidence."""


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateMigrationProvenance:
    """One explicit non-executable portable-schema transform recorded in schema-2 evidence."""

    transform_id: str
    from_schema_version: int
    to_schema_version: int
    input_document_sha256: str
    input_audit_sha256: str
    migration_runtime_version: str

    def to_json(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "from_schema_version": self.from_schema_version,
            "to_schema_version": self.to_schema_version,
            "input_document_sha256": self.input_document_sha256,
            "input_audit_sha256": self.input_audit_sha256,
            "migration_runtime_version": self.migration_runtime_version,
        }

    @classmethod
    def from_json(cls, raw: Any) -> "EnvironmentalFleetRuntimeStateMigrationProvenance":
        required = {
            "transform_id",
            "from_schema_version",
            "to_schema_version",
            "input_document_sha256",
            "input_audit_sha256",
            "migration_runtime_version",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state migration provenance fields are invalid"
            )
        transform_id = str(raw["transform_id"])
        migration_runtime_version = str(raw["migration_runtime_version"])
        if not transform_id or not migration_runtime_version:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state migration provenance text fields are empty"
            )
        try:
            from_schema_version = int(raw["from_schema_version"])
            to_schema_version = int(raw["to_schema_version"])
        except (TypeError, ValueError) as exc:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state migration provenance schema versions are invalid"
            ) from exc
        input_document_sha256 = _validate_sha256(
            raw["input_document_sha256"], "migration input document"
        )
        input_audit_sha256 = _validate_sha256(
            raw["input_audit_sha256"], "migration input audit"
        )
        return cls(
            transform_id=transform_id,
            from_schema_version=from_schema_version,
            to_schema_version=to_schema_version,
            input_document_sha256=input_document_sha256,
            input_audit_sha256=input_audit_sha256,
            migration_runtime_version=migration_runtime_version,
        )


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateExport:
    """Canonical JSON-only evidence of one quiescent committed runtime boundary."""

    schema_version: int
    audit_schema_version: int
    source_runtime_version: str
    audit_sha256: str
    migration_provenance: tuple[EnvironmentalFleetRuntimeStateMigrationProvenance, ...]
    audit: dict[str, Any]

    @property
    def world_time(self) -> float:
        return float(self.audit["world_time"])

    def to_bytes(self) -> bytes:
        if self.schema_version != _SCHEMA_VERSION:
            raise EnvironmentalFleetRuntimeStateExportMigrationRequired(
                f"runtime-state schema {self.schema_version} requires explicit migration; current schema is {_SCHEMA_VERSION}"
            )
        if self.audit_schema_version != _AUDIT_SCHEMA_VERSION:
            raise EnvironmentalFleetRuntimeStateExportMigrationRequired(
                f"runtime-state audit schema {self.audit_schema_version} requires explicit migration; current audit schema is {_AUDIT_SCHEMA_VERSION}"
            )
        _validate_audit_shape(self.audit)
        audit_bytes = _canonical_json(self.audit)
        if sha256(audit_bytes).hexdigest() != self.audit_sha256:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable state audit digest no longer matches audit content"
            )
        if not self.source_runtime_version:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state source runtime version is empty"
            )
        _validate_provenance_chain(self.migration_provenance, self.schema_version)
        payload = {
            "format": _FORMAT,
            "schema_version": self.schema_version,
            "audit_schema_version": self.audit_schema_version,
            "source_runtime_version": self.source_runtime_version,
            "audit_sha256": self.audit_sha256,
            "migration_provenance": [item.to_json() for item in self.migration_provenance],
            "audit": self.audit,
        }
        return _canonical_json(payload)

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "EnvironmentalFleetRuntimeStateExport":
        raw = _load_json_object(data)
        if raw.get("format") != _FORMAT:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state export format marker is invalid"
            )
        try:
            schema_version = int(raw.get("schema_version"))
        except (TypeError, ValueError) as exc:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state schema version is invalid"
            ) from exc
        if schema_version != _SCHEMA_VERSION:
            raise EnvironmentalFleetRuntimeStateExportMigrationRequired(
                f"runtime-state schema {schema_version} requires explicit migration; current schema is {_SCHEMA_VERSION}"
            )
        required = {
            "format",
            "schema_version",
            "audit_schema_version",
            "source_runtime_version",
            "audit_sha256",
            "migration_provenance",
            "audit",
        }
        if set(raw) != required:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state export top-level fields are invalid"
            )
        try:
            audit_schema_version = int(raw["audit_schema_version"])
        except (TypeError, ValueError) as exc:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state audit schema version is invalid"
            ) from exc
        if audit_schema_version != _AUDIT_SCHEMA_VERSION:
            raise EnvironmentalFleetRuntimeStateExportMigrationRequired(
                f"runtime-state audit schema {audit_schema_version} requires explicit migration; current audit schema is {_AUDIT_SCHEMA_VERSION}"
            )
        audit = raw["audit"]
        if not isinstance(audit, dict):
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state audit must be a JSON object"
            )
        _validate_audit_shape(audit)
        expected = _validate_sha256(raw["audit_sha256"], "audit")
        actual = sha256(_canonical_json(audit)).hexdigest()
        if actual != expected:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state audit SHA-256 mismatch"
            )
        source_runtime_version = str(raw["source_runtime_version"])
        if not source_runtime_version:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state source runtime version is empty"
            )
        raw_provenance = raw["migration_provenance"]
        if not isinstance(raw_provenance, list):
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state migration provenance must be an array"
            )
        provenance = tuple(
            EnvironmentalFleetRuntimeStateMigrationProvenance.from_json(item)
            for item in raw_provenance
        )
        _validate_provenance_chain(provenance, schema_version)
        return cls(
            schema_version=schema_version,
            audit_schema_version=audit_schema_version,
            source_runtime_version=source_runtime_version,
            audit_sha256=expected,
            migration_provenance=provenance,
            audit=audit,
        )


def export_environmental_fleet_runtime_state(
    session: EnvironmentalPersistentFleetControlSession,
) -> EnvironmentalFleetRuntimeStateExport:
    """Capture portable committed-state evidence without serializing executable Python objects."""

    if not isinstance(session, EnvironmentalPersistentFleetControlSession):
        raise TypeError("session must be EnvironmentalPersistentFleetControlSession")
    _validate_runtime_consistency(session)
    audit = _json_normalize(asdict(_audit_runtime(session)))
    _validate_audit_shape(audit)
    digest = sha256(_canonical_json(audit)).hexdigest()
    return EnvironmentalFleetRuntimeStateExport(
        schema_version=_SCHEMA_VERSION,
        audit_schema_version=_AUDIT_SCHEMA_VERSION,
        source_runtime_version=__version__,
        audit_sha256=digest,
        migration_provenance=(),
        audit=audit,
    )


def verify_environmental_fleet_runtime_state_export(
    session: EnvironmentalPersistentFleetControlSession,
    state_export: EnvironmentalFleetRuntimeStateExport,
) -> None:
    """Require a live runtime to reproduce one portable committed-state export exactly."""

    if not isinstance(state_export, EnvironmentalFleetRuntimeStateExport):
        raise TypeError("state_export must be EnvironmentalFleetRuntimeStateExport")
    live = export_environmental_fleet_runtime_state(session)
    if live.audit_sha256 != state_export.audit_sha256 or live.audit != state_export.audit:
        raise EnvironmentalFleetRuntimeStateExportMismatch(
            "live runtime does not match the portable committed-state export"
        )


def write_environmental_fleet_runtime_state_export(
    path: str | Path,
    session: EnvironmentalPersistentFleetControlSession,
) -> EnvironmentalFleetRuntimeStateExport:
    """Atomically publish one portable JSON state export."""

    state_export = export_environmental_fleet_runtime_state(session)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(state_export.to_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    return state_export


def read_environmental_fleet_runtime_state_export(
    path: str | Path,
) -> EnvironmentalFleetRuntimeStateExport:
    """Read current-schema portable JSON evidence; historical schemas require A132 migration."""

    return EnvironmentalFleetRuntimeStateExport.from_bytes(Path(path).read_bytes())


def _load_json_object(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("state export data must be bytes-like")
    try:
        raw = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "portable runtime-state export is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "portable runtime-state export must be a JSON object"
        )
    return raw


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_normalize(value: Any) -> Any:
    """Convert dataclass/asdict output into JSON primitives with deterministic container order."""

    if isinstance(value, dict):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_normalize(item) for item in value]
    if isinstance(value, list):
        return [_json_normalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise EnvironmentalFleetRuntimeStateExportIntegrityError(
        f"portable runtime-state audit contains unsupported value type {type(value).__name__}"
    )


def _validate_audit_shape(audit: dict[str, Any]) -> None:
    required = {
        "world_time",
        "vessel_ids",
        "commands",
        "queued_command_events",
        "registry_last_update_time",
        "registry_pairs",
        "fleet_observation_times",
        "avoidance_owners",
        "route_owners",
        "idle_active",
        "next_idle_review_time",
        "membership_events",
        "active_incarnations",
        "pending_incarnations",
        "next_incarnation_by_vessel",
        "route_targets",
        "profiles",
        "audit_counts",
    }
    if set(audit) != required:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "portable runtime-state audit fields do not match audit schema 1"
        )
    if not isinstance(audit["world_time"], (int, float)):
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "portable runtime-state world_time must be numeric"
        )
    for name in required - {"world_time", "registry_last_update_time", "idle_active", "next_idle_review_time"}:
        if not isinstance(audit[name], list):
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                f"portable runtime-state audit field {name!r} must be an array"
            )
    if not isinstance(audit["idle_active"], bool):
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "portable runtime-state idle_active must be boolean"
        )


def _validate_provenance_chain(
    provenance: tuple[EnvironmentalFleetRuntimeStateMigrationProvenance, ...],
    final_schema_version: int,
) -> None:
    if not provenance:
        return
    for previous, current in zip(provenance, provenance[1:]):
        if previous.to_schema_version != current.from_schema_version:
            raise EnvironmentalFleetRuntimeStateExportIntegrityError(
                "portable runtime-state migration provenance chain is discontinuous"
            )
    if provenance[-1].to_schema_version != final_schema_version:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "portable runtime-state migration provenance does not end at the export schema"
        )


def _validate_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            f"portable runtime-state {label} digest is invalid"
        )
    try:
        int(text, 16)
    except ValueError as exc:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            f"portable runtime-state {label} digest is invalid"
        ) from exc
    return text.lower()


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
