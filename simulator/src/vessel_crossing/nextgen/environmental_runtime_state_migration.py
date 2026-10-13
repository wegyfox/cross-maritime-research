"""Explicit migration and deterministic diff tooling for portable environmental runtime evidence.

This module operates only on A131/A132 JSON evidence. It never loads A129/A130 cloudpickle payloads,
never creates controller sessions and never grants command authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

from ..version import __version__
from .environmental_runtime_state_export import (
    EnvironmentalFleetRuntimeStateExport,
    EnvironmentalFleetRuntimeStateExportIntegrityError,
    EnvironmentalFleetRuntimeStateExportMigrationRequired,
    EnvironmentalFleetRuntimeStateMigrationProvenance,
    _AUDIT_SCHEMA_VERSION,
    _FORMAT,
    _SCHEMA_VERSION,
    _canonical_json,
    _load_json_object,
    _validate_audit_shape,
    _validate_sha256,
)

_SCHEMA_1_TO_2_TRANSFORM_ID = "environmental-runtime-state-schema-1-to-2@1"


class EnvironmentalFleetRuntimeStateMigrationError(RuntimeError):
    """Base error for A132 portable-schema migration/diff tooling."""


class EnvironmentalFleetRuntimeStateMigrationPathUnavailable(
    EnvironmentalFleetRuntimeStateExportMigrationRequired,
    EnvironmentalFleetRuntimeStateMigrationError,
):
    """No explicit built-in portable-schema path reaches the requested target schema."""


class EnvironmentalFleetRuntimeStateDiffKind(str, Enum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"
    CHANGED = "CHANGED"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateMigrationResult:
    """Verifiable evidence for one portable-schema migration chain."""

    source_schema_version: int
    target_schema_version: int
    source_document_sha256: str
    target_document_sha256: str
    steps: tuple[EnvironmentalFleetRuntimeStateMigrationProvenance, ...]
    state_export: EnvironmentalFleetRuntimeStateExport


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateDiffEntry:
    """One deterministic JSON-Pointer change between committed-state audits."""

    path: str
    kind: EnvironmentalFleetRuntimeStateDiffKind
    before: Any
    after: Any


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateDiff:
    """Deterministic audit-only comparison; envelope provenance is intentionally excluded."""

    left_audit_sha256: str
    right_audit_sha256: str
    entries: tuple[EnvironmentalFleetRuntimeStateDiffEntry, ...]

    @property
    def is_equal(self) -> bool:
        return not self.entries


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateDocumentComparison:
    """Migration evidence for two documents plus their current-schema audit diff."""

    left: EnvironmentalFleetRuntimeStateMigrationResult
    right: EnvironmentalFleetRuntimeStateMigrationResult
    diff: EnvironmentalFleetRuntimeStateDiff


def migrate_environmental_fleet_runtime_state_export(
    data: bytes | bytearray | memoryview,
    *,
    target_schema_version: int = _SCHEMA_VERSION,
) -> EnvironmentalFleetRuntimeStateMigrationResult:
    """Migrate portable JSON evidence through an explicit built-in schema path.

    The only historical path currently defined is schema 1 -> schema 2. Unknown schemas are rejected
    before their nested audit is interpreted. The result remains non-executable JSON evidence.
    """

    raw = _load_json_object(data)
    if raw.get("format") != _FORMAT:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "portable runtime-state export format marker is invalid"
        )
    try:
        source_schema = int(raw.get("schema_version"))
        target_schema = int(target_schema_version)
    except (TypeError, ValueError) as exc:
        raise EnvironmentalFleetRuntimeStateMigrationError(
            "portable runtime-state migration schema version is invalid"
        ) from exc
    if target_schema != _SCHEMA_VERSION:
        raise EnvironmentalFleetRuntimeStateMigrationPathUnavailable(
            f"no explicit portable migration path targets schema {target_schema}; current target is {_SCHEMA_VERSION}"
        )

    if source_schema == _SCHEMA_VERSION:
        current = EnvironmentalFleetRuntimeStateExport.from_bytes(data)
        canonical = current.to_bytes()
        digest = sha256(canonical).hexdigest()
        return EnvironmentalFleetRuntimeStateMigrationResult(
            source_schema_version=source_schema,
            target_schema_version=target_schema,
            source_document_sha256=digest,
            target_document_sha256=digest,
            steps=(),
            state_export=current,
        )

    if source_schema != 1:
        raise EnvironmentalFleetRuntimeStateMigrationPathUnavailable(
            f"no explicit portable migration path from schema {source_schema} to schema {target_schema}"
        )

    historical = _parse_schema_1(raw)
    canonical_source = _canonical_json(raw)
    source_document_sha256 = sha256(canonical_source).hexdigest()
    step = EnvironmentalFleetRuntimeStateMigrationProvenance(
        transform_id=_SCHEMA_1_TO_2_TRANSFORM_ID,
        from_schema_version=1,
        to_schema_version=2,
        input_document_sha256=source_document_sha256,
        input_audit_sha256=historical["audit_sha256"],
        migration_runtime_version=__version__,
    )
    migrated = EnvironmentalFleetRuntimeStateExport(
        schema_version=_SCHEMA_VERSION,
        audit_schema_version=_AUDIT_SCHEMA_VERSION,
        source_runtime_version=historical["source_runtime_version"],
        audit_sha256=historical["audit_sha256"],
        migration_provenance=(step,),
        audit=historical["audit"],
    )
    target_bytes = migrated.to_bytes()
    return EnvironmentalFleetRuntimeStateMigrationResult(
        source_schema_version=1,
        target_schema_version=_SCHEMA_VERSION,
        source_document_sha256=source_document_sha256,
        target_document_sha256=sha256(target_bytes).hexdigest(),
        steps=(step,),
        state_export=migrated,
    )


def diff_environmental_fleet_runtime_state_exports(
    left: EnvironmentalFleetRuntimeStateExport,
    right: EnvironmentalFleetRuntimeStateExport,
) -> EnvironmentalFleetRuntimeStateDiff:
    """Compare two current-schema exports by committed-state audit only."""

    if not isinstance(left, EnvironmentalFleetRuntimeStateExport):
        raise TypeError("left must be EnvironmentalFleetRuntimeStateExport")
    if not isinstance(right, EnvironmentalFleetRuntimeStateExport):
        raise TypeError("right must be EnvironmentalFleetRuntimeStateExport")
    # Validate both envelopes/digests before comparing nested state.
    left.to_bytes()
    right.to_bytes()
    entries: list[EnvironmentalFleetRuntimeStateDiffEntry] = []
    _diff_json(left.audit, right.audit, "", entries)
    return EnvironmentalFleetRuntimeStateDiff(
        left_audit_sha256=left.audit_sha256,
        right_audit_sha256=right.audit_sha256,
        entries=tuple(entries),
    )


def migrate_and_diff_environmental_fleet_runtime_state_documents(
    left: bytes | bytearray | memoryview,
    right: bytes | bytearray | memoryview,
) -> EnvironmentalFleetRuntimeStateDocumentComparison:
    """Explicitly migrate two portable documents to current schema, then compare their audits."""

    left_result = migrate_environmental_fleet_runtime_state_export(left)
    right_result = migrate_environmental_fleet_runtime_state_export(right)
    return EnvironmentalFleetRuntimeStateDocumentComparison(
        left=left_result,
        right=right_result,
        diff=diff_environmental_fleet_runtime_state_exports(
            left_result.state_export,
            right_result.state_export,
        ),
    )


def _parse_schema_1(raw: dict[str, Any]) -> dict[str, Any]:
    required = {"format", "schema_version", "source_runtime_version", "audit_sha256", "audit"}
    if set(raw) != required:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "historical schema-1 runtime-state export top-level fields are invalid"
        )
    if raw["format"] != _FORMAT or int(raw["schema_version"]) != 1:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "historical schema-1 runtime-state export marker is invalid"
        )
    audit = raw["audit"]
    if not isinstance(audit, dict):
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "historical schema-1 runtime-state audit must be a JSON object"
        )
    _validate_audit_shape(audit)
    expected = _validate_sha256(raw["audit_sha256"], "historical schema-1 audit")
    actual = sha256(_canonical_json(audit)).hexdigest()
    if expected != actual:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "historical schema-1 runtime-state audit SHA-256 mismatch"
        )
    source_runtime_version = str(raw["source_runtime_version"])
    if not source_runtime_version:
        raise EnvironmentalFleetRuntimeStateExportIntegrityError(
            "historical schema-1 runtime-state source runtime version is empty"
        )
    return {
        "source_runtime_version": source_runtime_version,
        "audit_sha256": expected,
        "audit": audit,
    }


def _diff_json(
    before: Any,
    after: Any,
    path: str,
    entries: list[EnvironmentalFleetRuntimeStateDiffEntry],
) -> None:
    if type(before) is not type(after):
        entries.append(
            EnvironmentalFleetRuntimeStateDiffEntry(
                path=path or "/",
                kind=EnvironmentalFleetRuntimeStateDiffKind.CHANGED,
                before=before,
                after=after,
            )
        )
        return
    if isinstance(before, dict):
        before_keys = set(before)
        after_keys = set(after)
        for key in sorted(before_keys - after_keys):
            entries.append(
                EnvironmentalFleetRuntimeStateDiffEntry(
                    path=_join_pointer(path, key),
                    kind=EnvironmentalFleetRuntimeStateDiffKind.REMOVED,
                    before=before[key],
                    after=None,
                )
            )
        for key in sorted(after_keys - before_keys):
            entries.append(
                EnvironmentalFleetRuntimeStateDiffEntry(
                    path=_join_pointer(path, key),
                    kind=EnvironmentalFleetRuntimeStateDiffKind.ADDED,
                    before=None,
                    after=after[key],
                )
            )
        for key in sorted(before_keys & after_keys):
            _diff_json(before[key], after[key], _join_pointer(path, key), entries)
        return
    if isinstance(before, list):
        shared = min(len(before), len(after))
        for index in range(shared):
            _diff_json(before[index], after[index], _join_pointer(path, str(index)), entries)
        for index in range(shared, len(before)):
            entries.append(
                EnvironmentalFleetRuntimeStateDiffEntry(
                    path=_join_pointer(path, str(index)),
                    kind=EnvironmentalFleetRuntimeStateDiffKind.REMOVED,
                    before=before[index],
                    after=None,
                )
            )
        for index in range(shared, len(after)):
            entries.append(
                EnvironmentalFleetRuntimeStateDiffEntry(
                    path=_join_pointer(path, str(index)),
                    kind=EnvironmentalFleetRuntimeStateDiffKind.ADDED,
                    before=None,
                    after=after[index],
                )
            )
        return
    if before != after:
        entries.append(
            EnvironmentalFleetRuntimeStateDiffEntry(
                path=path or "/",
                kind=EnvironmentalFleetRuntimeStateDiffKind.CHANGED,
                before=before,
                after=after,
            )
        )


def _join_pointer(base: str, token: str) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}"
