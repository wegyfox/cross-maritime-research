"""Semantic compatibility reporting over A132 portable environmental runtime-state evidence.

The report classifies structural committed-state differences by operational impact. It is deliberately
non-executable: successful compatibility analysis never authorizes A129/A130 checkpoint restore and
never constructs live command/controller state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any

from .environmental_runtime_state_migration import (
    EnvironmentalFleetRuntimeStateDiffEntry,
    EnvironmentalFleetRuntimeStateDocumentComparison,
    migrate_and_diff_environmental_fleet_runtime_state_documents,
)


class EnvironmentalFleetRuntimeStateSemanticDomain(str, Enum):
    PHYSICAL_WORLD = "PHYSICAL_WORLD"
    COMMAND_CHANNELS = "COMMAND_CHANNELS"
    ENCOUNTER_REGISTRY = "ENCOUNTER_REGISTRY"
    OWNERSHIP = "OWNERSHIP"
    SUPERVISION_SCHEDULER = "SUPERVISION_SCHEDULER"
    MEMBERSHIP_IDENTITY = "MEMBERSHIP_IDENTITY"
    MISSION_METADATA = "MISSION_METADATA"
    AUDIT_EVIDENCE = "AUDIT_EVIDENCE"


class EnvironmentalFleetRuntimeStateSemanticImpact(str, Enum):
    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    FUTURE_BEHAVIOR = "FUTURE_BEHAVIOR"
    CURRENT_STATE_OR_AUTHORITY = "CURRENT_STATE_OR_AUTHORITY"


class EnvironmentalFleetRuntimeStateCompatibility(str, Enum):
    EXACT = "EXACT"
    EVIDENCE_ONLY_DIFFERENCE = "EVIDENCE_ONLY_DIFFERENCE"
    FUTURE_BEHAVIOR_DIFFERENCE = "FUTURE_BEHAVIOR_DIFFERENCE"
    CURRENT_STATE_OR_AUTHORITY_DIFFERENCE = "CURRENT_STATE_OR_AUTHORITY_DIFFERENCE"


class EnvironmentalFleetRuntimeStateSchemaDisposition(str, Enum):
    CURRENT_SCHEMA = "CURRENT_SCHEMA"
    MIGRATED_TO_CURRENT = "MIGRATED_TO_CURRENT"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateSemanticFinding:
    path: str
    kind: str
    domain: EnvironmentalFleetRuntimeStateSemanticDomain
    impact: EnvironmentalFleetRuntimeStateSemanticImpact
    before: Any
    after: Any


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateCompatibilitySide:
    source_schema_version: int
    target_schema_version: int
    schema_disposition: EnvironmentalFleetRuntimeStateSchemaDisposition
    source_runtime_version: str
    source_document_sha256: str
    current_document_sha256: str
    audit_sha256: str
    migration_transform_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateCompatibilityReport:
    left: EnvironmentalFleetRuntimeStateCompatibilitySide
    right: EnvironmentalFleetRuntimeStateCompatibilitySide
    compatibility: EnvironmentalFleetRuntimeStateCompatibility
    findings: tuple[EnvironmentalFleetRuntimeStateSemanticFinding, ...]
    executable_restore_authorized: bool = False

    @property
    def is_exact(self) -> bool:
        return self.compatibility is EnvironmentalFleetRuntimeStateCompatibility.EXACT

    def to_json_bytes(self) -> bytes:
        """Return deterministic non-executable JSON report evidence."""

        payload = {
            "format": "vessel-crossing-environmental-runtime-state-compatibility-report",
            "left": _side_to_json(self.left),
            "right": _side_to_json(self.right),
            "compatibility": self.compatibility.value,
            "findings": [
                {
                    "path": finding.path,
                    "kind": finding.kind,
                    "domain": finding.domain.value,
                    "impact": finding.impact.value,
                    "before": finding.before,
                    "after": finding.after,
                }
                for finding in self.findings
            ],
            "executable_restore_authorized": False,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")


def report_environmental_fleet_runtime_state_compatibility(
    left: bytes | bytearray | memoryview,
    right: bytes | bytearray | memoryview,
) -> EnvironmentalFleetRuntimeStateCompatibilityReport:
    """Migrate portable documents explicitly, diff audits and classify operational impact."""

    comparison = migrate_and_diff_environmental_fleet_runtime_state_documents(left, right)
    findings = tuple(_classify_entry(entry) for entry in comparison.diff.entries)
    return EnvironmentalFleetRuntimeStateCompatibilityReport(
        left=_side(comparison, "left"),
        right=_side(comparison, "right"),
        compatibility=_overall_compatibility(findings),
        findings=findings,
        executable_restore_authorized=False,
    )


def _side(
    comparison: EnvironmentalFleetRuntimeStateDocumentComparison,
    which: str,
) -> EnvironmentalFleetRuntimeStateCompatibilitySide:
    result = comparison.left if which == "left" else comparison.right
    state_export = result.state_export
    return EnvironmentalFleetRuntimeStateCompatibilitySide(
        source_schema_version=result.source_schema_version,
        target_schema_version=result.target_schema_version,
        schema_disposition=(
            EnvironmentalFleetRuntimeStateSchemaDisposition.MIGRATED_TO_CURRENT
            if result.steps
            else EnvironmentalFleetRuntimeStateSchemaDisposition.CURRENT_SCHEMA
        ),
        source_runtime_version=state_export.source_runtime_version,
        source_document_sha256=result.source_document_sha256,
        current_document_sha256=result.target_document_sha256,
        audit_sha256=state_export.audit_sha256,
        migration_transform_ids=tuple(step.transform_id for step in result.steps),
    )


def _classify_entry(
    entry: EnvironmentalFleetRuntimeStateDiffEntry,
) -> EnvironmentalFleetRuntimeStateSemanticFinding:
    top = _top_level_token(entry.path)
    domain, impact = _FIELD_SEMANTICS[top]
    return EnvironmentalFleetRuntimeStateSemanticFinding(
        path=entry.path,
        kind=entry.kind.value,
        domain=domain,
        impact=impact,
        before=entry.before,
        after=entry.after,
    )


def _overall_compatibility(
    findings: tuple[EnvironmentalFleetRuntimeStateSemanticFinding, ...],
) -> EnvironmentalFleetRuntimeStateCompatibility:
    if not findings:
        return EnvironmentalFleetRuntimeStateCompatibility.EXACT
    impacts = {finding.impact for finding in findings}
    if EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY in impacts:
        return EnvironmentalFleetRuntimeStateCompatibility.CURRENT_STATE_OR_AUTHORITY_DIFFERENCE
    if EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR in impacts:
        return EnvironmentalFleetRuntimeStateCompatibility.FUTURE_BEHAVIOR_DIFFERENCE
    return EnvironmentalFleetRuntimeStateCompatibility.EVIDENCE_ONLY_DIFFERENCE


def _top_level_token(path: str) -> str:
    if not path.startswith("/") or path == "/":
        raise ValueError(f"semantic diff path has no auditable top-level field: {path!r}")
    token = path.split("/", 2)[1]
    token = token.replace("~1", "/").replace("~0", "~")
    if token not in _FIELD_SEMANTICS:
        raise ValueError(f"semantic diff references unknown audit field {token!r}")
    return token


def _side_to_json(side: EnvironmentalFleetRuntimeStateCompatibilitySide) -> dict[str, Any]:
    return {
        "source_schema_version": side.source_schema_version,
        "target_schema_version": side.target_schema_version,
        "schema_disposition": side.schema_disposition.value,
        "source_runtime_version": side.source_runtime_version,
        "source_document_sha256": side.source_document_sha256,
        "current_document_sha256": side.current_document_sha256,
        "audit_sha256": side.audit_sha256,
        "migration_transform_ids": list(side.migration_transform_ids),
    }


_FIELD_SEMANTICS: dict[
    str,
    tuple[EnvironmentalFleetRuntimeStateSemanticDomain, EnvironmentalFleetRuntimeStateSemanticImpact],
] = {
    "world_time": (
        EnvironmentalFleetRuntimeStateSemanticDomain.PHYSICAL_WORLD,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "vessel_ids": (
        EnvironmentalFleetRuntimeStateSemanticDomain.PHYSICAL_WORLD,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "commands": (
        EnvironmentalFleetRuntimeStateSemanticDomain.COMMAND_CHANNELS,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "queued_command_events": (
        EnvironmentalFleetRuntimeStateSemanticDomain.COMMAND_CHANNELS,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "registry_last_update_time": (
        EnvironmentalFleetRuntimeStateSemanticDomain.ENCOUNTER_REGISTRY,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "registry_pairs": (
        EnvironmentalFleetRuntimeStateSemanticDomain.ENCOUNTER_REGISTRY,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "fleet_observation_times": (
        EnvironmentalFleetRuntimeStateSemanticDomain.ENCOUNTER_REGISTRY,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "avoidance_owners": (
        EnvironmentalFleetRuntimeStateSemanticDomain.OWNERSHIP,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "route_owners": (
        EnvironmentalFleetRuntimeStateSemanticDomain.OWNERSHIP,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "idle_active": (
        EnvironmentalFleetRuntimeStateSemanticDomain.SUPERVISION_SCHEDULER,
        EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR,
    ),
    "next_idle_review_time": (
        EnvironmentalFleetRuntimeStateSemanticDomain.SUPERVISION_SCHEDULER,
        EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR,
    ),
    "membership_events": (
        EnvironmentalFleetRuntimeStateSemanticDomain.MEMBERSHIP_IDENTITY,
        EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR,
    ),
    "active_incarnations": (
        EnvironmentalFleetRuntimeStateSemanticDomain.MEMBERSHIP_IDENTITY,
        EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY,
    ),
    "pending_incarnations": (
        EnvironmentalFleetRuntimeStateSemanticDomain.MEMBERSHIP_IDENTITY,
        EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR,
    ),
    "next_incarnation_by_vessel": (
        EnvironmentalFleetRuntimeStateSemanticDomain.MEMBERSHIP_IDENTITY,
        EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR,
    ),
    "route_targets": (
        EnvironmentalFleetRuntimeStateSemanticDomain.MISSION_METADATA,
        EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR,
    ),
    "profiles": (
        EnvironmentalFleetRuntimeStateSemanticDomain.MISSION_METADATA,
        EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR,
    ),
    "audit_counts": (
        EnvironmentalFleetRuntimeStateSemanticDomain.AUDIT_EVIDENCE,
        EnvironmentalFleetRuntimeStateSemanticImpact.EVIDENCE_ONLY,
    ),
}
