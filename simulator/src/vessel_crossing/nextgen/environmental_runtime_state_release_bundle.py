"""Self-contained, non-executable release evidence over A131-A134 portable state tooling.

A135 bundles source portable documents, their explicitly migrated current-schema forms, the A133
semantic report, the selected A134 policy, and the resulting gate decision into one canonical JSON
envelope. It provides integrity and provenance linkage, not authenticity or command authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from ..version import __version__
from .environmental_runtime_state_compatibility import (
    EnvironmentalFleetRuntimeStateCompatibility,
    EnvironmentalFleetRuntimeStateSemanticDomain,
    EnvironmentalFleetRuntimeStateSemanticFinding,
    EnvironmentalFleetRuntimeStateSemanticImpact,
)
from .environmental_runtime_state_export import EnvironmentalFleetRuntimeStateExport
from .environmental_runtime_state_migration import migrate_environmental_fleet_runtime_state_export
from .environmental_runtime_state_policy import (
    EnvironmentalFleetRuntimeStateCompatibilityPolicy,
    EnvironmentalFleetRuntimeStateGateAction,
    EnvironmentalFleetRuntimeStateGateDecision,
    EnvironmentalFleetRuntimeStatePolicyDomainOverride,
    compare_and_gate_environmental_fleet_runtime_state_documents,
)

_FORMAT = "vessel-crossing-environmental-runtime-state-release-bundle"
_SCHEMA_VERSION = 1


class EnvironmentalFleetRuntimeStateReleaseBundleError(RuntimeError):
    """Base error for A135 portable release evidence."""


class EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
    EnvironmentalFleetRuntimeStateReleaseBundleError
):
    """Bundle JSON, hash links, report, policy, or gate evidence is inconsistent."""


class EnvironmentalFleetRuntimeStateReleaseBundleMigrationRequired(
    EnvironmentalFleetRuntimeStateReleaseBundleError
):
    """Bundle envelope schema is not understood by this runtime."""


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateReleaseBundle:
    schema_version: int
    content_sha256: str
    content: dict[str, Any]
    executable_restore_authorized: bool = False

    @property
    def bundle_id(self) -> str:
        return self.content_sha256

    @property
    def decision(self) -> EnvironmentalFleetRuntimeStateGateDecision:
        return EnvironmentalFleetRuntimeStateGateDecision(self.content["decision"])

    @property
    def portable_release_gate_passed(self) -> bool:
        return bool(self.content["portable_release_gate_passed"])

    @property
    def human_review_required(self) -> bool:
        return bool(self.content["human_review_required"])

    def to_bytes(self) -> bytes:
        if self.schema_version != _SCHEMA_VERSION:
            raise EnvironmentalFleetRuntimeStateReleaseBundleMigrationRequired(
                f"release-bundle schema {self.schema_version} requires explicit migration; current schema is {_SCHEMA_VERSION}"
            )
        actual = sha256(_canonical_json(self.content)).hexdigest()
        if actual != self.content_sha256:
            raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
                "release-bundle content SHA-256 mismatch"
            )
        if self.executable_restore_authorized:
            raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
                "portable release bundle cannot authorize executable restore"
            )
        return _canonical_json(
            {
                "format": _FORMAT,
                "schema_version": self.schema_version,
                "content_sha256": self.content_sha256,
                "content": self.content,
                "executable_restore_authorized": False,
            }
        )


def build_environmental_fleet_runtime_state_release_bundle(
    left: bytes | bytearray | memoryview,
    right: bytes | bytearray | memoryview,
    *,
    policy: EnvironmentalFleetRuntimeStateCompatibilityPolicy | None = None,
) -> EnvironmentalFleetRuntimeStateReleaseBundle:
    """Build one deterministic self-contained A131-A134 evidence package."""

    left_source = _load_json_object(left, "left source document")
    right_source = _load_json_object(right, "right source document")
    left_migrated = migrate_environmental_fleet_runtime_state_export(left)
    right_migrated = migrate_environmental_fleet_runtime_state_export(right)
    gate = compare_and_gate_environmental_fleet_runtime_state_documents(
        left,
        right,
        policy=policy,
    )
    selected_policy = gate.policy
    report = gate.compatibility_report

    left_current = _load_json_object(
        left_migrated.state_export.to_bytes(), "left current document"
    )
    right_current = _load_json_object(
        right_migrated.state_export.to_bytes(), "right current document"
    )
    report_object = _load_json_object(report.to_json_bytes(), "compatibility report")
    policy_object = selected_policy.to_json_object()
    gate_object = _load_json_object(gate.to_json_bytes(), "policy gate report")

    digests = {
        "left_source_document_sha256": _json_sha256(left_source),
        "right_source_document_sha256": _json_sha256(right_source),
        "left_current_document_sha256": _json_sha256(left_current),
        "right_current_document_sha256": _json_sha256(right_current),
        "compatibility_report_sha256": _json_sha256(report_object),
        "policy_sha256": _json_sha256(policy_object),
        "gate_report_sha256": _json_sha256(gate_object),
    }
    content = {
        "producer_runtime_version": __version__,
        "left_source_document": left_source,
        "right_source_document": right_source,
        "left_current_document": left_current,
        "right_current_document": right_current,
        "compatibility_report": report_object,
        "policy": policy_object,
        "gate_report": gate_object,
        "digests": digests,
        "decision": gate.decision.value,
        "portable_release_gate_passed": gate.portable_release_gate_passed,
        "human_review_required": gate.human_review_required,
        "executable_restore_authorized": False,
    }
    bundle = EnvironmentalFleetRuntimeStateReleaseBundle(
        schema_version=_SCHEMA_VERSION,
        content_sha256=sha256(_canonical_json(content)).hexdigest(),
        content=content,
        executable_restore_authorized=False,
    )
    # Run the same self-contained verification used by readers before returning builder output.
    return read_environmental_fleet_runtime_state_release_bundle(bundle.to_bytes())


def read_environmental_fleet_runtime_state_release_bundle(
    data: bytes | bytearray | memoryview,
) -> EnvironmentalFleetRuntimeStateReleaseBundle:
    """Read and verify one A135 bundle without constructing executable simulator state."""

    raw = _load_json_object(data, "release bundle")
    if raw.get("format") != _FORMAT:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle format marker is invalid"
        )
    try:
        schema_version = int(raw.get("schema_version"))
    except (TypeError, ValueError) as exc:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle schema version is invalid"
        ) from exc
    if schema_version != _SCHEMA_VERSION:
        raise EnvironmentalFleetRuntimeStateReleaseBundleMigrationRequired(
            f"release-bundle schema {schema_version} requires explicit migration; current schema is {_SCHEMA_VERSION}"
        )
    required = {
        "format",
        "schema_version",
        "content_sha256",
        "content",
        "executable_restore_authorized",
    }
    if set(raw) != required:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle top-level fields are invalid"
        )
    if raw["executable_restore_authorized"] is not False:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "portable release bundle cannot authorize executable restore"
        )
    content = raw["content"]
    if not isinstance(content, dict):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle content must be a JSON object"
        )
    expected = _validate_sha256(raw["content_sha256"], "release-bundle content")
    actual = sha256(_canonical_json(content)).hexdigest()
    if actual != expected:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle content SHA-256 mismatch"
        )
    _validate_content(content)
    return EnvironmentalFleetRuntimeStateReleaseBundle(
        schema_version=schema_version,
        content_sha256=expected,
        content=content,
        executable_restore_authorized=False,
    )


def _validate_content(content: dict[str, Any]) -> None:
    required = {
        "producer_runtime_version",
        "left_source_document",
        "right_source_document",
        "left_current_document",
        "right_current_document",
        "compatibility_report",
        "policy",
        "gate_report",
        "digests",
        "decision",
        "portable_release_gate_passed",
        "human_review_required",
        "executable_restore_authorized",
    }
    if set(content) != required:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle content fields are invalid"
        )
    if not isinstance(content["producer_runtime_version"], str) or not content[
        "producer_runtime_version"
    ]:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle producer runtime version is empty"
        )
    if content["executable_restore_authorized"] is not False:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle content cannot authorize executable restore"
        )

    object_fields = (
        "left_source_document",
        "right_source_document",
        "left_current_document",
        "right_current_document",
        "compatibility_report",
        "policy",
        "gate_report",
        "digests",
    )
    for field in object_fields:
        if not isinstance(content[field], dict):
            raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
                f"release-bundle {field} must be a JSON object"
            )

    digests = content["digests"]
    required_digests = {
        "left_source_document_sha256",
        "right_source_document_sha256",
        "left_current_document_sha256",
        "right_current_document_sha256",
        "compatibility_report_sha256",
        "policy_sha256",
        "gate_report_sha256",
    }
    if set(digests) != required_digests:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle digest fields are invalid"
        )
    calculated = {
        "left_source_document_sha256": _json_sha256(content["left_source_document"]),
        "right_source_document_sha256": _json_sha256(content["right_source_document"]),
        "left_current_document_sha256": _json_sha256(content["left_current_document"]),
        "right_current_document_sha256": _json_sha256(content["right_current_document"]),
        "compatibility_report_sha256": _json_sha256(content["compatibility_report"]),
        "policy_sha256": _json_sha256(content["policy"]),
        "gate_report_sha256": _json_sha256(content["gate_report"]),
    }
    for key, value in calculated.items():
        if _validate_sha256(digests.get(key), key) != value:
            raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
                f"release-bundle {key} does not match embedded evidence"
            )

    left_current = _read_current_export(content["left_current_document"], "left")
    right_current = _read_current_export(content["right_current_document"], "right")
    report = content["compatibility_report"]
    _validate_report_links(
        report,
        content["left_source_document"],
        content["right_source_document"],
        left_current,
        right_current,
        calculated,
    )
    policy = _policy_from_json(content["policy"])
    _validate_gate_links(content["gate_report"], report, policy, calculated)

    gate = content["gate_report"]
    if content["decision"] != gate.get("decision"):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle decision does not match gate report"
        )
    if content["portable_release_gate_passed"] is not gate.get(
        "portable_release_gate_passed"
    ):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle pass flag does not match gate report"
        )
    if content["human_review_required"] is not gate.get("human_review_required"):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "release-bundle review flag does not match gate report"
        )


def _validate_report_links(
    report: dict[str, Any],
    left_source: dict[str, Any],
    right_source: dict[str, Any],
    left_current: EnvironmentalFleetRuntimeStateExport,
    right_current: EnvironmentalFleetRuntimeStateExport,
    digests: dict[str, str],
) -> None:
    required = {
        "format",
        "left",
        "right",
        "compatibility",
        "findings",
        "executable_restore_authorized",
    }
    if set(report) != required or report.get("format") != (
        "vessel-crossing-environmental-runtime-state-compatibility-report"
    ):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A133 compatibility report shape is invalid"
        )
    if report["executable_restore_authorized"] is not False:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A133 report cannot authorize executable restore"
        )
    if not isinstance(report["findings"], list):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A133 findings must be an array"
        )

    _validate_report_side(
        report["left"], left_source, left_current, digests["left_source_document_sha256"], digests["left_current_document_sha256"]
    )
    _validate_report_side(
        report["right"], right_source, right_current, digests["right_source_document_sha256"], digests["right_current_document_sha256"]
    )

    impacts: set[EnvironmentalFleetRuntimeStateSemanticImpact] = set()
    for raw in report["findings"]:
        finding = _semantic_finding_from_json(raw)
        impacts.add(finding.impact)
    if not impacts:
        expected = EnvironmentalFleetRuntimeStateCompatibility.EXACT
    elif EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY in impacts:
        expected = EnvironmentalFleetRuntimeStateCompatibility.CURRENT_STATE_OR_AUTHORITY_DIFFERENCE
    elif EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR in impacts:
        expected = EnvironmentalFleetRuntimeStateCompatibility.FUTURE_BEHAVIOR_DIFFERENCE
    else:
        expected = EnvironmentalFleetRuntimeStateCompatibility.EVIDENCE_ONLY_DIFFERENCE
    if report["compatibility"] != expected.value:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A133 compatibility summary does not match findings"
        )


def _validate_report_side(
    side: Any,
    source_document: dict[str, Any],
    current_export: EnvironmentalFleetRuntimeStateExport,
    source_sha256: str,
    current_sha256: str,
) -> None:
    required = {
        "source_schema_version",
        "target_schema_version",
        "schema_disposition",
        "source_runtime_version",
        "source_document_sha256",
        "current_document_sha256",
        "audit_sha256",
        "migration_transform_ids",
    }
    if not isinstance(side, dict) or set(side) != required:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A133 report side fields are invalid"
        )
    try:
        source_schema = int(source_document.get("schema_version"))
    except (TypeError, ValueError) as exc:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "source portable document schema version is invalid"
        ) from exc
    if int(side["source_schema_version"]) != source_schema:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "A133 source schema does not match embedded source document"
        )
    if int(side["target_schema_version"]) != current_export.schema_version:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "A133 target schema does not match embedded current document"
        )
    if side["source_document_sha256"] != source_sha256:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "A133 source document SHA-256 does not match embedded source document"
        )
    if side["current_document_sha256"] != current_sha256:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "A133 current document SHA-256 does not match embedded current document"
        )
    if side["audit_sha256"] != current_export.audit_sha256:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "A133 audit SHA-256 does not match embedded current document"
        )
    if side["source_runtime_version"] != current_export.source_runtime_version:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "A133 source runtime version does not match embedded current document"
        )
    transform_ids = [item.transform_id for item in current_export.migration_provenance]
    if side["migration_transform_ids"] != transform_ids:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "A133 migration transform IDs do not match embedded current document"
        )
    expected_disposition = "CURRENT_SCHEMA" if source_schema == current_export.schema_version else "MIGRATED_TO_CURRENT"
    if side["schema_disposition"] != expected_disposition:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "A133 schema disposition does not match embedded documents"
        )
    if source_schema != current_export.schema_version:
        if not current_export.migration_provenance:
            raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
                "migrated current document lacks provenance"
            )
        first = current_export.migration_provenance[0]
        if first.input_document_sha256 != source_sha256:
            raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
                "current document migration provenance is not bound to embedded source document"
            )
        source_audit_sha256 = source_document.get("audit_sha256")
        if first.input_audit_sha256 != source_audit_sha256:
            raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
                "current document migration provenance is not bound to embedded source audit"
            )


def _validate_gate_links(
    gate: dict[str, Any],
    report: dict[str, Any],
    policy: EnvironmentalFleetRuntimeStateCompatibilityPolicy,
    digests: dict[str, str],
) -> None:
    required = {
        "format",
        "policy",
        "policy_sha256",
        "compatibility_report_sha256",
        "compatibility",
        "decision",
        "portable_release_gate_passed",
        "human_review_required",
        "findings",
        "executable_restore_authorized",
    }
    if not isinstance(gate, dict) or set(gate) != required or gate.get("format") != (
        "vessel-crossing-environmental-runtime-state-policy-gate-report"
    ):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 gate report shape is invalid"
        )
    if gate["executable_restore_authorized"] is not False:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 gate cannot authorize executable restore"
        )
    if gate["policy"] != policy.to_json_object():
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 gate policy does not match bundled policy"
        )
    if gate["policy_sha256"] != digests["policy_sha256"]:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 policy SHA-256 does not match bundled policy"
        )
    if gate["compatibility_report_sha256"] != digests["compatibility_report_sha256"]:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 report SHA-256 does not match bundled A133 report"
        )
    if gate["compatibility"] != report["compatibility"]:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 compatibility summary does not match A133 report"
        )
    if not isinstance(gate["findings"], list) or len(gate["findings"]) != len(report["findings"]):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 finding count does not match A133 report"
        )

    actions: list[EnvironmentalFleetRuntimeStateGateAction] = []
    for report_raw, gate_raw in zip(report["findings"], gate["findings"], strict=True):
        finding = _semantic_finding_from_json(report_raw)
        action = policy.action_for(finding)
        expected = {
            "path": finding.path,
            "kind": finding.kind,
            "domain": finding.domain.value,
            "impact": finding.impact.value,
            "action": action.value,
            "before": finding.before,
            "after": finding.after,
        }
        if gate_raw != expected:
            raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
                "embedded A134 finding does not match A133 evidence and policy"
            )
        actions.append(action)

    if EnvironmentalFleetRuntimeStateGateAction.BLOCK in actions:
        decision = EnvironmentalFleetRuntimeStateGateDecision.BLOCKED
    elif EnvironmentalFleetRuntimeStateGateAction.REQUIRE_REVIEW in actions:
        decision = EnvironmentalFleetRuntimeStateGateDecision.REVIEW_REQUIRED
    else:
        decision = EnvironmentalFleetRuntimeStateGateDecision.PASS
    if gate["decision"] != decision.value:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 decision does not match policy actions"
        )
    if gate["portable_release_gate_passed"] is not (decision is EnvironmentalFleetRuntimeStateGateDecision.PASS):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 pass flag does not match decision"
        )
    if gate["human_review_required"] is not (decision is EnvironmentalFleetRuntimeStateGateDecision.REVIEW_REQUIRED):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 review flag does not match decision"
        )


def _policy_from_json(raw: dict[str, Any]) -> EnvironmentalFleetRuntimeStateCompatibilityPolicy:
    required = {
        "format",
        "policy_id",
        "evidence_only_action",
        "future_behavior_action",
        "current_state_or_authority_action",
        "domain_overrides",
    }
    if set(raw) != required or raw.get("format") != (
        "vessel-crossing-environmental-runtime-state-compatibility-policy"
    ):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 policy shape is invalid"
        )
    overrides_raw = raw["domain_overrides"]
    if not isinstance(overrides_raw, list):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 policy domain overrides must be an array"
        )
    try:
        overrides = tuple(
            EnvironmentalFleetRuntimeStatePolicyDomainOverride(
                EnvironmentalFleetRuntimeStateSemanticDomain(item["domain"]),
                EnvironmentalFleetRuntimeStateGateAction(item["action"]),
            )
            for item in overrides_raw
            if isinstance(item, dict) and set(item) == {"domain", "action"}
        )
        if len(overrides) != len(overrides_raw):
            raise ValueError("invalid override shape")
        policy = EnvironmentalFleetRuntimeStateCompatibilityPolicy(
            policy_id=raw["policy_id"],
            evidence_only_action=EnvironmentalFleetRuntimeStateGateAction(raw["evidence_only_action"]),
            future_behavior_action=EnvironmentalFleetRuntimeStateGateAction(raw["future_behavior_action"]),
            current_state_or_authority_action=EnvironmentalFleetRuntimeStateGateAction(
                raw["current_state_or_authority_action"]
            ),
            domain_overrides=overrides,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 policy values are invalid"
        ) from exc
    if policy.to_json_object() != raw:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A134 policy is not canonical"
        )
    return policy


def _semantic_finding_from_json(raw: Any) -> EnvironmentalFleetRuntimeStateSemanticFinding:
    required = {"path", "kind", "domain", "impact", "before", "after"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A133 semantic finding shape is invalid"
        )
    try:
        return EnvironmentalFleetRuntimeStateSemanticFinding(
            path=str(raw["path"]),
            kind=str(raw["kind"]),
            domain=EnvironmentalFleetRuntimeStateSemanticDomain(raw["domain"]),
            impact=EnvironmentalFleetRuntimeStateSemanticImpact(raw["impact"]),
            before=raw["before"],
            after=raw["after"],
        )
    except ValueError as exc:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            "embedded A133 semantic finding enum value is invalid"
        ) from exc


def _read_current_export(raw: dict[str, Any], side: str) -> EnvironmentalFleetRuntimeStateExport:
    try:
        return EnvironmentalFleetRuntimeStateExport.from_bytes(_canonical_json(raw))
    except Exception as exc:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            f"embedded {side} current portable document is invalid"
        ) from exc


def _load_json_object(data: bytes | bytearray | memoryview, label: str) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes-like")
    try:
        raw = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            f"{label} must be a JSON object"
        )
    return raw


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            f"{label} SHA-256 is invalid"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise EnvironmentalFleetRuntimeStateReleaseBundleIntegrityError(
            f"{label} SHA-256 is invalid"
        ) from exc
    return value.lower()


def _json_sha256(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
