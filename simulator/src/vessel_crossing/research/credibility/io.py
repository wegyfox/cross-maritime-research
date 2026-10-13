"""Canonical JSON serialization and strict parsing for credibility records."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .records import (
    AIContribution,
    AIContributionCategory,
    AIResearchProvenance,
    ArtifactReference,
    AuditChallenge,
    AuditChallengeStatus,
    CredibilityKillAudit,
    CredibilityRecordError,
    EvidenceKind,
    EvidenceReference,
    ReproducibilityManifest,
    ReproducibilityStatus,
    TestFamilyRecord,
    TestFamilyStatus,
    ValidationMatrix,
    ValidationMatrixEntry,
    ValidationStatus,
    VerificationRegister,
    VerificationStatus,
)

Record = (
    VerificationRegister
    | ValidationMatrix
    | CredibilityKillAudit
    | ReproducibilityManifest
    | AIResearchProvenance
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def payload(record: Record) -> dict[str, Any]:
    """Convert a typed record to its JSON-compatible canonical payload."""
    return _jsonable(asdict(record))


def canonical_bytes(record: Record) -> bytes:
    """Return deterministic UTF-8 JSON bytes for hashing and archival."""
    rendered = json.dumps(
        payload(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (rendered + "\n").encode("utf-8")


def record_sha256(record: Record) -> str:
    return hashlib.sha256(canonical_bytes(record)).hexdigest()


def dump_record(record: Record, path: str | Path) -> None:
    """Atomically write one canonical credibility record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(record))
    os.replace(temporary, destination)


def _reject_extra_fields(
    data: dict[str, Any], allowed: set[str], record_name: str
) -> None:
    extra = set(data) - allowed
    if extra:
        raise CredibilityRecordError(
            f"{record_name} contains unsupported fields: {sorted(extra)!r}"
        )


def _enum_value(enum_type: type[Enum], value: object, field_name: str) -> Enum:
    try:
        return enum_type(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        choices = [item.value for item in enum_type]
        raise CredibilityRecordError(
            f"{field_name} must be one of {choices}"
        ) from exc


def _artifact(data: dict[str, Any] | None) -> ArtifactReference | None:
    if data is None:
        return None
    _reject_extra_fields(
        data,
        {"locator", "role", "sha256", "immutable"},
        "artifact_reference",
    )
    return ArtifactReference(
        locator=data["locator"],
        role=data["role"],
        sha256=data.get("sha256"),
        immutable=data.get("immutable", False),
    )


def _evidence(data: dict[str, Any]) -> EvidenceReference:
    _reject_extra_fields(
        data,
        {"evidence_id", "kind", "description", "artefact"},
        "evidence_reference",
    )
    return EvidenceReference(
        evidence_id=data["evidence_id"],
        kind=_enum_value(EvidenceKind, data["kind"], "kind"),  # type: ignore[arg-type]
        description=data["description"],
        artefact=_artifact(data.get("artefact")),
    )


def _verification_register(data: dict[str, Any]) -> VerificationRegister:
    _reject_extra_fields(
        data,
        {"record_type", "schema_version", "study_id", "families"},
        "verification_register",
    )
    families: list[TestFamilyRecord] = []
    for item in data["families"]:
        _reject_extra_fields(
            item,
            {
                "family_id",
                "title",
                "status",
                "failure_mode_detected",
                "does_not_prove",
                "scientific_relevance",
                "evidence",
            },
            "test_family",
        )
        families.append(
            TestFamilyRecord(
                family_id=item["family_id"],
                title=item["title"],
                status=_enum_value(  # type: ignore[arg-type]
                    TestFamilyStatus, item["status"], "status"
                ),
                failure_mode_detected=item["failure_mode_detected"],
                does_not_prove=item["does_not_prove"],
                scientific_relevance=item["scientific_relevance"],
                evidence=tuple(_evidence(entry) for entry in item.get("evidence", [])),
            )
        )
    return VerificationRegister(
        schema_version=data["schema_version"],
        study_id=data["study_id"],
        families=tuple(families),
    )


def _validation_matrix(data: dict[str, Any]) -> ValidationMatrix:
    _reject_extra_fields(
        data,
        {
            "record_type",
            "schema_version",
            "study_id",
            "intended_scientific_use",
            "entries",
        },
        "validation_matrix",
    )
    entries: list[ValidationMatrixEntry] = []
    for item in data["entries"]:
        _reject_extra_fields(
            item,
            {
                "component_id",
                "intended_role",
                "material_to_claim",
                "verification_status",
                "validation_status",
                "verification_evidence",
                "validation_evidence",
                "robustness_evidence",
                "known_limitations",
                "remaining_actions",
            },
            "validation_matrix_entry",
        )
        entries.append(
            ValidationMatrixEntry(
                component_id=item["component_id"],
                intended_role=item["intended_role"],
                material_to_claim=item["material_to_claim"],
                verification_status=_enum_value(  # type: ignore[arg-type]
                    VerificationStatus,
                    item["verification_status"],
                    "verification_status",
                ),
                validation_status=_enum_value(  # type: ignore[arg-type]
                    ValidationStatus,
                    item["validation_status"],
                    "validation_status",
                ),
                verification_evidence=tuple(
                    _evidence(entry)
                    for entry in item.get("verification_evidence", [])
                ),
                validation_evidence=tuple(
                    _evidence(entry)
                    for entry in item.get("validation_evidence", [])
                ),
                robustness_evidence=tuple(
                    _evidence(entry)
                    for entry in item.get("robustness_evidence", [])
                ),
                known_limitations=tuple(item.get("known_limitations", [])),
                remaining_actions=tuple(item.get("remaining_actions", [])),
            )
        )
    return ValidationMatrix(
        schema_version=data["schema_version"],
        study_id=data["study_id"],
        intended_scientific_use=data["intended_scientific_use"],
        entries=tuple(entries),
    )


def _kill_audit(data: dict[str, Any]) -> CredibilityKillAudit:
    _reject_extra_fields(
        data,
        {"record_type", "schema_version", "study_id", "challenges"},
        "credibility_kill_audit",
    )
    challenges: list[AuditChallenge] = []
    for item in data["challenges"]:
        _reject_extra_fields(
            item,
            {
                "challenge_id",
                "category",
                "question",
                "status",
                "evidence",
                "disposition",
            },
            "audit_challenge",
        )
        challenges.append(
            AuditChallenge(
                challenge_id=item["challenge_id"],
                category=item["category"],
                question=item["question"],
                status=_enum_value(  # type: ignore[arg-type]
                    AuditChallengeStatus, item["status"], "status"
                ),
                evidence=tuple(_evidence(entry) for entry in item.get("evidence", [])),
                disposition=item.get("disposition"),
            )
        )
    return CredibilityKillAudit(
        schema_version=data["schema_version"],
        study_id=data["study_id"],
        challenges=tuple(challenges),
    )


def _reproducibility_manifest(data: dict[str, Any]) -> ReproducibilityManifest:
    _reject_extra_fields(
        data,
        {
            "record_type",
            "schema_version",
            "study_id",
            "status",
            "source_version",
            "source_archive",
            "runtime_version",
            "platform_notes",
            "dependency_lock",
            "protocol",
            "configs",
            "seed_schedules",
            "estimands",
            "stopping_rules",
            "raw_evidence",
            "analysis_artefacts",
            "expected_outputs",
            "regeneration_steps",
            "deterministic_caveats",
        },
        "reproducibility_manifest",
    )
    return ReproducibilityManifest(
        schema_version=data["schema_version"],
        study_id=data["study_id"],
        status=_enum_value(  # type: ignore[arg-type]
            ReproducibilityStatus, data["status"], "status"
        ),
        source_version=data["source_version"],
        source_archive=_artifact(data.get("source_archive")),
        runtime_version=data["runtime_version"],
        platform_notes=data["platform_notes"],
        dependency_lock=_artifact(data.get("dependency_lock")),
        protocol=_artifact(data.get("protocol")),
        configs=tuple(
            _artifact(item) for item in data.get("configs", [])
        ),  # type: ignore[arg-type]
        seed_schedules=tuple(
            _artifact(item) for item in data.get("seed_schedules", [])
        ),  # type: ignore[arg-type]
        estimands=tuple(data.get("estimands", [])),
        stopping_rules=tuple(data.get("stopping_rules", [])),
        raw_evidence=tuple(
            _artifact(item) for item in data.get("raw_evidence", [])
        ),  # type: ignore[arg-type]
        analysis_artefacts=tuple(
            _artifact(item) for item in data.get("analysis_artefacts", [])
        ),  # type: ignore[arg-type]
        expected_outputs=tuple(
            _artifact(item) for item in data.get("expected_outputs", [])
        ),  # type: ignore[arg-type]
        regeneration_steps=tuple(data.get("regeneration_steps", [])),
        deterministic_caveats=tuple(data.get("deterministic_caveats", [])),
    )


def _ai_provenance(data: dict[str, Any]) -> AIResearchProvenance:
    _reject_extra_fields(
        data,
        {
            "record_type",
            "schema_version",
            "study_id",
            "ai_assistance_used",
            "contributions",
        },
        "ai_research_provenance",
    )
    contributions: list[AIContribution] = []
    for item in data.get("contributions", []):
        _reject_extra_fields(
            item,
            {
                "contribution_id",
                "system_or_tool",
                "category",
                "description",
                "human_review",
                "automated_checks",
            },
            "ai_contribution",
        )
        contributions.append(
            AIContribution(
                contribution_id=item["contribution_id"],
                system_or_tool=item["system_or_tool"],
                category=_enum_value(  # type: ignore[arg-type]
                    AIContributionCategory, item["category"], "category"
                ),
                description=item["description"],
                human_review=item["human_review"],
                automated_checks=tuple(item.get("automated_checks", [])),
            )
        )
    return AIResearchProvenance(
        schema_version=data["schema_version"],
        study_id=data["study_id"],
        ai_assistance_used=data["ai_assistance_used"],
        contributions=tuple(contributions),
    )


def record_from_payload(data: dict[str, Any]) -> Record:
    """Parse one JSON payload into a strict typed record."""
    if not isinstance(data, dict):
        raise CredibilityRecordError("record payload must be a JSON object")

    record_type = data.get("record_type")
    parsers = {
        "verification_register": _verification_register,
        "validation_matrix": _validation_matrix,
        "credibility_kill_audit": _kill_audit,
        "reproducibility_manifest": _reproducibility_manifest,
        "ai_research_provenance": _ai_provenance,
    }
    parser = parsers.get(record_type)
    if parser is None:
        raise CredibilityRecordError(f"unsupported record_type: {record_type!r}")
    return parser(data)


def load_record(path: str | Path) -> Record:
    """Load one UTF-8 JSON credibility record from disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return record_from_payload(data)
