"""Typed evidence records for model credibility and reproducibility.

The records in this module are intentionally strict. They represent publication
readiness evidence, so malformed status values or partially specified immutable
artifacts must fail closed instead of being silently normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CredibilityRecordError(ValueError):
    """Raised when a credibility record violates its schema or invariants."""


class VerificationStatus(str, Enum):
    NOT_ASSESSED = "NOT_ASSESSED"
    PARTIAL = "PARTIAL"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class ValidationStatus(str, Enum):
    NOT_ASSESSED = "NOT_ASSESSED"
    PARTIAL = "PARTIAL"
    SUPPORTED_FOR_INTENDED_USE = "SUPPORTED_FOR_INTENDED_USE"
    UNRESOLVED = "UNRESOLVED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class TestFamilyStatus(str, Enum):
    PLANNED = "PLANNED"
    PARTIAL = "PARTIAL"
    PASS = "PASS"
    FAIL = "FAIL"


class AuditChallengeStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    SURVIVED = "SURVIVED"
    FAILED = "FAILED"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ReproducibilityStatus(str, Enum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"


class EvidenceKind(str, Enum):
    TEST_REPORT = "TEST_REPORT"
    ANALYTICAL_CASE = "ANALYTICAL_CASE"
    DETERMINISTIC_REPLAY = "DETERMINISTIC_REPLAY"
    SENSITIVITY = "SENSITIVITY"
    REFERENCE_IMPLEMENTATION = "REFERENCE_IMPLEMENTATION"
    EXTERNAL_DATA = "EXTERNAL_DATA"
    EXTERNAL_MODEL = "EXTERNAL_MODEL"
    LITERATURE = "LITERATURE"
    RAW_RESULT = "RAW_RESULT"
    ANALYSIS = "ANALYSIS"
    DECISION_RECORD = "DECISION_RECORD"
    OTHER = "OTHER"


class AIContributionCategory(str, Enum):
    RESEARCH_CODE = "RESEARCH_CODE"
    LITERATURE_ANALYSIS = "LITERATURE_ANALYSIS"
    EXPERIMENTAL_DESIGN = "EXPERIMENTAL_DESIGN"
    STATISTICAL_ANALYSIS = "STATISTICAL_ANALYSIS"
    MANUSCRIPT = "MANUSCRIPT"
    DOCUMENTATION = "DOCUMENTATION"


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CredibilityRecordError(f"{field_name} must be a non-empty string")


def _require_schema_v1(value: object, field_name: str) -> None:
    # bool is a subclass of int in Python; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise CredibilityRecordError(f"{field_name} must be integer 1")


def _require_enum(value: object, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise CredibilityRecordError(
            f"{field_name} must be {enum_type.__name__}"
        )


def _require_unique(values: Iterable[str], field_name: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise CredibilityRecordError(f"{field_name} must contain unique values")


def _require_sha256(value: str | None, field_name: str) -> None:
    if value is not None and not _SHA256_RE.fullmatch(value):
        raise CredibilityRecordError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )


def _require_tuple(value: object, field_name: str) -> None:
    if type(value) is not tuple:
        raise CredibilityRecordError(f"{field_name} must be an immutable tuple")


def _require_typed_tuple(
    values: object,
    item_type: type[object],
    field_name: str,
) -> None:
    _require_tuple(values, field_name)
    if any(not isinstance(value, item_type) for value in values):
        raise CredibilityRecordError(
            f"{field_name} must contain only {item_type.__name__} values"
        )


def _require_optional_type(
    value: object,
    expected_type: type[object],
    field_name: str,
) -> None:
    if value is not None and not isinstance(value, expected_type):
        raise CredibilityRecordError(
            f"{field_name} must be {expected_type.__name__} or None"
        )


def _require_record_type(value: object, expected: str) -> None:
    if value != expected:
        raise CredibilityRecordError(f"record_type must be {expected!r}")


def _require_text_tuple(values: tuple[str, ...], field_name: str) -> None:
    _require_tuple(values, field_name)
    for value in values:
        _require_text(value, field_name)


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Reference to an evidence artifact, optionally bound by SHA-256."""

    locator: str
    role: str
    sha256: str | None = None
    immutable: bool = False

    def __post_init__(self) -> None:
        _require_text(self.locator, "artifact locator")
        _require_text(self.role, "artifact role")
        _require_sha256(self.sha256, "artifact sha256")
        if not isinstance(self.immutable, bool):
            raise CredibilityRecordError("immutable must be boolean")
        if self.immutable and self.sha256 is None:
            raise CredibilityRecordError("immutable artefact requires sha256")


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """One traceable item supporting verification, validation, or audit work."""

    evidence_id: str
    kind: EvidenceKind
    description: str
    artefact: ArtifactReference | None = None

    def __post_init__(self) -> None:
        _require_text(self.evidence_id, "evidence_id")
        _require_enum(self.kind, EvidenceKind, "kind")
        _require_text(self.description, "evidence description")
        _require_optional_type(self.artefact, ArtifactReference, "evidence artefact")


@dataclass(frozen=True, slots=True)
class TestFamilyRecord:
    """Scientific meaning of one verification test family."""

    family_id: str
    title: str
    status: TestFamilyStatus
    failure_mode_detected: str
    does_not_prove: str
    scientific_relevance: str
    evidence: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "family_id",
            "title",
            "failure_mode_detected",
            "does_not_prove",
            "scientific_relevance",
        ):
            _require_text(getattr(self, field_name), field_name)
        _require_enum(self.status, TestFamilyStatus, "test-family status")
        _require_typed_tuple(self.evidence, EvidenceReference, "test-family evidence")
        _require_unique(
            (item.evidence_id for item in self.evidence),
            "test-family evidence ids",
        )
        if self.status is TestFamilyStatus.PASS and not self.evidence:
            raise CredibilityRecordError("PASS test family requires evidence")


@dataclass(frozen=True, slots=True)
class VerificationRegister:
    """Verification evidence grouped by failure mode rather than test count."""

    schema_version: int
    study_id: str
    families: tuple[TestFamilyRecord, ...]
    record_type: str = "verification_register"

    def __post_init__(self) -> None:
        _require_schema_v1(
            self.schema_version, "verification register schema_version"
        )
        _require_text(self.study_id, "study_id")
        _require_typed_tuple(self.families, TestFamilyRecord, "verification families")
        _require_record_type(self.record_type, "verification_register")
        _require_unique(
            (family.family_id for family in self.families),
            "verification family ids",
        )


@dataclass(frozen=True, slots=True)
class ValidationMatrixEntry:
    """Verification/validation status for one model component or claim element."""

    component_id: str
    intended_role: str
    material_to_claim: bool
    verification_status: VerificationStatus
    validation_status: ValidationStatus
    verification_evidence: tuple[EvidenceReference, ...] = ()
    validation_evidence: tuple[EvidenceReference, ...] = ()
    robustness_evidence: tuple[EvidenceReference, ...] = ()
    known_limitations: tuple[str, ...] = ()
    remaining_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.component_id, "component_id")
        _require_text(self.intended_role, "intended_role")
        if not isinstance(self.material_to_claim, bool):
            raise CredibilityRecordError("material_to_claim must be boolean")
        _require_enum(
            self.verification_status,
            VerificationStatus,
            "verification_status",
        )
        _require_enum(
            self.validation_status,
            ValidationStatus,
            "validation_status",
        )
        _require_typed_tuple(
            self.verification_evidence, EvidenceReference, "verification evidence"
        )
        _require_typed_tuple(
            self.validation_evidence, EvidenceReference, "validation evidence"
        )
        _require_typed_tuple(
            self.robustness_evidence, EvidenceReference, "robustness evidence"
        )
        _require_text_tuple(self.known_limitations, "known limitation")
        _require_text_tuple(self.remaining_actions, "remaining action")

        if (
            self.verification_status is VerificationStatus.VERIFIED
            and not self.verification_evidence
        ):
            raise CredibilityRecordError(
                "VERIFIED component requires verification evidence"
            )
        if (
            self.validation_status
            is ValidationStatus.SUPPORTED_FOR_INTENDED_USE
            and not self.validation_evidence
        ):
            raise CredibilityRecordError(
                "SUPPORTED_FOR_INTENDED_USE requires validation evidence"
            )
        if self.validation_status in {
            ValidationStatus.PARTIAL,
            ValidationStatus.UNRESOLVED,
        } and not (self.known_limitations or self.remaining_actions):
            raise CredibilityRecordError(
                "PARTIAL/UNRESOLVED validation requires a limitation or remaining action"
            )


@dataclass(frozen=True, slots=True)
class ValidationMatrix:
    """Intended-use validation matrix for a publishable study."""

    schema_version: int
    study_id: str
    intended_scientific_use: str
    entries: tuple[ValidationMatrixEntry, ...]
    record_type: str = "validation_matrix"

    def __post_init__(self) -> None:
        _require_schema_v1(self.schema_version, "validation matrix schema_version")
        _require_text(self.study_id, "study_id")
        _require_text(self.intended_scientific_use, "intended_scientific_use")
        _require_typed_tuple(self.entries, ValidationMatrixEntry, "validation entries")
        _require_record_type(self.record_type, "validation_matrix")
        _require_unique(
            (entry.component_id for entry in self.entries),
            "validation component ids",
        )


@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    """Raw-to-paper reproducibility contract for one frozen study snapshot."""

    schema_version: int
    study_id: str
    status: ReproducibilityStatus
    source_version: str
    source_archive: ArtifactReference | None
    runtime_version: str
    platform_notes: str
    dependency_lock: ArtifactReference | None
    protocol: ArtifactReference | None
    configs: tuple[ArtifactReference, ...]
    seed_schedules: tuple[ArtifactReference, ...]
    estimands: tuple[str, ...]
    stopping_rules: tuple[str, ...]
    raw_evidence: tuple[ArtifactReference, ...]
    analysis_artefacts: tuple[ArtifactReference, ...]
    expected_outputs: tuple[ArtifactReference, ...]
    regeneration_steps: tuple[str, ...]
    deterministic_caveats: tuple[str, ...] = ()
    record_type: str = "reproducibility_manifest"

    def __post_init__(self) -> None:
        _require_schema_v1(
            self.schema_version, "reproducibility manifest schema_version"
        )
        _require_text(self.study_id, "study_id")
        _require_enum(self.status, ReproducibilityStatus, "reproducibility status")
        _require_text(self.source_version, "source_version")
        _require_text(self.runtime_version, "runtime_version")
        _require_text(self.platform_notes, "platform_notes")
        _require_optional_type(self.source_archive, ArtifactReference, "source_archive")
        _require_optional_type(self.dependency_lock, ArtifactReference, "dependency_lock")
        _require_optional_type(self.protocol, ArtifactReference, "protocol")
        artifact_collections = (
            self.configs,
            self.seed_schedules,
            self.raw_evidence,
            self.analysis_artefacts,
            self.expected_outputs,
        )
        for references in artifact_collections:
            _require_typed_tuple(
                references,
                ArtifactReference,
                "reproducibility artifact collections",
            )
        _require_text_tuple(self.estimands, "estimand")
        _require_text_tuple(self.stopping_rules, "stopping rule")
        _require_text_tuple(self.regeneration_steps, "regeneration step")
        _require_text_tuple(self.deterministic_caveats, "deterministic caveat")
        _require_record_type(self.record_type, "reproducibility_manifest")


@dataclass(frozen=True, slots=True)
class AuditChallenge:
    """One explicit attempt to falsify a material scientific result."""

    challenge_id: str
    category: str
    question: str
    status: AuditChallengeStatus
    evidence: tuple[EvidenceReference, ...] = ()
    disposition: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.challenge_id, "challenge_id")
        _require_text(self.category, "category")
        _require_text(self.question, "question")
        _require_enum(self.status, AuditChallengeStatus, "audit challenge status")
        _require_typed_tuple(self.evidence, EvidenceReference, "audit evidence")
        if self.disposition is not None:
            _require_text(self.disposition, "disposition")
        if self.status in {
            AuditChallengeStatus.FAILED,
            AuditChallengeStatus.UNRESOLVED,
        } and self.disposition is None:
            raise CredibilityRecordError(
                "FAILED/UNRESOLVED challenge requires disposition"
            )


@dataclass(frozen=True, slots=True)
class CredibilityKillAudit:
    """Result-credibility challenge register for a study."""

    schema_version: int
    study_id: str
    challenges: tuple[AuditChallenge, ...]
    record_type: str = "credibility_kill_audit"

    def __post_init__(self) -> None:
        _require_schema_v1(self.schema_version, "kill audit schema_version")
        _require_text(self.study_id, "study_id")
        _require_typed_tuple(self.challenges, AuditChallenge, "kill-audit challenges")
        _require_record_type(self.record_type, "credibility_kill_audit")
        _require_unique(
            (challenge.challenge_id for challenge in self.challenges),
            "kill-audit challenge ids",
        )


@dataclass(frozen=True, slots=True)
class AIContribution:
    """Practical provenance record for one category of AI-assisted work."""

    contribution_id: str
    system_or_tool: str
    category: AIContributionCategory
    description: str
    human_review: str
    automated_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.contribution_id, "contribution_id")
        _require_text(self.system_or_tool, "system_or_tool")
        _require_enum(self.category, AIContributionCategory, "AI contribution category")
        _require_text(self.description, "description")
        _require_text(self.human_review, "human_review")
        _require_text_tuple(self.automated_checks, "automated check")


@dataclass(frozen=True, slots=True)
class AIResearchProvenance:
    """Study-level disclosure-oriented summary of material AI assistance."""

    schema_version: int
    study_id: str
    ai_assistance_used: bool
    contributions: tuple[AIContribution, ...]
    record_type: str = "ai_research_provenance"

    def __post_init__(self) -> None:
        _require_schema_v1(self.schema_version, "AI provenance schema_version")
        _require_text(self.study_id, "study_id")
        if not isinstance(self.ai_assistance_used, bool):
            raise CredibilityRecordError("ai_assistance_used must be boolean")
        _require_typed_tuple(self.contributions, AIContribution, "AI contributions")
        _require_record_type(self.record_type, "ai_research_provenance")
        if self.ai_assistance_used and not self.contributions:
            raise CredibilityRecordError(
                "AI-assisted study requires at least one contribution record"
            )
        if not self.ai_assistance_used and self.contributions:
            raise CredibilityRecordError(
                "non-AI study cannot contain AI contribution records"
            )
        _require_unique(
            (item.contribution_id for item in self.contributions),
            "AI contribution ids",
        )
