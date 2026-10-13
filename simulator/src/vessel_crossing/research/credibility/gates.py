"""Fail-closed structural publication-readiness gates.

These gates do not decide whether a scientific claim is true. They only verify
that the evidence records required by the CROSS publication mandate are present
and internally consistent enough to proceed to human scientific review.
"""

from __future__ import annotations

from dataclasses import dataclass

from .records import (
    AIResearchProvenance,
    ArtifactReference,
    AuditChallengeStatus,
    CredibilityKillAudit,
    ReproducibilityManifest,
    ReproducibilityStatus,
    TestFamilyStatus,
    ValidationMatrix,
    ValidationStatus,
    VerificationRegister,
    VerificationStatus,
)
from .templates import required_kill_audit_challenge_ids


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    """Result of one structural publication-readiness gate."""

    gate_id: str
    passed: bool
    issues: tuple[str, ...]


def evaluate_model_credibility_gate(
    matrix: ValidationMatrix,
    audit: CredibilityKillAudit,
    verification_register: VerificationRegister | None = None,
) -> GateEvaluation:
    """Evaluate Gate A: model credibility / verification and validation."""
    issues: list[str] = []

    if matrix.study_id != audit.study_id:
        issues.append("validation matrix and kill audit study_id differ")

    if verification_register is None:
        issues.append("verification test-family register is required")
    elif verification_register.study_id != matrix.study_id:
        issues.append("verification register and validation matrix study_id differ")
    else:
        if not verification_register.families:
            issues.append("verification test-family register is empty")
        for family in verification_register.families:
            if family.status is not TestFamilyStatus.PASS:
                issues.append(
                    f"{family.family_id}: verification family status is "
                    f"{family.status.value}"
                )

    material_entries = [entry for entry in matrix.entries if entry.material_to_claim]
    if not material_entries:
        issues.append("validation matrix has no components marked material_to_claim")

    for entry in material_entries:
        if entry.verification_status is not VerificationStatus.VERIFIED:
            issues.append(f"{entry.component_id}: implementation is not VERIFIED")
        if entry.validation_status in {
            ValidationStatus.NOT_ASSESSED,
            ValidationStatus.FAILED,
        }:
            issues.append(
                f"{entry.component_id}: validation status is "
                f"{entry.validation_status.value}"
            )
        if entry.validation_status in {
            ValidationStatus.PARTIAL,
            ValidationStatus.UNRESOLVED,
        } and not entry.known_limitations:
            issues.append(
                f"{entry.component_id}: unresolved validation lacks explicit limitation"
            )

    if not audit.challenges:
        issues.append("credibility kill audit has no challenges")
    required_challenges = set(required_kill_audit_challenge_ids())
    present_challenges = {challenge.challenge_id for challenge in audit.challenges}
    for missing_id in sorted(required_challenges - present_challenges):
        issues.append(f"{missing_id}: required kill-audit challenge is missing")
    for challenge in audit.challenges:
        if challenge.status is AuditChallengeStatus.NOT_RUN:
            issues.append(f"{challenge.challenge_id}: kill-audit challenge not run")
        elif challenge.status is AuditChallengeStatus.FAILED:
            # A documented disposition explains the failure; it does not convert
            # a failed falsification challenge into a passing credibility gate.
            issues.append(f"{challenge.challenge_id}: kill-audit challenge FAILED")

    return GateEvaluation(
        gate_id="MODEL_CREDIBILITY_VV",
        passed=not issues,
        issues=tuple(issues),
    )


def _is_hashed(reference: ArtifactReference | None) -> bool:
    return reference is not None and reference.sha256 is not None


def evaluate_reproducibility_gate(
    manifest: ReproducibilityManifest,
    ai_provenance: AIResearchProvenance | None = None,
) -> GateEvaluation:
    """Evaluate Gate B: reproducibility and provenance completeness."""
    issues: list[str] = []

    if ai_provenance is None:
        issues.append("AI research provenance record is required")
    elif ai_provenance.study_id != manifest.study_id:
        issues.append("AI provenance and reproducibility manifest study_id differ")

    if manifest.status is not ReproducibilityStatus.FROZEN:
        issues.append("reproducibility manifest is not FROZEN")

    required_singletons = (
        (manifest.source_archive, "source archive with SHA-256 is required"),
        (manifest.dependency_lock, "dependency lock with SHA-256 is required"),
        (manifest.protocol, "frozen protocol with SHA-256 is required"),
    )
    for reference, message in required_singletons:
        if not _is_hashed(reference):
            issues.append(message)

    required_collections = (
        (manifest.configs, "all experiment configs must be present and hashed"),
        (manifest.seed_schedules, "seed schedule(s) must be present and hashed"),
        (manifest.raw_evidence, "raw evidence must be present and hashed"),
        (
            manifest.analysis_artefacts,
            "analysis artefacts must be present and hashed",
        ),
        (manifest.expected_outputs, "expected outputs must be present and hashed"),
    )
    for references, message in required_collections:
        if not references or any(reference.sha256 is None for reference in references):
            issues.append(message)

    if not manifest.estimands:
        issues.append("estimands must be explicit")
    if not manifest.stopping_rules:
        issues.append("stopping rules must be explicit")
    if len(manifest.regeneration_steps) < 5:
        issues.append("regeneration procedure is incomplete")

    return GateEvaluation(
        gate_id="REPRODUCIBILITY_PROVENANCE",
        passed=not issues,
        issues=tuple(issues),
    )
