"""Caller-selected release/audit policy over A133 portable compatibility reports.

This layer is deliberately non-executable. It can pass, require human review, or block a portable
release comparison, but it can never create, resume, or authorize simulator command authority.
Current-state/authority divergence is an invariant hard block and cannot be weakened by policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any

from .environmental_runtime_state_compatibility import (
    EnvironmentalFleetRuntimeStateCompatibilityReport,
    EnvironmentalFleetRuntimeStateSemanticDomain,
    EnvironmentalFleetRuntimeStateSemanticFinding,
    EnvironmentalFleetRuntimeStateSemanticImpact,
    report_environmental_fleet_runtime_state_compatibility,
)


class EnvironmentalFleetRuntimeStatePolicyError(ValueError):
    """Invalid portable compatibility policy configuration."""


class EnvironmentalFleetRuntimeStateGateAction(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_REVIEW = "REQUIRE_REVIEW"
    BLOCK = "BLOCK"


class EnvironmentalFleetRuntimeStateGateDecision(str, Enum):
    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStatePolicyDomainOverride:
    domain: EnvironmentalFleetRuntimeStateSemanticDomain
    action: EnvironmentalFleetRuntimeStateGateAction


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateCompatibilityPolicy:
    """Portable release policy selected by the caller.

    Evidence-only and future-behavior findings are configurable. Domain overrides take precedence for
    those two impacts. Current-state/authority findings always resolve to BLOCK regardless of policy.
    """

    policy_id: str = "portable-release-default-v1"
    evidence_only_action: EnvironmentalFleetRuntimeStateGateAction = EnvironmentalFleetRuntimeStateGateAction.ALLOW
    future_behavior_action: EnvironmentalFleetRuntimeStateGateAction = (
        EnvironmentalFleetRuntimeStateGateAction.REQUIRE_REVIEW
    )
    current_state_or_authority_action: EnvironmentalFleetRuntimeStateGateAction = (
        EnvironmentalFleetRuntimeStateGateAction.BLOCK
    )
    domain_overrides: tuple[EnvironmentalFleetRuntimeStatePolicyDomainOverride, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise EnvironmentalFleetRuntimeStatePolicyError("policy_id must be a non-empty string")
        for value, label in (
            (self.evidence_only_action, "evidence_only_action"),
            (self.future_behavior_action, "future_behavior_action"),
            (self.current_state_or_authority_action, "current_state_or_authority_action"),
        ):
            if not isinstance(value, EnvironmentalFleetRuntimeStateGateAction):
                raise EnvironmentalFleetRuntimeStatePolicyError(f"{label} must be EnvironmentalFleetRuntimeStateGateAction")
        if self.current_state_or_authority_action is not EnvironmentalFleetRuntimeStateGateAction.BLOCK:
            raise EnvironmentalFleetRuntimeStatePolicyError(
                "current_state_or_authority_action is an invariant hard BLOCK and cannot be weakened"
            )
        seen: set[EnvironmentalFleetRuntimeStateSemanticDomain] = set()
        for override in self.domain_overrides:
            if not isinstance(override, EnvironmentalFleetRuntimeStatePolicyDomainOverride):
                raise EnvironmentalFleetRuntimeStatePolicyError(
                    "domain_overrides must contain EnvironmentalFleetRuntimeStatePolicyDomainOverride values"
                )
            if override.domain in seen:
                raise EnvironmentalFleetRuntimeStatePolicyError(
                    f"duplicate policy domain override for {override.domain.value}"
                )
            seen.add(override.domain)

    def action_for(
        self,
        finding: EnvironmentalFleetRuntimeStateSemanticFinding,
    ) -> EnvironmentalFleetRuntimeStateGateAction:
        if finding.impact is EnvironmentalFleetRuntimeStateSemanticImpact.CURRENT_STATE_OR_AUTHORITY:
            return EnvironmentalFleetRuntimeStateGateAction.BLOCK
        for override in self.domain_overrides:
            if override.domain is finding.domain:
                return override.action
        if finding.impact is EnvironmentalFleetRuntimeStateSemanticImpact.EVIDENCE_ONLY:
            return self.evidence_only_action
        if finding.impact is EnvironmentalFleetRuntimeStateSemanticImpact.FUTURE_BEHAVIOR:
            return self.future_behavior_action
        raise EnvironmentalFleetRuntimeStatePolicyError(
            f"unknown semantic impact {finding.impact!r}"
        )

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": "vessel-crossing-environmental-runtime-state-compatibility-policy",
            "policy_id": self.policy_id,
            "evidence_only_action": self.evidence_only_action.value,
            "future_behavior_action": self.future_behavior_action.value,
            "current_state_or_authority_action": self.current_state_or_authority_action.value,
            "domain_overrides": [
                {"domain": override.domain.value, "action": override.action.value}
                for override in sorted(self.domain_overrides, key=lambda item: item.domain.value)
            ],
        }

    @property
    def sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateGateFinding:
    path: str
    kind: str
    domain: EnvironmentalFleetRuntimeStateSemanticDomain
    impact: EnvironmentalFleetRuntimeStateSemanticImpact
    action: EnvironmentalFleetRuntimeStateGateAction
    before: Any
    after: Any


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStatePolicyGateReport:
    policy: EnvironmentalFleetRuntimeStateCompatibilityPolicy
    compatibility_report: EnvironmentalFleetRuntimeStateCompatibilityReport
    decision: EnvironmentalFleetRuntimeStateGateDecision
    findings: tuple[EnvironmentalFleetRuntimeStateGateFinding, ...]
    policy_sha256: str
    compatibility_report_sha256: str
    executable_restore_authorized: bool = False

    @property
    def portable_release_gate_passed(self) -> bool:
        return self.decision is EnvironmentalFleetRuntimeStateGateDecision.PASS

    @property
    def human_review_required(self) -> bool:
        return self.decision is EnvironmentalFleetRuntimeStateGateDecision.REVIEW_REQUIRED

    def to_json_bytes(self) -> bytes:
        payload = {
            "format": "vessel-crossing-environmental-runtime-state-policy-gate-report",
            "policy": self.policy.to_json_object(),
            "policy_sha256": self.policy_sha256,
            "compatibility_report_sha256": self.compatibility_report_sha256,
            "compatibility": self.compatibility_report.compatibility.value,
            "decision": self.decision.value,
            "portable_release_gate_passed": self.portable_release_gate_passed,
            "human_review_required": self.human_review_required,
            "findings": [
                {
                    "path": finding.path,
                    "kind": finding.kind,
                    "domain": finding.domain.value,
                    "impact": finding.impact.value,
                    "action": finding.action.value,
                    "before": finding.before,
                    "after": finding.after,
                }
                for finding in self.findings
            ],
            "executable_restore_authorized": False,
        }
        return _canonical_json(payload)


def gate_environmental_fleet_runtime_state_compatibility_report(
    report: EnvironmentalFleetRuntimeStateCompatibilityReport,
    *,
    policy: EnvironmentalFleetRuntimeStateCompatibilityPolicy | None = None,
) -> EnvironmentalFleetRuntimeStatePolicyGateReport:
    """Evaluate one A133 report under a caller-selected non-executable release policy."""

    if not isinstance(report, EnvironmentalFleetRuntimeStateCompatibilityReport):
        raise TypeError("report must be EnvironmentalFleetRuntimeStateCompatibilityReport")
    selected = policy or EnvironmentalFleetRuntimeStateCompatibilityPolicy()
    if not isinstance(selected, EnvironmentalFleetRuntimeStateCompatibilityPolicy):
        raise TypeError("policy must be EnvironmentalFleetRuntimeStateCompatibilityPolicy")

    findings = tuple(
        EnvironmentalFleetRuntimeStateGateFinding(
            path=finding.path,
            kind=finding.kind,
            domain=finding.domain,
            impact=finding.impact,
            action=selected.action_for(finding),
            before=finding.before,
            after=finding.after,
        )
        for finding in report.findings
    )
    actions = {finding.action for finding in findings}
    if EnvironmentalFleetRuntimeStateGateAction.BLOCK in actions:
        decision = EnvironmentalFleetRuntimeStateGateDecision.BLOCKED
    elif EnvironmentalFleetRuntimeStateGateAction.REQUIRE_REVIEW in actions:
        decision = EnvironmentalFleetRuntimeStateGateDecision.REVIEW_REQUIRED
    else:
        decision = EnvironmentalFleetRuntimeStateGateDecision.PASS

    report_bytes = report.to_json_bytes()
    return EnvironmentalFleetRuntimeStatePolicyGateReport(
        policy=selected,
        compatibility_report=report,
        decision=decision,
        findings=findings,
        policy_sha256=selected.sha256,
        compatibility_report_sha256=sha256(report_bytes).hexdigest(),
        executable_restore_authorized=False,
    )


def compare_and_gate_environmental_fleet_runtime_state_documents(
    left: bytes | bytearray | memoryview,
    right: bytes | bytearray | memoryview,
    *,
    policy: EnvironmentalFleetRuntimeStateCompatibilityPolicy | None = None,
) -> EnvironmentalFleetRuntimeStatePolicyGateReport:
    """Build A133 compatibility evidence then apply the selected policy gate."""

    return gate_environmental_fleet_runtime_state_compatibility_report(
        report_environmental_fleet_runtime_state_compatibility(left, right),
        policy=policy,
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
