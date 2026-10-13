"""Detached Ed25519 authenticity/approval evidence over immutable A135 release bundles.

A136 deliberately keeps secret-key ownership outside simulator state.  It signs exact A135 bytes with
an external Ed25519 private key, stores only detached public evidence, and verifies that evidence under
an explicit caller-supplied trust policy.  A trusted approval may satisfy A135's human-review boundary,
but it can never override an A134 BLOCKED release gate and can never authorize executable restore.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import base64
import binascii
import json
from typing import Any, Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .environmental_runtime_state_policy import EnvironmentalFleetRuntimeStateGateDecision
from .environmental_runtime_state_release_bundle import (
    EnvironmentalFleetRuntimeStateReleaseBundle,
    read_environmental_fleet_runtime_state_release_bundle,
)

_SIGNATURE_FORMAT = "vessel-crossing-environmental-runtime-state-approval-signature"
_SIGNATURE_SCHEMA_VERSION = 1
_REPORT_FORMAT = "vessel-crossing-environmental-runtime-state-approval-verification-report"
_DOMAIN_SEPARATOR = b"vessel-crossing/a136/environmental-runtime-state-approval/ed25519/v1\x00"
_ALGORITHM = "Ed25519"
_APPROVAL = "APPROVE"


class EnvironmentalFleetRuntimeStateApprovalError(RuntimeError):
    """Base error for A136 detached approval evidence."""


class EnvironmentalFleetRuntimeStateApprovalIntegrityError(EnvironmentalFleetRuntimeStateApprovalError):
    """Detached approval JSON or cryptographic evidence is malformed/inconsistent."""


class EnvironmentalFleetRuntimeStateApprovalMigrationRequired(EnvironmentalFleetRuntimeStateApprovalError):
    """Detached approval schema is not understood by this runtime."""


class EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(ValueError):
    """Invalid A136 public-key trust policy."""


class EnvironmentalFleetRuntimeStateApprovalStatus(str, Enum):
    VALID = "VALID"
    WRONG_BUNDLE = "WRONG_BUNDLE"
    UNTRUSTED_SIGNER = "UNTRUSTED_SIGNER"
    PUBLIC_KEY_FINGERPRINT_MISMATCH = "PUBLIC_KEY_FINGERPRINT_MISMATCH"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"


class EnvironmentalFleetRuntimeStateApprovalDecision(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    TRUST_POLICY_UNSATISFIED = "TRUST_POLICY_UNSATISFIED"
    BLOCKED_BY_RELEASE_GATE = "BLOCKED_BY_RELEASE_GATE"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateTrustedSigner:
    """One trusted public verification key; never contains private key material."""

    signer_id: str
    key_id: str
    public_key: bytes

    def __post_init__(self) -> None:
        _validate_nonempty_text(self.signer_id, "signer_id", EnvironmentalFleetRuntimeStateApprovalTrustPolicyError)
        _validate_nonempty_text(self.key_id, "key_id", EnvironmentalFleetRuntimeStateApprovalTrustPolicyError)
        raw = bytes(self.public_key)
        if len(raw) != 32:
            raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                "Ed25519 public_key must contain exactly 32 raw bytes"
            )
        object.__setattr__(self, "public_key", raw)

    @property
    def public_key_sha256(self) -> str:
        return sha256(self.public_key).hexdigest()

    def to_json_object(self) -> dict[str, Any]:
        return {
            "signer_id": self.signer_id,
            "key_id": self.key_id,
            "algorithm": _ALGORITHM,
            "public_key_base64url": _b64url_encode(self.public_key),
            "public_key_sha256": self.public_key_sha256,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateApprovalTrustPolicy:
    """Caller-owned public trust policy for detached approvals.

    Quorum counts unique signer IDs, not keys, so rotating or duplicating keys cannot manufacture extra
    approvals.  Required signer IDs are an additional constraint.  The policy has no command authority.
    """

    trusted_signers: tuple[EnvironmentalFleetRuntimeStateTrustedSigner, ...]
    policy_id: str = "portable-approval-default-v1"
    minimum_approvals: int = 1
    required_signer_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_nonempty_text(self.policy_id, "policy_id", EnvironmentalFleetRuntimeStateApprovalTrustPolicyError)
        if not isinstance(self.minimum_approvals, int) or isinstance(self.minimum_approvals, bool):
            raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                "minimum_approvals must be an integer"
            )
        if self.minimum_approvals < 1:
            raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                "minimum_approvals must be at least 1"
            )
        if not self.trusted_signers:
            raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                "trusted_signers must not be empty"
            )
        pairs: set[tuple[str, str]] = set()
        public_key_fingerprints: set[str] = set()
        signer_ids: set[str] = set()
        for signer in self.trusted_signers:
            if not isinstance(signer, EnvironmentalFleetRuntimeStateTrustedSigner):
                raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                    "trusted_signers must contain EnvironmentalFleetRuntimeStateTrustedSigner values"
                )
            pair = (signer.signer_id, signer.key_id)
            if pair in pairs:
                raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                    f"duplicate trusted signer/key pair {pair!r}"
                )
            pairs.add(pair)
            fingerprint = signer.public_key_sha256
            if fingerprint in public_key_fingerprints:
                raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                    "the same Ed25519 public key cannot represent multiple trusted signer/key entries"
                )
            public_key_fingerprints.add(fingerprint)
            signer_ids.add(signer.signer_id)
        if self.minimum_approvals > len(signer_ids):
            raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                "minimum_approvals exceeds number of unique trusted signer IDs"
            )
        required = tuple(self.required_signer_ids)
        if len(set(required)) != len(required):
            raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                "required_signer_ids must not contain duplicates"
            )
        for signer_id in required:
            _validate_nonempty_text(signer_id, "required signer_id", EnvironmentalFleetRuntimeStateApprovalTrustPolicyError)
            if signer_id not in signer_ids:
                raise EnvironmentalFleetRuntimeStateApprovalTrustPolicyError(
                    f"required signer {signer_id!r} has no trusted key"
                )
        object.__setattr__(self, "required_signer_ids", required)

    def trusted_key(
        self, signer_id: str, key_id: str
    ) -> EnvironmentalFleetRuntimeStateTrustedSigner | None:
        for signer in self.trusted_signers:
            if signer.signer_id == signer_id and signer.key_id == key_id:
                return signer
        return None

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": "vessel-crossing-environmental-runtime-state-approval-trust-policy",
            "policy_id": self.policy_id,
            "minimum_approvals": self.minimum_approvals,
            "required_signer_ids": sorted(self.required_signer_ids),
            "trusted_signers": [
                signer.to_json_object()
                for signer in sorted(
                    self.trusted_signers,
                    key=lambda item: (item.signer_id, item.key_id),
                )
            ],
            "executable_restore_authorized": False,
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateDetachedApproval:
    schema_version: int
    signer_id: str
    key_id: str
    bundle_sha256: str
    bundle_id: str
    public_key_sha256: str
    signature: bytes
    executable_restore_authorized: bool = False

    @property
    def signature_document_sha256(self) -> str:
        return sha256(self.to_bytes()).hexdigest()

    def to_bytes(self) -> bytes:
        _validate_detached_approval_fields(self)
        return _canonical_json(_approval_to_json(self))


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateApprovalFinding:
    signer_id: str
    key_id: str
    status: EnvironmentalFleetRuntimeStateApprovalStatus
    signature_document_sha256: str


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRuntimeStateApprovalVerificationReport:
    bundle_sha256: str
    bundle_id: str
    bundle_gate_decision: EnvironmentalFleetRuntimeStateGateDecision
    trust_policy: EnvironmentalFleetRuntimeStateApprovalTrustPolicy
    decision: EnvironmentalFleetRuntimeStateApprovalDecision
    findings: tuple[EnvironmentalFleetRuntimeStateApprovalFinding, ...]
    valid_approval_signer_ids: tuple[str, ...]
    trust_policy_sha256: str
    executable_restore_authorized: bool = False

    @property
    def portable_release_authorized(self) -> bool:
        return self.decision is EnvironmentalFleetRuntimeStateApprovalDecision.AUTHORIZED

    @property
    def human_review_satisfied(self) -> bool:
        return (
            self.portable_release_authorized
            and self.bundle_gate_decision
            is EnvironmentalFleetRuntimeStateGateDecision.REVIEW_REQUIRED
        )

    def to_json_bytes(self) -> bytes:
        return _canonical_json(
            {
                "format": _REPORT_FORMAT,
                "bundle_sha256": self.bundle_sha256,
                "bundle_id": self.bundle_id,
                "bundle_gate_decision": self.bundle_gate_decision.value,
                "trust_policy": self.trust_policy.to_json_object(),
                "trust_policy_sha256": self.trust_policy_sha256,
                "decision": self.decision.value,
                "portable_release_authorized": self.portable_release_authorized,
                "human_review_satisfied": self.human_review_satisfied,
                "valid_approval_signer_ids": list(self.valid_approval_signer_ids),
                "findings": [
                    {
                        "signer_id": finding.signer_id,
                        "key_id": finding.key_id,
                        "status": finding.status.value,
                        "signature_document_sha256": finding.signature_document_sha256,
                    }
                    for finding in self.findings
                ],
                "executable_restore_authorized": False,
            }
        )


def sign_environmental_fleet_runtime_state_release_bundle(
    bundle_bytes: bytes | bytearray | memoryview,
    *,
    signer_id: str,
    key_id: str,
    private_key: Ed25519PrivateKey | bytes | bytearray | memoryview,
) -> EnvironmentalFleetRuntimeStateDetachedApproval:
    """Create detached approval over the exact immutable A135 bytes supplied by the caller.

    The private key is used transiently and is never stored in the returned evidence object.
    """

    bundle_raw = bytes(bundle_bytes)
    bundle = read_environmental_fleet_runtime_state_release_bundle(bundle_raw)
    _validate_nonempty_text(signer_id, "signer_id", EnvironmentalFleetRuntimeStateApprovalTrustPolicyError)
    _validate_nonempty_text(key_id, "key_id", EnvironmentalFleetRuntimeStateApprovalTrustPolicyError)
    key = _coerce_private_key(private_key)
    public_raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    statement = _signature_statement(
        signer_id=signer_id,
        key_id=key_id,
        bundle_sha256=sha256(bundle_raw).hexdigest(),
        bundle_id=bundle.bundle_id,
        public_key_sha256=sha256(public_raw).hexdigest(),
    )
    signature = key.sign(_signature_message(statement, bundle_raw))
    return EnvironmentalFleetRuntimeStateDetachedApproval(
        schema_version=_SIGNATURE_SCHEMA_VERSION,
        signer_id=signer_id,
        key_id=key_id,
        bundle_sha256=statement["bundle_sha256"],
        bundle_id=bundle.bundle_id,
        public_key_sha256=statement["public_key_sha256"],
        signature=signature,
        executable_restore_authorized=False,
    )


def read_environmental_fleet_runtime_state_detached_approval(
    data: bytes | bytearray | memoryview,
) -> EnvironmentalFleetRuntimeStateDetachedApproval:
    """Read detached approval syntax; cryptographic trust is evaluated separately."""

    raw = _load_json_object(data, "detached approval")
    if raw.get("format") != _SIGNATURE_FORMAT:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "detached approval format marker is invalid"
        )
    try:
        schema_version = int(raw.get("schema_version"))
    except (TypeError, ValueError) as exc:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "detached approval schema version is invalid"
        ) from exc
    if schema_version != _SIGNATURE_SCHEMA_VERSION:
        raise EnvironmentalFleetRuntimeStateApprovalMigrationRequired(
            f"approval schema {schema_version} requires explicit migration; current schema is {_SIGNATURE_SCHEMA_VERSION}"
        )
    required = {
        "format",
        "schema_version",
        "algorithm",
        "approval",
        "signer_id",
        "key_id",
        "bundle_sha256",
        "bundle_id",
        "public_key_sha256",
        "signature_base64url",
        "executable_restore_authorized",
    }
    if set(raw) != required:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "detached approval fields are invalid"
        )
    if raw["algorithm"] != _ALGORITHM or raw["approval"] != _APPROVAL:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "detached approval algorithm or approval statement is invalid"
        )
    if raw["executable_restore_authorized"] is not False:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "detached approval cannot authorize executable restore"
        )
    try:
        signature = _b64url_decode(raw["signature_base64url"])
    except (TypeError, ValueError, binascii.Error) as exc:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "detached approval signature encoding is invalid"
        ) from exc
    approval = EnvironmentalFleetRuntimeStateDetachedApproval(
        schema_version=schema_version,
        signer_id=raw["signer_id"],
        key_id=raw["key_id"],
        bundle_sha256=raw["bundle_sha256"],
        bundle_id=raw["bundle_id"],
        public_key_sha256=raw["public_key_sha256"],
        signature=signature,
        executable_restore_authorized=False,
    )
    _validate_detached_approval_fields(approval)
    if approval.to_bytes() != _canonical_json(raw):
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "detached approval is not canonical"
        )
    return approval


def verify_environmental_fleet_runtime_state_release_approvals(
    bundle_bytes: bytes | bytearray | memoryview,
    approvals: Iterable[bytes | bytearray | memoryview | EnvironmentalFleetRuntimeStateDetachedApproval],
    *,
    trust_policy: EnvironmentalFleetRuntimeStateApprovalTrustPolicy,
) -> EnvironmentalFleetRuntimeStateApprovalVerificationReport:
    """Verify detached approvals for one exact A135 bundle under explicit public trust policy.

    Any malformed/untrusted/wrong-bundle/invalid supplied signature makes the trust policy unsatisfied.
    A134 BLOCKED remains a hard block regardless of signatures or quorum.  REVIEW_REQUIRED may be
    satisfied only when this public-key approval policy succeeds.
    """

    if not isinstance(trust_policy, EnvironmentalFleetRuntimeStateApprovalTrustPolicy):
        raise TypeError("trust_policy must be EnvironmentalFleetRuntimeStateApprovalTrustPolicy")
    bundle_raw = bytes(bundle_bytes)
    bundle = read_environmental_fleet_runtime_state_release_bundle(bundle_raw)
    bundle_sha = sha256(bundle_raw).hexdigest()
    parsed: list[EnvironmentalFleetRuntimeStateDetachedApproval] = []
    document_digests: list[str] = []
    for approval in approvals:
        if isinstance(approval, EnvironmentalFleetRuntimeStateDetachedApproval):
            encoded = approval.to_bytes()
            parsed_approval = read_environmental_fleet_runtime_state_detached_approval(encoded)
        else:
            encoded = bytes(approval)
            parsed_approval = read_environmental_fleet_runtime_state_detached_approval(encoded)
        parsed.append(parsed_approval)
        document_digests.append(sha256(encoded).hexdigest())

    findings: list[EnvironmentalFleetRuntimeStateApprovalFinding] = []
    valid_signers: set[str] = set()
    all_evidence_valid = True
    for approval, document_sha in zip(parsed, document_digests, strict=True):
        status = _verify_one_approval(
            bundle_raw,
            bundle,
            bundle_sha,
            approval,
            trust_policy,
        )
        findings.append(
            EnvironmentalFleetRuntimeStateApprovalFinding(
                signer_id=approval.signer_id,
                key_id=approval.key_id,
                status=status,
                signature_document_sha256=document_sha,
            )
        )
        if status is EnvironmentalFleetRuntimeStateApprovalStatus.VALID:
            valid_signers.add(approval.signer_id)
        else:
            all_evidence_valid = False

    required = set(trust_policy.required_signer_ids)
    trust_satisfied = (
        all_evidence_valid
        and len(valid_signers) >= trust_policy.minimum_approvals
        and required.issubset(valid_signers)
    )
    bundle_gate_decision = bundle.decision
    if bundle_gate_decision is EnvironmentalFleetRuntimeStateGateDecision.BLOCKED:
        decision = EnvironmentalFleetRuntimeStateApprovalDecision.BLOCKED_BY_RELEASE_GATE
    elif trust_satisfied:
        decision = EnvironmentalFleetRuntimeStateApprovalDecision.AUTHORIZED
    else:
        decision = EnvironmentalFleetRuntimeStateApprovalDecision.TRUST_POLICY_UNSATISFIED

    return EnvironmentalFleetRuntimeStateApprovalVerificationReport(
        bundle_sha256=bundle_sha,
        bundle_id=bundle.bundle_id,
        bundle_gate_decision=bundle_gate_decision,
        trust_policy=trust_policy,
        decision=decision,
        findings=tuple(findings),
        valid_approval_signer_ids=tuple(sorted(valid_signers)),
        trust_policy_sha256=trust_policy.sha256,
        executable_restore_authorized=False,
    )


def _verify_one_approval(
    bundle_raw: bytes,
    bundle: EnvironmentalFleetRuntimeStateReleaseBundle,
    bundle_sha: str,
    approval: EnvironmentalFleetRuntimeStateDetachedApproval,
    policy: EnvironmentalFleetRuntimeStateApprovalTrustPolicy,
) -> EnvironmentalFleetRuntimeStateApprovalStatus:
    if approval.bundle_sha256 != bundle_sha or approval.bundle_id != bundle.bundle_id:
        return EnvironmentalFleetRuntimeStateApprovalStatus.WRONG_BUNDLE
    trusted = policy.trusted_key(approval.signer_id, approval.key_id)
    if trusted is None:
        return EnvironmentalFleetRuntimeStateApprovalStatus.UNTRUSTED_SIGNER
    if approval.public_key_sha256 != trusted.public_key_sha256:
        return EnvironmentalFleetRuntimeStateApprovalStatus.PUBLIC_KEY_FINGERPRINT_MISMATCH
    statement = _signature_statement(
        signer_id=approval.signer_id,
        key_id=approval.key_id,
        bundle_sha256=approval.bundle_sha256,
        bundle_id=approval.bundle_id,
        public_key_sha256=approval.public_key_sha256,
    )
    try:
        Ed25519PublicKey.from_public_bytes(trusted.public_key).verify(
            approval.signature,
            _signature_message(statement, bundle_raw),
        )
    except InvalidSignature:
        return EnvironmentalFleetRuntimeStateApprovalStatus.INVALID_SIGNATURE
    return EnvironmentalFleetRuntimeStateApprovalStatus.VALID


def _approval_to_json(
    approval: EnvironmentalFleetRuntimeStateDetachedApproval,
) -> dict[str, Any]:
    return {
        "format": _SIGNATURE_FORMAT,
        "schema_version": approval.schema_version,
        "algorithm": _ALGORITHM,
        "approval": _APPROVAL,
        "signer_id": approval.signer_id,
        "key_id": approval.key_id,
        "bundle_sha256": approval.bundle_sha256,
        "bundle_id": approval.bundle_id,
        "public_key_sha256": approval.public_key_sha256,
        "signature_base64url": _b64url_encode(approval.signature),
        "executable_restore_authorized": False,
    }


def _signature_statement(
    *,
    signer_id: str,
    key_id: str,
    bundle_sha256: str,
    bundle_id: str,
    public_key_sha256: str,
) -> dict[str, Any]:
    return {
        "format": _SIGNATURE_FORMAT,
        "schema_version": _SIGNATURE_SCHEMA_VERSION,
        "algorithm": _ALGORITHM,
        "approval": _APPROVAL,
        "signer_id": signer_id,
        "key_id": key_id,
        "bundle_sha256": bundle_sha256,
        "bundle_id": bundle_id,
        "public_key_sha256": public_key_sha256,
        "executable_restore_authorized": False,
    }


def _signature_message(statement: dict[str, Any], bundle_raw: bytes) -> bytes:
    return _DOMAIN_SEPARATOR + _canonical_json(statement) + b"\x00" + bundle_raw


def _validate_detached_approval_fields(
    approval: EnvironmentalFleetRuntimeStateDetachedApproval,
) -> None:
    if approval.schema_version != _SIGNATURE_SCHEMA_VERSION:
        raise EnvironmentalFleetRuntimeStateApprovalMigrationRequired(
            f"approval schema {approval.schema_version} requires explicit migration; current schema is {_SIGNATURE_SCHEMA_VERSION}"
        )
    _validate_nonempty_text(approval.signer_id, "signer_id", EnvironmentalFleetRuntimeStateApprovalIntegrityError)
    _validate_nonempty_text(approval.key_id, "key_id", EnvironmentalFleetRuntimeStateApprovalIntegrityError)
    _validate_sha256(approval.bundle_sha256, "bundle_sha256")
    _validate_sha256(approval.bundle_id, "bundle_id")
    _validate_sha256(approval.public_key_sha256, "public_key_sha256")
    if len(bytes(approval.signature)) != 64:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "Ed25519 signature must contain exactly 64 bytes"
        )
    if approval.executable_restore_authorized:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "detached approval cannot authorize executable restore"
        )


def _coerce_private_key(
    value: Ed25519PrivateKey | bytes | bytearray | memoryview,
) -> Ed25519PrivateKey:
    if isinstance(value, Ed25519PrivateKey):
        return value
    try:
        raw = bytes(value)
    except Exception as exc:  # pragma: no cover - defensive type boundary
        raise TypeError("private_key must be Ed25519PrivateKey or 32 raw bytes") from exc
    if len(raw) != 32:
        raise ValueError("raw Ed25519 private_key must contain exactly 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _load_json_object(data: bytes | bytearray | memoryview, label: str) -> dict[str, Any]:
    try:
        raw = bytes(data)
        parsed = json.loads(raw.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            f"{label} must be a JSON object"
        )
    return parsed


def _validate_nonempty_text(value: Any, label: str, error_type: type[Exception]) -> None:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be a non-empty string")


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            f"{label} must be a 64-character SHA-256 hex digest"
        )
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            f"{label} is not valid hexadecimal"
        ) from exc
    return value


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("base64url value must be a non-empty string")
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EnvironmentalFleetRuntimeStateApprovalIntegrityError(
            "approval evidence contains non-canonical JSON values"
        ) from exc
