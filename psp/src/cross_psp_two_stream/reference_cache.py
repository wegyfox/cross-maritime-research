"""Integrity-checked persistent cache for expensive dynamic B6 references.

The cache is a transport/performance layer only. It never changes reference semantics:
a cache hit is accepted only when the complete dynamic-reference identity matches and the
stored evidence bytes reproduce their declared SHA-256 exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from .b6 import (
    B6_REFERENCE_ALGORITHM_ID,
    B6ReferenceResult,
    B6RuntimeBinding,
)
from .contract import TwoStreamProfileContract
from .reference_arbiter import REFERENCE_ARBITER_ALGORITHM_ID

CACHE_SCHEMA = "cross-psp-dynamic-reference-cache-v1"


class ReferenceCacheError(ValueError):
    """Raised when a cache entry exists but cannot be trusted."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def default_reference_cache_dir() -> Path:
    """Return a platform-appropriate user cache location without creating it."""
    override = os.environ.get("CROSS_PSP_REFERENCE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "CROSS" / "reference-cache"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "cross-psp" / "reference-cache"


@dataclass(frozen=True, slots=True)
class ReferenceCacheIdentity:
    material: Mapping[str, object]
    sha256: str


def build_reference_cache_identity(
    *,
    physics_identity_sha256: str,
    profile: TwoStreamProfileContract,
    runtime: B6RuntimeBinding,
    b6_spec_sha256: str,
    implementation_manifest_sha256: str,
) -> ReferenceCacheIdentity:
    """Build the exact identity controlling reusable B6 evidence."""
    material = {
        "schema": CACHE_SCHEMA,
        "physics_identity_sha256": physics_identity_sha256,
        "profile_contract_sha256": profile.contract_sha256,
        "base_framework_version": profile.base_framework_version,
        "runtime_adapter_id": profile.runtime_adapter_id,
        "runtime_binding_id": runtime.binding_id,
        "runtime_binding_sha256": runtime.identity_sha256,
        "b6_algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "reference_arbiter_algorithm_id": REFERENCE_ARBITER_ALGORITHM_ID,
        "b6_spec_sha256": b6_spec_sha256,
        "implementation_manifest_sha256": implementation_manifest_sha256,
    }
    return ReferenceCacheIdentity(material=material, sha256=_sha(material))


def _validate_evidence(
    evidence: Mapping[str, object],
    identity: ReferenceCacheIdentity,
) -> None:
    material = identity.material
    expected = {
        "algorithm_id": material["b6_algorithm_id"],
        "profile_contract_sha256": material["profile_contract_sha256"],
        "b6_spec_sha256": material["b6_spec_sha256"],
        "implementation_manifest_sha256": material["implementation_manifest_sha256"],
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise ReferenceCacheError(f"cached reference {field} mismatch")

    reference_identity = evidence.get("reference_execution_identity")
    if not isinstance(reference_identity, Mapping):
        raise ReferenceCacheError("cached reference execution identity is missing")
    identity_checks = {
        "runtime_binding_id": material["runtime_binding_id"],
        "runtime_binding_sha256": material["runtime_binding_sha256"],
        "reference_arbiter_algorithm_id": material["reference_arbiter_algorithm_id"],
        "algorithm_id": material["b6_algorithm_id"],
        "profile_contract_sha256": material["profile_contract_sha256"],
    }
    for field, value in identity_checks.items():
        if reference_identity.get(field) != value:
            raise ReferenceCacheError(f"cached reference execution identity {field} mismatch")

    declared_reference_sha = evidence.get("reference_execution_identity_sha256")
    if declared_reference_sha != _sha(dict(reference_identity)):
        raise ReferenceCacheError("cached reference execution identity SHA mismatch")
    if not bool(evidence.get("promotable_reference", False)):
        raise ReferenceCacheError("non-promotable references are never reusable from persistent cache")


class DynamicReferenceCache:
    """File-backed cache keyed by exact dynamic-reference identity."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, identity: ReferenceCacheIdentity) -> Path:
        return self.root / f"{identity.sha256}.json"

    def load(self, identity: ReferenceCacheIdentity) -> B6ReferenceResult | None:
        path = self.path_for(identity)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReferenceCacheError(f"cannot read cached dynamic reference: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA:
            raise ReferenceCacheError("cached dynamic reference schema mismatch")
        if payload.get("cache_identity_sha256") != identity.sha256:
            raise ReferenceCacheError("cached dynamic reference identity SHA mismatch")
        if payload.get("cache_identity") != dict(identity.material):
            raise ReferenceCacheError("cached dynamic reference identity material mismatch")
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            raise ReferenceCacheError("cached dynamic reference evidence is missing")
        result = B6ReferenceResult(evidence)
        if payload.get("evidence_sha256") != result.evidence_sha256:
            raise ReferenceCacheError("cached dynamic reference evidence SHA mismatch")
        _validate_evidence(evidence, identity)
        return result

    def store(
        self,
        identity: ReferenceCacheIdentity,
        result: B6ReferenceResult,
    ) -> Path | None:
        """Atomically store only a promotable reference; return None for failed references."""
        if not result.promotable_reference:
            return None
        _validate_evidence(result.evidence, identity)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(identity)
        payload = {
            "schema": CACHE_SCHEMA,
            "cache_identity": dict(identity.material),
            "cache_identity_sha256": identity.sha256,
            "evidence_sha256": result.evidence_sha256,
            "evidence": dict(result.evidence),
        }
        encoded = _canonical_bytes(payload) + b"\n"
        fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, destination)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
        return destination


def resolve_dynamic_reference(
    cache: DynamicReferenceCache | None,
    identity: ReferenceCacheIdentity,
    compute,
) -> tuple[B6ReferenceResult, str]:
    """Return a verified cached reference or compute it fresh without weakening failures."""
    if cache is None:
        return compute(), "DISABLED_FRESH"
    try:
        cached = cache.load(identity)
    except ReferenceCacheError:
        result = compute()
        if result.promotable_reference:
            try:
                cache.store(identity, result)
                return result, "INVALID_RECOMPUTED_STORED"
            except OSError:
                return result, "INVALID_RECOMPUTED_CACHE_WRITE_FAILED"
        return result, "INVALID_RECOMPUTED_NONPROMOTABLE"
    if cached is not None:
        return cached, "HIT_VERIFIED"
    result = compute()
    if result.promotable_reference:
        try:
            cache.store(identity, result)
            return result, "MISS_COMPUTED_STORED"
        except OSError:
            return result, "MISS_COMPUTED_CACHE_WRITE_FAILED"
    return result, "MISS_COMPUTED_NONPROMOTABLE"
