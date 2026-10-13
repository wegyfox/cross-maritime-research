from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from cross_psp_two_stream import B6ReferenceResult
from cross_psp_two_stream.b6 import B6_REFERENCE_ALGORITHM_ID
from cross_psp_two_stream.q5_integration import (
    A69_SPEC_SHA256,
    A70_IMPLEMENTATION_MANIFEST_SHA256,
    prepare_q5_scenario,
)
from cross_psp_two_stream.reference_arbiter import REFERENCE_ARBITER_ALGORITHM_ID
from cross_psp_two_stream.reference_cache import (
    DynamicReferenceCache,
    ReferenceCacheError,
    build_reference_cache_identity,
    resolve_dynamic_reference,
)
from vessel_crossing.diagnostic.contract import load_diagnostic_scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "q5_scenario.yaml"


def _sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity_and_result():
    prepared = prepare_q5_scenario(load_diagnostic_scenario(SCENARIO))
    identity = build_reference_cache_identity(
        physics_identity_sha256=prepared.physical.physics_identity_sha256,
        profile=prepared.physical.profile,
        runtime=prepared.reference_runtime,
        b6_spec_sha256=A69_SPEC_SHA256,
        implementation_manifest_sha256=A70_IMPLEMENTATION_MANIFEST_SHA256,
    )
    reference_identity = {
        "algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "runtime_binding_id": prepared.reference_runtime.binding_id,
        "runtime_binding_sha256": prepared.reference_runtime.identity_sha256,
        "reference_arbiter_algorithm_id": REFERENCE_ARBITER_ALGORITHM_ID,
        "profile_contract_sha256": prepared.physical.profile.contract_sha256,
    }
    evidence = {
        "algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "profile_contract_sha256": prepared.physical.profile.contract_sha256,
        "b6_spec_sha256": A69_SPEC_SHA256,
        "implementation_manifest_sha256": A70_IMPLEMENTATION_MANIFEST_SHA256,
        "reference_execution_identity": reference_identity,
        "reference_execution_identity_sha256": _sha(reference_identity),
        "promotable_reference": True,
        "aggregate_status": "PASS_BALANCED_SATURATED_REFERENCE",
    }
    return identity, B6ReferenceResult(evidence)


def test_persistent_reference_cache_roundtrip_is_byte_verified(tmp_path: Path) -> None:
    identity, result = _identity_and_result()
    cache = DynamicReferenceCache(tmp_path)
    path = cache.store(identity, result)
    assert path is not None and path.exists()
    restored = cache.load(identity)
    assert restored is not None
    assert restored.evidence_sha256 == result.evidence_sha256
    assert restored.canonical_bytes == result.canonical_bytes


def test_cache_tampering_is_rejected(tmp_path: Path) -> None:
    identity, result = _identity_and_result()
    cache = DynamicReferenceCache(tmp_path)
    path = cache.store(identity, result)
    payload = json.loads(path.read_text())
    payload["evidence"]["aggregate_status"] = "TAMPERED"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceCacheError, match="evidence SHA"):
        cache.load(identity)


def test_resolver_uses_verified_hit_without_recompute(tmp_path: Path) -> None:
    identity, result = _identity_and_result()
    cache = DynamicReferenceCache(tmp_path)
    cache.store(identity, result)
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return result

    restored, status = resolve_dynamic_reference(cache, identity, compute)
    assert status == "HIT_VERIFIED"
    assert calls == 0
    assert restored.evidence_sha256 == result.evidence_sha256


def test_invalid_entry_is_recomputed_and_replaced(tmp_path: Path) -> None:
    identity, result = _identity_and_result()
    cache = DynamicReferenceCache(tmp_path)
    path = cache.store(identity, result)
    path.write_text("not json", encoding="utf-8")
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return result

    restored, status = resolve_dynamic_reference(cache, identity, compute)
    assert status == "INVALID_RECOMPUTED_STORED"
    assert calls == 1
    assert cache.load(identity).evidence_sha256 == restored.evidence_sha256
