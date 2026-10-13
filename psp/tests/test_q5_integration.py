from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json

import pytest

from vessel_crossing.cli import main
from vessel_crossing.diagnostic.bundle import verify_diagnostic_reproduction_bundle
from vessel_crossing.diagnostic.contract import load_diagnostic_scenario
from vessel_crossing.diagnostic.effects import _paired_estimate
from vessel_crossing.diagnostic.evidence import DiagnosticLevelEvidence, DiagnosticReplicationEvidence
from vessel_crossing.diagnostic.execution import (
    DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY,
    DiagnosticAdapterRegistry,
    DiagnosticExecutionAdapter,
    DiagnosticExecutionIdentity,
    run_diagnostic_orchestrator,
)
from vessel_crossing.diagnostic.nextgen_fifo_bindings import NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID
from vessel_crossing.nextgen.reference_envelope import ReferenceEnvelopeLevel

from cross_psp_two_stream import (
    B6ReferenceResult,
    B4EarliestSafeSlotController,
    B5BoundedSlotCostController,
    B4_POLICY_ALGORITHM_ID,
    B5_POLICY_ALGORITHM_ID,
    B6_REFERENCE_ALGORITHM_ID,
    PROFILE_VERSION,
)
from cross_psp_two_stream.q5_integration import (
    A69_SPEC_SHA256,
    A70_IMPLEMENTATION_MANIFEST_SHA256,
    A70_REFERENCE_EVIDENCE_SHA256,
    B4_D0_ID,
    B5_D0_ID,
    B6_D0_ID,
    Q5_CANDIDATE_VERSION,
    Q5_CENTER_ANCHOR_MODE,
    Q5_EXECUTION_IDENTITY_ID,
    Q5_SEMANTICS_PROFILE_ID,
    Q5IntegrationError,
    build_q5_bindings,
    build_q5_registry,
    load_qualified_profile,
    integration_contract_sha256,
    prepare_q5_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "q5_scenario.yaml"


def _scenario():
    return load_diagnostic_scenario(SCENARIO)



def _fake_dynamic_reference(profile, profile_classes, runtime, **kwargs):
    rate = 23.75 / 60.0
    reference_identity = {
        "algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "runtime_binding_id": runtime.binding_id,
        "runtime_binding_sha256": runtime.identity_sha256,
        "reference_arbiter_algorithm_id": "cross-b6-reference-fair-slot-arbiter-v1",
        "profile_contract_sha256": profile.contract_sha256,
    }
    identity_sha = sha256(
        json.dumps(reference_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return B6ReferenceResult({
        "algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "profile_contract_sha256": profile.contract_sha256,
        "b6_spec_sha256": kwargs["b6_spec_sha256"],
        "implementation_manifest_sha256": kwargs["implementation_manifest_sha256"],
        "reference_execution_identity": reference_identity,
        "reference_execution_identity_sha256": identity_sha,
        "promotable_reference": True,
        "aggregate_status": "PASS_BALANCED_SATURATED_REFERENCE",
        "runs": {
            "R0": {"run": {"measurement": {"service_rate_per_min": rate}}},
            "R1": {"run": {"measurement": {"service_rate_per_min": rate}}},
            "R2": {"run": {"measurement": {"service_rate_per_min": rate}}},
        },
        "causal": False,
        "field_calibrated": False,
        "deployment_recommendation": False,
        "global_optimum_claim": False,
    })


@pytest.fixture
def fake_dynamic_reference(monkeypatch):
    monkeypatch.setattr(
        "cross_psp_two_stream.q5_integration.run_balanced_saturated_reference",
        _fake_dynamic_reference,
    )

def test_q5_identity_and_binding_architecture_are_exact():
    scenario = _scenario()
    bindings = build_q5_bindings(scenario)
    profile = load_qualified_profile()
    prepared = prepare_q5_scenario(scenario)
    identity = bindings.execution_identity
    assert identity is not None
    assert Q5_CANDIDATE_VERSION == "0.1.0a8"
    assert PROFILE_VERSION == "0.1.0a7"  # frozen scientific profile, new integration package
    assert identity.identity_id == Q5_EXECUTION_IDENTITY_ID
    assert identity.components["psp_package_candidate_version"] == Q5_CANDIDATE_VERSION
    assert identity.components["psp_profile"]["contract_sha256"] == prepared.physical.profile.contract_sha256
    assert identity.components["center_anchor_mode"] == Q5_CENTER_ANCHOR_MODE
    assert identity.components["a69_spec_sha256"] == A69_SPEC_SHA256
    assert identity.components["a70_implementation_manifest_sha256"] == A70_IMPLEMENTATION_MANIFEST_SHA256
    assert identity.components["a70_reference_evidence_sha256"] == A70_REFERENCE_EVIDENCE_SHA256
    assert identity.components["q5_integration_contract_sha256"] == integration_contract_sha256(prepared.physical.profile)
    assert identity.components["dynamic_physics"]["scope_status"] == "QUALIFIED_PROFILE"
    assert identity.components["dynamic_physics"]["physics_identity_sha256"] == prepared.physical.physics_identity_sha256
    assert identity.components["dynamic_physics"]["frozen_a70_reference_value_reused"] is False
    assert bindings.intervention_sequences == frozenset({(B4_D0_ID,), (B4_D0_ID, B5_D0_ID)})
    assert bindings.reference_ids == frozenset({B6_D0_ID})
    assert DEFAULT_DIAGNOSTIC_ADAPTER_REGISTRY.adapter_ids == ()


def test_q5_config_fails_closed_outside_frozen_zero_drift_domain():
    scenario = _scenario()
    payload = json.loads(json.dumps(dict(scenario.model_parameters)))
    payload["scenario"]["default_environmental_drift_nm_per_min"] = [0.01, 0.0]
    changed = replace(scenario, model_parameters=payload)
    with pytest.raises(Q5IntegrationError, match="zero environmental drift"):
        build_q5_bindings(changed)


def test_q5_real_runtime_executes_b4_cumulative_b5_and_separate_b6(fake_dynamic_reference):
    scenario = _scenario()
    result = run_diagnostic_orchestrator(scenario, registry=build_q5_registry(scenario))
    b4 = result.study.level(ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION)
    b5 = result.study.level(ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)
    b6 = result.study.level(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE)
    assert b4.causal is True and b4.intervention_id == B4_D0_ID
    assert b5.causal is True and b5.intervention_id == B5_D0_ID
    assert b4.reason == "REGISTERED_CANONICAL_SOURCE_ADMISSION_CONTROLLER"
    assert b5.reason == "REGISTERED_CANONICAL_CUMULATIVE_SOURCE_ADMISSION_CONTROLLER"
    assert b6.causal is False and b6.intervention_id == B6_D0_ID
    assert b6.primary_kpi == pytest.approx(23.75)
    assert b6.aggregate_metrics["a70_reference_evidence_sha256"] == A70_REFERENCE_EVIDENCE_SHA256
    assert b6.aggregate_metrics["reference_mode"] == "DYNAMIC_SAME_PHYSICS"
    assert b6.aggregate_metrics["frozen_a70_reference_value_reused"] is False
    assert len(b6.aggregate_metrics["physics_identity_sha256"]) == 64
    assert b6.aggregate_metrics["promotable_reference"] is True
    assert b6.aggregate_metrics["deployment_b4_executed"] is False
    assert b6.aggregate_metrics["deployment_b5_executed"] is False
    assert b6.aggregate_metrics["field_calibrated"] is False
    assert b6.aggregate_metrics["deployment_recommendation"] is False
    assert b6.aggregate_metrics["global_optimum_claim"] is False
    # The frozen integration fixture itself is a legitimate NULL B5-vs-B4 realization.
    effect = next(item for item in result.effects if item.to_level is ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)
    assert effect.mean_effect == pytest.approx(0.0)
    assert effect.zero_count == scenario.experiment.replications


def _level(level, values, *, intervention_id=None, target_factor=None, causal=True):
    reps = tuple(DiagnosticReplicationEvidence(f"R{i:03d}", value) for i, value in enumerate(values))
    return DiagnosticLevelEvidence(
        level,
        {"mean_service_yield_per_hour": sum(values) / len(values)},
        reps,
        intervention_id=intervention_id,
        target_factor=target_factor,
        causal=causal,
    )


def test_q5_downstream_effect_sign_is_not_forced():
    b4 = _level(ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION, (10.0, 10.0), intervention_id=B4_D0_ID, target_factor="global_slot_conflict")
    for values, expected in [((11.0, 12.0), 1), ((10.0, 10.0), 0), ((9.0, 8.0), -1)]:
        b5 = _level(ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE, values, intervention_id=B5_D0_ID, target_factor="queue_order_structure")
        effect = _paired_estimate(b4, b5, confidence_level=0.95)
        assert (effect.mean_effect > 0) - (effect.mean_effect < 0) == expected


def test_q5_product_lifecycle_fresh_bundle_resume_and_identity_guard(tmp_path, capsys, fake_dynamic_reference):
    scenario = _scenario()
    registry = build_q5_registry(scenario)
    output = tmp_path / "product"
    common = [
        "diagnostic-run", "--scenario", str(SCENARIO), "--output", str(output),
        "--date", "2026-10-03", "--scope", "CROSS PSP A72 Q5 qualification",
    ]
    assert main(common, diagnostic_registry=registry) == 0
    fresh = json.loads(capsys.readouterr().out)
    bundle = Path(fresh["reproduction_bundle"]["path"])
    assert verify_diagnostic_reproduction_bundle(bundle).passed
    fresh_bundle_sha = fresh["reproduction_bundle"]["sha256"]

    assert main(common + ["--resume"], diagnostic_registry=build_q5_registry(scenario)) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["resume"]["complete_prefix"] is True
    assert resumed["reproduction_bundle"]["sha256"] == fresh_bundle_sha

    adapter = registry.resolve(NEXTGEN_ENVIRONMENTAL_NETWORK_FIFO_ADAPTER_ID)
    altered = DiagnosticExecutionIdentity("cross-psp-two-stream-q5-integration-v1-altered", {"changed": True})
    bad_adapter = replace(adapter, execution_identity=altered)
    assert main(common + ["--resume"], diagnostic_registry=DiagnosticAdapterRegistry((bad_adapter,))) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["passed"] is False
    assert "identity" in failure["error"].lower()
