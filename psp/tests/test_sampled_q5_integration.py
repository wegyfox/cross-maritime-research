from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from vessel_crossing.diagnostic.contract import load_diagnostic_scenario
from vessel_crossing.diagnostic.execution import run_diagnostic_orchestrator
from vessel_crossing.nextgen.reference_envelope import ReferenceEnvelopeLevel

import cross_psp_two_stream.custom_physics_binding as binding_mod
from cross_psp_two_stream.b6 import B6ReferenceResult, B6_REFERENCE_ALGORITHM_ID
from cross_psp_two_stream.custom_physics_binding import (
    bind_distribution_scenario,
    bind_fixed_scenario,
    bind_fleet_mixture_scenario,
)
from cross_psp_two_stream.fleet_mixtures import fleet_mixture_plan_from_mapping
from cross_psp_two_stream.physics_distributions import physics_distribution_plan_from_mapping
from cross_psp_two_stream.q5_integration import B4_D0_ID, B5_D0_ID, B6_D0_ID, prepare_q5_scenario
from cross_psp_two_stream.sampled_q5_integration import (
    SAMPLED_Q5_EXECUTION_IDENTITY_ID,
    build_sampled_q5_bindings,
    build_sampled_q5_registry,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "q5_scenario.yaml"


def _fake_reference(profile, profile_classes, runtime, **kwargs):
    rate = 23.5 / 60.0
    identity = {
        "algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "runtime_binding_id": runtime.binding_id,
        "runtime_binding_sha256": runtime.identity_sha256,
        "profile_contract_sha256": profile.contract_sha256,
    }
    identity_sha = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return B6ReferenceResult({
        "algorithm_id": B6_REFERENCE_ALGORITHM_ID,
        "profile_contract_sha256": profile.contract_sha256,
        "b6_spec_sha256": kwargs["b6_spec_sha256"],
        "implementation_manifest_sha256": kwargs["implementation_manifest_sha256"],
        "reference_execution_identity": identity,
        "reference_execution_identity.identity_sha256": identity_sha,
        "promotable_reference": True,
        "aggregate_status": "PASS_BALANCED_SATURATED_REFERENCE",
        "runs": {
            name: {
                "replay_equal": True,
                "run": {"measurement": {"service_rate_per_min": rate}},
            }
            for name in ("R0", "R1", "R2")
        },
        "causal": False,
        "field_calibrated": False,
        "deployment_recommendation": False,
        "global_optimum_claim": False,
    })


@pytest.fixture
def fake_reference(monkeypatch):
    monkeypatch.setattr(binding_mod, "run_balanced_saturated_reference", _fake_reference)


def _distribution_plan(seed: int = 42):
    return physics_distribution_plan_from_mapping({
        "schema": "cross-psp-physics-distribution-v1",
        "master_seed": seed,
        "streams": [
            {
                "stream_id": stream_id,
                "distribution": {
                    "parameter": "length_m",
                    "unit": "m",
                    "family": "uniform",
                    "parameters": {},
                    "bounds": {"lower": 119.0, "upper": 121.0},
                    "sampling_scope": "per_vessel",
                },
            }
            for stream_id in ("alpha", "beta")
        ],
    })


def _profile(class_id: str, length: float):
    return {
        "class_id": class_id,
        "dimensions_m": {"length_m": length, "beam_m": 20.0},
        "speed": {"cruise_speed_knots": 6.0, "maximum_speed_knots": 10.0},
        "maneuver_limits": {
            "max_acceleration_knots_per_min": 2.0,
            "max_deceleration_knots_per_min": 2.0,
            "max_turn_rate_deg_per_min": 30.0,
        },
        "safety_domain": {
            "type": "four_quadrant_nm",
            "ahead_nm": 0.1,
            "astern_nm": 0.1,
            "port_nm": 0.1,
            "starboard_nm": 0.1,
        },
    }


def _mixture_plan(seed: int = 5):
    return fleet_mixture_plan_from_mapping({
        "schema": "cross-psp-fleet-mixture-v1",
        "master_seed": seed,
        "streams": [
            {
                "stream_id": stream_id,
                "classes": [
                    {"weight": 1, "profile": _profile("short", 110.0)},
                    {"weight": 1, "profile": _profile("long", 130.0)},
                ],
            }
            for stream_id in ("alpha", "beta")
        ],
    })


def test_distribution_runs_full_q5_ladder_with_sampled_identity(fake_reference):
    scenario = load_diagnostic_scenario(SCENARIO)
    prepared = prepare_q5_scenario(scenario)
    plan = _distribution_plan()
    bound = bind_distribution_scenario(prepared, plan)
    bindings = build_sampled_q5_bindings(bound, distribution_plan=plan)
    identity = bindings.execution_identity
    assert identity is not None
    assert identity.identity_id == SAMPLED_Q5_EXECUTION_IDENTITY_ID
    assert identity.components["dynamic_physics"]["physics_identity_sha256"] == bound.physics_identity_sha256
    assert identity.components["dynamic_physics"]["scope_status"] == "SUPPORTED_EXPLORATORY"
    assert identity.components["dynamic_physics"]["robustness_qualified"] is False
    assert identity.components["custom_physics_binding"]["plan_identity_sha256"] == plan.plan_identity_sha256

    result = run_diagnostic_orchestrator(
        scenario,
        registry=build_sampled_q5_registry(bound, distribution_plan=plan),
    )
    b0 = result.study.level(ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK)
    b4 = result.study.level(ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION)
    b5 = result.study.level(ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)
    b6 = result.study.level(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE)
    assert b0.aggregate_metrics["physics_identity_sha256"] == bound.physics_identity_sha256
    assert b4.intervention_id == B4_D0_ID and b4.causal is True
    assert b5.intervention_id == B5_D0_ID and b5.causal is True
    assert all(
        item.metrics["physics_identity_sha256"] == bound.physics_identity_sha256
        for item in b4.replications + b5.replications
    )
    assert b6.intervention_id == B6_D0_ID and b6.causal is False
    assert b6.aggregate_metrics["physics_identity_sha256"] == bound.physics_identity_sha256
    assert b6.aggregate_metrics["scope_status"] == "SUPPORTED_EXPLORATORY"
    assert b6.aggregate_metrics["robustness_qualified"] is False
    assert b6.primary_kpi == pytest.approx(23.5)


def test_fleet_mixture_runs_full_q5_ladder_and_execution_identity_changes(fake_reference):
    scenario = load_diagnostic_scenario(SCENARIO)
    prepared = prepare_q5_scenario(scenario)
    first_plan = _mixture_plan(seed=5)
    second_plan = _mixture_plan(seed=6)
    first = bind_fleet_mixture_scenario(prepared, first_plan)
    second = bind_fleet_mixture_scenario(prepared, second_plan)
    first_bindings = build_sampled_q5_bindings(first, fleet_mixture_plan=first_plan)
    second_bindings = build_sampled_q5_bindings(second, fleet_mixture_plan=second_plan)
    assert first_bindings.execution_identity.identity_sha256 != second_bindings.execution_identity.identity_sha256

    result = run_diagnostic_orchestrator(
        scenario,
        registry=build_sampled_q5_registry(first, fleet_mixture_plan=first_plan),
    )
    b4 = result.study.level(ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION)
    b5 = result.study.level(ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)
    b6 = result.study.level(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE)
    assert b4.reason.endswith("SAMPLED_PHYSICS")
    assert b5.reason.endswith("SAMPLED_PHYSICS")
    assert b6.aggregate_metrics["custom_physics_mode"] == "fleet_mixture"
    assert b6.aggregate_metrics["custom_physics_plan_identity_sha256"] == first_plan.plan_identity_sha256


def test_sampled_registry_rejects_fixed_passthrough():
    prepared = prepare_q5_scenario(load_diagnostic_scenario(SCENARIO))
    fixed = bind_fixed_scenario(prepared)
    with pytest.raises(ValueError, match="distribution/fleet-mixture"):
        build_sampled_q5_bindings(fixed)
