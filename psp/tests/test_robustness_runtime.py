from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import cross_psp_two_stream.robustness_runtime as runtime_mod
from cross_psp_two_stream.robustness_matrix import robustness_campaign_plan_from_mapping
from cross_psp_two_stream.robustness_runtime import (
    ROBUSTNESS_RUNTIME_SCHEMA,
    RobustnessRuntimeError,
    build_robustness_runtime_adapter,
    parameter_binding_manifest,
    runtime_adapter_code_identity_sha256,
)
from cross_psp_two_stream.robustness_wizard import baseline_source_from_files
from vessel_crossing.nextgen.reference_envelope import ReferenceEnvelopeLevel

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "q5_scenario.yaml"


def _campaign(source, parameter_id: str, baseline: float, *, unit: str = "m", percent: float = 5):
    return robustness_campaign_plan_from_mapping({
        "schema": "cross-psp-robustness-campaign-v1",
        "baseline": {
            "mode": source["mode"],
            "physics_identity_sha256": source["expected_physics_identity_sha256"],
            "scope_status": "QUALIFIED_PROFILE" if source["mode"] == "fixed" else "SUPPORTED_EXPLORATORY",
            "plan_identity_sha256": source["expected_plan_identity_sha256"],
        },
        "scientific_question": "A93 runtime seam test",
        "campaign_intent": "exploratory",
        "design": "full_factorial_grid",
        "master_seed": 101,
        "replications": 1,
        "confidence_target": 0.95,
        "estimated_seconds_per_work_unit": 1,
        "max_work_units": 10,
        "parameters": [{
            "parameter_id": parameter_id,
            "unit": unit,
            "baseline_value": baseline,
            "reference_value": baseline,
            "supported_bounds": {"lower": 0.001, "upper": baseline * 2},
            "variation": {"mode": "symmetric_percent", "percent": percent, "levels": 3},
        }],
    })


def _distribution_mapping(seed: int = 17):
    return {
        "schema": "cross-psp-physics-distribution-v1",
        "master_seed": seed,
        "streams": [{
            "stream_id": "alpha",
            "distribution": {
                "parameter": "length_m",
                "unit": "m",
                "family": "uniform",
                "parameters": {},
                "bounds": {"lower": 119, "upper": 121},
                "sampling_scope": "per_vessel",
            },
        }],
    }


def _profile(class_id: str, length: float = 120.0):
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


def _mixture_mapping(seed: int = 5):
    return {
        "schema": "cross-psp-fleet-mixture-v1",
        "master_seed": seed,
        "streams": [
            {
                "stream_id": stream_id,
                "classes": [
                    {"weight": 1, "profile": _profile(f"{stream_id}-standard", 120)},
                    {"weight": 1, "profile": _profile(f"{stream_id}-large", 120)},
                ],
            }
            for stream_id in ("alpha", "beta")
        ],
    }


def _fake_orchestrator_result():
    values = {
        ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK: (10.0, False, None, "B0"),
        ReferenceEnvelopeLevel.B4_SLOT_VTS_COORDINATION: (11.0, True, "global_slot_coordination", "B4"),
        ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE: (12.0, True, "type_order_staging", "B5"),
        ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE: (23.5, False, "physical_balanced_reference", "B6"),
    }
    levels = {
        key: SimpleNamespace(
            primary_kpi=kpi,
            aggregate_metrics=(
                {"mean_service_yield_per_hour": kpi, "replication_count": 2, "seed_schedule": [101, 202]}
                if key is ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE
                else {"mean_service_yield_per_hour": kpi}
            ),
            causal=causal,
            intervention_id=intervention,
            reason=reason,
        )
        for key, (kpi, causal, intervention, reason) in values.items()
    }
    return SimpleNamespace(study=SimpleNamespace(level=lambda key: levels[key]))


def test_binding_manifest_and_code_identity_are_canonical():
    manifest = parameter_binding_manifest()
    assert manifest["schema"] == "cross-psp-robustness-parameter-bindings-v1"
    assert "cruise_speed_knots" in manifest["fixed"]["fields"]
    first = runtime_adapter_code_identity_sha256()
    second = runtime_adapter_code_identity_sha256()
    assert first == second and len(first) == 64


def test_fixed_baseline_combination_prepares_exact_existing_q5_identity():
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    plan = _campaign(source, "stream.alpha.cruise_speed_knots", 6.0, unit="knots")
    prepared = build_robustness_runtime_adapter(source).prepare(plan.work_units[1])
    assert prepared.bound.physics_identity_sha256 == source["expected_physics_identity_sha256"]
    assert prepared.derived_plan_identity_sha256 is None
    assert len(prepared.execution_identity_sha256) == 64
    assert prepared.traffic_seed == 359


def test_fixed_stream_class_field_changes_only_explicit_physics_identity():
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    plan = _campaign(source, "stream.alpha.length_m", 120.0)
    adapter = build_robustness_runtime_adapter(source)
    baseline = adapter.prepare(plan.work_units[1])
    changed = adapter.prepare(plan.work_units[2])
    assert baseline.bound.physics_identity_sha256 == source["expected_physics_identity_sha256"]
    assert changed.bound.physics_identity_sha256 != source["expected_physics_identity_sha256"]
    assert changed.bound.scope_status == "SUPPORTED_EXPLORATORY"
    assert changed.traffic_seed == baseline.traffic_seed == 359


def test_distribution_work_unit_rebuilds_valid_plan_and_distinct_sample_identity(tmp_path):
    plan_path = tmp_path / "distribution.yaml"
    plan_path.write_text(yaml.safe_dump(_distribution_mapping(), sort_keys=False), encoding="utf-8")
    source = baseline_source_from_files(mode="distributions", scenario_path=SCENARIO, plan_path=plan_path)
    campaign = _campaign(source, "distribution.alpha.bounds.lower", 119.0, percent=0.5)
    adapter = build_robustness_runtime_adapter(source)
    first = adapter.prepare(campaign.work_units[0])
    second = adapter.prepare(campaign.work_units[1])
    assert first.mode == "distributions"
    assert first.derived_plan_identity_sha256 != source["expected_plan_identity_sha256"]
    assert first.derived_plan_identity_sha256 != second.derived_plan_identity_sha256
    assert first.sample_seed is not None
    assert first.bound.scope_status == "SUPPORTED_EXPLORATORY"


def test_fleet_work_unit_revalidates_coherent_profile(tmp_path):
    plan_path = tmp_path / "mixture.yaml"
    plan_path.write_text(yaml.safe_dump(_mixture_mapping(), sort_keys=False), encoding="utf-8")
    source = baseline_source_from_files(mode="fleet_mixture", scenario_path=SCENARIO, plan_path=plan_path)
    campaign = _campaign(source, "fleet.alpha.class.alpha-standard.length_m", 120.0)
    prepared = build_robustness_runtime_adapter(source).prepare(campaign.work_units[-1])
    assert prepared.mode == "fleet_mixture"
    assert prepared.bound.scope_status == "SUPPORTED_EXPLORATORY"
    assert prepared.derived_plan_identity_sha256 is not None


def test_execute_returns_a90_a91_machine_mapping_without_reinterpreting_metrics(monkeypatch):
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    plan = _campaign(source, "stream.alpha.cruise_speed_knots", 6.0, unit="knots")
    monkeypatch.setattr(runtime_mod, "run_diagnostic_orchestrator", lambda scenario, registry: _fake_orchestrator_result())
    result = build_robustness_runtime_adapter(source).execute(plan.work_units[1])
    assert result["schema"] == ROBUSTNESS_RUNTIME_SCHEMA
    assert result["metrics"]["primary"] == 12.0
    assert result["metrics"]["b5_delta_vs_b0_per_hour"] == 2.0
    assert result["replication_semantics"]["campaign_replication_axis"] == "NONE_FIXED_Q5_INTERNAL_ONLY"
    assert result["replication_semantics"]["q5_internal_replication_count"] == 2
    assert result["replication_semantics"]["q5_internal_seed_schedule"] == [101, 202]
    assert result["levels"]["B4"]["causal"] is True
    assert result["levels"]["B6"]["causal"] is False
    assert result["claim_boundary"]["automatic_qualification_promotion"] is False


def test_unsupported_arbitrary_path_fails_closed_before_q5_execution():
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    campaign = _campaign(source, "model.parameters.streams.0.cruise_speed_knots", 6.0, unit="knots")
    adapter = build_robustness_runtime_adapter(source)
    with pytest.raises(RobustnessRuntimeError, match="unsupported fixed parameter binding"):
        adapter.prepare(campaign.work_units[1])


def test_saved_baseline_identity_change_is_rejected(tmp_path):
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_bytes(SCENARIO.read_bytes())
    source = baseline_source_from_files(mode="fixed", scenario_path=scenario_path)
    payload = yaml.safe_load(scenario_path.read_text())
    payload["model"]["parameters"]["streams"][0]["cruise_speed_knots"] = 7.0
    scenario_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(RobustnessRuntimeError, match="baseline physics identity mismatch"):
        build_robustness_runtime_adapter(source)


def test_fixed_outer_replication_index_is_rejected_before_execution():
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    mapping = {
        "schema": "cross-psp-robustness-campaign-v1",
        "baseline": {
            "mode": "fixed",
            "physics_identity_sha256": source["expected_physics_identity_sha256"],
            "scope_status": "QUALIFIED_PROFILE",
            "plan_identity_sha256": None,
        },
        "scientific_question": "outer replication rejection",
        "campaign_intent": "exploratory",
        "design": "full_factorial_grid",
        "master_seed": 101,
        "replications": 2,
        "confidence_target": 0.95,
        "estimated_seconds_per_work_unit": 1,
        "max_work_units": 10,
        "parameters": [{
            "parameter_id": "stream.alpha.length_m",
            "unit": "m",
            "baseline_value": 120,
            "reference_value": 120,
            "supported_bounds": {"lower": 100, "upper": 140},
            "variation": {"mode": "symmetric_percent", "percent": 1, "levels": 3},
        }],
    }
    plan = robustness_campaign_plan_from_mapping(mapping)
    with pytest.raises(RobustnessRuntimeError, match="replication_index must be 0"):
        build_robustness_runtime_adapter(source).prepare(plan.work_units[1])
