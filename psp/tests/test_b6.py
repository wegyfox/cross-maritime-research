from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

import cross_psp_two_stream.b6 as b6mod
from cross_psp_two_stream import profile_contract_from_mapping
from cross_psp_two_stream.b6 import B6RuntimeBinding, run_balanced_saturated_reference
from vessel_crossing.nextgen import (
    EnvironmentalNetworkSourceFutureIngressConfig,
    PairSafetyMetricConfig,
    ShipDomain,
    VesselClassProfile,
    VesselDimensions,
)
from vessel_crossing.nextgen.encounter_registry import EncounterDiscoveryConfig
from vessel_crossing.nextgen.risk import RiskGateConfig
from vessel_crossing.research.r1_p4b_experiment_spec import build_publication_config

ROOT = Path(__file__).resolve().parents[1]
SPEC_SHA = "03829b884b3533366bc040330db298be5e19865a131ebfef953874b3721a592d"


def _mapping():
    return yaml.safe_load((ROOT / "examples" / "profile.yaml").read_text())


def _binding():
    cfg = build_publication_config(room_nm=1.375, demand_per_hour=4.0, traffic_seed=0)
    pair = cfg.scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    discovery = EncounterDiscoveryConfig(current_distance_nm=0.001, dcpa_distance_nm=0.001, tcpa_horizon_min=60.0)
    lifecycle = replace(
        cfg.scenario.lifecycle,
        risk_gate=RiskGateConfig(dcpa_threshold_nm=0.001, tcpa_horizon_min=60.0, immediate_distance_nm=0.001),
    )
    return B6RuntimeBinding(
        binding_id="q4-physical-fixture-v1",
        identity_material={"fixture": "frozen-d0-narrow-encounter-gate-v1"},
        discovery=discovery,
        lifecycle=lifecycle,
        recapture_config=cfg.scenario.recapture,
        runtime_config=cfg.scenario.runtime,
        recheck_config=cfg.recheck,
        safety_config=PairSafetyMetricConfig(
            near_miss_margin_nm=0.0,
            time_tolerance_min=1e-4,
            margin_tolerance_nm=1e-8,
            max_subdivisions=32,
        ),
        same_route_dynamic_ingress=cfg.scenario.recapture.same_route_following,
        future_ingress_clearance=EnvironmentalNetworkSourceFutureIngressConfig(pair_clearance=pair),
        persistent_same_route_following=cfg.scenario.recapture.same_route_following,
    ), cfg


def _small_physical_profile():
    data = _mapping()
    for stream in data["streams"]:
        stream["source_to_center_distance_nm"] = 0.15
        stream["center_to_exit_distance_nm"] = 0.15
    for template in data["b6_reference"]["supply_templates"]:
        heading = template["cycle"][0]["through_water_heading_deg"]
        template["cycle"] = [{
            "profile_class_id": "q4",
            "through_water_heading_deg": heading,
            "through_water_speed_nm_per_min": 0.10,
        }]
    data["b6_reference"].update(
        warmup_min=0.0,
        measurement_duration_min=5.0,
        drain_min=3.0,
        saturated_demands_per_stream=64,
    )
    return profile_contract_from_mapping(data)


def test_a70_contract_binds_reference_fairness_into_profile_identity():
    profile = profile_contract_from_mapping(_mapping())
    assert profile.schema == "cross-two-stream-profile-contract-v4"
    assert profile.profile_version == "0.1.0a7"
    assert profile.b6_reference.max_share_deficit_passages == 0.5
    assert profile.b6_reference.fairness_comparison_tolerance == 1e-12
    assert "reference_arbiter" in profile.component_contract_ids


def test_a70_aggregate_evidence_is_reference_only_and_requires_stable_replays(monkeypatch):
    data = _mapping()
    binding, cfg = _binding()
    vp = cfg.streams[0].vessel_class
    for template in data["b6_reference"]["supply_templates"]:
        for entry in template["cycle"]:
            entry["profile_class_id"] = vp.class_id
    profile = profile_contract_from_mapping(data)

    def fake_run(*args, measurement_min: float, **kwargs):
        return {
            "status": "PASS_REFERENCE_RUN",
            "semantic_supply_manifest_sha256": "2" * 64,
            "measurement": {"service_rate_per_min": 1.0},
            "semantic_measurement": {
                "service_rate_per_min": 1.0,
                "per_stream_key_center_passages": {"a": 10, "b": 10},
            },
        }

    monkeypatch.setattr(b6mod, "_run_once", fake_run)
    result = run_balanced_saturated_reference(
        profile,
        {vp.class_id: vp},
        binding,
        b6_spec_sha256=SPEC_SHA,
        implementation_manifest_sha256="1" * 64,
    )
    assert result.promotable_reference
    assert result.evidence["aggregate_status"] == "PASS_BALANCED_SATURATED_REFERENCE"
    assert result.evidence["causal"] is False
    assert result.evidence["deployment_b4_executed"] is False
    assert result.evidence["deployment_b5_executed"] is False
    assert result.evidence["global_optimum_claim"] is False
    assert all(result.evidence["runs"][name]["replay_equal"] for name in ("R0", "R1", "R2"))


def test_a70_symmetric_saturated_50_50_physical_smoke_passes_balance():
    profile = _small_physical_profile()
    binding, cfg = _binding()
    vp = VesselClassProfile(
        class_id="q4",
        dimensions=VesselDimensions.from_meters(length_m=30.0, beam_m=6.0),
        maneuver_limits=cfg.streams[0].limits,
        safety_domain=ShipDomain.circular(0.025),
        speed_envelope=cfg.streams[0].vessel_class.speed_envelope,
    )
    run = b6mod._run_once(
        profile,
        {"q4": vp},
        binding,
        warmup_min=0.0,
        measurement_min=12.0,
        drain_min=4.0,
    )
    assert run["status"] == "PASS_REFERENCE_RUN"
    assert run["saturation"]["exhausted_before_cutoff"] is False
    assert run["safety"]["pass"] is True
    assert run["reference_arbiter"]["deployment_b4_executed"] is False
    assert run["reference_arbiter"]["deployment_b5_executed"] is False
    shares = run["measurement"]["achieved_stream_shares"]
    assert all(abs(value - 0.5) <= profile.b6_reference.balance_tolerance_fraction for value in shares.values())
