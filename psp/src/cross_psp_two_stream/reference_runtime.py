"""Frozen A70 reference-runtime reconstruction for dynamic B6 execution.

The A70 qualification used one narrow D0 physical-runtime fixture.  Dynamic references
must reuse that same reference machinery while substituting only the user-supplied
physical profile.  The historical research-fixture dependency is intentionally isolated
in this module so the rest of the product does not depend on research namespace details.
"""
from __future__ import annotations

from dataclasses import replace

from vessel_crossing.nextgen import EnvironmentalNetworkSourceFutureIngressConfig, PairSafetyMetricConfig
from vessel_crossing.nextgen.encounter_registry import EncounterDiscoveryConfig
from vessel_crossing.nextgen.risk import RiskGateConfig
from vessel_crossing.research.r1_p4b_experiment_spec import build_publication_config

from .b6 import B6RuntimeBinding

A70_REFERENCE_RUNTIME_BINDING_ID = "q4-physical-fixture-v1"
A70_REFERENCE_RUNTIME_IDENTITY_MATERIAL = {"fixture": "frozen-d0-narrow-encounter-gate-v1"}


def build_a70_reference_runtime_binding() -> B6RuntimeBinding:
    """Reconstruct the exact runtime binding used by the qualified A70 Q4 fixture."""
    cfg = build_publication_config(room_nm=1.375, demand_per_hour=4.0, traffic_seed=0)
    pair_clearance = cfg.scenario.recapture.joint_cycle.replanning.semantic_config.physics.pair_clearance
    discovery = EncounterDiscoveryConfig(
        current_distance_nm=0.001,
        dcpa_distance_nm=0.001,
        tcpa_horizon_min=60.0,
    )
    lifecycle = replace(
        cfg.scenario.lifecycle,
        risk_gate=RiskGateConfig(
            dcpa_threshold_nm=0.001,
            tcpa_horizon_min=60.0,
            immediate_distance_nm=0.001,
        ),
    )
    following = cfg.scenario.recapture.same_route_following
    return B6RuntimeBinding(
        binding_id=A70_REFERENCE_RUNTIME_BINDING_ID,
        identity_material=A70_REFERENCE_RUNTIME_IDENTITY_MATERIAL,
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
        same_route_dynamic_ingress=following,
        future_ingress_clearance=EnvironmentalNetworkSourceFutureIngressConfig(
            pair_clearance=pair_clearance
        ),
        persistent_same_route_following=following,
    )
