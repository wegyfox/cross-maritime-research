from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vessel_crossing.diagnostic.contract import load_diagnostic_scenario
from vessel_crossing.nextgen.environmental_network_scenario import (
    build_environmental_network_scenario_ingress,
)

from cross_psp_two_stream.custom_physics_binding import (
    CUSTOM_PHYSICS_BINDING_SCHEMA,
    CustomPhysicsBindingError,
    bind_distribution_scenario,
    bind_fixed_scenario,
    bind_fleet_mixture_scenario,
    materialize_bound_traffic,
    run_bound_fifo_baseline,
)
from cross_psp_two_stream.fleet_mixtures import fleet_mixture_plan_from_mapping
from cross_psp_two_stream.physics_distributions import physics_distribution_plan_from_mapping
from cross_psp_two_stream.q5_integration import prepare_q5_scenario, qualified_q5_physics_identity_sha256
from cross_psp_two_stream.reference_cache import build_reference_cache_identity

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "q5_scenario.yaml"


def _prepared():
    return prepare_q5_scenario(load_diagnostic_scenario(SCENARIO))


def _distribution_plan(*, seed: int = 42, parameter: str = "length_m", unit: str = "m"):
    return physics_distribution_plan_from_mapping({
        "schema": "cross-psp-physics-distribution-v1",
        "master_seed": seed,
        "streams": [
            {
                "stream_id": stream_id,
                "distribution": {
                    "parameter": parameter,
                    "unit": unit,
                    "family": "uniform",
                    "parameters": {},
                    "bounds": {
                        "lower": 119.0 if parameter == "length_m" else 5.8,
                        "upper": 121.0 if parameter == "length_m" else 6.2,
                    },
                    "sampling_scope": "per_vessel",
                },
            }
            for stream_id in ("alpha", "beta")
        ],
    })


def _profile(class_id: str, *, length: float, beam: float, cruise: float = 6.0):
    return {
        "class_id": class_id,
        "dimensions_m": {"length_m": length, "beam_m": beam},
        "speed": {"cruise_speed_knots": cruise, "maximum_speed_knots": 10.0},
        "maneuver_limits": {
            "max_acceleration_knots_per_min": 2.0,
            "max_deceleration_knots_per_min": 2.0,
            "max_turn_rate_deg_per_min": 30.0,
        },
        "safety_domain": {
            "type": "four_quadrant_nm",
            "ahead_nm": 0.10,
            "astern_nm": 0.10,
            "port_nm": 0.10,
            "starboard_nm": 0.10,
        },
    }


def _mixture_plan(*, seed: int = 91):
    classes = [
        {"weight": 3, "profile": _profile("small", length=100.0, beam=18.0, cruise=5.5)},
        {"weight": 1, "profile": _profile("large", length=150.0, beam=24.0, cruise=6.5)},
    ]
    return fleet_mixture_plan_from_mapping({
        "schema": "cross-psp-fleet-mixture-v1",
        "master_seed": seed,
        "streams": [
            {"stream_id": "alpha", "classes": classes},
            {"stream_id": "beta", "classes": classes},
        ],
    })


def test_fixed_binding_is_identity_preserving_passthrough():
    prepared = _prepared()
    bound = bind_fixed_scenario(prepared)
    assert bound.mode == "fixed"
    assert bound.physics_identity_sha256 == qualified_q5_physics_identity_sha256()
    assert bound.physics_identity_sha256 == prepared.physical.physics_identity_sha256
    assert bound.scope_status == "QUALIFIED_PROFILE"
    assert bound.profile.contract_sha256 == prepared.physical.profile.contract_sha256


def test_distribution_binding_is_deterministic_and_never_inherits_default_identity():
    prepared = _prepared()
    plan = _distribution_plan(seed=42)
    first = bind_distribution_scenario(prepared, plan)
    second = bind_distribution_scenario(prepared, plan)
    changed = bind_distribution_scenario(prepared, _distribution_plan(seed=43))
    assert first.identity_material["schema"] == CUSTOM_PHYSICS_BINDING_SCHEMA
    assert first.physics_identity_sha256 == second.physics_identity_sha256
    assert first.profile.contract_sha256 == second.profile.contract_sha256
    assert first.physics_identity_sha256 != prepared.physical.physics_identity_sha256
    assert first.physics_identity_sha256 != changed.physics_identity_sha256
    assert first.scope_status == "SUPPORTED_EXPLORATORY"
    assert len(first.profile_classes) == 512  # 256 deterministic samples per stream


def test_distribution_traffic_preserves_poisson_times_ids_and_binds_per_arrival_profiles():
    prepared = _prepared()
    plan = _distribution_plan()
    bound = bind_distribution_scenario(prepared, plan)
    traffic_a = materialize_bound_traffic(bound, distribution_plan=plan)
    traffic_b = materialize_bound_traffic(bound, distribution_plan=plan)
    assert traffic_a.streams == traffic_b.streams
    assert [a.at for a in traffic_a.arrivals] == [a.at for a in traffic_b.arrivals]
    assert [a.vessel.vessel_id for a in traffic_a.arrivals] == [a.vessel.vessel_id for a in traffic_b.arrivals]
    assert [a.vessel_class.class_id for a in traffic_a.arrivals] == [
        a.vessel_class.class_id for a in traffic_b.arrivals
    ]
    assert all(a.vessel_class.class_id.startswith("dist-") for a in traffic_a.arrivals)
    assert any(abs(a.vessel_class.dimensions.length_m - 120.0) > 1e-9 for a in traffic_a.arrivals)
    # D0's real environmental ingress path consumes the per-arrival class/limits; this
    # validates the heterogeneous realization at the supported simulator boundary.
    ingress = build_environmental_network_scenario_ingress(traffic_a, prepared.config.scenario)
    assert len(ingress) == traffic_a.count
    assert {item.record.profile.class_id for item in ingress} == {
        a.vessel_class.class_id for a in traffic_a.arrivals
    }


def test_benign_distribution_executes_through_public_pre_realized_fifo_seam():
    prepared = _prepared()
    # The frozen Q5 B0 fixture itself can hit a canonical no-safe-candidate fail-closed
    # path with both streams active.  Keep the exact alpha process and set beta demand to
    # zero so this test isolates binding/execution rather than a collision-avoidance case.
    plan = physics_distribution_plan_from_mapping({
        "schema": "cross-psp-physics-distribution-v1",
        "master_seed": 42,
        "streams": [{
            "stream_id": "alpha",
            "distribution": {
                "parameter": "length_m",
                "unit": "m",
                "family": "uniform",
                "parameters": {},
                "bounds": {"lower": 119.0, "upper": 121.0},
                "sampling_scope": "per_vessel",
            },
        }],
    })
    bound = bind_distribution_scenario(prepared, plan)
    streams = tuple(
        replace(item, rate_per_minute=(item.rate_per_minute if item.stream_id == "alpha" else 0.0))
        for item in prepared.config.streams
    )
    config = replace(prepared.config, streams=streams)
    result = run_bound_fifo_baseline(bound, config=config, distribution_plan=plan)
    assert result.summary.realized_demand_count == 1
    assert result.summary.physical_admission_count == 1


def test_distribution_binding_fails_closed_on_unit_and_invalid_resolved_profile():
    prepared = _prepared()
    bad_unit = _distribution_plan(parameter="cruise_speed_knots", unit="m")
    with pytest.raises(CustomPhysicsBindingError, match="requires unit"):
        bind_distribution_scenario(prepared, bad_unit)

    bad_speed = physics_distribution_plan_from_mapping({
        "schema": "cross-psp-physics-distribution-v1",
        "master_seed": 7,
        "streams": [{
            "stream_id": "alpha",
            "distribution": {
                "parameter": "maximum_speed_knots",
                "unit": "knots",
                "family": "uniform",
                "parameters": {},
                "bounds": {"lower": 1.0, "upper": 2.0},
                "sampling_scope": "per_vessel",
            },
        }],
    })
    with pytest.raises(CustomPhysicsBindingError, match="invalid vessel profile"):
        bind_distribution_scenario(prepared, bad_speed)


def test_fleet_mixture_binding_keeps_complete_named_profiles_coherent():
    prepared = _prepared()
    plan = _mixture_plan()
    bound = bind_fleet_mixture_scenario(prepared, plan)
    traffic = materialize_bound_traffic(bound, fleet_mixture_plan=plan)
    assert bound.scope_status == "SUPPORTED_EXPLORATORY"
    assert bound.physics_identity_sha256 != prepared.physical.physics_identity_sha256
    assert set(bound.profile_classes).issubset({
        f"mix-{plan.plan_identity_sha256[:12]}-small",
        f"mix-{plan.plan_identity_sha256[:12]}-large",
    })
    assert traffic.arrivals
    for arrival in traffic.arrivals:
        profile = arrival.vessel_class
        assert profile is not None and profile.safety_domain is not None
        assert profile.speed_envelope is not None
        assert arrival.limits == profile.maneuver_limits
        assert profile.speed_envelope.permits(arrival.vessel.speed)
        if profile.class_id.endswith("small"):
            assert profile.dimensions.length_m == pytest.approx(100.0)
            assert arrival.vessel.speed * 60.0 == pytest.approx(5.5)
        else:
            assert profile.dimensions.length_m == pytest.approx(150.0)
            assert arrival.vessel.speed * 60.0 == pytest.approx(6.5)


def test_sampled_binding_changes_reference_cache_identity_and_plan_mismatch_fails_closed():
    prepared = _prepared()
    plan = _distribution_plan()
    bound = bind_distribution_scenario(prepared, plan)
    base_cache = build_reference_cache_identity(
        physics_identity_sha256=prepared.physical.physics_identity_sha256,
        profile=prepared.physical.profile,
        runtime=prepared.reference_runtime,
        b6_spec_sha256="spec",
        implementation_manifest_sha256="impl",
    )
    sampled_cache = build_reference_cache_identity(
        physics_identity_sha256=bound.physics_identity_sha256,
        profile=bound.profile,
        runtime=prepared.reference_runtime,
        b6_spec_sha256="spec",
        implementation_manifest_sha256="impl",
    )
    assert sampled_cache.sha256 != base_cache.sha256
    with pytest.raises(CustomPhysicsBindingError, match="plan identity does not match"):
        materialize_bound_traffic(bound, distribution_plan=_distribution_plan(seed=999))


def test_fleet_mixture_requires_full_stream_definition():
    prepared = _prepared()
    plan = fleet_mixture_plan_from_mapping({
        "schema": "cross-psp-fleet-mixture-v1",
        "master_seed": 1,
        "streams": [{"stream_id": "alpha", "classes": [{"weight": 1, "profile": _profile("only", length=120.0, beam=20.0)}]}],
    })
    with pytest.raises(CustomPhysicsBindingError, match="requires exactly"):
        bind_fleet_mixture_scenario(prepared, plan)
