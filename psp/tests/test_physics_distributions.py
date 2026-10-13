from __future__ import annotations

from copy import deepcopy

import pytest

from cross_psp_two_stream.physics_distributions import (
    PHYSICS_DISTRIBUTION_SCHEMA,
    PhysicsDistributionError,
    _canonicalize_sample_float,
    physics_distribution_plan_from_mapping,
    sample_distribution_plan,
)


def _plan():
    return {
        "schema": PHYSICS_DISTRIBUTION_SCHEMA,
        "master_seed": 359,
        "streams": [
            {
                "stream_id": "beta",
                "distribution": {
                    "parameter": "cruise_speed_knots",
                    "unit": "knots",
                    "family": "triangular",
                    "parameters": {"mode": 6.0},
                    "bounds": {"lower": 4.0, "upper": 8.0},
                    "sampling_scope": "per_vessel",
                },
            },
            {
                "stream_id": "alpha",
                "distribution": {
                    "parameter": "length_m",
                    "unit": "m",
                    "family": "normal",
                    "parameters": {"mean": 120.0, "stddev": 8.0},
                    "bounds": {"lower": 90.0, "upper": 150.0},
                    "sampling_scope": "per_vessel",
                },
            },
        ],
    }


def test_distribution_plan_is_canonical_and_order_independent():
    a = physics_distribution_plan_from_mapping(_plan())
    reversed_plan = deepcopy(_plan())
    reversed_plan["streams"].reverse()
    b = physics_distribution_plan_from_mapping(reversed_plan)
    assert a.plan_identity_sha256 == b.plan_identity_sha256
    assert [row.stream_id for row in a.streams] == ["alpha", "beta"]


def test_distribution_samples_are_deterministic_per_index_and_identity():
    plan = physics_distribution_plan_from_mapping(_plan())
    first = sample_distribution_plan(plan, 12)
    replay = sample_distribution_plan(plan, 12)
    assert first == replay
    assert all(len(row.sample_identity_sha256) == 64 for row in first)
    assert {row.derived_seed for row in first}.isdisjoint({row.derived_seed for row in sample_distribution_plan(plan, 13)})


def test_uniform_normal_and_triangular_are_bounded():
    base = _plan()
    families = [
        ("uniform", {}, 1.0, 2.0),
        ("normal", {"mean": 1.5, "stddev": 0.2}, 1.0, 2.0),
        ("triangular", {"mode": 1.4}, 1.0, 2.0),
    ]
    for family, parameters, lower, upper in families:
        mapping = deepcopy(base)
        mapping["streams"] = [mapping["streams"][0]]
        dist = mapping["streams"][0]["distribution"]
        dist.update({"family": family, "parameters": parameters, "bounds": {"lower": lower, "upper": upper}})
        plan = physics_distribution_plan_from_mapping(mapping)
        values = [sample_distribution_plan(plan, i)[0].value for i in range(20)]
        assert all(lower <= value <= upper for value in values)


def test_distribution_rejects_unknown_schema_and_scope():
    bad = _plan()
    bad["schema"] = "wrong"
    with pytest.raises(PhysicsDistributionError, match="schema"):
        physics_distribution_plan_from_mapping(bad)
    bad = _plan()
    bad["streams"][0]["distribution"]["sampling_scope"] = "per_run"
    with pytest.raises(PhysicsDistributionError, match="sampling_scope"):
        physics_distribution_plan_from_mapping(bad)


def test_distribution_rejects_invalid_family_parameters_and_bounds():
    bad = _plan()
    bad["streams"][0]["distribution"]["bounds"] = {"lower": 8.0, "upper": 4.0}
    with pytest.raises(PhysicsDistributionError, match="lower < upper"):
        physics_distribution_plan_from_mapping(bad)
    bad = _plan()
    bad["streams"][1]["distribution"]["parameters"]["stddev"] = 0.0
    with pytest.raises(PhysicsDistributionError, match="stddev"):
        physics_distribution_plan_from_mapping(bad)


def test_distribution_rejects_duplicate_streams_and_bad_seed():
    bad = _plan()
    bad["streams"][1]["stream_id"] = bad["streams"][0]["stream_id"]
    with pytest.raises(PhysicsDistributionError, match="duplicate"):
        physics_distribution_plan_from_mapping(bad)
    bad = _plan()
    bad["master_seed"] = -1
    with pytest.raises(PhysicsDistributionError, match="master_seed"):
        physics_distribution_plan_from_mapping(bad)


def test_distribution_rejects_nonfinite_public_values_and_bad_sample_index():
    bad = _plan()
    bad["streams"][0]["distribution"]["bounds"]["upper"] = float("inf")
    with pytest.raises(PhysicsDistributionError, match="finite"):
        physics_distribution_plan_from_mapping(bad)
    plan = physics_distribution_plan_from_mapping(_plan())
    with pytest.raises(PhysicsDistributionError, match="sample_index"):
        sample_distribution_plan(plan, -1)


def test_sample_float_canonicalization_collapses_observed_cross_platform_ulp_drift():
    linux_a = 110.48212284672647
    windows_a = 110.48212284672645
    linux_b = 111.28358744482289
    windows_b = 111.2835874448229

    assert _canonicalize_sample_float(linux_a) == _canonicalize_sample_float(windows_a)
    assert _canonicalize_sample_float(linux_b) == _canonicalize_sample_float(windows_b)
    assert _canonicalize_sample_float(linux_a) == 110.482122846726
    assert _canonicalize_sample_float(linux_b) == 111.283587444823


def test_distribution_samples_apply_canonical_float_before_identity(monkeypatch):
    import cross_psp_two_stream.physics_distributions as distributions

    mapping = _plan()
    mapping["streams"] = [mapping["streams"][1]]
    plan = physics_distribution_plan_from_mapping(mapping)

    monkeypatch.setattr(distributions, "_sample_value", lambda spec, rng: 110.48212284672647)
    linux_like = sample_distribution_plan(plan, 178)[0]
    monkeypatch.setattr(distributions, "_sample_value", lambda spec, rng: 110.48212284672645)
    windows_like = sample_distribution_plan(plan, 178)[0]

    assert linux_like.value == windows_like.value == 110.482122846726
    assert linux_like.sample_identity_sha256 == windows_like.sample_identity_sha256
