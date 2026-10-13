from __future__ import annotations

from copy import deepcopy

import pytest

from cross_psp_two_stream.fleet_mixtures import (
    FLEET_MIXTURE_SCHEMA,
    FleetMixtureError,
    fleet_mixture_plan_from_mapping,
    sample_fleet_mixture,
)


def _profile(class_id: str, *, length=120.0, beam=20.0, cruise=6.0, maximum=10.0):
    return {
        "class_id": class_id,
        "dimensions_m": {"length_m": length, "beam_m": beam},
        "speed": {"cruise_speed_knots": cruise, "maximum_speed_knots": maximum},
        "maneuver_limits": {
            "max_acceleration_knots_per_min": 2.0,
            "max_deceleration_knots_per_min": 2.0,
            "max_turn_rate_deg_per_min": 30.0,
        },
        "safety_domain": {
            "type": "four_quadrant_nm",
            "ahead_nm": 0.12,
            "astern_nm": 0.08,
            "port_nm": 0.09,
            "starboard_nm": 0.10,
        },
    }


def _plan(weights=(1, 3)):
    return {
        "schema": FLEET_MIXTURE_SCHEMA,
        "master_seed": 777,
        "streams": [
            {
                "stream_id": "alpha",
                "classes": [
                    {"weight": weights[0], "profile": _profile("small", length=90.0, beam=15.0, cruise=5.0)},
                    {"weight": weights[1], "profile": _profile("large", length=180.0, beam=30.0, cruise=7.0, maximum=12.0)},
                ],
            },
            {
                "stream_id": "beta",
                "classes": [{"weight": 1, "profile": _profile("beta-reference")}],
            },
        ],
    }


def test_fleet_mixture_samples_complete_coherent_named_profiles():
    plan = fleet_mixture_plan_from_mapping(_plan())
    sample = sample_fleet_mixture(plan, 4)
    assert {row.stream_id for row in sample} == {"alpha", "beta"}
    for row in sample:
        profile = row.profile
        assert profile.class_id
        assert profile.length_m > profile.beam_m > 0
        assert profile.cruise_speed_knots <= profile.maximum_speed_knots
        assert profile.safety_ahead_nm > 0 and profile.safety_starboard_nm > 0
        assert len(row.sample_identity_sha256) == 64


def test_proportional_weights_have_same_identity_and_sample_set():
    a = fleet_mixture_plan_from_mapping(_plan((1, 3)))
    b = fleet_mixture_plan_from_mapping(_plan((2, 6)))
    assert a.plan_identity_sha256 == b.plan_identity_sha256
    assert [sample_fleet_mixture(a, i) for i in range(20)] == [sample_fleet_mixture(b, i) for i in range(20)]


def test_fleet_mixture_rejects_duplicate_classes_and_streams():
    bad = _plan()
    bad["streams"][0]["classes"][1]["profile"]["class_id"] = "small"
    with pytest.raises(FleetMixtureError, match="duplicate class"):
        fleet_mixture_plan_from_mapping(bad)
    bad = _plan()
    bad["streams"][1]["stream_id"] = "alpha"
    with pytest.raises(FleetMixtureError, match="duplicate stream"):
        fleet_mixture_plan_from_mapping(bad)


def test_fleet_mixture_rejects_partial_profile_and_bad_dimension_ratio():
    bad = _plan()
    del bad["streams"][0]["classes"][0]["profile"]["maneuver_limits"]
    with pytest.raises(FleetMixtureError, match="fields mismatch"):
        fleet_mixture_plan_from_mapping(bad)
    bad = _plan()
    bad["streams"][0]["classes"][0]["profile"]["dimensions_m"] = {"length_m": 10.0, "beam_m": 10.0}
    with pytest.raises(FleetMixtureError, match="length/beam"):
        fleet_mixture_plan_from_mapping(bad)


def test_fleet_mixture_rejects_invalid_weights_speed_and_safety_domain():
    bad = _plan()
    bad["streams"][0]["classes"][0]["weight"] = -1
    with pytest.raises(FleetMixtureError, match="weight"):
        fleet_mixture_plan_from_mapping(bad)
    bad = _plan()
    bad["streams"][0]["classes"][0]["profile"]["speed"] = {"cruise_speed_knots": 11.0, "maximum_speed_knots": 10.0}
    with pytest.raises(FleetMixtureError, match="cruise_speed"):
        fleet_mixture_plan_from_mapping(bad)
    bad = _plan()
    bad["streams"][0]["classes"][0]["profile"]["safety_domain"]["ahead_nm"] = 0.0
    with pytest.raises(FleetMixtureError, match="safety_ahead_nm"):
        fleet_mixture_plan_from_mapping(bad)


def test_fleet_mixture_replay_and_sample_index_are_deterministic():
    plan = fleet_mixture_plan_from_mapping(_plan())
    assert sample_fleet_mixture(plan, 17) == sample_fleet_mixture(plan, 17)
    assert sample_fleet_mixture(plan, 17) != sample_fleet_mixture(plan, 18)
    with pytest.raises(FleetMixtureError, match="sample_index"):
        sample_fleet_mixture(plan, -1)
