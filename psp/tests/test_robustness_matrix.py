from __future__ import annotations

import pytest

from cross_psp_two_stream.robustness_matrix import (
    ROBUSTNESS_CAMPAIGN_SCHEMA,
    RobustnessMatrixError,
    robustness_campaign_plan_from_mapping,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _parameter(
    parameter_id: str,
    *,
    baseline: float = 100.0,
    variation: dict | None = None,
    lower: float | None = 0.0,
    upper: float | None = 200.0,
):
    return {
        "parameter_id": parameter_id,
        "unit": "m",
        "baseline_value": baseline,
        "reference_value": baseline,
        "supported_bounds": {"lower": lower, "upper": upper},
        "variation": variation or {"mode": "symmetric_percent", "percent": 10, "levels": 3},
    }


def _campaign(*, parameters=None, baseline_mode="fixed", plan_sha=None, **overrides):
    payload = {
        "schema": ROBUSTNESS_CAMPAIGN_SCHEMA,
        "baseline": {
            "mode": baseline_mode,
            "physics_identity_sha256": SHA_A,
            "scope_status": "QUALIFIED_PROFILE" if baseline_mode == "fixed" else "SUPPORTED_EXPLORATORY",
            "plan_identity_sha256": plan_sha,
        },
        "scientific_question": "Does the conclusion remain stable over the declared envelope?",
        "campaign_intent": "exploratory",
        "design": "full_factorial_grid",
        "master_seed": 1234,
        "replications": 2,
        "confidence_target": 0.95,
        "estimated_seconds_per_work_unit": 12.5,
        "max_work_units": 1000,
        "parameters": parameters or [_parameter("stream.alpha.length_m")],
    }
    payload.update(overrides)
    return payload


def test_symmetric_grid_is_deterministic_and_explicitly_contains_baseline():
    first = robustness_campaign_plan_from_mapping(_campaign())
    second = robustness_campaign_plan_from_mapping(_campaign())
    grid = first.parameters[0]
    assert grid.values == ("90", "100", "110")
    assert grid.requested_lower == "90"
    assert grid.requested_upper == "110"
    assert first.campaign_identity_sha256 == second.campaign_identity_sha256
    assert [unit.derived_seed for unit in first.work_units] == [unit.derived_seed for unit in second.work_units]
    assert first.combination_count == 3
    assert first.work_unit_count == 6
    assert first.estimated_total_seconds == "75"


def test_asymmetric_two_sided_and_one_sided_percentage_grids():
    two_sided = _parameter(
        "stream.alpha.length_m",
        variation={"mode": "asymmetric_percent", "lower_percent": 20, "upper_percent": 10, "levels": 5},
    )
    plan = robustness_campaign_plan_from_mapping(_campaign(parameters=[two_sided]))
    assert plan.parameters[0].values == ("80", "90", "100", "105", "110")

    one_sided = _parameter(
        "stream.alpha.length_m",
        variation={"mode": "asymmetric_percent", "lower_percent": 0, "upper_percent": 30, "levels": 4},
    )
    plan = robustness_campaign_plan_from_mapping(_campaign(parameters=[one_sided]))
    assert plan.parameters[0].values == ("100", "110", "120", "130")


def test_absolute_bounds_support_one_sided_grid_and_fail_closed_outside_supported_range():
    parameter = _parameter(
        "stream.alpha.length_m",
        variation={"mode": "absolute_bounds", "lower": 100, "upper": 160, "levels": 4},
    )
    plan = robustness_campaign_plan_from_mapping(_campaign(parameters=[parameter]))
    assert plan.parameters[0].values == ("100", "120", "140", "160")

    bad = _parameter(
        "stream.alpha.length_m",
        variation={"mode": "absolute_bounds", "lower": -1, "upper": 120, "levels": 3},
        lower=0,
    )
    with pytest.raises(RobustnessMatrixError, match="below supported lower bound"):
        robustness_campaign_plan_from_mapping(_campaign(parameters=[bad]))


def test_full_factorial_count_replications_and_seed_identity_are_exact():
    parameters = [
        _parameter("stream.alpha.length_m"),
        _parameter(
            "stream.beta.beam_m",
            baseline=20,
            variation={"mode": "absolute_bounds", "lower": 18, "upper": 22, "levels": 3},
            lower=10,
            upper=40,
        ),
    ]
    plan = robustness_campaign_plan_from_mapping(_campaign(parameters=parameters, replications=3))
    assert plan.combination_count == 9
    assert plan.work_unit_count == 27
    assert len({unit.work_unit_identity_sha256 for unit in plan.work_units}) == 27
    assert len({unit.derived_seed for unit in plan.work_units}) == 27
    assert all(set(unit.parameter_values) == {"stream.alpha.length_m", "stream.beta.beam_m"} for unit in plan.work_units)


def test_parameter_input_order_is_canonical_but_seed_or_range_change_changes_identity():
    a = _parameter("stream.alpha.length_m")
    b = _parameter("stream.beta.length_m")
    first = robustness_campaign_plan_from_mapping(_campaign(parameters=[a, b]))
    reordered = robustness_campaign_plan_from_mapping(_campaign(parameters=[b, a]))
    changed_seed = robustness_campaign_plan_from_mapping(_campaign(parameters=[a, b], master_seed=1235))
    changed_range = robustness_campaign_plan_from_mapping(
        _campaign(parameters=[
            _parameter("stream.alpha.length_m", variation={"mode": "symmetric_percent", "percent": 20, "levels": 3}),
            b,
        ])
    )
    assert first.campaign_identity_sha256 == reordered.campaign_identity_sha256
    assert first.campaign_identity_sha256 != changed_seed.campaign_identity_sha256
    assert first.campaign_identity_sha256 != changed_range.campaign_identity_sha256


def test_distribution_and_fleet_mixture_baselines_require_plan_identity_and_remain_exploratory():
    distribution = robustness_campaign_plan_from_mapping(
        _campaign(baseline_mode="distributions", plan_sha=SHA_B)
    )
    mixture = robustness_campaign_plan_from_mapping(
        _campaign(baseline_mode="fleet_mixture", plan_sha=SHA_B)
    )
    for plan in (distribution, mixture):
        preflight = plan.preflight_dict()
        assert preflight["resulting_claim_status"] == "SUPPORTED_EXPLORATORY"
        assert preflight["automatic_qualification_promotion"] is False
        assert preflight["baseline"]["plan_identity_sha256"] == SHA_B
    with pytest.raises(RobustnessMatrixError, match="requires plan_identity_sha256"):
        robustness_campaign_plan_from_mapping(_campaign(baseline_mode="distributions", plan_sha=None))


def test_campaign_size_guard_fails_before_execution():
    parameters = [
        _parameter("stream.alpha.length_m", variation={"mode": "symmetric_percent", "percent": 10, "levels": 5}),
        _parameter("stream.beta.length_m", variation={"mode": "symmetric_percent", "percent": 10, "levels": 5}),
    ]
    with pytest.raises(RobustnessMatrixError, match="exceeding max_work_units=10"):
        robustness_campaign_plan_from_mapping(
            _campaign(parameters=parameters, replications=2, max_work_units=10)
        )


def test_validation_rejects_even_two_sided_levels_zero_percentage_and_bad_confidence():
    even = _parameter(
        "stream.alpha.length_m",
        variation={"mode": "symmetric_percent", "percent": 10, "levels": 4},
    )
    with pytest.raises(RobustnessMatrixError, match="odd levels"):
        robustness_campaign_plan_from_mapping(_campaign(parameters=[even]))

    zero = _parameter(
        "stream.alpha.length_m",
        variation={"mode": "symmetric_percent", "percent": 0, "levels": 3},
    )
    with pytest.raises(RobustnessMatrixError, match="percent must be > 0"):
        robustness_campaign_plan_from_mapping(_campaign(parameters=[zero]))

    with pytest.raises(RobustnessMatrixError, match="strictly between 0 and 1"):
        robustness_campaign_plan_from_mapping(_campaign(confidence_target=1.0))
