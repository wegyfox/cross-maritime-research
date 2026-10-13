from __future__ import annotations

import json

import pytest

from cross_psp_two_stream.robustness_aggregation import (
    ROBUSTNESS_AGGREGATION_SCHEMA,
    REGION_BOUNDARY,
    REGION_SENSITIVE,
    REGION_STABLE,
    RobustnessAggregationError,
    aggregate_robustness_campaign,
)
from cross_psp_two_stream.robustness_execution import run_robustness_campaign
from cross_psp_two_stream.robustness_matrix import ROBUSTNESS_CAMPAIGN_SCHEMA, robustness_campaign_plan_from_mapping

BASE_SHA = "a" * 64
CODE_SHA = "c" * 64


def _parameter(parameter_id: str):
    return {
        "parameter_id": parameter_id,
        "unit": "u",
        "baseline_value": 0,
        "reference_value": 0,
        "supported_bounds": {"lower": -2, "upper": 2},
        "variation": {"mode": "absolute_bounds", "lower": -1, "upper": 1, "levels": 3},
    }


def _plan(*, replications: int = 1, parameters=None):
    return robustness_campaign_plan_from_mapping({
        "schema": ROBUSTNESS_CAMPAIGN_SCHEMA,
        "baseline": {
            "mode": "fixed",
            "physics_identity_sha256": BASE_SHA,
            "scope_status": "QUALIFIED_PROFILE",
            "plan_identity_sha256": None,
        },
        "scientific_question": "synthetic explainability",
        "campaign_intent": "exploratory",
        "design": "full_factorial_grid",
        "master_seed": 17,
        "replications": replications,
        "confidence_target": None,
        "estimated_seconds_per_work_unit": 1,
        "max_work_units": 100,
        "parameters": parameters or [_parameter("a"), _parameter("b")],
    })


def _config(**overrides):
    value = {
        "schema": ROBUSTNESS_AGGREGATION_SCHEMA,
        "execution_code_identity_sha256": CODE_SHA,
        "primary_metric_path": "metrics.primary",
        "stable_absolute_delta": 0.25,
        "sensitive_absolute_delta": 2.5,
        "conclusion_rule": {"operator": ">=", "threshold": 9.0},
    }
    value.update(overrides)
    return value


def _execute(plan, root, fn):
    def executor(unit):
        values = {k: float(v) for k, v in unit.parameter_values.items()}
        return {"metrics": {"primary": fn(values, unit.replication_index)}}
    summary = run_robustness_campaign(
        plan, root, executor, execution_code_identity_sha256=CODE_SHA
    )
    assert summary.status == "COMPLETE"


def test_additive_campaign_reports_transparent_main_effects_zero_interaction_and_stable_conclusion(tmp_path):
    plan = _plan()
    _execute(plan, tmp_path, lambda v, r: 10 + 2 * v["a"] + 1 * v["b"])
    result = aggregate_robustness_campaign(plan, tmp_path, _config())
    assert result.baseline_mean == pytest.approx(10)
    assert result.parameter_effects[0].parameter_id == "a"
    assert result.parameter_effects[0].max_abs_effect == pytest.approx(2)
    assert result.parameter_effects[0].direction == "POSITIVE_OVER_RANGE"
    assert result.parameter_effects[1].parameter_id == "b"
    assert result.interactions[0].max_abs_interaction == pytest.approx(0)
    assert result.conclusion_stable is False  # low-a/low-b combination falls below 9
    assert result.claim_status == "SUPPORTED_EXPLORATORY"
    assert result.human_report_dict()["qualification_note"].startswith("Robustness evidence")


def test_pairwise_interaction_is_detected_by_second_order_residual(tmp_path):
    plan = _plan()
    _execute(plan, tmp_path, lambda v, r: 10 + v["a"] + v["b"] + 4 * v["a"] * v["b"])
    result = aggregate_robustness_campaign(plan, tmp_path, _config(conclusion_rule={"operator": ">=", "threshold": -100}))
    assert result.interactions[0].parameter_a == "a"
    assert result.interactions[0].parameter_b == "b"
    assert result.interactions[0].max_abs_interaction == pytest.approx(4)
    assert result.conclusion_stable is True


def test_replication_summary_reports_mean_stddev_and_standard_error(tmp_path):
    plan = _plan(replications=3, parameters=[_parameter("a")])
    _execute(plan, tmp_path, lambda v, r: 10 + v["a"] + (r - 1))
    result = aggregate_robustness_campaign(plan, tmp_path, _config(conclusion_rule={"operator": ">=", "threshold": 0}))
    baseline = next(item for item in result.combination_summaries if item.combination_index == result.baseline_combination_index)
    assert baseline.replication_count == 3
    assert baseline.mean == pytest.approx(10)
    assert baseline.sample_stddev == pytest.approx(1)
    assert baseline.standard_error == pytest.approx(1 / (3 ** 0.5))


def test_region_classification_uses_only_declared_absolute_thresholds(tmp_path):
    plan = _plan(parameters=[_parameter("a")])
    _execute(plan, tmp_path, lambda v, r: 10 + 3 * v["a"])
    result = aggregate_robustness_campaign(
        plan,
        tmp_path,
        _config(stable_absolute_delta=0.1, sensitive_absolute_delta=2.0, conclusion_rule={"operator": ">=", "threshold": -100}),
    )
    assert result.region_counts[REGION_STABLE] == 1
    assert result.region_counts[REGION_SENSITIVE] == 2
    assert result.region_counts[REGION_BOUNDARY] == 0


def test_nonmonotonic_parameter_reports_direction_and_nonlinearity(tmp_path):
    plan = _plan(parameters=[_parameter("a")])
    _execute(plan, tmp_path, lambda v, r: 10 - 5 * (v["a"] ** 2))
    result = aggregate_robustness_campaign(plan, tmp_path, _config(conclusion_rule={"operator": ">=", "threshold": 0}))
    effect = result.parameter_effects[0]
    assert effect.direction == "NON_MONOTONIC"
    assert effect.max_linearity_residual_abs == pytest.approx(5)


def test_nested_metric_path_and_conclusion_flip_indices_are_explicit(tmp_path):
    plan = _plan(parameters=[_parameter("a")])
    _execute(plan, tmp_path, lambda v, r: 10 + 2 * v["a"])
    result = aggregate_robustness_campaign(plan, tmp_path, _config(conclusion_rule={"operator": ">=", "threshold": 10}))
    assert result.conclusion_stable is False
    assert len(result.conclusion_flip_combination_indices) == 1
    assert result.human_report_dict()["headline"].startswith("Primary conclusion changed")


def test_incomplete_or_tampered_campaign_fails_closed(tmp_path):
    plan = _plan(parameters=[_parameter("a")])
    run_robustness_campaign(
        plan,
        tmp_path,
        lambda unit: {"metrics": {"primary": 10}},
        execution_code_identity_sha256=CODE_SHA,
        max_new_units=1,
    )
    with pytest.raises(RobustnessAggregationError, match="campaign is incomplete"):
        aggregate_robustness_campaign(plan, tmp_path, _config())

    # Complete it, then corrupt one result without updating hashes.
    run_robustness_campaign(
        plan,
        tmp_path,
        lambda unit: {"metrics": {"primary": 10}},
        execution_code_identity_sha256=CODE_SHA,
    )
    path = sorted((tmp_path / "units").glob("*.json"))[0]
    payload = json.loads(path.read_text())
    payload["result"]["metrics"]["primary"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(RobustnessAggregationError, match="result hash mismatch"):
        aggregate_robustness_campaign(plan, tmp_path, _config())


def test_bad_thresholds_and_wrong_execution_identity_are_rejected(tmp_path):
    plan = _plan(parameters=[_parameter("a")])
    _execute(plan, tmp_path, lambda v, r: 10)
    with pytest.raises(RobustnessAggregationError, match="thresholds must satisfy"):
        aggregate_robustness_campaign(
            plan, tmp_path, _config(stable_absolute_delta=2, sensitive_absolute_delta=1)
        )
    with pytest.raises(RobustnessAggregationError, match="execution_code_identity_sha256"):
        aggregate_robustness_campaign(
            plan, tmp_path, _config(execution_code_identity_sha256="d" * 64)
        )


def test_runtime_replication_semantics_are_reported_without_fake_outer_uncertainty(tmp_path):
    plan = _plan(replications=1, parameters=[_parameter("a")])

    def executor(unit):
        value = 10 + float(unit.parameter_values["a"])
        return {
            "metrics": {"primary": value},
            "replication_semantics": {
                "schema": "cross-psp-robustness-replication-semantics-v1",
                "campaign_replication_index": unit.replication_index,
                "campaign_replication_axis": "NONE_FIXED_Q5_INTERNAL_ONLY",
                "q5_internal_replication_axis": "TRAFFIC_REALIZATION",
                "q5_internal_replication_count": 2,
                "q5_internal_seed_schedule": [101, 202],
                "q5_internal_paired": True,
                "q5_internal_confidence_level": 0.95,
                "uncertainty_note": "Q5 internal paired replications own traffic uncertainty.",
            },
        }

    summary = run_robustness_campaign(
        plan, tmp_path, executor, execution_code_identity_sha256=CODE_SHA
    )
    assert summary.status == "COMPLETE"
    result = aggregate_robustness_campaign(
        plan,
        tmp_path,
        _config(conclusion_rule={"operator": ">=", "threshold": 0}),
    )
    replication = result.replication_summary
    assert replication is not None
    assert replication["campaign_replication_count"] == 1
    assert replication["campaign_replication_axis"] == "NONE_FIXED_Q5_INTERNAL_ONLY"
    assert replication["q5_internal_replication_count"] == 2
    assert replication["q5_internal_seed_schedule"] == [101, 202]
    assert replication["standard_error_scope"] == "NOT_AVAILABLE_AT_CAMPAIGN_LEVEL"
    assert result.human_report_dict()["replication_summary"]["q5_internal_paired"] is True
