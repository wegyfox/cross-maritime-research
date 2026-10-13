from __future__ import annotations

import json
from pathlib import Path

import pytest

from cross_psp_two_stream.robustness_execution import (
    STATUS_CANCELLED_RESUMABLE,
    STATUS_COMPLETE,
    STATUS_FAILED_RESUMABLE,
    STATUS_PAUSED_RESUMABLE,
    RobustnessExecutionError,
    run_robustness_campaign,
)
from cross_psp_two_stream.robustness_matrix import (
    ROBUSTNESS_CAMPAIGN_SCHEMA,
    robustness_campaign_plan_from_mapping,
)

BASE_SHA = "a" * 64
CODE_SHA = "c" * 64
OTHER_CODE_SHA = "d" * 64


def _plan(*, seed: int = 7):
    return robustness_campaign_plan_from_mapping({
        "schema": ROBUSTNESS_CAMPAIGN_SCHEMA,
        "baseline": {
            "mode": "fixed",
            "physics_identity_sha256": BASE_SHA,
            "scope_status": "QUALIFIED_PROFILE",
            "plan_identity_sha256": None,
        },
        "scientific_question": "checkpoint semantics",
        "campaign_intent": "exploratory",
        "design": "full_factorial_grid",
        "master_seed": seed,
        "replications": 1,
        "confidence_target": None,
        "estimated_seconds_per_work_unit": 1,
        "max_work_units": 100,
        "parameters": [{
            "parameter_id": "stream.alpha.length_m",
            "unit": "m",
            "baseline_value": 100,
            "reference_value": 100,
            "supported_bounds": {"lower": 50, "upper": 150},
            "variation": {"mode": "symmetric_percent", "percent": 10, "levels": 3},
        }],
    })


def _executor(calls: list[int]):
    def execute(unit):
        calls.append(unit.work_unit_index)
        return {"metric": unit.work_unit_index + 0.5, "seed": unit.derived_seed}
    return execute


def _checkpoint(root: Path):
    return json.loads((root / "checkpoint.json").read_text())


def test_complete_execution_writes_atomic_unit_records_and_complete_checkpoint(tmp_path):
    calls: list[int] = []
    plan = _plan()
    summary = run_robustness_campaign(
        plan, tmp_path, _executor(calls), execution_code_identity_sha256=CODE_SHA
    )
    assert summary.status == STATUS_COMPLETE
    assert calls == [0, 1, 2]
    assert len(list((tmp_path / "units").glob("*.json"))) == 3
    checkpoint = _checkpoint(tmp_path)
    assert checkpoint["status"] == STATUS_COMPLETE
    assert checkpoint["completed_count"] == 3
    assert checkpoint["remaining_count"] == 0
    assert checkpoint["baseline"]["physics_identity_sha256"] == BASE_SHA


def test_bounded_pause_and_resume_do_not_recompute_valid_units(tmp_path):
    calls: list[int] = []
    plan = _plan()
    first = run_robustness_campaign(
        plan,
        tmp_path,
        _executor(calls),
        execution_code_identity_sha256=CODE_SHA,
        max_new_units=1,
    )
    assert first.status == STATUS_PAUSED_RESUMABLE
    assert calls == [0]
    second = run_robustness_campaign(
        plan,
        tmp_path,
        _executor(calls),
        execution_code_identity_sha256=CODE_SHA,
    )
    assert second.status == STATUS_COMPLETE
    assert calls == [0, 1, 2]


def test_cancel_callback_leaves_latest_checkpoint_resumable(tmp_path):
    calls: list[int] = []
    plan = _plan()
    checks = {"count": 0}
    def cancel():
        checks["count"] += 1
        return checks["count"] > 1
    summary = run_robustness_campaign(
        plan,
        tmp_path,
        _executor(calls),
        execution_code_identity_sha256=CODE_SHA,
        should_cancel=cancel,
    )
    assert summary.status == STATUS_CANCELLED_RESUMABLE
    assert calls == [0]
    assert _checkpoint(tmp_path)["status"] == STATUS_CANCELLED_RESUMABLE


def test_resume_fails_closed_on_execution_code_or_campaign_identity_mismatch(tmp_path):
    plan = _plan()
    run_robustness_campaign(
        plan, tmp_path, lambda unit: {"ok": True}, execution_code_identity_sha256=CODE_SHA, max_new_units=1
    )
    with pytest.raises(RobustnessExecutionError, match="execution_code_identity_sha256"):
        run_robustness_campaign(
            plan, tmp_path, lambda unit: {"ok": True}, execution_code_identity_sha256=OTHER_CODE_SHA
        )
    with pytest.raises(RobustnessExecutionError, match="campaign_identity_sha256"):
        run_robustness_campaign(
            _plan(seed=8), tmp_path, lambda unit: {"ok": True}, execution_code_identity_sha256=CODE_SHA
        )


def test_tampered_unit_record_is_rejected_before_executor_runs(tmp_path):
    calls: list[int] = []
    plan = _plan()
    run_robustness_campaign(
        plan, tmp_path, _executor(calls), execution_code_identity_sha256=CODE_SHA, max_new_units=1
    )
    unit_path = next((tmp_path / "units").glob("*.json"))
    payload = json.loads(unit_path.read_text())
    payload["result"]["metric"] = 999
    unit_path.write_text(json.dumps(payload))
    with pytest.raises(RobustnessExecutionError, match="result hash mismatch"):
        run_robustness_campaign(
            plan, tmp_path, _executor(calls), execution_code_identity_sha256=CODE_SHA
        )
    assert calls == [0]


def test_stale_checkpoint_repairs_from_newer_atomic_unit_record_without_recompute(tmp_path):
    calls: list[int] = []
    plan = _plan()
    run_robustness_campaign(
        plan, tmp_path, _executor(calls), execution_code_identity_sha256=CODE_SHA, max_new_units=1
    )
    stale = (tmp_path / "checkpoint.json").read_bytes()
    run_robustness_campaign(
        plan, tmp_path, _executor(calls), execution_code_identity_sha256=CODE_SHA, max_new_units=1
    )
    assert calls == [0, 1]
    (tmp_path / "checkpoint.json").write_bytes(stale)
    run_robustness_campaign(
        plan, tmp_path, _executor(calls), execution_code_identity_sha256=CODE_SHA
    )
    assert calls == [0, 1, 2]
    assert _checkpoint(tmp_path)["completed_count"] == 3


def test_failed_unit_is_recorded_but_not_completed_and_can_be_retried(tmp_path):
    plan = _plan()
    bad_calls: list[int] = []
    def fail_first(unit):
        bad_calls.append(unit.work_unit_index)
        raise ValueError("synthetic failure")
    with pytest.raises(RobustnessExecutionError, match="synthetic failure"):
        run_robustness_campaign(
            plan, tmp_path, fail_first, execution_code_identity_sha256=CODE_SHA
        )
    checkpoint = _checkpoint(tmp_path)
    assert checkpoint["status"] == STATUS_FAILED_RESUMABLE
    assert checkpoint["completed_count"] == 0
    assert len(list((tmp_path / "failures").glob("*.json"))) == 1

    good_calls: list[int] = []
    summary = run_robustness_campaign(
        plan, tmp_path, _executor(good_calls), execution_code_identity_sha256=CODE_SHA
    )
    assert summary.status == STATUS_COMPLETE
    assert good_calls == [0, 1, 2]


def test_noncanonical_executor_result_fails_closed_without_completion(tmp_path):
    plan = _plan()
    with pytest.raises(RobustnessExecutionError, match="canonical JSON"):
        run_robustness_campaign(
            plan,
            tmp_path,
            lambda unit: {"bad": float("nan")},
            execution_code_identity_sha256=CODE_SHA,
        )
    assert _checkpoint(tmp_path)["completed_count"] == 0
