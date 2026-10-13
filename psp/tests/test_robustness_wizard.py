from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from cross_psp_two_stream.robustness_execution import run_robustness_campaign
from cross_psp_two_stream.robustness_wizard import (
    ROBUSTNESS_WIZARD_SCHEMA,
    RobustnessWizardError,
    baseline_source_from_files,
    preflight_robustness_wizard_config,
    run_robustness_wizard,
    write_robustness_wizard_config,
)
from cross_psp_two_stream.wizard import ConsoleIO, run_wizard

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "q5_scenario.yaml"
CODE_SHA = "c" * 64
OTHER_CODE_SHA = "d" * 64


def _parameter(parameter_id: str = "stream.alpha.length_m"):
    return {
        "parameter_id": parameter_id,
        "unit": "m",
        "baseline_value": 120,
        "reference_value": 120,
        "supported_bounds": {"lower": 80, "upper": 180},
        "variation": {"mode": "symmetric_percent", "percent": 10, "levels": 3},
    }


def _mapping(source, *, replications: int = 1, help_key: str = "stream.alpha.length_m"):
    return {
        "schema": ROBUSTNESS_WIZARD_SCHEMA,
        "baseline_source": source,
        "campaign": {
            "scientific_question": "Does the primary conclusion remain stable over the declared envelope?",
            "campaign_intent": "exploratory",
            "design": "full_factorial_grid",
            "master_seed": 991,
            "replications": replications,
            "confidence_target": 0.95,
            "estimated_seconds_per_work_unit": 2.5,
            "max_work_units": 100,
            "parameters": [_parameter()],
        },
        "parameter_help": {help_key: "Alpha vessel length at the declared physical baseline."},
        "aggregation": {
            "schema": "cross-psp-robustness-aggregation-v1",
            "primary_metric_path": "metrics.primary",
            "stable_absolute_delta": 1.0,
            "sensitive_absolute_delta": 20.0,
            "conclusion_rule": {"operator": ">=", "threshold": 100.0},
        },
    }


def _write_config(tmp_path: Path, source, **kwargs) -> Path:
    return write_robustness_wizard_config(_mapping(source, **kwargs), tmp_path / "robustness.yaml")


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
                "bounds": {"lower": 110, "upper": 130},
                "sampling_scope": "per_vessel",
            },
        }],
    }


def _profile(class_id: str):
    return {
        "class_id": class_id,
        "dimensions_m": {"length_m": 120, "beam_m": 20},
        "speed": {"cruise_speed_knots": 6, "maximum_speed_knots": 10},
        "maneuver_limits": {
            "max_acceleration_knots_per_min": 2,
            "max_deceleration_knots_per_min": 2,
            "max_turn_rate_deg_per_min": 30,
        },
        "safety_domain": {
            "type": "four_quadrant_nm",
            "ahead_nm": 0.10,
            "astern_nm": 0.10,
            "port_nm": 0.10,
            "starboard_nm": 0.10,
        },
    }


def _mixture_mapping(seed: int = 18):
    return {
        "schema": "cross-psp-fleet-mixture-v1",
        "master_seed": seed,
        "streams": [
            {"stream_id": "alpha", "classes": [{"weight": 1, "profile": _profile("alpha-standard")}]},
            {"stream_id": "beta", "classes": [{"weight": 1, "profile": _profile("beta-standard")}]},
        ],
    }


def test_fixed_robustness_preflight_is_identity_pinned_and_shows_exact_workload(tmp_path):
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    path = _write_config(tmp_path, source)
    pf = preflight_robustness_wizard_config(path, output_dir=tmp_path / "out")
    assert pf.plan.baseline.mode == "fixed"
    assert pf.plan.baseline.scope_status == "QUALIFIED_PROFILE"
    assert pf.plan.combination_count == 3
    assert pf.plan.replications == 1
    assert pf.plan.work_unit_count == 3
    assert pf.plan.estimated_total_seconds == "7.5"
    assert pf.parameter_help["stream.alpha.length_m"].startswith("Alpha vessel")
    assert pf.checkpoint_path == tmp_path / "out" / "checkpoint.json"
    assert len(pf.config_identity_sha256) == 64


def test_fixed_preflight_rejects_duplicate_outer_replications(tmp_path):
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    path = _write_config(tmp_path, source, replications=2)
    with pytest.raises(RobustnessWizardError, match="campaign.replications=1"):
        preflight_robustness_wizard_config(path)


def test_distribution_and_fleet_mixture_baseline_modes_are_publicly_resolved_as_exploratory(tmp_path):
    dist_path = tmp_path / "distribution.yaml"
    dist_path.write_text(yaml.safe_dump(_distribution_mapping(), sort_keys=False), encoding="utf-8")
    dist_source = baseline_source_from_files(
        mode="distributions", scenario_path=SCENARIO, plan_path=dist_path
    )
    dist_config = _write_config(tmp_path / "dist", dist_source)
    dist_pf = preflight_robustness_wizard_config(dist_config)
    assert dist_pf.plan.baseline.mode == "distributions"
    assert dist_pf.plan.baseline.scope_status == "SUPPORTED_EXPLORATORY"
    assert dist_pf.plan.baseline.plan_identity_sha256 == dist_source["expected_plan_identity_sha256"]

    mix_path = tmp_path / "mixture.yaml"
    mix_path.write_text(yaml.safe_dump(_mixture_mapping(), sort_keys=False), encoding="utf-8")
    mix_source = baseline_source_from_files(
        mode="fleet_mixture", scenario_path=SCENARIO, plan_path=mix_path
    )
    mix_config = _write_config(tmp_path / "mix", mix_source)
    mix_pf = preflight_robustness_wizard_config(mix_config)
    assert mix_pf.plan.baseline.mode == "fleet_mixture"
    assert mix_pf.plan.baseline.scope_status == "SUPPORTED_EXPLORATORY"
    assert mix_pf.plan.baseline.plan_identity_sha256 == mix_source["expected_plan_identity_sha256"]


def test_saved_sampled_config_fails_closed_if_underlying_plan_changes(tmp_path):
    dist_path = tmp_path / "distribution.yaml"
    dist_path.write_text(yaml.safe_dump(_distribution_mapping(seed=17), sort_keys=False), encoding="utf-8")
    source = baseline_source_from_files(mode="distributions", scenario_path=SCENARIO, plan_path=dist_path)
    config = _write_config(tmp_path, source)
    dist_path.write_text(yaml.safe_dump(_distribution_mapping(seed=999), sort_keys=False), encoding="utf-8")
    with pytest.raises(RobustnessWizardError, match="plan identity mismatch|physics identity mismatch"):
        preflight_robustness_wizard_config(config)


def test_parameter_help_must_cover_exact_campaign_surface(tmp_path):
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    path = _write_config(tmp_path, source, help_key="wrong.parameter")
    with pytest.raises(RobustnessWizardError, match="parameter_help keys"):
        preflight_robustness_wizard_config(path)


def test_guided_execution_reports_checkpoint_progress_aggregates_and_opens_report(tmp_path):
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    config = _write_config(tmp_path, source, replications=1)
    answers = iter(["y"])
    output: list[str] = []
    opened: list[Path] = []
    calls: list[int] = []
    io = ConsoleIO(read=lambda prompt: next(answers), write=output.append)

    def execute(unit):
        calls.append(unit.work_unit_index)
        length = float(unit.parameter_values["stream.alpha.length_m"])
        return {"metrics": {"primary": length + unit.replication_index}}

    status = run_robustness_wizard(
        io=io,
        config_path=config,
        output_dir=tmp_path / "out",
        execute_unit=execute,
        execution_code_identity_sha256=CODE_SHA,
        open_result=opened.append,
    )
    assert status == 0
    assert calls == list(range(3))
    assert any("Robustness progress: completed=1/3" in line for line in output)
    assert any("ROBUSTNESS CAMPAIGN COMPLETE" in line for line in output)
    assert (tmp_path / "out" / "checkpoint.json").is_file()
    assert (tmp_path / "out" / "ROBUSTNESS_RESULT.json").is_file()
    report = tmp_path / "out" / "ROBUSTNESS_REPORT.md"
    assert report.is_file()
    assert opened == [report.resolve()]
    payload = json.loads((tmp_path / "out" / "ROBUSTNESS_RESULT.json").read_text())
    assert payload["automatic_qualification_promotion"] is False
    assert payload["claim_status"] == "SUPPORTED_EXPLORATORY"


def test_resume_rejects_changed_execution_code_identity_without_recomputing_valid_units(tmp_path):
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    config = _write_config(tmp_path, source, replications=1)
    out_dir = tmp_path / "out"
    pf = preflight_robustness_wizard_config(config, output_dir=out_dir)
    first_calls: list[int] = []
    run_robustness_campaign(
        pf.plan,
        out_dir,
        lambda unit: first_calls.append(unit.work_unit_index) or {"metrics": {"primary": 120}},
        execution_code_identity_sha256=CODE_SHA,
        max_new_units=1,
    )
    assert first_calls == [0]

    output: list[str] = []
    later_calls: list[int] = []
    io = ConsoleIO(read=lambda prompt: "y", write=output.append)
    status = run_robustness_wizard(
        io=io,
        config_path=config,
        output_dir=out_dir,
        execute_unit=lambda unit: later_calls.append(unit.work_unit_index) or {"metrics": {"primary": 120}},
        execution_code_identity_sha256=OTHER_CODE_SHA,
    )
    assert status == 2
    assert later_calls == []
    assert any("FAIL-CLOSED robustness campaign" in line for line in output)
    assert any("identity" in line.lower() for line in output)


def test_existing_wizard_routes_robustness_mode_without_starting_when_executor_is_absent(tmp_path):
    source = baseline_source_from_files(mode="fixed", scenario_path=SCENARIO)
    config = _write_config(tmp_path, source)
    output: list[str] = []
    io = ConsoleIO(read=lambda prompt: "", write=output.append)
    status = run_wizard(
        mode="robustness",
        io=io,
        config_path=config,
        output_dir=tmp_path / "out",
    )
    assert status == 0
    assert any("Robustness preflight PASS" in line for line in output)
    assert any("Scientific work-unit execution is not attached" in line for line in output)


def test_guided_builder_creates_reloadable_config_without_manual_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter([
        str(SCENARIO),  # baseline diagnostic YAML
        "",             # fixed mode
        "",             # scientific question
        "",             # exploratory intent
        "",             # master seed
        "",             # replications
        "",             # confidence
        "",             # sec/work unit
        "",             # max work units
        "",             # one parameter
        "",             # parameter id
        "",             # meaning
        "",             # unit
        "",             # baseline
        "",             # reference
        "",             # supported lower = none
        "",             # supported upper = none
        "",             # symmetric variation
        "",             # 10 percent
        "",             # 3 levels
        "",             # metric path
        "",             # stable threshold
        "",             # sensitive threshold
        "",             # conclusion operator
        "",             # conclusion threshold
    ])
    output: list[str] = []
    io = ConsoleIO(read=lambda prompt: next(answers), write=output.append)
    status = run_wizard(mode="robustness", io=io)
    assert status == 0
    saved = list((tmp_path / "cross_custom_configs").glob("cross_robustness_*.yaml"))
    assert len(saved) == 1
    pf = preflight_robustness_wizard_config(saved[0])
    assert pf.plan.baseline.mode == "fixed"
    assert pf.plan.work_unit_count == 3
    assert any("Saved robustness configuration" in line for line in output)
    assert any("Robustness preflight PASS" in line for line in output)


def test_guided_builder_cancel_starts_nothing_and_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output: list[str] = []
    io = ConsoleIO(read=lambda prompt: "cancel", write=output.append)
    status = run_wizard(mode="robustness", io=io)
    assert status == 0
    assert not (tmp_path / "cross_custom_configs").exists()
    assert any("Cancelled" in line for line in output)
