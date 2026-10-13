from __future__ import annotations

from pathlib import Path

from cross_psp_two_stream.robustness_runtime import build_robustness_runtime_adapter
from cross_psp_two_stream.robustness_wizard import (
    load_robustness_wizard_config,
    preflight_robustness_wizard_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "robustness" / "a94_fixed_max_speed_acceptance.yaml"


def test_a94_acceptance_campaign_is_bounded_portable_and_exploratory(monkeypatch):
    monkeypatch.chdir(ROOT)
    mapping = load_robustness_wizard_config(CONFIG)
    assert mapping["baseline_source"]["scenario_path"] == "examples/q5_scenario.yaml"
    pf = preflight_robustness_wizard_config(CONFIG)
    assert pf.plan.work_unit_count == 3
    assert pf.plan.combination_count == 3
    assert pf.plan.replications == 1
    assert pf.plan.baseline.mode == "fixed"
    assert pf.plan.baseline.scope_status == "QUALIFIED_PROFILE"
    assert pf.plan.campaign_intent == "exploratory"
    assert [item.parameter_id for item in pf.plan.parameters] == ["stream.alpha.maximum_speed_knots"]
    assert pf.plan.parameters[0].values == ("9.8", "10", "10.2")
    assert pf.aggregation_config["primary_metric_path"] == "metrics.primary"


def test_a94_every_work_unit_prepares_through_a93_without_execution(monkeypatch):
    monkeypatch.chdir(ROOT)
    mapping = load_robustness_wizard_config(CONFIG)
    pf = preflight_robustness_wizard_config(CONFIG)
    adapter = build_robustness_runtime_adapter(mapping["baseline_source"])
    prepared = [adapter.prepare(unit) for unit in pf.plan.work_units]
    assert prepared[1].bound.physics_identity_sha256 == pf.plan.baseline.physics_identity_sha256
    assert prepared[0].bound.physics_identity_sha256 != pf.plan.baseline.physics_identity_sha256
    assert prepared[2].bound.physics_identity_sha256 != pf.plan.baseline.physics_identity_sha256
    assert all(item.traffic_seed == 359 for item in prepared)
    assert all(item.derived_plan_identity_sha256 is None for item in prepared)
    assert len(adapter.code_identity_sha256) == 64


def test_a94_acceptance_runner_and_launcher_are_packaged_and_syntax_valid():
    bundle_root = ROOT.parent
    runner = bundle_root / "acceptance" / "A94_PUBLIC_ROBUSTNESS_SMOKE_RUNNER.py"
    launcher = bundle_root / "acceptance" / "RUN_A94_PUBLIC_ROBUSTNESS_SMOKE.ps1"
    assert runner.is_file() and launcher.is_file()
    compile(runner.read_text(encoding="utf-8"), str(runner), "exec")
    text = launcher.read_text(encoding="utf-8")
    assert "A94_PUBLIC_ROBUSTNESS_SMOKE_RUNNER.py" in text
    assert ".venv\\Scripts\\python.exe" in text
