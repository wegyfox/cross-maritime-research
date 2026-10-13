from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import yaml

from cross_psp_two_stream.fleet_mixtures import fleet_mixture_plan_from_mapping
from cross_psp_two_stream.robustness_runtime import build_robustness_runtime_adapter
from cross_psp_two_stream.robustness_wizard import (
    load_robustness_wizard_config,
    preflight_robustness_wizard_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PROJECT_ROOT.parent
CONFIG = PROJECT_ROOT / "examples" / "robustness" / "a103_final_fleet_realism_gate.yaml"
PLAN = PROJECT_ROOT / "examples" / "robustness" / "plans" / "a103_final_fleet_realism_baseline.yaml"
RUNNER = BUNDLE_ROOT / "acceptance" / "A103_FINAL_FLEET_REALISM_GATE_RUNNER.py"
LAUNCHER = BUNDLE_ROOT / "acceptance" / "RUN_A103_FINAL_FLEET_REALISM_GATE.ps1"


def test_a103_is_exactly_three_work_units_and_one_sampled_replication(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    preflight = preflight_robustness_wizard_config(CONFIG)
    assert preflight.plan.baseline.mode == "fleet_mixture"
    assert preflight.plan.replications == 1
    assert preflight.plan.work_unit_count == 3
    grid = preflight.plan.parameters[0]
    assert grid.parameter_id == "fleet.alpha.class.alpha-autonomous-standard.weight"
    assert grid.values == ("0.2", "2", "20")
    assert preflight.plan.estimated_total_seconds == "5400"


def test_a103_profiles_are_coherent_and_explicitly_synthetic():
    raw = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    plan = fleet_mixture_plan_from_mapping(raw)
    assert plan.plan_identity_sha256 == "d4e6b7bdeeb5e0e460f847dc5be8165739428dff4d390c567abcfd45ee28cf55"
    streams = {stream.stream_id: stream for stream in plan.streams}
    assert set(streams) == {"alpha", "beta"}
    alpha_ids = {member.profile.class_id for member in streams["alpha"].classes}
    beta_ids = {member.profile.class_id for member in streams["beta"].classes}
    assert alpha_ids == {
        "alpha-small-conventional-like",
        "alpha-large-conventional-like",
        "alpha-autonomous-standard",
    }
    assert beta_ids == {
        "beta-small-conventional-like",
        "beta-medium-conventional-like",
        "beta-large-conventional-like",
    }
    config_text = CONFIG.read_text(encoding="utf-8")
    assert "are synthetic" in config_text
    assert "representative coherent profiles, not field-calibrated vessel populations" in config_text


def test_a103_prepares_three_distinct_sampled_physics_units_without_execution(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    mapping = load_robustness_wizard_config(CONFIG)
    preflight = preflight_robustness_wizard_config(CONFIG)
    adapter = build_robustness_runtime_adapter(mapping["baseline_source"])
    prepared = [adapter.prepare(unit) for unit in preflight.plan.work_units]
    assert len(prepared) == 3
    assert len({item.sample_seed for item in prepared}) == 3
    assert len({item.derived_plan_identity_sha256 for item in prepared}) == 3
    assert len({item.bound.physics_identity_sha256 for item in prepared}) == 3


def test_a103_runner_is_preflightable_and_fail_closed_to_three_units():
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--preflight-only"],
        cwd=BUNDLE_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "A103 FINAL FLEET REALISM PREFLIGHT PASS" in completed.stdout
    assert "work_unit_count=3" in completed.stdout
    assert "alpha_autonomous_share_approx=9%,50%,91%" in completed.stdout
    assert "automatic_qualification_promotion=false" in completed.stdout
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    assert "A103_FINAL_FLEET_REALISM_GATE_RUNNER.py" in launcher_text
    assert "--preflight-only" in launcher_text
