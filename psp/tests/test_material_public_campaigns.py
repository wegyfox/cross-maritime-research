from __future__ import annotations

from pathlib import Path

import yaml

from cross_psp_two_stream.robustness_runtime import build_robustness_runtime_adapter
from cross_psp_two_stream.robustness_wizard import (
    load_robustness_wizard_config,
    preflight_robustness_wizard_config,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples" / "robustness" / "a96_material_campaign_suite.yaml"


def _suite():
    return yaml.safe_load(SUITE.read_text(encoding="utf-8"))


def _config(item):
    return ROOT / item["config"].removeprefix("examples/") if item["config"].startswith("examples/") else ROOT / item["config"]


def test_a96_suite_is_preflight_only_bounded_and_machine_calibrated():
    suite = _suite()
    assert suite["schema"] == "cross-psp-public-material-robustness-suite-v1"
    assert suite["status"] == "PREFLIGHT_ONLY"
    assert suite["machine_calibration"]["source"] == "A94 Windows acceptance evidence"
    assert suite["machine_calibration"]["planning_seconds_per_work_unit"] == 1800
    assert len(suite["campaigns"]) == 5
    assert sum(item["expected_work_units"] for item in suite["campaigns"]) == 45
    assert suite["claim_boundary"]["automatic_qualification_promotion"] is False


def test_a96_every_campaign_preflights_to_exact_nine_units_and_prepares_without_execution(monkeypatch):
    monkeypatch.chdir(ROOT)
    for item in _suite()["campaigns"]:
        path = Path(item["config"])
        mapping = load_robustness_wizard_config(path)
        preflight = preflight_robustness_wizard_config(path)
        assert preflight.plan.work_unit_count == item["expected_work_units"] == 9
        assert preflight.plan.estimated_total_seconds == "16200"
        adapter = build_robustness_runtime_adapter(mapping["baseline_source"])
        prepared = [adapter.prepare(unit) for unit in preflight.plan.work_units]
        assert len(prepared) == 9


def test_a96_fixed_and_sampled_replication_axes_are_deliberately_distinct(monkeypatch):
    monkeypatch.chdir(ROOT)
    suite = _suite()
    fixed = [item for item in suite["campaigns"] if item["category"] != "sampled_physics_robustness"]
    sampled = [item for item in suite["campaigns"] if item["category"] == "sampled_physics_robustness"]
    assert len(fixed) == 3 and len(sampled) == 2
    for item in fixed:
        pf = preflight_robustness_wizard_config(Path(item["config"]))
        assert pf.plan.baseline.mode == "fixed"
        assert pf.plan.replications == 1
    for item in sampled:
        path = Path(item["config"])
        mapping = load_robustness_wizard_config(path)
        pf = preflight_robustness_wizard_config(path)
        assert pf.plan.baseline.mode in {"distributions", "fleet_mixture"}
        assert pf.plan.replications == 3
        adapter = build_robustness_runtime_adapter(mapping["baseline_source"])
        # first three units are the three outer replications of one combination
        prepared = [adapter.prepare(unit) for unit in pf.plan.work_units[:3]]
        assert len({item.sample_seed for item in prepared}) == 3
        assert len({item.derived_plan_identity_sha256 for item in prepared}) == 3


def test_a96_primary_metrics_match_the_scientific_question(monkeypatch):
    monkeypatch.chdir(ROOT)
    by_id = {item["id"]: item for item in _suite()["campaigns"]}
    safety = load_robustness_wizard_config(Path(by_id["fixed-safety-domain-interaction"]["config"]))
    assert safety["aggregation"]["primary_metric_path"] == "metrics.b6_reference_rate_per_hour"
    for campaign_id in (
        "fixed-cruise-interaction",
        "fixed-maneuverability-interaction",
        "distribution-length-variability",
        "fleet-mixture-composition",
    ):
        mapping = load_robustness_wizard_config(Path(by_id[campaign_id]["config"]))
        assert mapping["aggregation"]["primary_metric_path"] == "metrics.primary"
