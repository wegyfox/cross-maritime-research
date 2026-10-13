from __future__ import annotations

from copy import deepcopy
from importlib.resources import as_file, files
import json
from pathlib import Path

import pytest
import yaml

from cross_psp_two_stream.analysis import (
    ANALYSIS_SURFACE_ID,
    AnalysisSurfaceError,
    load_baseline_analysis_scenario,
    validate_analysis_scenario,
    classify_analysis_failure,
    write_analysis_failure,
    write_analysis_summary,
)


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_analysis_accepts_demand_seed_horizon_replication_changes(tmp_path: Path) -> None:
    data = deepcopy(load_baseline_analysis_scenario())
    data["scenario_id"] = "CUSTOM"
    data["model"]["parameters"]["streams"][0]["rate_per_hour"] = 4.0
    data["model"]["parameters"]["streams"][1]["rate_per_hour"] = 9.0
    data["model"]["parameters"]["scenario"]["seed"] = 777
    data["model"]["parameters"]["scenario"]["traffic_horizon_min"] = 20.0
    data["model"]["parameters"]["scenario"]["end_time"] = 70.0
    data["experiment"]["replications"] = 6
    data["experiment"]["confidence_level"] = 0.9
    result = validate_analysis_scenario(_write(tmp_path, data))
    assert result.rate_per_hour_by_stream == {"alpha": 4.0, "beta": 9.0}
    assert result.replications == 6
    assert result.confidence_level == 0.9
    assert result.changed_paths


def test_analysis_marks_physics_change_for_scientific_preflight(tmp_path: Path) -> None:
    data = deepcopy(load_baseline_analysis_scenario())
    data["model"]["parameters"]["streams"][0]["cruise_speed_knots"] = 7.0
    result = validate_analysis_scenario(_write(tmp_path, data))
    assert any(path.endswith("cruise_speed_knots") for path in result.physical_changed_paths)


def test_analysis_allows_drift_attempt_but_marks_it_physical(tmp_path: Path) -> None:
    data = deepcopy(load_baseline_analysis_scenario())
    data["model"]["parameters"]["scenario"]["default_environmental_drift_nm_per_min"] = [0.01, 0.0]
    result = validate_analysis_scenario(_write(tmp_path, data))
    assert any("default_environmental_drift" in path for path in result.physical_changed_paths)


def test_analysis_summary_projects_canonical_decision_view(tmp_path: Path) -> None:
    data = deepcopy(load_baseline_analysis_scenario())
    data["experiment"]["replications"] = 4
    validation = validate_analysis_scenario(_write(tmp_path, data))
    decision = {
        "source": {"decomposition_sha256": "a" * 64},
        "dominant_modeled_bottleneck": {"classification": "GLOBAL_SLOT_COORDINATION_CONFLICT", "basis": "x", "caveat": "y"},
        "canonical_next_counterfactual": {"for_B0_like_operation": "global_slot_coordination", "deployment_recommendation": "false"},
        "claim_scope": {"field_calibrated": False, "deployment_recommendation": False},
        "scenarios": [{"scenario_id": "S", "current_modeled_value": 3.0, "managed_modeled_value": 4.0, "managed_recoverable_gap": 1.0, "physical_reference_value": 85.0, "residual_reference_gap": 81.0}],
    }
    (tmp_path / "DECISION_VIEW.json").write_text(json.dumps(decision))
    json_path, md_path = write_analysis_summary(tmp_path, validation)
    payload = json.loads(json_path.read_text())
    assert payload["analysis_surface_id"] == ANALYSIS_SURFACE_ID
    assert payload["dominant_modeled_bottleneck"]["classification"] == "GLOBAL_SLOT_COORDINATION_CONFLICT"
    assert "modeled research diagnosis" in md_path.read_text().lower()


def test_failure_taxonomy_distinguishes_semantic_and_action_space(tmp_path: Path) -> None:
    semantic = classify_analysis_failure("crossing classification does not yield a unique starboard-side give-way vessel")
    action = classify_analysis_failure("environmental pair batch contains no safe progress-capable row")
    assert semantic[0] == "SEMANTIC_AMBIGUITY_FAIL_CLOSED"
    assert action[0] == "MODEL_ACTION_SPACE_BOUNDARY"
    sparse = classify_analysis_failure("B4 reservoir streams must exactly match the two profile streams")
    assert sparse[0] == "INSUFFICIENT_REALIZED_DEMAND"
    data = deepcopy(load_baseline_analysis_scenario())
    validation = validate_analysis_scenario(_write(tmp_path, data))
    _, md_path, terminal = write_analysis_failure(tmp_path, validation, error="crossing classification does not yield a unique starboard-side give-way vessel")
    assert terminal == "SEMANTIC_AMBIGUITY_FAIL_CLOSED"
    assert "not a field-capacity" in md_path.read_text()
