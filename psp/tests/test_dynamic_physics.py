from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

import cross_psp_two_stream.b6 as b6mod
from cross_psp_two_stream.contract import profile_contract_from_mapping
from cross_psp_two_stream.q5_integration import Q5IntegrationError, prepare_q5_scenario
from cross_psp_two_stream.reference_runtime import build_a70_reference_runtime_binding
from vessel_crossing.diagnostic.contract import load_diagnostic_scenario

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples" / "q5_scenario.yaml"


def _payload() -> dict:
    return yaml.safe_load(BASE.read_text())


def _load(tmp_path: Path, payload: dict):
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_diagnostic_scenario(path)


def _set_route(payload: dict, stream: str, start: tuple[float, float], end: tuple[float, float]) -> None:
    params = payload["model"]["parameters"]
    stream_row = next(row for row in params["streams"] if row["stream_id"] == stream)
    origin = stream_row["origin_node_id"]
    destination = stream_row["destination_node_id"]
    for node in params["network"]["nodes"]:
        if node["node_id"] == origin:
            node["position_nm"] = list(start)
        elif node["node_id"] == destination:
            node["position_nm"] = list(end)
    edge = next(
        row
        for row in params["network"]["edges"]
        if row["from_node_id"] == origin and row["to_node_id"] == destination
    )
    edge["waypoints_nm"] = [list(start), list(end)]


def test_reference_runtime_identity_is_exact_a70_fixture() -> None:
    binding = build_a70_reference_runtime_binding()
    assert binding.binding_id == "q4-physical-fixture-v1"
    assert binding.identity_sha256 == "faf41114fec96817562f01f439847cc71f4c57095d3d678f2f2c90d5ab29182e"


def test_baseline_q5_physics_is_qualified_but_never_reuses_85_scalar() -> None:
    prepared = prepare_q5_scenario(load_diagnostic_scenario(BASE))
    assert prepared.physical.scope_status == "QUALIFIED_PROFILE"
    assert prepared.physical.frozen_a70_reference_reuse_eligible is False
    assert prepared.physical.profile.b6_reference.measurement_duration_min == 180.0


def test_changed_geometry_speed_domain_and_shares_are_supported_exploratory(tmp_path: Path) -> None:
    data = _payload()
    _set_route(data, "alpha", (0.0, -1.2), (0.0, 1.8))
    _set_route(data, "beta", (1.4, 0.0), (-1.1, 0.0))
    data["model"]["parameters"]["streams"][0]["cruise_speed_knots"] = 7.0
    data["model"]["parameters"]["vessel_classes"]["q5"]["safety_domain"] = {
        "type": "four_quadrant_m",
        "ahead_m": 220.0,
        "astern_m": 120.0,
        "port_m": 90.0,
        "starboard_m": 110.0,
    }
    data["metadata"]["b6_target_stream_shares"] = {"alpha": 0.7, "beta": 0.3}
    prepared = prepare_q5_scenario(_load(tmp_path, data))
    assert prepared.physical.scope_status == "SUPPORTED_EXPLORATORY"
    profile = prepared.physical.profile
    geometry = {row.stream_id: row for row in profile.streams}
    assert geometry["alpha"].source_to_center_distance_nm == pytest.approx(1.2)
    assert geometry["alpha"].center_to_exit_distance_nm == pytest.approx(1.8)
    assert geometry["beta"].source_to_center_distance_nm == pytest.approx(1.4)
    assert geometry["beta"].center_to_exit_distance_nm == pytest.approx(1.1)
    shares = {row.stream_id: row.share for row in profile.b6_reference.stream_shares}
    assert shares == pytest.approx({"alpha": 0.7, "beta": 0.3})


def test_nonzero_drift_fails_closed_before_campaign(tmp_path: Path) -> None:
    data = _payload()
    data["model"]["parameters"]["scenario"]["default_environmental_drift_nm_per_min"] = [0.001, 0.0]
    with pytest.raises(Q5IntegrationError, match="zero environmental drift"):
        prepare_q5_scenario(_load(tmp_path, data))


def test_nonorthogonal_route_fails_closed_before_campaign(tmp_path: Path) -> None:
    data = _payload()
    _set_route(data, "beta", (1.5, -0.2), (-1.5, 0.2))
    with pytest.raises(Q5IntegrationError, match="orthogonal"):
        prepare_q5_scenario(_load(tmp_path, data))


def test_speed_above_envelope_fails_closed_before_campaign(tmp_path: Path) -> None:
    data = _payload()
    data["model"]["parameters"]["streams"][0]["cruise_speed_knots"] = 12.0
    with pytest.raises(Q5IntegrationError, match="speed envelope"):
        prepare_q5_scenario(_load(tmp_path, data))


def test_changed_physics_reference_smoke_is_deterministic_and_safe(tmp_path: Path) -> None:
    data = _payload()
    _set_route(data, "alpha", (0.0, -0.15), (0.0, 0.15))
    _set_route(data, "beta", (0.15, 0.0), (-0.15, 0.0))
    vessel = data["model"]["parameters"]["vessel_classes"]["q5"]
    vessel["dimensions_m"] = {"length_m": 30.0, "beam_m": 6.0}
    vessel["safety_domain"] = {"type": "circular_nm", "radius_nm": 0.025}
    prepared = prepare_q5_scenario(_load(tmp_path, data))

    profile_payload = prepared.physical.profile.canonical_dict()
    profile_payload.pop("component_contract_ids", None)
    profile_payload["b6_reference"].update(
        warmup_min=2.0,
        measurement_duration_min=12.0,
        drain_min=4.0,
        saturated_demands_per_stream=64,
    )
    smoke_profile = profile_contract_from_mapping(profile_payload)
    first = b6mod._run_once(
        smoke_profile,
        prepared.physical.profile_classes,
        prepared.reference_runtime,
        warmup_min=2.0,
        measurement_min=12.0,
        drain_min=4.0,
    )
    second = b6mod._run_once(
        smoke_profile,
        prepared.physical.profile_classes,
        prepared.reference_runtime,
        warmup_min=2.0,
        measurement_min=12.0,
        drain_min=4.0,
    )
    assert first["status"] == "PASS_REFERENCE_RUN"
    assert first["safety"]["pass"] is True
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
