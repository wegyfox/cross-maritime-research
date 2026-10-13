from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

import cross_psp_two_stream.contract as c


def _mapping():
    return yaml.safe_load((Path(__file__).parents[1] / "examples" / "profile.yaml").read_text())


def test_example_contract_loads_and_uses_arbitrary_stream_names():
    contract = c.load_profile_contract(Path(__file__).parents[1] / "examples" / "profile.yaml")
    assert tuple(row.stream_id for row in contract.streams) == ("alpha", "beta")
    assert contract.profile_id == c.PROFILE_ID
    assert contract.base_framework_version == "1.1.0a36"
    assert contract.profile_version == "0.1.0a7"
    assert contract.planned_slot_prediction_scope_id == c.PLANNED_SLOT_PREDICTION_SCOPE_ID
    assert contract.runtime_adapter_id == "nextgen.environmental_network_fifo.v1"
    assert set(contract.component_contract_ids) == {
        "slot_solver", "b4_policy", "b5_policy", "b6_reference", "reference_arbiter"
    }
    assert len(contract.contract_sha256) == 64


def test_canonical_hash_is_independent_of_json_key_order(tmp_path):
    mapping = _mapping()
    first = c.profile_contract_from_mapping(mapping)
    reordered = {key: mapping[key] for key in reversed(list(mapping))}
    second = c.profile_contract_from_mapping(reordered)
    assert first.contract_sha256 == second.contract_sha256
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(reordered))
    assert c.load_profile_contract(path).contract_sha256 == first.contract_sha256


def test_profile_v1_rejects_nonorthogonal_crossing():
    mapping = _mapping()
    mapping["crossing_angle_deg"] = 60.0
    with pytest.raises(c.ProfileContractError, match="orthogonal"):
        c.profile_contract_from_mapping(mapping)


def test_profile_rejects_duplicate_stream_ids_and_share_mismatch():
    duplicate = _mapping()
    duplicate["streams"][1]["stream_id"] = "alpha"
    with pytest.raises(c.ProfileContractError, match="distinct"):
        c.profile_contract_from_mapping(duplicate)

    mismatch = _mapping()
    mismatch["b6_reference"]["stream_shares"][1]["stream_id"] = "gamma"
    with pytest.raises(c.ProfileContractError, match="exactly match"):
        c.profile_contract_from_mapping(mismatch)


def test_profile_rejects_invalid_numerical_and_fairness_bounds():
    mapping = _mapping()
    mapping["slot_solver"]["separation_grid_min"] = 31.0
    with pytest.raises(c.ProfileContractError, match="must not exceed"):
        c.profile_contract_from_mapping(mapping)

    mapping = _mapping()
    mapping["b5_fairness"]["lookahead_depth"] = 1
    with pytest.raises(c.ProfileContractError, match="lookahead_depth"):
        c.profile_contract_from_mapping(mapping)


def test_profile_rejects_invalid_b6_balance_contract():
    mapping = _mapping()
    mapping["b6_reference"]["stream_shares"][0]["share"] = 0.6
    with pytest.raises(c.ProfileContractError, match="sum to 1"):
        c.profile_contract_from_mapping(mapping)

    mapping = _mapping()
    mapping["b6_reference"]["balance_tolerance_fraction"] = 0.5
    with pytest.raises(c.ProfileContractError, match="must be < 0.5"):
        c.profile_contract_from_mapping(mapping)


def test_profile_rejects_invalid_reference_fairness_contract():
    mapping = _mapping()
    mapping["b6_reference"]["max_share_deficit_passages"] = 0.0
    with pytest.raises(c.ProfileContractError, match="max_share_deficit_passages"):
        c.profile_contract_from_mapping(mapping)

    mapping = _mapping()
    mapping["b6_reference"]["fairness_comparison_tolerance"] = mapping["b6_reference"]["max_share_deficit_passages"]
    with pytest.raises(c.ProfileContractError, match="fairness_comparison_tolerance"):
        c.profile_contract_from_mapping(mapping)


def test_unknown_keys_fail_closed():
    mapping = _mapping()
    mapping["r2_delta_arm"] = "POS050"
    with pytest.raises(c.ProfileContractError, match="extra"):
        c.profile_contract_from_mapping(mapping)


def test_base_framework_pin_is_explicit(monkeypatch):
    c.assert_compatible_base_framework()
    monkeypatch.setattr(c, "_base_framework_version", "1.1.0a37")
    with pytest.raises(RuntimeError, match="qualified only"):
        c.assert_compatible_base_framework()


def test_profile_requires_explicit_exit_distance_and_zero_drift_prediction_scope():
    mapping = _mapping()
    del mapping["streams"][0]["center_to_exit_distance_nm"]
    with pytest.raises(c.ProfileContractError, match="missing"):
        c.profile_contract_from_mapping(mapping)

    mapping = _mapping()
    mapping["planned_slot_prediction_scope_id"] = "uniform-drift-planned-slot-v1"
    with pytest.raises(c.ProfileContractError, match="planned_slot_prediction_scope_id"):
        c.profile_contract_from_mapping(mapping)


def test_profile_rejects_invalid_clearance_numerics():
    mapping = _mapping()
    mapping["slot_solver"]["clearance_margin_tolerance_nm"] = -1e-6
    with pytest.raises(c.ProfileContractError, match="non-negative"):
        c.profile_contract_from_mapping(mapping)

    mapping = _mapping()
    mapping["slot_solver"]["max_clearance_subdivisions"] = 0
    with pytest.raises(c.ProfileContractError, match="max_clearance_subdivisions"):
        c.profile_contract_from_mapping(mapping)
