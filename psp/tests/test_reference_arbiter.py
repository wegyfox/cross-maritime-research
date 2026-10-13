from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

import cross_psp_two_stream.b6 as b6mod
from cross_psp_two_stream import profile_contract_from_mapping
from cross_psp_two_stream.reference_arbiter import (
    ReferenceFairSlotArbiter,
    build_reference_stream_keys,
)
from vessel_crossing.nextgen import ShipDomain, VesselClassProfile, VesselDimensions
from vessel_crossing.research.r1_p4b_experiment_spec import build_publication_config

ROOT = Path(__file__).resolve().parents[1]


def _mapping():
    return yaml.safe_load((ROOT / "examples" / "profile.yaml").read_text())


def _profile_class():
    cfg = build_publication_config(room_nm=1.375, demand_per_hour=4.0, traffic_seed=0)
    base = cfg.streams[0]
    return VesselClassProfile(
        class_id="reference-class",
        dimensions=VesselDimensions.from_meters(length_m=30.0, beam_m=6.0),
        maneuver_limits=base.limits,
        safety_domain=ShipDomain.circular(0.025),
        speed_envelope=base.vessel_class.speed_envelope,
    )


def _ctx(time: float = 0.0):
    return SimpleNamespace(snapshot=SimpleNamespace(time=time))


def _rename(mapping):
    out = deepcopy(mapping)
    ren = {"alpha": "zeta", "beta": "eta"}
    for row in out["streams"]:
        row["stream_id"] = ren[row["stream_id"]]
    for row in out["b6_reference"]["stream_shares"]:
        row["stream_id"] = ren[row["stream_id"]]
    for row in out["b6_reference"]["supply_templates"]:
        row["stream_id"] = ren[row["stream_id"]]
    return out


def test_reference_stream_keys_are_display_name_invariant():
    vp = _profile_class()
    original = profile_contract_from_mapping(_mapping())
    renamed = profile_contract_from_mapping(_rename(_mapping()))
    first = build_reference_stream_keys(original, {vp.class_id: vp})
    second = build_reference_stream_keys(renamed, {vp.class_id: vp})
    assert set(first.values()) == set(second.values())


def test_fairness_projection_forces_other_stream_after_symmetric_first_commitment():
    vp = _profile_class()
    profile = profile_contract_from_mapping(_mapping())
    keys = build_reference_stream_keys(profile, {vp.class_id: vp})
    arb = ReferenceFairSlotArbiter(profile, center_observation_provider=lambda: (), reference_stream_keys_by_stream=keys)
    ka, kb = keys["alpha"], keys["beta"]
    assert arb.fairness_feasible(ka)[0]
    assert arb.fairness_feasible(kb)[0]
    arb.commitments_by_key[ka] = 1
    assert arb.fairness_feasible(ka)[0] is False
    assert arb.fairness_feasible(kb)[0] is True


def test_symmetric_tie_selection_is_invariant_to_display_rename():
    vp = _profile_class()
    selected = []
    for mapping in (_mapping(), _rename(_mapping())):
        profile = profile_contract_from_mapping(mapping)
        keys = build_reference_stream_keys(profile, {vp.class_id: vp})
        reservoir, *_ = b6mod._supply(profile, {vp.class_id: vp}, keys)
        arb = ReferenceFairSlotArbiter(profile, center_observation_provider=lambda: (), reference_stream_keys_by_stream=keys)
        prepared = arb.prepare_reservoir(reservoir, _ctx())
        assert prepared is not None
        selected.append(arb.reservation.stream_key)
    assert selected[0] == selected[1]


def test_asymmetric_70_30_deficit_guard_has_no_hardcoded_alternation():
    data = _mapping()
    data["b6_reference"]["stream_shares"][0]["share"] = 0.7
    data["b6_reference"]["stream_shares"][1]["share"] = 0.3
    data["b6_reference"]["max_share_deficit_passages"] = 0.7
    profile = profile_contract_from_mapping(data)
    vp = _profile_class()
    keys = build_reference_stream_keys(profile, {vp.class_id: vp})
    arb = ReferenceFairSlotArbiter(profile, center_observation_provider=lambda: (), reference_stream_keys_by_stream=keys)
    ka, kb = keys["alpha"], keys["beta"]
    # Two alpha commitments can be feasible in the generic deficit band; this is not strict alternation.
    arb.commitments_by_key[ka] = 1
    assert arb.fairness_feasible(ka)[0]
    # But continued overservice eventually makes beta mandatory.
    arb.commitments_by_key[ka] = 2
    assert arb.fairness_feasible(ka)[0] is False
    assert arb.fairness_feasible(kb)[0] is True


def test_duplicate_reference_stream_keys_fail_closed():
    import pytest
    from cross_psp_two_stream.reference_arbiter import ReferenceArbiterError
    vp = _profile_class()
    profile = profile_contract_from_mapping(_mapping())
    with pytest.raises(ReferenceArbiterError, match="FAIL_NON_UNIQUE_REFERENCE_STREAM_KEY"):
        ReferenceFairSlotArbiter(
            profile,
            center_observation_provider=lambda: (),
            reference_stream_keys_by_stream={"alpha": "a" * 64, "beta": "a" * 64},
        )


def test_denial_creates_no_phantom_commitment_and_admission_creates_exactly_one():
    from types import SimpleNamespace
    vp = _profile_class()
    profile = profile_contract_from_mapping(_mapping())
    keys = build_reference_stream_keys(profile, {vp.class_id: vp})
    reservoir, *_ = b6mod._supply(profile, {vp.class_id: vp}, keys)
    arb = ReferenceFairSlotArbiter(profile, center_observation_provider=lambda: (), reference_stream_keys_by_stream=keys)
    arb.prepare_reservoir(reservoir, _ctx())
    selected = arb.reservation
    assert selected is not None
    before = dict(arb.commitments_by_key)
    denial = SimpleNamespace(
        snapshot_time=selected.release_target_min,
        admissions=(),
        decisions=(SimpleNamespace(vessel_id=selected.vessel_id, status=SimpleNamespace(value="BLOCKED")),),
    )
    arb.observe_resolution(denial, _ctx(selected.release_target_min))
    assert arb.commitments_by_key == before
    assert arb.reservation is None

    arb.prepare_reservoir(reservoir, _ctx(selected.release_target_min))
    selected2 = arb.reservation
    assert selected2 is not None
    admission = SimpleNamespace(vessel_id=selected2.vessel_id, admission_time=selected2.release_target_min)
    resolution = SimpleNamespace(snapshot_time=selected2.release_target_min, admissions=(admission,), decisions=())
    arb.observe_resolution(resolution, _ctx(selected2.release_target_min))
    assert sum(arb.commitments_by_key.values()) == 1
    assert arb.commitments_by_key[selected2.stream_key] == 1


def test_fairness_infeasible_faster_stream_is_rejected():
    vp = _profile_class()
    profile = profile_contract_from_mapping(_mapping())
    keys = build_reference_stream_keys(profile, {vp.class_id: vp})
    reservoir, *_ = b6mod._supply(profile, {vp.class_id: vp}, keys)
    arb = ReferenceFairSlotArbiter(profile, center_observation_provider=lambda: (), reference_stream_keys_by_stream=keys)
    ka, kb = keys["alpha"], keys["beta"]
    arb.commitments_by_key[ka] = 1
    arb.prepare_reservoir(reservoir, _ctx())
    assert arb.reservation is not None
    assert arb.reservation.stream_key == kb


def test_70_30_analytical_no_starvation_bound():
    data = _mapping()
    data["b6_reference"]["stream_shares"][0]["share"] = 0.7
    data["b6_reference"]["stream_shares"][1]["share"] = 0.3
    data["b6_reference"]["max_share_deficit_passages"] = 0.7
    profile = profile_contract_from_mapping(data)
    vp = _profile_class()
    keys = build_reference_stream_keys(profile, {vp.class_id: vp})
    arb = ReferenceFairSlotArbiter(profile, center_observation_provider=lambda: (), reference_stream_keys_by_stream=keys)
    ka, kb = keys["alpha"], keys["beta"]
    # floor(B/q_beta)=floor(0.7/0.3)=2 additional alpha commitments are feasible.
    assert arb.fairness_feasible(ka)[0]
    arb.commitments_by_key[ka] = 1
    assert arb.fairness_feasible(ka)[0]
    arb.commitments_by_key[ka] = 2
    assert arb.fairness_feasible(ka)[0] is False
    assert arb.fairness_feasible(kb)[0] is True


def test_reference_arbiter_fails_closed_on_nonzero_drift():
    from dataclasses import replace
    import pytest
    from cross_psp_two_stream.reference_arbiter import ReferenceArbiterError
    from vessel_crossing.nextgen import EnvironmentalNetworkSourceReservoir
    vp = _profile_class()
    profile = profile_contract_from_mapping(_mapping())
    keys = build_reference_stream_keys(profile, {vp.class_id: vp})
    reservoir, *_ = b6mod._supply(profile, {vp.class_id: vp}, keys)
    rows = list(reservoir.demands)
    rows[0] = replace(rows[0], environmental_drift_velocity=(0.001, 0.0))
    bad = EnvironmentalNetworkSourceReservoir(
        demands=tuple(rows),
        admissions=reservoir.admissions,
        service_order_by_stream=reservoir.service_order_by_stream,
    )
    arb = ReferenceFairSlotArbiter(profile, center_observation_provider=lambda: (), reference_stream_keys_by_stream=keys)
    with pytest.raises(ReferenceArbiterError, match="FAIL_UNSUPPORTED_DOMAIN"):
        arb.prepare_reservoir(bad, _ctx())
