from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from vessel_crossing.nextgen import (
    CrossTrackRecovery,
    EnvironmentalNetworkSourceDemand,
    EnvironmentalNetworkSourceReservoir,
    EnvironmentalRouteHandoffTarget,
    EnvironmentalWorldSnapshot,
    ManeuverLimits,
    Route2D,
    RouteEndAction,
    RouteFollower,
    RouteNavigator,
    ShipDomain,
    VesselClassProfile,
    VesselDimensions,
    WaterRelativeVesselState,
    resolve_environmental_network_source_reservoir,
)
from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
    EnvironmentalNetworkSourceControllerContext,
)

from cross_psp_two_stream import (
    B4EarliestSafeSlotController,
    B5BoundedSlotCostController,
    profile_contract_from_mapping,
)

ROOT = Path(__file__).resolve().parents[1]


def _profile(*, lookahead=3, max_bypass=1, max_extra_wait=5.0):
    data = yaml.safe_load((ROOT / "examples" / "profile.yaml").read_text())
    data["b5_fairness"]["lookahead_depth"] = lookahead
    data["b5_fairness"]["max_bypass_per_demand"] = max_bypass
    data["b5_fairness"]["max_extra_wait_min"] = max_extra_wait
    return profile_contract_from_mapping(data)


def _limits():
    return ManeuverLimits.from_knots_per_minute(
        max_acceleration_knots_per_min=2.0,
        max_deceleration_knots_per_min=2.0,
        max_turn_rate_deg_per_min=30.0,
    )


def _demand(vessel_id: str, *, stream_id: str, demand_time: float, speed: float):
    room = 1.5
    if stream_id == "alpha":
        start, end, heading = (0.0, -room), (0.0, room), 0.0
    else:
        start, end, heading = (room, 0.0), (-room, 0.0), 270.0
    route = Route2D((start, end))
    profile = VesselClassProfile(
        class_id=f"class-{speed}",
        dimensions=VesselDimensions.from_meters(length_m=120.0, beam_m=20.0),
        maneuver_limits=_limits(),
        safety_domain=ShipDomain.circular(0.10),
    )
    return EnvironmentalNetworkSourceDemand(
        stream_id=stream_id,
        nominal_vessel_id=vessel_id,
        demand_time=demand_time,
        vessel=WaterRelativeVesselState(vessel_id, start, heading, speed),
        route_target=EnvironmentalRouteHandoffTarget(
            vessel_id=vessel_id,
            navigator=RouteNavigator(RouteFollower(route, speed), CrossTrackRecovery(route)),
            end_action=RouteEndAction.CONTINUE,
        ),
        limits=profile.maneuver_limits,
        environmental_drift_velocity=(0.0, 0.0),
        profile=profile,
    )


def _context(at: float, vessels=()):
    return EnvironmentalNetworkSourceControllerContext(
        snapshot=EnvironmentalWorldSnapshot(time=at, vessels=tuple(vessels)),
        active_profiles={}, active_route_targets={}, active_limits={},
        active_environmental_drifts={}, active_commands={},
        active_motion_backend=None, active_motion_states={},
        same_route_dynamic_ingress=None, future_ingress_clearance=None,
    )


def _controller(**kwargs):
    return B5BoundedSlotCostController(
        _profile(**kwargs), center_observation_provider=lambda: ()
    )


def _reservoir(alpha_speeds, *, beta_time=100.0, beta_speed=0.1):
    demands = tuple(
        _demand(f"a{i}", stream_id="alpha", demand_time=0.0, speed=speed)
        for i, speed in enumerate(alpha_speeds)
    ) + (_demand("b0", stream_id="beta", demand_time=beta_time, speed=beta_speed),)
    return EnvironmentalNetworkSourceReservoir(demands)


def test_q3_help_candidate_reorders_and_exact_b4_reserves_staged_head():
    reservoir = _reservoir([0.05, 0.10])
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert dict(prepared.service_order_by_stream)["alpha"][:2] == ("a1", "a0")
    assert ctl.reservation is not None and ctl.reservation.vessel_id == "a1"
    assert ctl.reservation.center_target_min == pytest.approx(15.0)
    assert ctl.bypass_count["a0"] == 1
    assert any(x.classification == "HELP" for x in ctl.last_candidate_evaluations)
    assert any(x.event == "B5_SLOT_COST_REORDER" for x in ctl.trace)


def test_q3_null_candidate_is_measured_but_not_reordered():
    reservoir = _reservoir([0.10, 0.10])
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert prepared.service_order_by_stream == ()
    assert ctl.reservation is not None and ctl.reservation.vessel_id == "a0"
    assert [x.classification for x in ctl.last_candidate_evaluations] == ["NULL"]
    assert not any(x.event == "B5_SLOT_COST_REORDER" for x in ctl.trace)


def test_q3_hurt_candidate_is_measured_but_rejected():
    reservoir = _reservoir([0.10, 0.05])
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert prepared.service_order_by_stream == ()
    assert ctl.reservation is not None and ctl.reservation.vessel_id == "a0"
    assert [x.classification for x in ctl.last_candidate_evaluations] == ["HURT"]
    assert not any(x.event == "B5_SLOT_COST_REORDER" for x in ctl.trace)


def test_q3_null_case_is_output_equivalent_to_exact_b4_controller():
    reservoir = _reservoir([0.10, 0.10])
    b4 = B4EarliestSafeSlotController(_profile(), center_observation_provider=lambda: ())
    b5 = _controller()
    p4 = b4.prepare_reservoir(reservoir, _context(0.0))
    p5 = b5.prepare_reservoir(reservoir, _context(0.0))
    assert p5 == p4
    assert b5.reservation is not None and b4.reservation is not None
    assert (b5.reservation.vessel_id, b5.reservation.center_target_min, b5.reservation.release_target_min) == (
        b4.reservation.vessel_id, b4.reservation.center_target_min, b4.reservation.release_target_min
    )


def test_q3_lookahead_bound_excludes_better_candidate_outside_prefix():
    reservoir = _reservoir([0.05, 0.05, 0.10])
    ctl = _controller(lookahead=2)
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert prepared.service_order_by_stream == ()
    assert ctl.reservation is not None and ctl.reservation.vessel_id == "a0"
    # a2 would help but is outside depth=2; only a1 is evaluated and null.
    assert [x.candidate_vessel_id for x in ctl.last_candidate_evaluations] == ["a1"]
    assert ctl.last_candidate_evaluations[0].classification == "NULL"


def test_q3_max_bypass_and_bypass_deadline_force_fairness_promotion():
    reservoir = _reservoir([0.05, 0.10])
    ctl = _controller(max_bypass=1, max_extra_wait=5.0)
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert dict(prepared.service_order_by_stream)["alpha"][:2] == ("a1", "a0")

    blocker = WaterRelativeVesselState("blocker", (50.0, 50.0), 0.0, 0.1)
    denied = resolve_environmental_network_source_reservoir(
        prepared, _context(0.0, (blocker,)).snapshot, {}
    )
    assert not denied.admissions
    ctl.observe_resolution(denied, _context(0.0, (blocker,)))
    assert ctl.reservation is None

    prepared5 = ctl.prepare_reservoir(denied.reservoir, _context(5.0))
    assert dict(prepared5.service_order_by_stream)["alpha"][:2] == ("a0", "a1")
    assert ctl.reservation is not None and ctl.reservation.vessel_id == "a0"
    assert ctl.bypass_count["a0"] == 1
    assert any(x.event == "B5_FAIRNESS_PROMOTION" and x.vessel_id == "a0" for x in ctl.trace)


def test_q3_b5_changes_only_service_order_and_b4_not_before_evidence():
    reservoir = _reservoir([0.05, 0.10])
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert prepared.admissions == reservoir.admissions
    assert prepared.service_order_by_stream != reservoir.service_order_by_stream
    for before, after in zip(reservoir.demands, prepared.demands, strict=True):
        assert replace(after, release_not_before_time=before.release_not_before_time) == before


def test_q3_deterministic_replay_including_reorder_and_b4_reservation():
    reservoir = _reservoir([0.05, 0.10])
    outputs = []
    for _ in range(2):
        ctl = _controller()
        prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
        outputs.append((prepared, ctl.reservation, tuple(ctl.trace), dict(ctl.bypass_count)))
    assert outputs[0] == outputs[1]


def test_q3_frozen_d0_guard_accepts_b5_service_order_and_canonical_gate_admits_selected_head():
    from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
        _apply_source_admission_controller,
    )
    reservoir = _reservoir([0.05, 0.10])
    ctl = _controller()
    prepared = _apply_source_admission_controller(ctl, reservoir, _context(0.0))
    assert dict(prepared.service_order_by_stream)["alpha"][:2] == ("a1", "a0")
    resolved = resolve_environmental_network_source_reservoir(prepared, _context(0.0).snapshot, {})
    assert [x.vessel_id for x in resolved.admissions] == ["a1"]
    ctl.observe_resolution(resolved, _context(0.0))
    assert any(x.event == "B4_CANONICAL_ADMISSION_COMMITTED" and x.vessel_id == "a1" for x in ctl.trace)
