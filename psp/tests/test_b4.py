from __future__ import annotations

from copy import deepcopy
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
    B4CenterPassObservation,
    B4EarliestSafeSlotController,
    B4PolicyError,
    profile_contract_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


def _mapping(*, room: float = 1.5):
    data = yaml.safe_load((ROOT / "examples" / "profile.yaml").read_text())
    for stream in data["streams"]:
        stream["source_to_center_distance_nm"] = room
        stream["center_to_exit_distance_nm"] = room
    return data


def _profile(*, room: float = 1.5):
    return profile_contract_from_mapping(_mapping(room=room))


def _limits() -> ManeuverLimits:
    return ManeuverLimits.from_knots_per_minute(
        max_acceleration_knots_per_min=2.0,
        max_deceleration_knots_per_min=2.0,
        max_turn_rate_deg_per_min=30.0,
    )


def _vessel_profile(class_id: str = "p0") -> VesselClassProfile:
    return VesselClassProfile(
        class_id=class_id,
        dimensions=VesselDimensions.from_meters(length_m=120.0, beam_m=20.0),
        maneuver_limits=_limits(),
        safety_domain=ShipDomain.circular(0.10),
    )


def _demand(vessel_id: str, *, stream_id: str, demand_time: float, room: float = 1.5,
            drift=(0.0, 0.0), heading: float | None = None,
            release_not_before_time: float | None = None) -> EnvironmentalNetworkSourceDemand:
    if stream_id == "alpha":
        h = 0.0 if heading is None else heading
        start = (0.0, -room)
        route = Route2D((start, (0.0, room)))
    else:
        h = 270.0 if heading is None else heading
        start = (room, 0.0)
        route = Route2D((start, (-room, 0.0)))
    vp = _vessel_profile()
    return EnvironmentalNetworkSourceDemand(
        stream_id=stream_id,
        nominal_vessel_id=vessel_id,
        demand_time=demand_time,
        vessel=WaterRelativeVesselState(vessel_id, start, h, 0.10),
        route_target=EnvironmentalRouteHandoffTarget(
            vessel_id=vessel_id,
            navigator=RouteNavigator(RouteFollower(route, 0.10), CrossTrackRecovery(route)),
            end_action=RouteEndAction.CONTINUE,
        ),
        limits=vp.maneuver_limits,
        environmental_drift_velocity=drift,
        profile=vp,
        release_not_before_time=release_not_before_time,
    )


def _context(at: float, vessels=()) -> EnvironmentalNetworkSourceControllerContext:
    return EnvironmentalNetworkSourceControllerContext(
        snapshot=EnvironmentalWorldSnapshot(time=at, vessels=tuple(vessels)),
        active_profiles={},
        active_route_targets={},
        active_limits={},
        active_environmental_drifts={},
        active_commands={},
        active_motion_backend=None,
        active_motion_states={},
        same_route_dynamic_ingress=None,
        future_ingress_clearance=None,
    )


def _controller(*, room=1.5, observations=None):
    rows = [] if observations is None else observations
    return B4EarliestSafeSlotController(_profile(room=room), center_observation_provider=lambda: tuple(rows))


def test_q2_noop_equivalence_when_b4_does_not_bind():
    a = _demand("a0", stream_id="alpha", demand_time=0.0)
    b = _demand("b0", stream_id="beta", demand_time=100.0)
    reservoir = EnvironmentalNetworkSourceReservoir((a, b))
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert prepared == reservoir
    assert ctl.reservation is not None
    assert ctl.reservation.vessel_id == "a0"


def test_q2_deterministic_tie_break_and_temporary_competing_head_hold():
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0),
        _demand("b0", stream_id="beta", demand_time=0.0),
    ))
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert ctl.reservation is not None and ctl.reservation.vessel_id == "a0"
    a, b = prepared.demands
    assert a.release_not_before_time is None
    assert b.release_not_before_time == pytest.approx(15.0)


def test_q2_canonical_admission_commits_anchor_from_actual_admission_time_and_solver_binds_next():
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0),
        _demand("b0", stream_id="beta", demand_time=0.0),
    ))
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    # Resolve later than the reservation creation time. The committed center anchor must
    # use actual canonical source admission time 1.0 + 15 min, not stale target 15.0.
    result = resolve_environmental_network_source_reservoir(prepared, _context(1.0).snapshot, {})
    assert [x.vessel_id for x in result.admissions] == ["a0"]
    ctl.observe_resolution(result, _context(1.0))
    commit = next(x for x in ctl.trace if x.event == "B4_CANONICAL_ADMISSION_COMMITTED")
    assert commit.center_target_min == pytest.approx(16.0)

    ctl.prepare_reservoir(result.reservoir, _context(1.0))
    reservation = ctl.reservation
    assert reservation is not None and reservation.vessel_id == "b0"
    assert reservation.solver_certificate is not None
    assert reservation.solver_certificate.certified
    assert reservation.center_target_min == pytest.approx(
        16.0 + reservation.solver_certificate.separation_min
    )


def test_q2_actual_center_observation_reanchors_future_reservation():
    room = 0.05
    observations: list[B4CenterPassObservation] = []
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0, room=room),
        _demand("b0", stream_id="beta", demand_time=2.0, room=room),
    ))
    ctl = _controller(room=room, observations=observations)
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    result = resolve_environmental_network_source_reservoir(prepared, _context(1.0).snapshot, {})
    ctl.observe_resolution(result, _context(1.0))
    observations.append(B4CenterPassObservation("a0", "alpha", 2.0, 0.0, 0.10))

    ctl.prepare_reservoir(result.reservoir, _context(2.0))
    reservation = ctl.reservation
    assert reservation is not None and reservation.vessel_id == "b0"
    assert reservation.solver_certificate is not None
    assert reservation.center_target_min == pytest.approx(
        2.0 + reservation.solver_certificate.separation_min
    )
    assert any(x.event == "B4_ACTUAL_CENTER_REANCHOR" for x in ctl.trace)


def test_q2_canonical_denial_clears_reservation_for_replan():
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0),
        _demand("b0", stream_id="beta", demand_time=100.0),
    ))
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    blocker = WaterRelativeVesselState("blocker", (50.0, 50.0), 0.0, 0.1)
    result = resolve_environmental_network_source_reservoir(
        prepared, _context(0.0, (blocker,)).snapshot, {}
    )
    assert not result.admissions
    assert result.decisions[0].status.value == "BLOCKED_MISSING_ACTIVE_DOMAIN"
    ctl.observe_resolution(result, _context(0.0, (blocker,)))
    assert ctl.reservation is None
    assert any(x.event == "B4_SOURCE_DENIAL_REPLAN" and x.detail == "BLOCKED_MISSING_ACTIVE_DOMAIN" for x in ctl.trace)

    # The next prepare creates a fresh reservation rather than preserving the denied one.
    ctl.prepare_reservoir(result.reservoir, _context(0.0))
    assert ctl.reservation is not None
    assert sum(x.event == "B4_SLOT_RESERVED" for x in ctl.trace) == 2


def test_q2_b4_mutates_only_not_before_evidence_never_service_order_or_physics():
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0),
        _demand("b0", stream_id="beta", demand_time=0.0),
    ), service_order_by_stream=(("alpha", ("a0",)), ("beta", ("b0",))))
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    assert prepared.admissions == reservoir.admissions
    assert prepared.service_order_by_stream == reservoir.service_order_by_stream
    for before, after in zip(reservoir.demands, prepared.demands, strict=True):
        assert replace(after, release_not_before_time=before.release_not_before_time) == before


def test_q2_fail_closed_outside_zero_drift_and_orthogonal_fixed_stream_domain():
    ctl = _controller()
    bad_drift = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0, drift=(0.001, 0.0)),
        _demand("b0", stream_id="beta", demand_time=0.0),
    ))
    with pytest.raises(B4PolicyError, match="zero environmental drift"):
        ctl.prepare_reservoir(bad_drift, _context(0.0))

    ctl = _controller()
    bad_angle = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0),
        _demand("b0", stream_id="beta", demand_time=0.0, heading=260.0),
    ))
    with pytest.raises(B4PolicyError, match="orthogonal"):
        ctl.prepare_reservoir(bad_angle, _context(0.0))


def test_q2_center_observation_requires_prior_canonical_admission_and_matching_motion():
    observations = [B4CenterPassObservation("a0", "alpha", 0.0, 0.0, 0.10)]
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0),
        _demand("b0", stream_id="beta", demand_time=100.0),
    ))
    ctl = _controller(observations=observations)
    with pytest.raises(B4PolicyError, match="prior canonical source admission"):
        ctl.prepare_reservoir(reservoir, _context(0.0))


def test_q2_deterministic_replay_produces_identical_gates_and_trace():
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0),
        _demand("b0", stream_id="beta", demand_time=0.0),
    ))
    outputs = []
    for _ in range(2):
        ctl = _controller()
        prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
        resolved = resolve_environmental_network_source_reservoir(prepared, _context(0.0).snapshot, {})
        ctl.observe_resolution(resolved, _context(0.0))
        prepared_next = ctl.prepare_reservoir(resolved.reservoir, _context(0.0))
        outputs.append((prepared, resolved, prepared_next, tuple(ctl.trace)))
    assert outputs[0] == outputs[1]


def test_q2_existing_not_before_floor_is_never_moved_earlier():
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0, release_not_before_time=5.0),
        _demand("b0", stream_id="beta", demand_time=100.0),
    ))
    ctl = _controller()
    prepared = ctl.prepare_reservoir(reservoir, _context(0.0))
    a0 = next(x for x in prepared.demands if x.vessel_id == "a0")
    assert a0.release_not_before_time == pytest.approx(5.0)


def test_q2_frozen_d0_controller_guard_accepts_b4_surface():
    from vessel_crossing.nextgen.environmental_network_source_fifo_runtime import (
        _apply_source_admission_controller,
    )
    reservoir = EnvironmentalNetworkSourceReservoir((
        _demand("a0", stream_id="alpha", demand_time=0.0),
        _demand("b0", stream_id="beta", demand_time=0.0),
    ))
    ctl = _controller()
    prepared = _apply_source_admission_controller(ctl, reservoir, _context(0.0))
    assert isinstance(prepared, EnvironmentalNetworkSourceReservoir)
    assert prepared.admissions == reservoir.admissions
