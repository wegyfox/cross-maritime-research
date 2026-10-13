from __future__ import annotations

from dataclasses import replace
from math import isclose

import pytest

from cross_psp_two_stream import (
    CERTIFIED_GRID_SLOT,
    NO_CERTIFIED_SLOT_WITHIN_SEARCH,
    PlannedCrossingMotion,
    SafeConflictSlotSolver,
    SafeSlotNumericalContract,
    SafeSlotTable,
)
from vessel_crossing.nextgen.domain_clearance import certify_pair_domain_clearance
from vessel_crossing.nextgen.ship_domain import ShipDomain
from vessel_crossing.research.r2_b0_b6_reference_controller import (
    derive_r2_free_flow_slot_separation_min,
)


def _motion(
    stream_id: str,
    heading_deg: float,
    *,
    speed: float = 0.10,
    source: float = 1.375,
    exit: float = 1.375,
    domain: ShipDomain | None = None,
    identity: str | None = None,
) -> PlannedCrossingMotion:
    return PlannedCrossingMotion(
        stream_id=stream_id,
        heading_deg=heading_deg,
        speed_nm_per_min=speed,
        source_to_center_distance_nm=source,
        center_to_exit_distance_nm=exit,
        ship_domain=domain or ShipDomain.circular(0.10),
        profile_physics_identity=identity or f"{stream_id}-physics-v1",
    )


def _numerics(
    grid: float = 0.10,
    *,
    max_search: float = 10.0,
    time_tolerance: float = 0.001,
    margin_tolerance: float = 1e-8,
    subdivisions: int = 24,
) -> SafeSlotNumericalContract:
    return SafeSlotNumericalContract(
        separation_grid_min=grid,
        max_search_min=max_search,
        clearance_time_tolerance_min=time_tolerance,
        clearance_margin_tolerance_nm=margin_tolerance,
        max_clearance_subdivisions=subdivisions,
    )


def test_s1_exact_linear_state_construction_heterogeneous_asymmetric_extents():
    east = _motion("east", 90.0, speed=0.20, source=1.2, exit=2.4)
    center = 7.5
    start, end = east.active_interval(center_time_min=center)
    assert start == pytest.approx(1.5)
    assert end == pytest.approx(19.5)
    assert east.state_at(global_time_min=center, center_time_min=center).position == pytest.approx((0.0, 0.0))
    assert east.state_at(global_time_min=start, center_time_min=center).position == pytest.approx((-1.2, 0.0))
    assert east.state_at(global_time_min=end, center_time_min=center).position == pytest.approx((2.4, 0.0))

    north = _motion("north", 0.0, speed=0.12, source=0.6, exit=1.8)
    n_start, n_end = north.active_interval(center_time_min=3.0)
    assert n_start == pytest.approx(-2.0)
    assert n_end == pytest.approx(18.0)
    assert north.state_at(global_time_min=n_start, center_time_min=3.0).position == pytest.approx((0.0, -0.6))
    assert north.state_at(global_time_min=n_end, center_time_min=3.0).position == pytest.approx((0.0, 1.8))


def test_s2_directional_case_reuses_canonical_domain_certificate_exactly():
    predecessor = _motion(
        "north",
        0.0,
        domain=ShipDomain(ahead_nm=0.14, astern_nm=0.08, port_nm=0.10, starboard_nm=0.12),
        source=1.5,
        exit=2.0,
    )
    candidate = _motion(
        "east",
        90.0,
        speed=0.12,
        domain=ShipDomain(ahead_nm=0.12, astern_nm=0.09, port_nm=0.11, starboard_nm=0.08),
        source=1.2,
        exit=1.8,
    )
    solver = SafeConflictSlotSolver(_numerics(grid=0.10, time_tolerance=0.005, subdivisions=16))
    delta = 3.0
    point = solver.evaluate_separation(
        predecessor,
        candidate,
        grid_index=30,
        separation_min=delta,
    )
    assert point.clearance is not None

    pred_start, pred_end = predecessor.active_interval(center_time_min=0.0)
    cand_start, cand_end = candidate.active_interval(center_time_min=delta)
    overlap_start = max(pred_start, cand_start)
    overlap_end = min(pred_end, cand_end)
    duration = overlap_end - overlap_start

    direct = certify_pair_domain_clearance(
        first_state_at=lambda t: predecessor.state_at(global_time_min=overlap_start + t, center_time_min=0.0),
        second_state_at=lambda t: candidate.state_at(global_time_min=overlap_start + t, center_time_min=delta),
        first_domain=predecessor.ship_domain,
        second_domain=candidate.ship_domain,
        duration_min=duration,
        first_speed_bound_nm_per_min=predecessor.speed_nm_per_min,
        second_speed_bound_nm_per_min=candidate.speed_nm_per_min,
        first_turn_rate_bound_deg_per_min=0.0,
        second_turn_rate_bound_deg_per_min=0.0,
        time_tolerance_min=solver.numerical_contract.clearance_time_tolerance_min,
        margin_tolerance_nm=solver.numerical_contract.clearance_margin_tolerance_nm,
        max_subdivisions=solver.numerical_contract.max_clearance_subdivisions,
    )
    assert point.clearance == direct
    assert point.certified_clear == direct.certified_clear


def test_s2_no_overlap_is_explicit_clear_by_construction_not_fake_certificate():
    predecessor = _motion("north", 0.0, source=0.1, exit=0.1)
    candidate = _motion("east", 90.0, source=0.1, exit=0.1)
    point = SafeConflictSlotSolver(_numerics()).evaluate_separation(
        predecessor,
        candidate,
        grid_index=100,
        separation_min=10.0,
    )
    assert point.certified_clear
    assert point.clear_by_nonoverlap
    assert point.clearance is None
    assert point.overlap_start_min is None
    assert point.overlap_end_min is None


def test_s3_grid_search_returns_first_certified_point_and_exhausts_fail_closed():
    predecessor = _motion("north", 0.0)
    candidate = _motion("west", 270.0)
    solver = SafeConflictSlotSolver(_numerics(grid=0.25, max_search=5.0))
    result = solver.solve(predecessor, candidate, use_cache=False)

    assert result.status == CERTIFIED_GRID_SLOT
    assert result.grid_index is not None
    assert result.separation_min == pytest.approx(result.grid_index * 0.25)
    assert result.evaluated_candidate_count == result.grid_index + 1
    assert result.not_certified_candidate_count == result.grid_index
    assert result.certified_candidate_count == 1
    for index in range(result.grid_index):
        earlier = solver.evaluate_separation(
            predecessor,
            candidate,
            grid_index=index,
            separation_min=index * 0.25,
        )
        assert not earlier.certified_clear

    exhausted = SafeConflictSlotSolver(_numerics(grid=0.25, max_search=2.50)).solve(
        predecessor,
        candidate,
        use_cache=False,
    )
    assert exhausted.status == NO_CERTIFIED_SLOT_WITHIN_SEARCH
    assert exhausted.separation_min is None
    assert exhausted.grid_index is None
    assert exhausted.evaluated_candidate_count == 11
    assert exhausted.not_certified_candidate_count == 11


def test_s3_numerical_contract_and_physics_identity_change_execution_cache_key():
    predecessor = _motion("north", 0.0)
    candidate = _motion("west", 270.0)
    base = SafeConflictSlotSolver(_numerics(grid=0.10))
    finer = SafeConflictSlotSolver(_numerics(grid=0.05))
    tighter = SafeConflictSlotSolver(_numerics(grid=0.10, time_tolerance=0.0005))

    base_key = base.cache_key_sha256(predecessor, candidate)
    assert base_key != finer.cache_key_sha256(predecessor, candidate)
    assert base_key != tighter.cache_key_sha256(predecessor, candidate)
    changed_candidate = replace(candidate, profile_physics_identity="west-physics-v2")
    assert base_key != base.cache_key_sha256(predecessor, changed_candidate)


def test_s3_v1_fails_closed_outside_distinct_orthogonal_competing_stream_domain():
    solver = SafeConflictSlotSolver(_numerics())
    north = _motion("north", 0.0)
    same_stream = _motion("north", 90.0, identity="north-other")
    with pytest.raises(ValueError, match="distinct competing streams"):
        solver.solve(north, same_stream, use_cache=False)

    skew = _motion("skew", 45.0)
    with pytest.raises(ValueError, match="orthogonal"):
        solver.solve(north, skew, use_cache=False)


def test_s4_grid_refinement_matrix_converges_at_declared_0005_min_grid():
    cases = (
        (
            _motion("north-circular", 0.0),
            _motion("west-circular", 270.0),
            (2.84, 2.84, 2.835),
        ),
        (
            _motion(
                "north-directional",
                0.0,
                source=1.5,
                exit=2.0,
                domain=ShipDomain(0.14, 0.08, 0.10, 0.12),
            ),
            _motion(
                "east-directional",
                90.0,
                speed=0.12,
                source=1.2,
                exit=1.8,
                domain=ShipDomain(0.12, 0.09, 0.11, 0.08),
            ),
            (2.64, 2.64, 2.64),
        ),
        (
            _motion(
                "north-heterogeneous",
                0.0,
                speed=0.08,
                source=0.8,
                exit=2.2,
                domain=ShipDomain(0.16, 0.07, 0.09, 0.13),
            ),
            _motion(
                "east-heterogeneous",
                90.0,
                speed=0.15,
                source=2.0,
                exit=0.9,
                domain=ShipDomain(0.11, 0.10, 0.15, 0.07),
            ),
            (3.02, 3.02, 3.02),
        ),
    )
    grids = (0.02, 0.01, 0.005)
    for predecessor, candidate, expected in cases:
        separations = []
        for grid in grids:
            result = SafeConflictSlotSolver(
                _numerics(
                    grid=grid,
                    time_tolerance=0.0025,
                    margin_tolerance=1e-6,
                    subdivisions=24,
                )
            ).solve(predecessor, candidate, use_cache=False)
            assert result.certified
            assert result.separation_min is not None
            separations.append(result.separation_min)
        assert separations == pytest.approx(expected)
        assert abs(separations[-1] - separations[-2]) <= grids[1] + 1e-12
        nominal_pair_slot_rates_per_hour = [60.0 / value for value in separations]
        relative_span = (
            max(nominal_pair_slot_rates_per_hour) - min(nominal_pair_slot_rates_per_hour)
        ) / max(nominal_pair_slot_rates_per_hour)
        assert relative_span < 0.002


def test_s5_r2_subset_records_conservative_certificate_delta_without_tuning_it_away():
    domain = ShipDomain.circular(0.10)
    heading_pairs = ((0.0, 270.0), (270.0, 0.0), (0.0, 90.0), (90.0, 0.0))
    for predecessor_heading, candidate_heading in heading_pairs:
        predecessor = _motion("predecessor", predecessor_heading, domain=domain)
        candidate = _motion("candidate", candidate_heading, domain=domain)
        psp = SafeConflictSlotSolver(
            _numerics(
                grid=0.01,
                time_tolerance=0.0025,
                margin_tolerance=1e-6,
                subdivisions=24,
            )
        ).solve(predecessor, candidate, use_cache=False)
        r2 = derive_r2_free_flow_slot_separation_min(
            predecessor_heading_deg=predecessor_heading,
            successor_heading_deg=candidate_heading,
            predecessor_domain=domain,
            successor_domain=domain,
            speed_nm_per_min=0.10,
            time_tolerance_min=0.01,
            max_search_min=10.0,
        )
        assert r2 == pytest.approx(2.83)
        assert psp.separation_min == pytest.approx(2.84)
        assert psp.separation_min >= r2
        assert psp.separation_min - r2 <= 0.01 + 1e-12


def test_s6_cached_and_uncached_evidence_is_byte_identical():
    predecessor = _motion("north", 0.0)
    candidate = _motion("west", 270.0)
    table = SafeSlotTable()
    solver = SafeConflictSlotSolver(_numerics(grid=0.05), table=table)

    uncached = solver.solve(predecessor, candidate, use_cache=False)
    first_cached_path = solver.solve(predecessor, candidate, use_cache=True)
    second_cached_path = solver.solve(predecessor, candidate, use_cache=True)

    assert len(table) == 1
    assert uncached.canonical_bytes() == first_cached_path.canonical_bytes()
    assert first_cached_path.canonical_bytes() == second_cached_path.canonical_bytes()
    assert uncached.evidence_sha256 == second_cached_path.evidence_sha256
