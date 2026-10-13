"""Finite adjacent-pair combinatorics and large-N CROSS derivations for R2-S1."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from .r2_s1_pair_semantics import AS, HS, SourcePairIntervalMatrix

_EPS = 1e-12


def _validate_type(value: str) -> None:
    if value not in {AS, HS}:
        raise ValueError("sequence values must be AS or HS")


@dataclass(frozen=True, slots=True)
class SourcePairCounts:
    as_follower_as_leader: int = 0
    as_follower_hs_leader: int = 0
    hs_follower_as_leader: int = 0
    hs_follower_hs_leader: int = 0

    def __post_init__(self) -> None:
        values = (
            self.as_follower_as_leader,
            self.as_follower_hs_leader,
            self.hs_follower_as_leader,
            self.hs_follower_hs_leader,
        )
        if any((not isinstance(x, int) or isinstance(x, bool) or x < 0) for x in values):
            raise ValueError("pair counts must be non-negative integers")

    @property
    def total(self) -> int:
        return (
            self.as_follower_as_leader
            + self.as_follower_hs_leader
            + self.hs_follower_as_leader
            + self.hs_follower_hs_leader
        )

    def count(self, *, follower_type: str, leader_type: str) -> int:
        _validate_type(follower_type)
        _validate_type(leader_type)
        if follower_type == AS and leader_type == AS:
            return self.as_follower_as_leader
        if follower_type == AS and leader_type == HS:
            return self.as_follower_hs_leader
        if follower_type == HS and leader_type == AS:
            return self.hs_follower_as_leader
        return self.hs_follower_hs_leader


@dataclass(frozen=True, slots=True)
class FiniteSequenceIntervalSummary:
    leader_to_trailer: tuple[str, ...]
    pair_counts: SourcePairCounts
    total_interval_min: float
    mean_adjacent_interval_min: float


@dataclass(frozen=True, slots=True)
class FiniteExtremalIntervalSummary:
    n_as: int
    n_hs: int
    min_total_interval_min: float
    max_total_interval_min: float
    min_mean_adjacent_interval_min: float
    max_mean_adjacent_interval_min: float
    minimizing_leader_to_trailer: tuple[str, ...]
    maximizing_leader_to_trailer: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LargeNExtremalIntervalSummary:
    p_as: float
    delta_min: float
    clustered_mean_interval_min: float
    max_cross_mean_interval_min: float
    min_mean_interval_min: float
    max_mean_interval_min: float
    min_arrangement_family: str
    max_arrangement_family: str


def count_adjacent_source_pairs(leader_to_trailer: Sequence[str]) -> SourcePairCounts:
    sequence = tuple(leader_to_trailer)
    if len(sequence) < 2:
        raise ValueError("leader_to_trailer must contain at least two ships")
    for value in sequence:
        _validate_type(value)
    counts = {(AS, AS): 0, (AS, HS): 0, (HS, AS): 0, (HS, HS): 0}
    for leader, follower in zip(sequence, sequence[1:]):
        counts[(follower, leader)] += 1
    result = SourcePairCounts(
        as_follower_as_leader=counts[(AS, AS)],
        as_follower_hs_leader=counts[(AS, HS)],
        hs_follower_as_leader=counts[(HS, AS)],
        hs_follower_hs_leader=counts[(HS, HS)],
    )
    if result.total != len(sequence) - 1:
        raise AssertionError("adjacent-pair count invariant failed")
    return result


def summarize_finite_sequence(
    leader_to_trailer: Sequence[str],
    *,
    matrix: SourcePairIntervalMatrix,
) -> FiniteSequenceIntervalSummary:
    sequence = tuple(leader_to_trailer)
    counts = count_adjacent_source_pairs(sequence)
    total = 0.0
    for follower in (AS, HS):
        for leader in (AS, HS):
            total += counts.count(follower_type=follower, leader_type=leader) * matrix.interval_for_source_roles(
                follower_type=follower, leader_type=leader
            )
    return FiniteSequenceIntervalSummary(
        leader_to_trailer=sequence,
        pair_counts=counts,
        total_interval_min=total,
        mean_adjacent_interval_min=total / (len(sequence) - 1),
    )


def _reconstruct_sequence(
    state: tuple[int, int, str],
    previous: dict[tuple[int, int, str], str | None],
) -> tuple[str, ...]:
    a, h, last = state
    reversed_sequence = [last]
    while a + h > 1:
        prev = previous[state]
        if prev is None:
            raise AssertionError("missing DP predecessor")
        if last == AS:
            a -= 1
        else:
            h -= 1
        state = (a, h, prev)
        last = prev
        reversed_sequence.append(last)
    reversed_sequence.reverse()
    return tuple(reversed_sequence)


def exact_finite_extrema(
    *,
    n_as: int,
    n_hs: int,
    matrix: SourcePairIntervalMatrix,
) -> FiniteExtremalIntervalSummary:
    """Compute exact finite-chain min/max adjacent interval sums for fixed type counts."""
    if not isinstance(n_as, int) or isinstance(n_as, bool) or n_as < 0:
        raise ValueError("n_as must be a non-negative integer")
    if not isinstance(n_hs, int) or isinstance(n_hs, bool) or n_hs < 0:
        raise ValueError("n_hs must be a non-negative integer")
    total_ships = n_as + n_hs
    if total_ships < 2:
        raise ValueError("at least two ships are required")

    min_cost: dict[tuple[int, int, str], float] = {}
    max_cost: dict[tuple[int, int, str], float] = {}
    min_prev: dict[tuple[int, int, str], str | None] = {}
    max_prev: dict[tuple[int, int, str], str | None] = {}
    if n_as:
        state = (1, 0, AS)
        min_cost[state] = max_cost[state] = 0.0
        min_prev[state] = max_prev[state] = None
    if n_hs:
        state = (0, 1, HS)
        min_cost[state] = max_cost[state] = 0.0
        min_prev[state] = max_prev[state] = None

    for used in range(1, total_ships):
        states = sorted(
            (state for state in min_cost if state[0] + state[1] == used),
            key=lambda x: (x[0], x[1], x[2]),
        )
        for a, h, last in states:
            for follower in (AS, HS):
                if follower == AS and a >= n_as:
                    continue
                if follower == HS and h >= n_hs:
                    continue
                na = a + (follower == AS)
                nh = h + (follower == HS)
                nxt = (int(na), int(nh), follower)
                step = matrix.interval_for_source_roles(follower_type=follower, leader_type=last)
                candidate_min = min_cost[(a, h, last)] + step
                candidate_max = max_cost[(a, h, last)] + step
                if nxt not in min_cost or candidate_min < min_cost[nxt] - _EPS:
                    min_cost[nxt] = candidate_min
                    min_prev[nxt] = last
                if nxt not in max_cost or candidate_max > max_cost[nxt] + _EPS:
                    max_cost[nxt] = candidate_max
                    max_prev[nxt] = last

    final_states = [(n_as, n_hs, last) for last in (AS, HS) if (n_as, n_hs, last) in min_cost]
    min_state = min(final_states, key=lambda s: (min_cost[s], s[2]))
    max_state = max(final_states, key=lambda s: (max_cost[s], -ord(s[2][0])))
    min_total = min_cost[min_state]
    max_total = max_cost[max_state]
    denominator = total_ships - 1
    return FiniteExtremalIntervalSummary(
        n_as=n_as,
        n_hs=n_hs,
        min_total_interval_min=min_total,
        max_total_interval_min=max_total,
        min_mean_adjacent_interval_min=min_total / denominator,
        max_mean_adjacent_interval_min=max_total / denominator,
        minimizing_leader_to_trailer=_reconstruct_sequence(min_state, min_prev),
        maximizing_leader_to_trailer=_reconstruct_sequence(max_state, max_prev),
    )


def large_n_extrema(
    p_as: float,
    *,
    matrix: SourcePairIntervalMatrix,
) -> LargeNExtremalIntervalSummary:
    """Return CROSS-derived asymptotic clustering/max-cross candidate extrema."""
    if not isfinite(p_as) or not 0.0 <= p_as <= 1.0:
        raise ValueError("p_as must be finite and within [0, 1]")
    a = matrix.interval_for_source_roles(follower_type=AS, leader_type=AS)
    b = matrix.interval_for_source_roles(follower_type=AS, leader_type=HS)
    c = matrix.interval_for_source_roles(follower_type=HS, leader_type=AS)
    d = matrix.interval_for_source_roles(follower_type=HS, leader_type=HS)
    delta = (b + c) - (a + d)
    clustered = p_as * a + (1.0 - p_as) * d
    if p_as <= 0.5:
        max_cross = (1.0 - 2.0 * p_as) * d + p_as * (b + c)
    else:
        max_cross = (2.0 * p_as - 1.0) * a + (1.0 - p_as) * (b + c)
    if p_as <= _EPS or p_as >= 1.0 - _EPS:
        min_mean = max_mean = clustered
        min_family = max_family = "SINGLE_TYPE_DEGENERATE"
    elif delta > _EPS:
        min_mean, max_mean = clustered, max_cross
        min_family, max_family = "CLUSTERED", "MAX_CROSS"
    elif delta < -_EPS:
        min_mean, max_mean = max_cross, clustered
        min_family, max_family = "MAX_CROSS", "CLUSTERED"
    else:
        min_mean = max_mean = clustered
        min_family = max_family = "RUN_COUNT_NEUTRAL"
    return LargeNExtremalIntervalSummary(
        p_as=p_as,
        delta_min=delta,
        clustered_mean_interval_min=clustered,
        max_cross_mean_interval_min=max_cross,
        min_mean_interval_min=min_mean,
        max_mean_interval_min=max_mean,
        min_arrangement_family=min_family,
        max_arrangement_family=max_family,
    )
