"""Exact finite-chain bound theory for CROSS R2-S2.

This module is CROSS-owned mathematics built on the qualified R2-S1 role-safe
pair semantics.  It is not a recovered Guo 2024 Section-3 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Iterable

from .r2_s1_pair_semantics import AS, HS, SourcePairIntervalMatrix
from .r2_s1_sequence_extrema import (
    SourcePairCounts,
    count_adjacent_source_pairs,
    summarize_finite_sequence,
)

_EPS: Final = 1e-12
_ENDPOINT_ORDER: Final = ("AA", "AH", "HA", "HH")


def _validate_count(value: int, *, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _abcd(matrix: SourcePairIntervalMatrix) -> tuple[float, float, float, float]:
    a = matrix.interval_for_source_roles(follower_type=AS, leader_type=AS)
    b = matrix.interval_for_source_roles(follower_type=AS, leader_type=HS)
    c = matrix.interval_for_source_roles(follower_type=HS, leader_type=AS)
    d = matrix.interval_for_source_roles(follower_type=HS, leader_type=HS)
    return a, b, c, d


@dataclass(frozen=True, slots=True)
class EndpointRunFamily:
    """One feasible endpoint-conditioned affine family H(q)."""

    endpoint_class: str
    n_as: int
    n_hs: int
    q_min: int
    q_max: int
    base_without_q_min: float
    delta_min: float

    def __post_init__(self) -> None:
        if self.endpoint_class not in _ENDPOINT_ORDER:
            raise ValueError("endpoint_class must be one of AA/AH/HA/HH")
        _validate_count(self.n_as, field="n_as")
        _validate_count(self.n_hs, field="n_hs")
        if self.q_min < 1 or self.q_max < self.q_min:
            raise ValueError("invalid q range")
        if not isfinite(self.base_without_q_min) or not isfinite(self.delta_min):
            raise ValueError("affine coefficients must be finite")

    def total_for_q(self, q: int) -> float:
        if not isinstance(q, int) or isinstance(q, bool) or not self.q_min <= q <= self.q_max:
            raise ValueError("q is outside the feasible family range")
        return self.base_without_q_min + q * self.delta_min

    def pair_counts_for_q(self, q: int) -> SourcePairCounts:
        if not isinstance(q, int) or isinstance(q, bool) or not self.q_min <= q <= self.q_max:
            raise ValueError("q is outside the feasible family range")
        m, k = self.n_as, self.n_hs
        if self.endpoint_class == "AA":
            return SourcePairCounts(m - q - 1, q, q, k - q)
        if self.endpoint_class == "HH":
            return SourcePairCounts(m - q, q, q, k - q - 1)
        if self.endpoint_class == "AH":
            return SourcePairCounts(m - q, q - 1, q, k - q)
        return SourcePairCounts(m - q, q, q - 1, k - q)

    def witness_for_q(self, q: int) -> tuple[str, ...]:
        """Construct one physical leader->trailer witness for this family/q."""
        counts = self.pair_counts_for_q(q)
        # The pair-count call above validates q.  Construct by positive run lengths.
        if self.endpoint_class == "AA":
            r_a, r_h, start = q + 1, q, AS
        elif self.endpoint_class == "HH":
            r_a, r_h, start = q, q + 1, HS
        elif self.endpoint_class == "AH":
            r_a = r_h = q
            start = AS
        else:
            r_a = r_h = q
            start = HS

        def lengths(total: int, runs: int) -> list[int]:
            if runs <= 0 or total < runs:
                raise AssertionError("invalid run allocation")
            return [total - runs + 1] + [1] * (runs - 1)

        a_lengths = lengths(self.n_as, r_a)
        h_lengths = lengths(self.n_hs, r_h)
        ia = ih = 0
        out: list[str] = []
        current = start
        while ia < len(a_lengths) or ih < len(h_lengths):
            if current == AS:
                if ia >= len(a_lengths):
                    raise AssertionError("A run allocation exhausted")
                out.extend([AS] * a_lengths[ia])
                ia += 1
                current = HS
            else:
                if ih >= len(h_lengths):
                    raise AssertionError("H run allocation exhausted")
                out.extend([HS] * h_lengths[ih])
                ih += 1
                current = AS
        witness = tuple(out)
        endpoint_code = ("A" if witness[0] == AS else "H") + ("A" if witness[-1] == AS else "H")
        if endpoint_code != self.endpoint_class:
            raise AssertionError("witness endpoint invariant failed")
        if witness.count(AS) != self.n_as or witness.count(HS) != self.n_hs:
            raise AssertionError("witness composition invariant failed")
        if count_adjacent_source_pairs(witness) != counts:
            raise AssertionError("witness pair-count invariant failed")
        return witness


@dataclass(frozen=True, slots=True)
class EndpointExtremumCandidate:
    endpoint_class: str
    q: int
    total_interval_min: float
    mean_adjacent_interval_min: float
    pair_counts: SourcePairCounts
    leader_to_trailer: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClosedFormFiniteExtrema:
    n_as: int
    n_hs: int
    delta_min: float
    min_candidate: EndpointExtremumCandidate
    max_candidate: EndpointExtremumCandidate
    endpoint_families: tuple[EndpointRunFamily, ...]

    @property
    def min_total_interval_min(self) -> float:
        return self.min_candidate.total_interval_min

    @property
    def max_total_interval_min(self) -> float:
        return self.max_candidate.total_interval_min

    @property
    def min_mean_adjacent_interval_min(self) -> float:
        return self.min_candidate.mean_adjacent_interval_min

    @property
    def max_mean_adjacent_interval_min(self) -> float:
        return self.max_candidate.mean_adjacent_interval_min


@dataclass(frozen=True, slots=True)
class FixedSequenceTransposeSensitivity:
    leader_to_trailer: tuple[str, ...]
    mixed_transition_imbalance: int
    mixed_cell_difference_min: float
    total_difference_min: float
    mean_difference_min: float
    absolute_total_bound_min: float
    absolute_mean_bound_min: float


def endpoint_run_families(
    *, n_as: int, n_hs: int, matrix: SourcePairIntervalMatrix
) -> tuple[EndpointRunFamily, ...]:
    """Return all feasible two-type endpoint-conditioned H(q) families."""
    _validate_count(n_as, field="n_as")
    _validate_count(n_hs, field="n_hs")
    if n_as + n_hs < 2:
        raise ValueError("at least two ships are required")
    if n_as == 0 or n_hs == 0:
        return ()

    a, b, c, d = _abcd(matrix)
    delta = (b + c) - (a + d)
    base = a * n_as + d * n_hs
    rows: list[EndpointRunFamily] = []
    if n_as >= 2:
        rows.append(EndpointRunFamily("AA", n_as, n_hs, 1, min(n_hs, n_as - 1), base - a, delta))
    rows.append(EndpointRunFamily("AH", n_as, n_hs, 1, min(n_as, n_hs), base - b, delta))
    rows.append(EndpointRunFamily("HA", n_as, n_hs, 1, min(n_as, n_hs), base - c, delta))
    if n_hs >= 2:
        rows.append(EndpointRunFamily("HH", n_as, n_hs, 1, min(n_as, n_hs - 1), base - d, delta))
    return tuple(rows)


def _candidate(family: EndpointRunFamily, q: int, matrix: SourcePairIntervalMatrix) -> EndpointExtremumCandidate:
    witness = family.witness_for_q(q)
    summary = summarize_finite_sequence(witness, matrix=matrix)
    expected = family.total_for_q(q)
    if abs(summary.total_interval_min - expected) > 1e-9:
        raise AssertionError("closed-form witness total disagrees with S1 pair counter")
    return EndpointExtremumCandidate(
        endpoint_class=family.endpoint_class,
        q=q,
        total_interval_min=expected,
        mean_adjacent_interval_min=expected / (family.n_as + family.n_hs - 1),
        pair_counts=family.pair_counts_for_q(q),
        leader_to_trailer=witness,
    )


def exact_closed_form_extrema(
    *, n_as: int, n_hs: int, matrix: SourcePairIntervalMatrix
) -> ClosedFormFiniteExtrema:
    """Exact finite min/max using the R2-S2 endpoint/run theorem."""
    _validate_count(n_as, field="n_as")
    _validate_count(n_hs, field="n_hs")
    n = n_as + n_hs
    if n < 2:
        raise ValueError("at least two ships are required")
    a, b, c, d = _abcd(matrix)
    delta = (b + c) - (a + d)

    if n_hs == 0:
        seq = (AS,) * n_as
        total = (n_as - 1) * a
        cand = EndpointExtremumCandidate("AA", 0, total, total / (n - 1), SourcePairCounts(n_as - 1, 0, 0, 0), seq)
        return ClosedFormFiniteExtrema(n_as, n_hs, delta, cand, cand, ())
    if n_as == 0:
        seq = (HS,) * n_hs
        total = (n_hs - 1) * d
        cand = EndpointExtremumCandidate("HH", 0, total, total / (n - 1), SourcePairCounts(0, 0, 0, n_hs - 1), seq)
        return ClosedFormFiniteExtrema(n_as, n_hs, delta, cand, cand, ())

    families = endpoint_run_families(n_as=n_as, n_hs=n_hs, matrix=matrix)
    min_candidates: list[EndpointExtremumCandidate] = []
    max_candidates: list[EndpointExtremumCandidate] = []
    for family in families:
        if delta > _EPS:
            q_for_min, q_for_max = family.q_min, family.q_max
        elif delta < -_EPS:
            q_for_min, q_for_max = family.q_max, family.q_min
        else:
            q_for_min = q_for_max = family.q_min
        min_candidates.append(_candidate(family, q_for_min, matrix))
        max_candidates.append(_candidate(family, q_for_max, matrix))

    rank = {name: i for i, name in enumerate(_ENDPOINT_ORDER)}
    min_cand = min(min_candidates, key=lambda x: (x.total_interval_min, rank[x.endpoint_class], x.q))
    max_cand = max(max_candidates, key=lambda x: (x.total_interval_min, -rank[x.endpoint_class], -x.q))
    return ClosedFormFiniteExtrema(n_as, n_hs, delta, min_cand, max_cand, families)


def mixed_transition_imbalance(leader_to_trailer: Iterable[str]) -> int:
    """Return N_AH - N_HA in source follower->leader notation."""
    seq = tuple(leader_to_trailer)
    counts = count_adjacent_source_pairs(seq)
    imbalance = counts.as_follower_hs_leader - counts.hs_follower_as_leader
    if imbalance not in {-1, 0, 1}:
        raise AssertionError("binary-chain transition imbalance invariant failed")
    endpoint_expected = 0
    if seq[0] == HS and seq[-1] == AS:
        endpoint_expected = 1
    elif seq[0] == AS and seq[-1] == HS:
        endpoint_expected = -1
    if imbalance != endpoint_expected:
        raise AssertionError("transition imbalance is inconsistent with endpoints")
    return imbalance


def fixed_sequence_transpose_sensitivity(
    leader_to_trailer: Iterable[str], *, matrix: SourcePairIntervalMatrix
) -> FixedSequenceTransposeSensitivity:
    """Exact effect of swapping only the two mixed directed cells for one sequence.

    The returned difference is H(original) - H(mixed-cell-transposed).
    """
    seq = tuple(leader_to_trailer)
    if len(seq) < 2:
        raise ValueError("leader_to_trailer must contain at least two ships")
    _, b, c, _ = _abcd(matrix)
    imbalance = mixed_transition_imbalance(seq)
    cell_difference = b - c
    total_difference = cell_difference * imbalance
    mean_difference = total_difference / (len(seq) - 1)
    return FixedSequenceTransposeSensitivity(
        leader_to_trailer=seq,
        mixed_transition_imbalance=imbalance,
        mixed_cell_difference_min=cell_difference,
        total_difference_min=total_difference,
        mean_difference_min=mean_difference,
        absolute_total_bound_min=abs(cell_difference),
        absolute_mean_bound_min=abs(cell_difference) / (len(seq) - 1),
    )
