"""Exact random-ordering theory for CROSS R2-S3.

This module is CROSS-owned probability/combinatorics built on the qualified
R2-S2 finite endpoint/run theorem.  It is not a recovered Guo 2024 Section-3
implementation and does not authorize physical simulator campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, isfinite
from typing import Final, Iterable

from .r2_s1_pair_semantics import AS, HS, SourcePairIntervalMatrix
from .r2_s2_finite_bounds import EndpointRunFamily, endpoint_run_families, exact_closed_form_extrema

_EPS: Final = 1e-12
_MINUTES_PER_DAY: Final = 1440.0
_SUPPORT_DIGITS: Final = 12


def _validate_count(value: int, *, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")


def _validate_probability(value: float, *, field: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")


def _abcd(matrix: SourcePairIntervalMatrix) -> tuple[float, float, float, float]:
    a = matrix.interval_for_source_roles(follower_type=AS, leader_type=AS)
    b = matrix.interval_for_source_roles(follower_type=AS, leader_type=HS)
    c = matrix.interval_for_source_roles(follower_type=HS, leader_type=AS)
    d = matrix.interval_for_source_roles(follower_type=HS, leader_type=HS)
    return a, b, c, d


def _physical_weight(matrix: SourcePairIntervalMatrix, *, leader: str, follower: str) -> float:
    return matrix.interval_for_source_roles(follower_type=follower, leader_type=leader)


def _falling(value: int, order: int) -> int:
    if order < 0:
        raise ValueError("order must be non-negative")
    if order > value:
        return 0
    out = 1
    for x in range(order):
        out *= value - x
    return out


def _ordered_tuple_probability(pattern: tuple[str, ...], *, n_as: int, n_hs: int) -> float:
    r = len(pattern)
    n = n_as + n_hs
    if r > n:
        return 0.0
    need_as = pattern.count(AS)
    need_hs = r - need_as
    numerator = _falling(n_as, need_as) * _falling(n_hs, need_hs)
    denominator = _falling(n, r)
    return numerator / denominator if denominator else 0.0


def _iid_tuple_probability(pattern: tuple[str, ...], *, p_as: float) -> float:
    n_as = pattern.count(AS)
    return (p_as ** n_as) * ((1.0 - p_as) ** (len(pattern) - n_as))


def _support_key(value: float) -> float:
    return round(value, _SUPPORT_DIGITS)


@dataclass(frozen=True, slots=True)
class ExpectedPairCounts:
    aa: float
    ah: float
    ha: float
    hh: float

    @property
    def total(self) -> float:
        return self.aa + self.ah + self.ha + self.hh


@dataclass(frozen=True, slots=True)
class IntervalMomentSummary:
    expected_total_interval_min: float
    variance_total_interval_min2: float
    expected_mean_adjacent_interval_min: float
    variance_mean_adjacent_interval_min2: float


@dataclass(frozen=True, slots=True)
class CapacityMomentSummary:
    expected_capacity_ships_per_day: float
    capacity_at_expected_mean_ships_per_day: float
    jensen_gap_ships_per_day: float


@dataclass(frozen=True, slots=True)
class EndpointQProbabilityMass:
    endpoint_class: str
    q: int
    sequence_count: int
    probability: float
    total_interval_min: float
    mean_adjacent_interval_min: float
    capacity_ships_per_day: float


@dataclass(frozen=True, slots=True)
class IntervalSupportMass:
    total_interval_min: float
    sequence_count: int
    probability: float
    mean_adjacent_interval_min: float
    capacity_ships_per_day: float


@dataclass(frozen=True, slots=True)
class FixedCompositionDistribution:
    n_as: int
    n_hs: int
    total_sequences: int
    endpoint_q_masses: tuple[EndpointQProbabilityMass, ...]
    support: tuple[IntervalSupportMass, ...]
    interval_moments: IntervalMomentSummary
    capacity_moments: CapacityMomentSummary

    @property
    def n(self) -> int:
        return self.n_as + self.n_hs


@dataclass(frozen=True, slots=True)
class BernoulliCompositionMass:
    n_as: int
    probability: float


@dataclass(frozen=True, slots=True)
class BernoulliSupportMass:
    total_interval_min: float
    probability: float
    mean_adjacent_interval_min: float
    capacity_ships_per_day: float


@dataclass(frozen=True, slots=True)
class BernoulliSequenceDistribution:
    n: int
    p_as: float
    composition_masses: tuple[BernoulliCompositionMass, ...]
    support: tuple[BernoulliSupportMass, ...]
    interval_moments: IntervalMomentSummary
    capacity_moments: CapacityMomentSummary


def expected_pair_counts_fixed_composition(*, n_as: int, n_hs: int) -> ExpectedPairCounts:
    """Exact expected directed adjacent counts under a uniform fixed-composition permutation."""
    _validate_count(n_as, field="n_as")
    _validate_count(n_hs, field="n_hs")
    n = n_as + n_hs
    if n < 2:
        raise ValueError("at least two ships are required")
    return ExpectedPairCounts(
        aa=n_as * (n_as - 1) / n,
        ah=n_as * n_hs / n,
        ha=n_as * n_hs / n,
        hh=n_hs * (n_hs - 1) / n,
    )


def expected_total_fixed_composition(
    *, n_as: int, n_hs: int, matrix: SourcePairIntervalMatrix
) -> float:
    counts = expected_pair_counts_fixed_composition(n_as=n_as, n_hs=n_hs)
    a, b, c, d = _abcd(matrix)
    return a * counts.aa + b * counts.ah + c * counts.ha + d * counts.hh


def fixed_composition_moment_oracle(
    *, n_as: int, n_hs: int, matrix: SourcePairIntervalMatrix
) -> IntervalMomentSummary:
    """Constant-size exact first/second moments from local tuple probabilities."""
    _validate_count(n_as, field="n_as")
    _validate_count(n_hs, field="n_hs")
    n = n_as + n_hs
    if n < 2:
        raise ValueError("at least two ships are required")

    edge_mean = 0.0
    edge_second = 0.0
    for x, y in product((AS, HS), repeat=2):
        p = _ordered_tuple_probability((x, y), n_as=n_as, n_hs=n_hs)
        w = _physical_weight(matrix, leader=x, follower=y)
        edge_mean += p * w
        edge_second += p * w * w

    overlap = 0.0
    if n >= 3:
        for x, y, z in product((AS, HS), repeat=3):
            p = _ordered_tuple_probability((x, y, z), n_as=n_as, n_hs=n_hs)
            overlap += p * _physical_weight(matrix, leader=x, follower=y) * _physical_weight(
                matrix, leader=y, follower=z
            )

    disjoint = 0.0
    if n >= 4:
        for x, y, z, t in product((AS, HS), repeat=4):
            p = _ordered_tuple_probability((x, y, z, t), n_as=n_as, n_hs=n_hs)
            disjoint += p * _physical_weight(matrix, leader=x, follower=y) * _physical_weight(
                matrix, leader=z, follower=t
            )

    expected_total = (n - 1) * edge_mean
    expected_square = (n - 1) * edge_second + 2 * (n - 2) * overlap + (n - 2) * (n - 3) * disjoint
    variance = expected_square - expected_total * expected_total
    if variance < 0.0 and abs(variance) <= 1e-9:
        variance = 0.0
    if variance < -1e-9:
        raise AssertionError("negative fixed-composition variance")
    denominator = n - 1
    return IntervalMomentSummary(
        expected_total_interval_min=expected_total,
        variance_total_interval_min2=variance,
        expected_mean_adjacent_interval_min=expected_total / denominator,
        variance_mean_adjacent_interval_min2=variance / (denominator * denominator),
    )


def _family_sequence_count(family: EndpointRunFamily, q: int) -> int:
    m, k = family.n_as, family.n_hs
    if not family.q_min <= q <= family.q_max:
        raise ValueError("q outside family range")
    if family.endpoint_class == "AA":
        return comb(m - 1, q) * comb(k - 1, q - 1)
    if family.endpoint_class == "HH":
        return comb(m - 1, q - 1) * comb(k - 1, q)
    return comb(m - 1, q - 1) * comb(k - 1, q - 1)


def _capacity_for_total(*, total_interval_min: float, n: int) -> float:
    if total_interval_min <= 0.0:
        raise ValueError("total_interval_min must be positive")
    return _MINUTES_PER_DAY * (n - 1) / total_interval_min


def fixed_composition_distribution(
    *, n_as: int, n_hs: int, matrix: SourcePairIntervalMatrix
) -> FixedCompositionDistribution:
    """Exact distribution over all uniform fixed-composition physical orderings."""
    _validate_count(n_as, field="n_as")
    _validate_count(n_hs, field="n_hs")
    n = n_as + n_hs
    if n < 2:
        raise ValueError("at least two ships are required")
    total_sequences = comb(n, n_as)
    masses: list[EndpointQProbabilityMass] = []

    if n_as == 0 or n_hs == 0:
        extrema = exact_closed_form_extrema(n_as=n_as, n_hs=n_hs, matrix=matrix)
        cand = extrema.min_candidate
        masses.append(
            EndpointQProbabilityMass(
                endpoint_class=cand.endpoint_class,
                q=0,
                sequence_count=1,
                probability=1.0,
                total_interval_min=cand.total_interval_min,
                mean_adjacent_interval_min=cand.mean_adjacent_interval_min,
                capacity_ships_per_day=_capacity_for_total(total_interval_min=cand.total_interval_min, n=n),
            )
        )
    else:
        for family in endpoint_run_families(n_as=n_as, n_hs=n_hs, matrix=matrix):
            for q in range(family.q_min, family.q_max + 1):
                sequence_count = _family_sequence_count(family, q)
                total = family.total_for_q(q)
                masses.append(
                    EndpointQProbabilityMass(
                        endpoint_class=family.endpoint_class,
                        q=q,
                        sequence_count=sequence_count,
                        probability=sequence_count / total_sequences,
                        total_interval_min=total,
                        mean_adjacent_interval_min=total / (n - 1),
                        capacity_ships_per_day=_capacity_for_total(total_interval_min=total, n=n),
                    )
                )

    if sum(row.sequence_count for row in masses) != total_sequences:
        raise AssertionError("endpoint/q multiplicities do not sum to C(n,m)")
    p_total = sum(row.probability for row in masses)
    if abs(p_total - 1.0) > 1e-10:
        raise AssertionError("fixed-composition probabilities do not sum to one")

    buckets: dict[float, dict[str, float | int]] = {}
    for row in masses:
        key = _support_key(row.total_interval_min)
        bucket = buckets.setdefault(key, {"total": row.total_interval_min, "count": 0, "prob": 0.0})
        if abs(float(bucket["total"]) - row.total_interval_min) > 1e-9:
            raise AssertionError("support rounding collision exceeded tolerance")
        bucket["count"] = int(bucket["count"]) + row.sequence_count
        bucket["prob"] = float(bucket["prob"]) + row.probability

    support = tuple(
        IntervalSupportMass(
            total_interval_min=float(bucket["total"]),
            sequence_count=int(bucket["count"]),
            probability=float(bucket["prob"]),
            mean_adjacent_interval_min=float(bucket["total"]) / (n - 1),
            capacity_ships_per_day=_capacity_for_total(total_interval_min=float(bucket["total"]), n=n),
        )
        for _, bucket in sorted(buckets.items())
    )

    expected_total = sum(row.probability * row.total_interval_min for row in support)
    variance_total = sum(row.probability * (row.total_interval_min - expected_total) ** 2 for row in support)
    denominator = n - 1
    interval_moments = IntervalMomentSummary(
        expected_total_interval_min=expected_total,
        variance_total_interval_min2=variance_total,
        expected_mean_adjacent_interval_min=expected_total / denominator,
        variance_mean_adjacent_interval_min2=variance_total / (denominator * denominator),
    )
    expected_capacity = sum(row.probability * row.capacity_ships_per_day for row in support)
    capacity_at_expected_mean = _MINUTES_PER_DAY / interval_moments.expected_mean_adjacent_interval_min
    gap = expected_capacity - capacity_at_expected_mean
    if gap < 0.0 and abs(gap) <= 1e-9:
        gap = 0.0
    if gap < -1e-8:
        raise AssertionError("Jensen gap unexpectedly negative")
    capacity_moments = CapacityMomentSummary(
        expected_capacity_ships_per_day=expected_capacity,
        capacity_at_expected_mean_ships_per_day=capacity_at_expected_mean,
        jensen_gap_ships_per_day=gap,
    )

    return FixedCompositionDistribution(
        n_as=n_as,
        n_hs=n_hs,
        total_sequences=total_sequences,
        endpoint_q_masses=tuple(masses),
        support=support,
        interval_moments=interval_moments,
        capacity_moments=capacity_moments,
    )


def bernoulli_moment_oracle(
    *, n: int, p_as: float, matrix: SourcePairIntervalMatrix
) -> IntervalMomentSummary:
    """Exact first/second total-interval moments for iid Bernoulli ship types."""
    _validate_n(n)
    _validate_probability(p_as, field="p_as")

    edge_mean = 0.0
    edge_second = 0.0
    for x, y in product((AS, HS), repeat=2):
        p = _iid_tuple_probability((x, y), p_as=p_as)
        w = _physical_weight(matrix, leader=x, follower=y)
        edge_mean += p * w
        edge_second += p * w * w

    overlap = 0.0
    for x, y, z in product((AS, HS), repeat=3):
        p = _iid_tuple_probability((x, y, z), p_as=p_as)
        overlap += p * _physical_weight(matrix, leader=x, follower=y) * _physical_weight(
            matrix, leader=y, follower=z
        )

    expected_total = (n - 1) * edge_mean
    variance = (n - 1) * (edge_second - edge_mean * edge_mean) + 2 * (n - 2) * (
        overlap - edge_mean * edge_mean
    )
    if variance < 0.0 and abs(variance) <= 1e-9:
        variance = 0.0
    if variance < -1e-9:
        raise AssertionError("negative Bernoulli variance")
    denominator = n - 1
    return IntervalMomentSummary(
        expected_total_interval_min=expected_total,
        variance_total_interval_min2=variance,
        expected_mean_adjacent_interval_min=expected_total / denominator,
        variance_mean_adjacent_interval_min2=variance / (denominator * denominator),
    )


def bernoulli_sequence_distribution(
    *, n: int, p_as: float, matrix: SourcePairIntervalMatrix
) -> BernoulliSequenceDistribution:
    """Exact iid Bernoulli sequence distribution via the binomial composition mixture."""
    _validate_n(n)
    _validate_probability(p_as, field="p_as")
    compositions: list[BernoulliCompositionMass] = []
    buckets: dict[float, dict[str, float]] = {}

    for n_as in range(n + 1):
        n_hs = n - n_as
        weight = comb(n, n_as) * (p_as ** n_as) * ((1.0 - p_as) ** n_hs)
        if weight == 0.0:
            continue
        compositions.append(BernoulliCompositionMass(n_as=n_as, probability=weight))
        conditional = fixed_composition_distribution(n_as=n_as, n_hs=n_hs, matrix=matrix)
        for row in conditional.support:
            joint_probability = weight * row.probability
            key = _support_key(row.total_interval_min)
            bucket = buckets.setdefault(key, {"total": row.total_interval_min, "prob": 0.0})
            if abs(bucket["total"] - row.total_interval_min) > 1e-9:
                raise AssertionError("Bernoulli support rounding collision exceeded tolerance")
            bucket["prob"] += joint_probability

    if abs(sum(row.probability for row in compositions) - 1.0) > 1e-10:
        raise AssertionError("binomial composition probabilities do not sum to one")

    support = tuple(
        BernoulliSupportMass(
            total_interval_min=bucket["total"],
            probability=bucket["prob"],
            mean_adjacent_interval_min=bucket["total"] / (n - 1),
            capacity_ships_per_day=_capacity_for_total(total_interval_min=bucket["total"], n=n),
        )
        for _, bucket in sorted(buckets.items())
    )
    if abs(sum(row.probability for row in support) - 1.0) > 1e-10:
        raise AssertionError("Bernoulli support probabilities do not sum to one")

    expected_total = sum(row.probability * row.total_interval_min for row in support)
    variance_total = sum(row.probability * (row.total_interval_min - expected_total) ** 2 for row in support)
    denominator = n - 1
    interval_moments = IntervalMomentSummary(
        expected_total_interval_min=expected_total,
        variance_total_interval_min2=variance_total,
        expected_mean_adjacent_interval_min=expected_total / denominator,
        variance_mean_adjacent_interval_min2=variance_total / (denominator * denominator),
    )
    expected_capacity = sum(row.probability * row.capacity_ships_per_day for row in support)
    capacity_at_expected_mean = _MINUTES_PER_DAY / interval_moments.expected_mean_adjacent_interval_min
    gap = expected_capacity - capacity_at_expected_mean
    if gap < 0.0 and abs(gap) <= 1e-9:
        gap = 0.0
    if gap < -1e-8:
        raise AssertionError("Bernoulli Jensen gap unexpectedly negative")
    capacity_moments = CapacityMomentSummary(
        expected_capacity_ships_per_day=expected_capacity,
        capacity_at_expected_mean_ships_per_day=capacity_at_expected_mean,
        jensen_gap_ships_per_day=gap,
    )
    return BernoulliSequenceDistribution(
        n=n,
        p_as=p_as,
        composition_masses=tuple(compositions),
        support=support,
        interval_moments=interval_moments,
        capacity_moments=capacity_moments,
    )
