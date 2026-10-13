"""Exact dependent binary-sequence theory for CROSS R2-S4.

This module is CROSS-owned mathematics built on the qualified R2-S1/S2/S3
stack.  It generalizes iid Bernoulli type sequences to a two-state first-order
Markov chain.  It is not a recovered Guo 2024 Section-3 implementation and it
does not authorize physical simulator campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Iterable

from .r2_s1_pair_semantics import AS, HS, SourcePairIntervalMatrix
from .r2_s3_random_sequence_distribution import CapacityMomentSummary, IntervalMomentSummary

_EPS: Final = 1e-12
_SUPPORT_DIGITS: Final = 12
_MINUTES_PER_DAY: Final = 1440.0
_STATES: Final = (AS, HS)


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")


def _validate_probability(value: float, *, field: str) -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")


def _support_key(value: float) -> float:
    return round(value, _SUPPORT_DIGITS)


def _physical_weight(matrix: SourcePairIntervalMatrix, *, leader: str, follower: str) -> float:
    return matrix.interval_for_source_roles(follower_type=follower, leader_type=leader)


def _abcd(matrix: SourcePairIntervalMatrix) -> tuple[float, float, float, float]:
    a = matrix.interval_for_source_roles(follower_type=AS, leader_type=AS)
    b = matrix.interval_for_source_roles(follower_type=AS, leader_type=HS)
    c = matrix.interval_for_source_roles(follower_type=HS, leader_type=AS)
    d = matrix.interval_for_source_roles(follower_type=HS, leader_type=HS)
    return a, b, c, d


def _capacity_for_total(*, total_interval_min: float, n: int) -> float:
    if total_interval_min <= 0.0:
        raise ValueError("total_interval_min must be positive")
    return _MINUTES_PER_DAY * (n - 1) / total_interval_min


@dataclass(frozen=True, slots=True)
class BinaryMarkovModel:
    """General two-state Markov law in physical leader->trailer order."""

    initial_as_probability: float
    p_as_to_as: float
    p_as_to_hs: float
    p_hs_to_as: float
    p_hs_to_hs: float

    def __post_init__(self) -> None:
        for field in (
            "initial_as_probability",
            "p_as_to_as",
            "p_as_to_hs",
            "p_hs_to_as",
            "p_hs_to_hs",
        ):
            _validate_probability(getattr(self, field), field=field)
        if abs((self.p_as_to_as + self.p_as_to_hs) - 1.0) > 1e-10:
            raise ValueError("AS transition row must sum to one")
        if abs((self.p_hs_to_as + self.p_hs_to_hs) - 1.0) > 1e-10:
            raise ValueError("HS transition row must sum to one")

    @property
    def initial_hs_probability(self) -> float:
        return 1.0 - self.initial_as_probability

    def initial_probability(self, state: str) -> float:
        if state == AS:
            return self.initial_as_probability
        if state == HS:
            return self.initial_hs_probability
        raise ValueError("state must be AS or HS")

    def transition_probability(self, *, current: str, next_state: str) -> float:
        if current == AS and next_state == AS:
            return self.p_as_to_as
        if current == AS and next_state == HS:
            return self.p_as_to_hs
        if current == HS and next_state == AS:
            return self.p_hs_to_as
        if current == HS and next_state == HS:
            return self.p_hs_to_hs
        raise ValueError("current and next_state must be AS or HS")


@dataclass(frozen=True, slots=True)
class StationaryMarkovParameterization:
    p_as: float
    rho: float
    rho_min: float
    rho_max: float
    cross_flux: float
    model: BinaryMarkovModel

    @property
    def p_hs(self) -> float:
        return 1.0 - self.p_as


@dataclass(frozen=True, slots=True)
class MarkovSupportMass:
    total_interval_min: float
    probability: float
    mean_adjacent_interval_min: float
    capacity_ships_per_day: float


@dataclass(frozen=True, slots=True)
class MarkovSequenceDistribution:
    n: int
    model: BinaryMarkovModel
    support: tuple[MarkovSupportMass, ...]
    interval_moments: IntervalMomentSummary
    capacity_moments: CapacityMomentSummary


@dataclass(frozen=True, slots=True)
class StationaryDeltaSensitivity:
    p_as: float
    rho: float
    delta_min: float
    cross_flux: float
    expected_edge_interval_min: float
    capacity_at_expected_mean_ships_per_day: float
    d_expected_edge_interval_d_rho_min: float
    d_capacity_at_expected_mean_d_rho_ships_per_day: float


def iid_binary_markov_model(*, p_as: float) -> BinaryMarkovModel:
    """Return the Markov representation of an iid Bernoulli type sequence."""
    _validate_probability(p_as, field="p_as")
    q = 1.0 - p_as
    return BinaryMarkovModel(
        initial_as_probability=p_as,
        p_as_to_as=p_as,
        p_as_to_hs=q,
        p_hs_to_as=p_as,
        p_hs_to_hs=q,
    )


def stationary_markov_parameterization(*, p_as: float, rho: float) -> StationaryMarkovParameterization:
    """Construct a stationary binary chain from AS fraction and lag-1 correlation.

    The correlation is the Pearson correlation of consecutive AS indicators.
    This is CROSS-owned type-dependence semantics, not a Guo trust parameter.
    """
    if not isfinite(p_as) or not 0.0 < p_as < 1.0:
        raise ValueError("p_as must be finite and strictly within (0, 1)")
    if not isfinite(rho):
        raise ValueError("rho must be finite")
    q = 1.0 - p_as
    rho_min = -min(p_as, q) / max(p_as, q)
    rho_max = 1.0
    if rho < rho_min - 1e-12 or rho > rho_max + 1e-12:
        raise ValueError(f"rho must be within [{rho_min}, 1]")
    rho = min(max(rho, rho_min), rho_max)
    cross_flux = p_as * q * (1.0 - rho)
    p_aa = p_as + q * rho
    p_ah = q * (1.0 - rho)
    p_ha = p_as * (1.0 - rho)
    p_hh = q + p_as * rho

    # Eliminate harmless boundary round-off before validation.
    def clip(x: float) -> float:
        if -1e-12 < x < 0.0:
            return 0.0
        if 1.0 < x < 1.0 + 1e-12:
            return 1.0
        return x

    model = BinaryMarkovModel(
        initial_as_probability=p_as,
        p_as_to_as=clip(p_aa),
        p_as_to_hs=clip(p_ah),
        p_hs_to_as=clip(p_ha),
        p_hs_to_hs=clip(p_hh),
    )
    return StationaryMarkovParameterization(
        p_as=p_as,
        rho=rho,
        rho_min=rho_min,
        rho_max=rho_max,
        cross_flux=cross_flux,
        model=model,
    )


def stationary_edge_probabilities(*, p_as: float, rho: float) -> dict[tuple[str, str], float]:
    """Physical (current,next) edge probabilities under the stationary p/rho model."""
    param = stationary_markov_parameterization(p_as=p_as, rho=rho)
    q = 1.0 - p_as
    m = param.model
    return {
        (AS, AS): p_as * m.p_as_to_as,
        (AS, HS): p_as * m.p_as_to_hs,
        (HS, AS): q * m.p_hs_to_as,
        (HS, HS): q * m.p_hs_to_hs,
    }


def stationary_lag1_as_correlation(*, p_as: float, model: BinaryMarkovModel) -> float:
    """Recover lag-1 AS-indicator correlation when `(p,1-p)` is stationary."""
    if not isfinite(p_as) or not 0.0 < p_as < 1.0:
        raise ValueError("p_as must be finite and strictly within (0, 1)")
    q = 1.0 - p_as
    next_as = p_as * model.p_as_to_as + q * model.p_hs_to_as
    if abs(next_as - p_as) > 1e-10:
        raise ValueError("provided p_as is not stationary for the model")
    p_aa = p_as * model.p_as_to_as
    return (p_aa - p_as * p_as) / (p_as * q)


def markov_sequence_probability(sequence: Iterable[str], *, model: BinaryMarkovModel) -> float:
    seq = tuple(sequence)
    if not seq:
        raise ValueError("sequence must be non-empty")
    for state in seq:
        if state not in _STATES:
            raise ValueError("sequence states must be AS or HS")
    p = model.initial_probability(seq[0])
    for current, nxt in zip(seq, seq[1:]):
        p *= model.transition_probability(current=current, next_state=nxt)
    return p


def markov_moment_oracle(
    *, n: int, matrix: SourcePairIntervalMatrix, model: BinaryMarkovModel
) -> IntervalMomentSummary:
    """Exact O(n) first/second moments for a general binary Markov sequence."""
    _validate_n(n)
    alpha = {AS: model.initial_as_probability, HS: model.initial_hs_probability}
    m1 = {AS: 0.0, HS: 0.0}
    m2 = {AS: 0.0, HS: 0.0}

    for _ in range(n - 1):
        next_alpha = {AS: 0.0, HS: 0.0}
        next_m1 = {AS: 0.0, HS: 0.0}
        next_m2 = {AS: 0.0, HS: 0.0}
        for current in _STATES:
            for nxt in _STATES:
                tp = model.transition_probability(current=current, next_state=nxt)
                if tp == 0.0:
                    continue
                w = _physical_weight(matrix, leader=current, follower=nxt)
                next_alpha[nxt] += alpha[current] * tp
                next_m1[nxt] += tp * (m1[current] + alpha[current] * w)
                next_m2[nxt] += tp * (
                    m2[current] + 2.0 * w * m1[current] + alpha[current] * w * w
                )
        alpha, m1, m2 = next_alpha, next_m1, next_m2

    if abs(sum(alpha.values()) - 1.0) > 1e-10:
        raise AssertionError("Markov state probability mass does not sum to one")
    expected_total = sum(m1.values())
    expected_square = sum(m2.values())
    variance = expected_square - expected_total * expected_total
    if variance < 0.0 and abs(variance) <= 1e-9:
        variance = 0.0
    if variance < -1e-9:
        raise AssertionError("negative Markov variance")
    denom = n - 1
    return IntervalMomentSummary(
        expected_total_interval_min=expected_total,
        variance_total_interval_min2=variance,
        expected_mean_adjacent_interval_min=expected_total / denom,
        variance_mean_adjacent_interval_min2=variance / (denom * denom),
    )


def markov_sequence_distribution(
    *, n: int, matrix: SourcePairIntervalMatrix, model: BinaryMarkovModel
) -> MarkovSequenceDistribution:
    """Exact finite total-interval distribution for a binary Markov chain.

    The forward state uses integer directed-pair counts rather than cumulative
    floating-point totals.  This makes path merging independent of floating
    summation order; interval totals are evaluated only after the exact
    combinatorial state has been formed.
    """
    _validate_n(n)
    # State key: (last_state, N_AA, N_AH, N_HA, N_HH), where pair counts use
    # canonical source follower->leader orientation.
    current: dict[tuple[str, int, int, int, int], float] = {}
    for state in _STATES:
        prob = model.initial_probability(state)
        if prob > 0.0:
            current[(state, 0, 0, 0, 0)] = prob

    for _ in range(n - 1):
        nxt_map: dict[tuple[str, int, int, int, int], float] = {}
        for (current_state, aa, ah, ha, hh), prob in current.items():
            for nxt in _STATES:
                tp = model.transition_probability(current=current_state, next_state=nxt)
                if tp == 0.0:
                    continue
                naa, nah, nha, nhh = aa, ah, ha, hh
                if current_state == AS and nxt == AS:
                    naa += 1
                elif current_state == AS and nxt == HS:
                    # physical AS->HS = HS follower behind AS leader = source HA
                    nha += 1
                elif current_state == HS and nxt == AS:
                    # physical HS->AS = AS follower behind HS leader = source AH
                    nah += 1
                else:
                    nhh += 1
                key = (nxt, naa, nah, nha, nhh)
                nxt_map[key] = nxt_map.get(key, 0.0) + prob * tp
        current = nxt_map

    a, b, c, d = _abcd(matrix)
    buckets: dict[float, tuple[float, float]] = {}
    for (_, aa, ah, ha, hh), prob in current.items():
        total = aa * a + ah * b + ha * c + hh * d
        key = _support_key(total)
        if key in buckets:
            representative, existing_prob = buckets[key]
            if abs(representative - total) > 1e-9:
                raise AssertionError("Markov aggregate support rounding collision exceeded tolerance")
            buckets[key] = (representative, existing_prob + prob)
        else:
            buckets[key] = (total, prob)

    probability_sum = sum(prob for _, prob in buckets.values())
    if abs(probability_sum - 1.0) > 1e-10:
        raise AssertionError("Markov distribution probabilities do not sum to one")

    support = tuple(
        MarkovSupportMass(
            total_interval_min=total,
            probability=prob,
            mean_adjacent_interval_min=total / (n - 1),
            capacity_ships_per_day=_capacity_for_total(total_interval_min=total, n=n),
        )
        for _, (total, prob) in sorted(buckets.items())
    )

    expected_total = sum(row.probability * row.total_interval_min for row in support)
    variance_total = sum(row.probability * (row.total_interval_min - expected_total) ** 2 for row in support)
    denom = n - 1
    interval_moments = IntervalMomentSummary(
        expected_total_interval_min=expected_total,
        variance_total_interval_min2=variance_total,
        expected_mean_adjacent_interval_min=expected_total / denom,
        variance_mean_adjacent_interval_min2=variance_total / (denom * denom),
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

    oracle = markov_moment_oracle(n=n, matrix=matrix, model=model)
    if abs(oracle.expected_total_interval_min - expected_total) > 1e-8:
        raise AssertionError("distribution mean disagrees with Markov moment oracle")
    if abs(oracle.variance_total_interval_min2 - variance_total) > 1e-7:
        raise AssertionError("distribution variance disagrees with Markov moment oracle")

    return MarkovSequenceDistribution(
        n=n,
        model=model,
        support=support,
        interval_moments=interval_moments,
        capacity_moments=capacity_moments,
    )


def stationary_expected_edge_interval(
    *, p_as: float, rho: float, matrix: SourcePairIntervalMatrix
) -> float:
    """Closed-form stationary expected interval per adjacent edge."""
    param = stationary_markov_parameterization(p_as=p_as, rho=rho)
    q = 1.0 - p_as
    a, b, c, d = _abcd(matrix)
    delta = (b + c) - (a + d)
    return p_as * a + q * d + param.cross_flux * delta


def stationary_delta_sensitivity(
    *, p_as: float, rho: float, matrix: SourcePairIntervalMatrix
) -> StationaryDeltaSensitivity:
    """Exact S2-Delta sensitivity of stationary mean interval and mean-capacity."""
    param = stationary_markov_parameterization(p_as=p_as, rho=rho)
    q = 1.0 - p_as
    a, b, c, d = _abcd(matrix)
    delta = (b + c) - (a + d)
    mu = p_as * a + q * d + param.cross_flux * delta
    d_mu = -p_as * q * delta
    capacity = _MINUTES_PER_DAY / mu
    d_capacity = -_MINUTES_PER_DAY * d_mu / (mu * mu)
    return StationaryDeltaSensitivity(
        p_as=p_as,
        rho=param.rho,
        delta_min=delta,
        cross_flux=param.cross_flux,
        expected_edge_interval_min=mu,
        capacity_at_expected_mean_ships_per_day=capacity,
        d_expected_edge_interval_d_rho_min=d_mu,
        d_capacity_at_expected_mean_d_rho_ships_per_day=d_capacity,
    )
