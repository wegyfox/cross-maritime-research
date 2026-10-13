"""Exact observation-design conditioning for CROSS R2-S6.

This CROSS-owned layer quantifies how strongly the qualified S5 mixed-sum
identification amplifies mean errors and how frequently the direction-labelled
mixed endpoint classes needed for mixed-difference identification appear in
finite stationary Markov blocks.

It is not a recovered Guo 2024 protocol and it does not authorize physical
simulator campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, isfinite

from .r2_s4_markov_clustering import stationary_markov_parameterization

_EPS = 1e-12


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")


def _validate_blocks(blocks: int) -> None:
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 0:
        raise ValueError("blocks must be an integer >= 0")


def _validate_min_each(min_each: int) -> None:
    if not isinstance(min_each, int) or isinstance(min_each, bool) or min_each < 1:
        raise ValueError("min_each must be an integer >= 1")


def _pow_probability(base: float, exponent: int) -> float:
    if exponent == 0:
        return 1.0
    if base == 0.0:
        return 0.0
    return base**exponent


@dataclass(frozen=True, slots=True)
class MixedSumConditioning:
    p_as: float
    rho: float
    cross_flux: float
    mean_error_amplification_to_mixed_sum: float


@dataclass(frozen=True, slots=True)
class DirectionalEndpointExposure:
    n: int
    p_as: float
    rho: float
    ha_probability: float
    ah_probability: float
    same_endpoint_probability: float
    either_mixed_endpoint_probability: float


@dataclass(frozen=True, slots=True)
class DirectionalCoverage:
    n: int
    blocks: int
    min_each_direction: int
    directional_endpoint_probability: float
    probability_both_directions_observed: float
    expected_ha_blocks: float
    expected_ah_blocks: float


@dataclass(frozen=True, slots=True)
class MinimumDirectionalBlockDesign:
    n: int
    min_each_direction: int
    target_probability: float
    required_blocks: int
    achieved_probability: float
    previous_probability: float
    directional_endpoint_probability: float
    expected_ha_blocks: float
    expected_ah_blocks: float


def mixed_sum_conditioning(*, p_as: float, rho: float) -> MixedSumConditioning:
    """Return exact S=b+c mean-error amplification under stationary S4 law.

    Qualified S5 gives S = ... + mu/r, so dS/dmu = 1/r where
    r=p(1-p)(1-rho).  Zero cross flux is a structural non-identifiability
    boundary and therefore fails closed.
    """
    param = stationary_markov_parameterization(p_as=p_as, rho=rho)
    r = param.cross_flux
    if r <= _EPS:
        raise ValueError("stationary cross flux must be positive for mixed-sum conditioning")
    return MixedSumConditioning(
        p_as=param.p_as,
        rho=param.rho,
        cross_flux=r,
        mean_error_amplification_to_mixed_sum=1.0 / r,
    )


def directional_endpoint_exposure(*, n: int, p_as: float, rho: float) -> DirectionalEndpointExposure:
    """Exact stationary finite-block probabilities for HA/AH endpoint classes."""
    _validate_n(n)
    param = stationary_markov_parameterization(p_as=p_as, rho=rho)
    p = param.p_as
    q = 1.0 - p
    transitions = n - 1
    e = p * q * (1.0 - param.rho**transitions)
    # Remove only harmless boundary round-off; mathematically e is in [0, 1/2].
    if -_EPS < e < 0.0:
        e = 0.0
    if 0.5 < e < 0.5 + _EPS:
        e = 0.5
    if e < -_EPS or e > 0.5 + _EPS:
        raise AssertionError("directional endpoint probability left the valid range")
    same = 1.0 - 2.0 * e
    if -_EPS < same < 0.0:
        same = 0.0
    if same < -_EPS or same > 1.0 + _EPS:
        raise AssertionError("same-endpoint probability left the valid range")
    return DirectionalEndpointExposure(
        n=n,
        p_as=p,
        rho=param.rho,
        ha_probability=e,
        ah_probability=e,
        same_endpoint_probability=same,
        either_mixed_endpoint_probability=2.0 * e,
    )


def _prob_one_direction_below(*, blocks: int, min_each: int, e: float) -> float:
    # Marginal count is Binomial(blocks, e).
    upper = min(min_each - 1, blocks)
    other = 1.0 - e
    return sum(
        comb(blocks, i) * _pow_probability(e, i) * _pow_probability(other, blocks - i)
        for i in range(upper + 1)
    )


def _prob_both_directions_below(*, blocks: int, min_each: int, e: float) -> float:
    # Joint lower rectangle under Multinomial(B; e, e, 1-2e).
    same = 1.0 - 2.0 * e
    total = 0.0
    upper = min(min_each - 1, blocks)
    for i in range(upper + 1):
        max_j = min(upper, blocks - i)
        for j in range(max_j + 1):
            k = blocks - i - j
            coeff = comb(blocks, i) * comb(blocks - i, j)
            total += (
                coeff
                * _pow_probability(e, i + j)
                * _pow_probability(same, k)
            )
    return total


def directional_coverage_probability(
    *, n: int, p_as: float, rho: float, blocks: int, min_each_direction: int = 1
) -> DirectionalCoverage:
    """Exact P(N_HA>=k and N_AH>=k) for independent stationary blocks."""
    _validate_blocks(blocks)
    _validate_min_each(min_each_direction)
    exposure = directional_endpoint_exposure(n=n, p_as=p_as, rho=rho)
    e = exposure.ha_probability
    k = min_each_direction
    if blocks < 2 * k or e <= 0.0:
        probability = 0.0
    else:
        low_one = _prob_one_direction_below(blocks=blocks, min_each=k, e=e)
        low_both = _prob_both_directions_below(blocks=blocks, min_each=k, e=e)
        probability = 1.0 - 2.0 * low_one + low_both
        if -1e-12 < probability < 0.0:
            probability = 0.0
        if 1.0 < probability < 1.0 + 1e-12:
            probability = 1.0
        if probability < -1e-10 or probability > 1.0 + 1e-10:
            raise AssertionError("directional coverage probability left [0,1]")
        probability = min(max(probability, 0.0), 1.0)
    expected = blocks * e
    return DirectionalCoverage(
        n=n,
        blocks=blocks,
        min_each_direction=k,
        directional_endpoint_probability=e,
        probability_both_directions_observed=probability,
        expected_ha_blocks=expected,
        expected_ah_blocks=expected,
    )


def minimum_blocks_for_directional_coverage(
    *,
    n: int,
    p_as: float,
    rho: float,
    min_each_direction: int,
    target_probability: float,
    max_blocks: int = 1_000_000,
) -> MinimumDirectionalBlockDesign:
    """Smallest independent-block count reaching an exact directional coverage target."""
    _validate_n(n)
    _validate_min_each(min_each_direction)
    if not isfinite(target_probability) or not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must be finite and strictly within (0,1)")
    if not isinstance(max_blocks, int) or isinstance(max_blocks, bool) or max_blocks < 2 * min_each_direction:
        raise ValueError("max_blocks must be an integer >= 2*min_each_direction")

    exposure = directional_endpoint_exposure(n=n, p_as=p_as, rho=rho)
    e = exposure.ha_probability
    if e <= _EPS:
        raise ValueError("directional endpoint exposure is zero; D cannot be sampled by this block design")

    low = 2 * min_each_direction - 1  # guaranteed infeasible by count alone
    high = max(low + 1, 1)
    while high <= max_blocks:
        cov = directional_coverage_probability(
            n=n,
            p_as=p_as,
            rho=rho,
            blocks=high,
            min_each_direction=min_each_direction,
        ).probability_both_directions_observed
        if cov >= target_probability:
            break
        low = high
        high *= 2
    if high > max_blocks:
        high = max_blocks
        cov = directional_coverage_probability(
            n=n,
            p_as=p_as,
            rho=rho,
            blocks=high,
            min_each_direction=min_each_direction,
        ).probability_both_directions_observed
        if cov < target_probability:
            raise ValueError("target_probability is not reached within max_blocks")

    while high - low > 1:
        mid = (low + high) // 2
        cov = directional_coverage_probability(
            n=n,
            p_as=p_as,
            rho=rho,
            blocks=mid,
            min_each_direction=min_each_direction,
        ).probability_both_directions_observed
        if cov >= target_probability:
            high = mid
        else:
            low = mid

    achieved = directional_coverage_probability(
        n=n, p_as=p_as, rho=rho, blocks=high, min_each_direction=min_each_direction
    )
    previous_probability = 0.0
    if high > 0:
        previous_probability = directional_coverage_probability(
            n=n, p_as=p_as, rho=rho, blocks=high - 1, min_each_direction=min_each_direction
        ).probability_both_directions_observed
    return MinimumDirectionalBlockDesign(
        n=n,
        min_each_direction=min_each_direction,
        target_probability=target_probability,
        required_blocks=high,
        achieved_probability=achieved.probability_both_directions_observed,
        previous_probability=previous_probability,
        directional_endpoint_probability=e,
        expected_ha_blocks=achieved.expected_ha_blocks,
        expected_ah_blocks=achieved.expected_ah_blocks,
    )
