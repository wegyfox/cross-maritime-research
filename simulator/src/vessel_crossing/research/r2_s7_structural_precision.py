"""Exact structural estimation precision for CROSS R2-S7.

This CROSS-owned layer quantifies finite-sample variance of the S5 mixed-sum
and mixed-difference estimators under the qualified S4 stationary Markov model
and the S6 independent-block observation design.

It is not a Guo-2024 protocol, does not include measurement noise, and does not
authorize physical simulator campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, isfinite, lgamma, log, log1p, sqrt

from .r2_s1_pair_semantics import SourcePairIntervalMatrix
from .r2_s4_markov_clustering import markov_moment_oracle, stationary_markov_parameterization
from .r2_s5_identifiability import (
    markov_endpoint_conditioned_distributions,
    mixed_cell_coordinates,
)
from .r2_s6_observation_design import directional_endpoint_exposure

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class MixedSumStructuralPrecision:
    n: int
    blocks: int
    p_as: float
    rho: float
    cross_flux: float
    true_mixed_sum_min: float
    expected_estimator_mixed_sum_min: float
    one_block_total_variance_min2: float
    estimator_variance_min2: float
    estimator_standard_error_min: float


@dataclass(frozen=True, slots=True)
class MinimumMixedSumBlocks:
    n: int
    p_as: float
    rho: float
    target_standard_error_min: float
    required_blocks: int
    achieved_standard_error_min: float
    previous_standard_error_min: float | None
    cross_flux: float
    one_block_total_variance_min2: float


@dataclass(frozen=True, slots=True)
class MixedDifferenceEndpointVariance:
    n: int
    p_as: float
    rho: float
    ha_endpoint_probability: float
    ah_endpoint_probability: float
    ha_mean_total_min: float
    ah_mean_total_min: float
    true_mixed_difference_min: float
    ha_variance_total_min2: float
    ah_variance_total_min2: float
    translation_variance_invariant_holds: bool


@dataclass(frozen=True, slots=True)
class MixedDifferenceConditionalPrecision:
    n: int
    ha_count: int
    ah_count: int
    true_mixed_difference_min: float
    expected_estimator_mixed_difference_min: float
    estimator_variance_min2: float
    estimator_standard_error_min: float


@dataclass(frozen=True, slots=True)
class MixedDifferencePrecisionDesign:
    n: int
    p_as: float
    rho: float
    target_standard_error_min: float
    target_coverage_probability: float
    min_each_direction: int
    required_blocks: int
    achieved_coverage_probability: float
    previous_coverage_probability: float
    directional_endpoint_probability: float
    guaranteed_conditional_variance_bound_min2: float
    guaranteed_conditional_standard_error_bound_min: float
    ha_variance_total_min2: float
    ah_variance_total_min2: float


def _validate_blocks(blocks: int) -> None:
    if not isinstance(blocks, int) or isinstance(blocks, bool) or blocks < 1:
        raise ValueError("blocks must be an integer >= 1")


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")


def _validate_positive_finite(value: float, *, field: str) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be finite and > 0")


def _conditional_variance(row) -> float:
    mean = row.expected_total_interval_min
    variance = sum(
        mass.conditional_probability * (mass.total_interval_min - mean) ** 2
        for mass in row.support
    )
    if variance < 0.0 and abs(variance) <= 1e-12:
        variance = 0.0
    if variance < -1e-10:
        raise AssertionError("negative endpoint-conditioned variance")
    return max(variance, 0.0)



def _logaddexp(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + log1p(exp(lo - hi))


def _binomial_lower_tail_stable(*, blocks: int, upper: int, probability: float) -> float:
    """Stable exact finite sum P[Binomial(blocks,p)<=upper] in float arithmetic."""
    if upper < 0:
        return 0.0
    if upper >= blocks:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    lp = log(probability)
    lq = log(1.0 - probability)
    base = lgamma(blocks + 1.0)
    acc = float("-inf")
    for i in range(upper + 1):
        term = base - lgamma(i + 1.0) - lgamma(blocks - i + 1.0) + i * lp + (blocks - i) * lq
        acc = _logaddexp(acc, term)
    return 0.0 if acc < -745.0 else min(max(exp(acc), 0.0), 1.0)


def _joint_below_stable(*, blocks: int, min_each: int, e: float) -> float:
    """Stable finite multinomial lower rectangle P[N_HA<k,N_AH<k]."""
    upper = min(min_each - 1, blocks)
    if upper < 0:
        return 0.0
    same = 1.0 - 2.0 * e
    if e <= 0.0:
        return 1.0
    if same <= _EPS:
        # All blocks are directional. Both counts can be <k only if B <= 2(k-1).
        if blocks > 2 * upper:
            return 0.0
        acc = 0.0
        for i in range(max(0, blocks - upper), upper + 1):
            j = blocks - i
            if 0 <= j <= upper:
                acc += exp(
                    lgamma(blocks + 1.0) - lgamma(i + 1.0) - lgamma(j + 1.0)
                    + blocks * log(e)
                )
        return min(max(acc, 0.0), 1.0)

    le = log(e)
    ls = log(same)
    base = lgamma(blocks + 1.0)
    logfac_small = [lgamma(i + 1.0) for i in range(upper + 1)]
    # k_same ranges from max(0, B-2*upper) through B.
    low_same = max(0, blocks - 2 * upper)
    logfac_same = {k: lgamma(k + 1.0) for k in range(low_same, blocks + 1)}
    acc = float("-inf")
    for i in range(upper + 1):
        max_j = min(upper, blocks - i)
        for j in range(max_j + 1):
            k_same = blocks - i - j
            term = (
                base - logfac_small[i] - logfac_small[j] - logfac_same[k_same]
                + (i + j) * le + k_same * ls
            )
            acc = _logaddexp(acc, term)
    return 0.0 if acc < -745.0 else min(max(exp(acc), 0.0), 1.0)


def _stable_directional_coverage(*, blocks: int, min_each: int, e: float) -> float:
    if blocks < 2 * min_each or e <= 0.0:
        return 0.0
    low_one = _binomial_lower_tail_stable(blocks=blocks, upper=min_each - 1, probability=e)
    low_both = _joint_below_stable(blocks=blocks, min_each=min_each, e=e)
    value = 1.0 - 2.0 * low_one + low_both
    return min(max(value, 0.0), 1.0)


def _minimum_blocks_for_stable_coverage(
    *, n: int, p_as: float, rho: float, min_each: int, target: float, max_blocks: int
) -> tuple[int, float, float, float]:
    if not isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target_coverage_probability must be finite and strictly within (0,1)")
    if not isinstance(max_blocks, int) or isinstance(max_blocks, bool) or max_blocks < 2 * min_each:
        raise ValueError("max_blocks must be an integer >= 2*min_each")
    exposure = directional_endpoint_exposure(n=n, p_as=p_as, rho=rho)
    e = exposure.ha_probability
    if e <= _EPS:
        raise ValueError("directional endpoint exposure is zero; D cannot be sampled by this block design")
    low = 2 * min_each - 1
    high = max(2 * min_each, 1)
    while high <= max_blocks and _stable_directional_coverage(blocks=high, min_each=min_each, e=e) < target:
        low = high
        high *= 2
    if high > max_blocks:
        high = max_blocks
        if _stable_directional_coverage(blocks=high, min_each=min_each, e=e) < target:
            raise ValueError("target_coverage_probability is not reached within max_blocks")
    while high - low > 1:
        mid = (low + high) // 2
        if _stable_directional_coverage(blocks=mid, min_each=min_each, e=e) >= target:
            high = mid
        else:
            low = mid
    achieved = _stable_directional_coverage(blocks=high, min_each=min_each, e=e)
    previous = _stable_directional_coverage(blocks=high - 1, min_each=min_each, e=e) if high > 0 else 0.0
    return high, achieved, previous, e

def mixed_sum_structural_precision(
    *,
    n: int,
    blocks: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
) -> MixedSumStructuralPrecision:
    """Exact finite-sample variance of the stationary S=b+c estimator."""
    _validate_n(n)
    _validate_blocks(blocks)
    param = stationary_markov_parameterization(p_as=p_as, rho=rho)
    r = param.cross_flux
    if r <= _EPS:
        raise ValueError("stationary cross flux must be positive to estimate mixed sum")

    moments = markov_moment_oracle(n=n, matrix=matrix, model=param.model)
    m = n - 1
    variance = moments.variance_total_interval_min2 / (blocks * m * m * r * r)
    if variance < 0.0 and abs(variance) <= 1e-12:
        variance = 0.0
    if variance < -1e-10:
        raise AssertionError("negative mixed-sum estimator variance")

    coords = mixed_cell_coordinates(matrix)
    return MixedSumStructuralPrecision(
        n=n,
        blocks=blocks,
        p_as=param.p_as,
        rho=param.rho,
        cross_flux=r,
        true_mixed_sum_min=coords.mixed_sum_min,
        expected_estimator_mixed_sum_min=coords.mixed_sum_min,
        one_block_total_variance_min2=moments.variance_total_interval_min2,
        estimator_variance_min2=max(variance, 0.0),
        estimator_standard_error_min=sqrt(max(variance, 0.0)),
    )


def minimum_blocks_for_mixed_sum_precision(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    target_standard_error_min: float,
) -> MinimumMixedSumBlocks:
    """Smallest independent block count meeting a structural S standard error."""
    _validate_n(n)
    _validate_positive_finite(target_standard_error_min, field="target_standard_error_min")
    one = mixed_sum_structural_precision(n=n, blocks=1, p_as=p_as, rho=rho, matrix=matrix)
    numerator = one.one_block_total_variance_min2
    denominator = (n - 1) ** 2 * one.cross_flux**2 * target_standard_error_min**2
    required = 1 if numerator <= _EPS else max(1, int(ceil(numerator / denominator - 1e-15)))
    achieved = mixed_sum_structural_precision(
        n=n, blocks=required, p_as=p_as, rho=rho, matrix=matrix
    ).estimator_standard_error_min
    previous = None
    if required > 1:
        previous = mixed_sum_structural_precision(
            n=n, blocks=required - 1, p_as=p_as, rho=rho, matrix=matrix
        ).estimator_standard_error_min
    return MinimumMixedSumBlocks(
        n=n,
        p_as=one.p_as,
        rho=one.rho,
        target_standard_error_min=target_standard_error_min,
        required_blocks=required,
        achieved_standard_error_min=achieved,
        previous_standard_error_min=previous,
        cross_flux=one.cross_flux,
        one_block_total_variance_min2=numerator,
    )


def mixed_difference_endpoint_variance(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
) -> MixedDifferenceEndpointVariance:
    """Exact conditional HA/AH total variances used by the D estimator."""
    _validate_n(n)
    param = stationary_markov_parameterization(p_as=p_as, rho=rho)
    rows = {
        row.endpoint_class: row
        for row in markov_endpoint_conditioned_distributions(
            n=n, matrix=matrix, model=param.model
        )
    }
    if "HA" not in rows or "AH" not in rows:
        raise ValueError("both directed mixed endpoint classes must have positive probability")
    ha = rows["HA"]
    ah = rows["AH"]
    vha = _conditional_variance(ha)
    vah = _conditional_variance(ah)
    tolerance = 1e-9 * max(1.0, vha, vah)
    invariant = abs(vha - vah) <= tolerance
    if not invariant:
        raise AssertionError("S5 endpoint translation variance invariant failed")
    coords = mixed_cell_coordinates(matrix)
    return MixedDifferenceEndpointVariance(
        n=n,
        p_as=param.p_as,
        rho=param.rho,
        ha_endpoint_probability=ha.unconditional_endpoint_probability,
        ah_endpoint_probability=ah.unconditional_endpoint_probability,
        ha_mean_total_min=ha.expected_total_interval_min,
        ah_mean_total_min=ah.expected_total_interval_min,
        true_mixed_difference_min=coords.mixed_difference_min,
        ha_variance_total_min2=vha,
        ah_variance_total_min2=vah,
        translation_variance_invariant_holds=True,
    )


def mixed_difference_conditional_precision(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    ha_count: int,
    ah_count: int,
) -> MixedDifferenceConditionalPrecision:
    """Exact variance of D_hat conditioned on positive HA/AH block counts."""
    for value, field in ((ha_count, "ha_count"), (ah_count, "ah_count")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be an integer >= 1")
    ep = mixed_difference_endpoint_variance(n=n, p_as=p_as, rho=rho, matrix=matrix)
    variance = ep.ha_variance_total_min2 / ha_count + ep.ah_variance_total_min2 / ah_count
    return MixedDifferenceConditionalPrecision(
        n=n,
        ha_count=ha_count,
        ah_count=ah_count,
        true_mixed_difference_min=ep.true_mixed_difference_min,
        expected_estimator_mixed_difference_min=ep.true_mixed_difference_min,
        estimator_variance_min2=variance,
        estimator_standard_error_min=sqrt(variance),
    )


def minimum_direction_count_for_precision(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    target_standard_error_min: float,
) -> int:
    """Smallest symmetric count k guaranteeing the conditional D SE target."""
    _validate_positive_finite(target_standard_error_min, field="target_standard_error_min")
    ep = mixed_difference_endpoint_variance(n=n, p_as=p_as, rho=rho, matrix=matrix)
    total_var = ep.ha_variance_total_min2 + ep.ah_variance_total_min2
    if total_var <= _EPS:
        return 1
    return max(1, int(ceil(total_var / (target_standard_error_min**2) - 1e-15)))


def mixed_difference_precision_design(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    target_standard_error_min: float,
    target_coverage_probability: float,
    max_blocks: int = 1_000_000,
) -> MixedDifferencePrecisionDesign:
    """Couple the exact D precision count threshold to the S6 coverage design."""
    _validate_positive_finite(target_standard_error_min, field="target_standard_error_min")
    k = minimum_direction_count_for_precision(
        n=n,
        p_as=p_as,
        rho=rho,
        matrix=matrix,
        target_standard_error_min=target_standard_error_min,
    )
    ep = mixed_difference_endpoint_variance(n=n, p_as=p_as, rho=rho, matrix=matrix)
    required_blocks, achieved, previous, endpoint_probability = _minimum_blocks_for_stable_coverage(
        n=n,
        p_as=p_as,
        rho=rho,
        min_each=k,
        target=target_coverage_probability,
        max_blocks=max_blocks,
    )
    bound_var = (ep.ha_variance_total_min2 + ep.ah_variance_total_min2) / k
    return MixedDifferencePrecisionDesign(
        n=n,
        p_as=ep.p_as,
        rho=ep.rho,
        target_standard_error_min=target_standard_error_min,
        target_coverage_probability=target_coverage_probability,
        min_each_direction=k,
        required_blocks=required_blocks,
        achieved_coverage_probability=achieved,
        previous_coverage_probability=previous,
        directional_endpoint_probability=endpoint_probability,
        guaranteed_conditional_variance_bound_min2=bound_var,
        guaranteed_conditional_standard_error_bound_min=sqrt(bound_var),
        ha_variance_total_min2=ep.ha_variance_total_min2,
        ah_variance_total_min2=ep.ah_variance_total_min2,
    )
