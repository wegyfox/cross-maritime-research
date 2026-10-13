"""Measurement-noise propagation and finite-variance coverage for CROSS R2-S8.

This CROSS-owned layer adds an explicit additive measurement-error model on top
of the qualified R2-S7 structural estimators.  V1 assumes block-total error is
zero mean, homoscedastic, independent across observed blocks, and independent
of the structural sequence process.  The variance is an externally supplied
model parameter; R2-S8 does not claim empirical calibration of that value.

Coverage intervals use Chebyshev's inequality.  They are therefore finite-
sample lower-bound guarantees under the stated finite-variance assumptions and
must not be described as Gaussian intervals or as Guo-2024 protocol content.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, sqrt

from .r2_s1_pair_semantics import SourcePairIntervalMatrix
from .r2_s7_structural_precision import (
    _minimum_blocks_for_stable_coverage,
    mixed_difference_endpoint_variance,
    mixed_sum_structural_precision,
)

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class AdditiveBlockMeasurementNoise:
    block_total_variance_min2: float
    zero_mean: bool = True
    independent_across_blocks: bool = True
    independent_of_structure: bool = True
    homoscedastic_across_endpoint_classes: bool = True
    calibration_status: str = "UNBOUND_PARAMETER"


@dataclass(frozen=True, slots=True)
class NoisyMixedSumPrecision:
    n: int
    blocks: int
    p_as: float
    rho: float
    cross_flux: float
    true_mixed_sum_min: float
    expected_estimator_mixed_sum_min: float
    structural_variance_min2: float
    measurement_variance_contribution_min2: float
    total_variance_min2: float
    total_standard_error_min: float
    block_total_noise_variance_min2: float


@dataclass(frozen=True, slots=True)
class NoisyMixedDifferencePrecision:
    n: int
    ha_count: int
    ah_count: int
    true_mixed_difference_min: float
    expected_estimator_mixed_difference_min: float
    structural_variance_min2: float
    measurement_variance_contribution_min2: float
    total_variance_min2: float
    total_standard_error_min: float
    block_total_noise_variance_min2: float


@dataclass(frozen=True, slots=True)
class ChebyshevCoverageInterval:
    center_estimate_min: float
    lower_min: float
    upper_min: float
    halfwidth_min: float
    alpha: float
    coverage_lower_bound: float
    variance_min2: float
    guarantee_type: str = "CHEBYSHEV_FINITE_VARIANCE_LOWER_BOUND"


@dataclass(frozen=True, slots=True)
class MinimumNoisyMixedSumDesign:
    n: int
    p_as: float
    rho: float
    alpha: float
    target_halfwidth_min: float
    required_blocks: int
    achieved_halfwidth_min: float
    previous_halfwidth_min: float | None
    structural_one_block_variance_min2: float
    block_total_noise_variance_min2: float


@dataclass(frozen=True, slots=True)
class NoisyMixedDifferenceDesign:
    n: int
    p_as: float
    rho: float
    alpha: float
    target_halfwidth_min: float
    target_directional_coverage_probability: float
    min_each_direction: int
    required_blocks: int
    achieved_directional_coverage_probability: float
    previous_directional_coverage_probability: float
    conditional_chebyshev_coverage_lower_bound: float
    joint_precision_and_sampling_coverage_lower_bound: float
    directional_endpoint_probability: float
    achieved_conditional_halfwidth_bound_min: float
    structural_ha_variance_min2: float
    structural_ah_variance_min2: float
    block_total_noise_variance_min2: float


def _validate_nonnegative_finite(value: float, *, field: str) -> float:
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{field} must be finite and >= 0")
    return value


def _validate_positive_finite(value: float, *, field: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be finite and > 0")
    return value


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly within (0,1)")
    return alpha


def additive_block_measurement_noise(*, block_total_variance_min2: float) -> AdditiveBlockMeasurementNoise:
    """Construct the frozen S8 V1 measurement-noise assumption set."""
    variance = _validate_nonnegative_finite(
        block_total_variance_min2, field="block_total_variance_min2"
    )
    return AdditiveBlockMeasurementNoise(block_total_variance_min2=variance)


def chebyshev_halfwidth(*, variance_min2: float, alpha: float) -> float:
    """Two-sided Chebyshev halfwidth with coverage at least 1-alpha."""
    variance = _validate_nonnegative_finite(variance_min2, field="variance_min2")
    alpha = _validate_alpha(alpha)
    return sqrt(variance / alpha)


def chebyshev_interval(
    *, center_estimate_min: float, variance_min2: float, alpha: float
) -> ChebyshevCoverageInterval:
    """Return a model-conditional finite-variance Chebyshev interval."""
    center = float(center_estimate_min)
    if not isfinite(center):
        raise ValueError("center_estimate_min must be finite")
    variance = _validate_nonnegative_finite(variance_min2, field="variance_min2")
    alpha = _validate_alpha(alpha)
    halfwidth = chebyshev_halfwidth(variance_min2=variance, alpha=alpha)
    return ChebyshevCoverageInterval(
        center_estimate_min=center,
        lower_min=center - halfwidth,
        upper_min=center + halfwidth,
        halfwidth_min=halfwidth,
        alpha=alpha,
        coverage_lower_bound=1.0 - alpha,
        variance_min2=variance,
    )


def noisy_mixed_sum_precision(
    *,
    n: int,
    blocks: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    noise: AdditiveBlockMeasurementNoise,
) -> NoisyMixedSumPrecision:
    """Exact variance propagation for S=b+c under S8 additive block noise."""
    structural = mixed_sum_structural_precision(
        n=n, blocks=blocks, p_as=p_as, rho=rho, matrix=matrix
    )
    m = n - 1
    scale2 = blocks * m * m * structural.cross_flux * structural.cross_flux
    measurement = noise.block_total_variance_min2 / scale2
    total = structural.estimator_variance_min2 + measurement
    return NoisyMixedSumPrecision(
        n=n,
        blocks=blocks,
        p_as=structural.p_as,
        rho=structural.rho,
        cross_flux=structural.cross_flux,
        true_mixed_sum_min=structural.true_mixed_sum_min,
        expected_estimator_mixed_sum_min=structural.expected_estimator_mixed_sum_min,
        structural_variance_min2=structural.estimator_variance_min2,
        measurement_variance_contribution_min2=measurement,
        total_variance_min2=total,
        total_standard_error_min=sqrt(max(total, 0.0)),
        block_total_noise_variance_min2=noise.block_total_variance_min2,
    )


def noisy_mixed_sum_chebyshev_interval(
    *,
    center_estimate_min: float,
    n: int,
    blocks: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    noise: AdditiveBlockMeasurementNoise,
    alpha: float,
) -> ChebyshevCoverageInterval:
    precision = noisy_mixed_sum_precision(
        n=n, blocks=blocks, p_as=p_as, rho=rho, matrix=matrix, noise=noise
    )
    return chebyshev_interval(
        center_estimate_min=center_estimate_min,
        variance_min2=precision.total_variance_min2,
        alpha=alpha,
    )


def minimum_blocks_for_noisy_mixed_sum_interval(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    noise: AdditiveBlockMeasurementNoise,
    target_halfwidth_min: float,
    alpha: float,
) -> MinimumNoisyMixedSumDesign:
    """Smallest B meeting a Chebyshev halfwidth target for the S estimator."""
    target = _validate_positive_finite(target_halfwidth_min, field="target_halfwidth_min")
    alpha = _validate_alpha(alpha)
    one = mixed_sum_structural_precision(n=n, blocks=1, p_as=p_as, rho=rho, matrix=matrix)
    m = n - 1
    numerator = one.one_block_total_variance_min2 + noise.block_total_variance_min2
    denominator = m * m * one.cross_flux * one.cross_flux * alpha * target * target
    required = 1 if numerator <= _EPS else max(1, int(ceil(numerator / denominator - 1e-15)))
    achieved_precision = noisy_mixed_sum_precision(
        n=n, blocks=required, p_as=p_as, rho=rho, matrix=matrix, noise=noise
    )
    achieved = chebyshev_halfwidth(
        variance_min2=achieved_precision.total_variance_min2, alpha=alpha
    )
    previous = None
    if required > 1:
        previous_precision = noisy_mixed_sum_precision(
            n=n, blocks=required - 1, p_as=p_as, rho=rho, matrix=matrix, noise=noise
        )
        previous = chebyshev_halfwidth(
            variance_min2=previous_precision.total_variance_min2, alpha=alpha
        )
    return MinimumNoisyMixedSumDesign(
        n=n,
        p_as=one.p_as,
        rho=one.rho,
        alpha=alpha,
        target_halfwidth_min=target,
        required_blocks=required,
        achieved_halfwidth_min=achieved,
        previous_halfwidth_min=previous,
        structural_one_block_variance_min2=one.one_block_total_variance_min2,
        block_total_noise_variance_min2=noise.block_total_variance_min2,
    )


def noisy_mixed_difference_conditional_precision(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    noise: AdditiveBlockMeasurementNoise,
    ha_count: int,
    ah_count: int,
) -> NoisyMixedDifferencePrecision:
    """Conditional D=b-c variance with additive error on each observed block."""
    for value, field in ((ha_count, "ha_count"), (ah_count, "ah_count")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be an integer >= 1")
    ep = mixed_difference_endpoint_variance(n=n, p_as=p_as, rho=rho, matrix=matrix)
    structural = ep.ha_variance_total_min2 / ha_count + ep.ah_variance_total_min2 / ah_count
    measurement = noise.block_total_variance_min2 * (1.0 / ha_count + 1.0 / ah_count)
    total = structural + measurement
    return NoisyMixedDifferencePrecision(
        n=n,
        ha_count=ha_count,
        ah_count=ah_count,
        true_mixed_difference_min=ep.true_mixed_difference_min,
        expected_estimator_mixed_difference_min=ep.true_mixed_difference_min,
        structural_variance_min2=structural,
        measurement_variance_contribution_min2=measurement,
        total_variance_min2=total,
        total_standard_error_min=sqrt(max(total, 0.0)),
        block_total_noise_variance_min2=noise.block_total_variance_min2,
    )


def noisy_mixed_difference_chebyshev_interval(
    *,
    center_estimate_min: float,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    noise: AdditiveBlockMeasurementNoise,
    ha_count: int,
    ah_count: int,
    alpha: float,
) -> ChebyshevCoverageInterval:
    precision = noisy_mixed_difference_conditional_precision(
        n=n,
        p_as=p_as,
        rho=rho,
        matrix=matrix,
        noise=noise,
        ha_count=ha_count,
        ah_count=ah_count,
    )
    return chebyshev_interval(
        center_estimate_min=center_estimate_min,
        variance_min2=precision.total_variance_min2,
        alpha=alpha,
    )


def minimum_direction_count_for_noisy_difference_interval(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    noise: AdditiveBlockMeasurementNoise,
    target_halfwidth_min: float,
    alpha: float,
) -> int:
    """Smallest symmetric endpoint count k meeting a Chebyshev D halfwidth."""
    target = _validate_positive_finite(target_halfwidth_min, field="target_halfwidth_min")
    alpha = _validate_alpha(alpha)
    ep = mixed_difference_endpoint_variance(n=n, p_as=p_as, rho=rho, matrix=matrix)
    numerator = (
        ep.ha_variance_total_min2
        + ep.ah_variance_total_min2
        + 2.0 * noise.block_total_variance_min2
    )
    if numerator <= _EPS:
        return 1
    return max(1, int(ceil(numerator / (alpha * target * target) - 1e-15)))


def noisy_mixed_difference_design(
    *,
    n: int,
    p_as: float,
    rho: float,
    matrix: SourcePairIntervalMatrix,
    noise: AdditiveBlockMeasurementNoise,
    target_halfwidth_min: float,
    alpha: float,
    target_directional_coverage_probability: float,
    max_blocks: int = 1_000_000,
) -> NoisyMixedDifferenceDesign:
    """Couple noisy D precision to stable finite directional sampling coverage."""
    alpha = _validate_alpha(alpha)
    target = _validate_positive_finite(target_halfwidth_min, field="target_halfwidth_min")
    k = minimum_direction_count_for_noisy_difference_interval(
        n=n,
        p_as=p_as,
        rho=rho,
        matrix=matrix,
        noise=noise,
        target_halfwidth_min=target,
        alpha=alpha,
    )
    ep = mixed_difference_endpoint_variance(n=n, p_as=p_as, rho=rho, matrix=matrix)
    required, achieved, previous, endpoint_probability = _minimum_blocks_for_stable_coverage(
        n=n,
        p_as=p_as,
        rho=rho,
        min_each=k,
        target=target_directional_coverage_probability,
        max_blocks=max_blocks,
    )
    precision_at_k = noisy_mixed_difference_conditional_precision(
        n=n,
        p_as=p_as,
        rho=rho,
        matrix=matrix,
        noise=noise,
        ha_count=k,
        ah_count=k,
    )
    halfwidth = chebyshev_halfwidth(
        variance_min2=precision_at_k.total_variance_min2,
        alpha=alpha,
    )
    conditional_coverage = 1.0 - alpha
    joint_lower = achieved * conditional_coverage
    return NoisyMixedDifferenceDesign(
        n=n,
        p_as=ep.p_as,
        rho=ep.rho,
        alpha=alpha,
        target_halfwidth_min=target,
        target_directional_coverage_probability=target_directional_coverage_probability,
        min_each_direction=k,
        required_blocks=required,
        achieved_directional_coverage_probability=achieved,
        previous_directional_coverage_probability=previous,
        conditional_chebyshev_coverage_lower_bound=conditional_coverage,
        joint_precision_and_sampling_coverage_lower_bound=joint_lower,
        directional_endpoint_probability=endpoint_probability,
        achieved_conditional_halfwidth_bound_min=halfwidth,
        structural_ha_variance_min2=ep.ha_variance_total_min2,
        structural_ah_variance_min2=ep.ah_variance_total_min2,
        block_total_noise_variance_min2=noise.block_total_variance_min2,
    )
