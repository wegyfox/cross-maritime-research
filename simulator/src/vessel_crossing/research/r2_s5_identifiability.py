"""Endpoint-conditioned identifiability theory for CROSS R2-S5.

This module is CROSS-owned mathematics built on the qualified R2-S1..S4
stack. It formalizes which aggregate observations can identify the two
source-oriented mixed cells and which observations are invariant to swapping
those cells. It is not a recovered Guo 2024 Section-3 implementation and it
does not authorize physical simulator campaigns.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Iterable

from .r2_s1_pair_semantics import AS, HS, SourcePairIntervalMatrix
from .r2_s1_sequence_extrema import count_adjacent_source_pairs
from .r2_s4_markov_clustering import BinaryMarkovModel

_EPS: Final = 1e-12
_SUPPORT_DIGITS: Final = 12
_STATES: Final = (AS, HS)


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")


def _validate_probability(value: float, *, field: str) -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")


def _support_key(value: float) -> float:
    return round(value, _SUPPORT_DIGITS)


def _abcd(matrix: SourcePairIntervalMatrix) -> tuple[float, float, float, float]:
    a = matrix.interval_for_source_roles(follower_type=AS, leader_type=AS)
    b = matrix.interval_for_source_roles(follower_type=AS, leader_type=HS)
    c = matrix.interval_for_source_roles(follower_type=HS, leader_type=AS)
    d = matrix.interval_for_source_roles(follower_type=HS, leader_type=HS)
    return a, b, c, d


@dataclass(frozen=True, slots=True)
class MixedCellCoordinates:
    aa_min: float
    ah_min: float
    ha_min: float
    hh_min: float
    mixed_sum_min: float
    mixed_difference_min: float
    delta_min: float


@dataclass(frozen=True, slots=True)
class SequenceMixedDecomposition:
    leader_to_trailer: tuple[str, ...]
    endpoint_class: str
    mixed_transition_count: int
    mixed_transition_imbalance: int
    homogeneous_component_min: float
    mixed_symmetric_component_min: float
    mixed_orientation_component_min: float
    total_interval_min: float


@dataclass(frozen=True, slots=True)
class MarkovEndpointMarginals:
    n: int
    initial_as_probability: float
    final_as_probability: float
    expected_mixed_transition_imbalance: float


@dataclass(frozen=True, slots=True)
class StationaryMixedSumIdentification:
    p_as: float
    rho: float
    cross_flux: float
    expected_edge_interval_min: float
    identified_mixed_sum_min: float


@dataclass(frozen=True, slots=True)
class EndpointSupportMass:
    total_interval_min: float
    conditional_probability: float


@dataclass(frozen=True, slots=True)
class EndpointConditionedDistribution:
    endpoint_class: str
    unconditional_endpoint_probability: float
    support: tuple[EndpointSupportMass, ...]
    expected_total_interval_min: float


@dataclass(frozen=True, slots=True)
class EndpointMixedDifferenceIdentification:
    mean_total_ha_min: float
    mean_total_ah_min: float
    identified_mixed_difference_min: float


@dataclass(frozen=True, slots=True)
class ReconstructedMixedCells:
    mixed_sum_min: float
    mixed_difference_min: float
    ah_min: float
    ha_min: float


@dataclass(frozen=True, slots=True)
class IdentifiabilitySummary:
    aggregate_stationary_mean_identifies_mixed_sum: bool
    aggregate_stationary_mean_identifies_mixed_difference: bool
    stationary_unlabelled_distribution_identifies_difference_sign: bool
    endpoint_direction_conditioned_totals_identify_mixed_difference: bool
    combined_sum_and_difference_identify_directed_mixed_cells: bool


def mixed_cell_coordinates(matrix: SourcePairIntervalMatrix) -> MixedCellCoordinates:
    a, b, c, d = _abcd(matrix)
    s = b + c
    diff = b - c
    return MixedCellCoordinates(
        aa_min=a,
        ah_min=b,
        ha_min=c,
        hh_min=d,
        mixed_sum_min=s,
        mixed_difference_min=diff,
        delta_min=s - (a + d),
    )


def decompose_sequence_total(
    leader_to_trailer: Iterable[str], *, matrix: SourcePairIntervalMatrix
) -> SequenceMixedDecomposition:
    """Exact S/D decomposition of one finite physical leader->trailer chain."""
    seq = tuple(leader_to_trailer)
    if len(seq) < 2:
        raise ValueError("leader_to_trailer must contain at least two ships")
    for state in seq:
        if state not in _STATES:
            raise ValueError("sequence states must be AS or HS")

    counts = count_adjacent_source_pairs(seq)
    coords = mixed_cell_coordinates(matrix)
    mixed_count = counts.as_follower_hs_leader + counts.hs_follower_as_leader
    imbalance = counts.as_follower_hs_leader - counts.hs_follower_as_leader
    expected_imbalance = (1 if seq[-1] == AS else 0) - (1 if seq[0] == AS else 0)
    if imbalance != expected_imbalance or imbalance not in {-1, 0, 1}:
        raise AssertionError("binary endpoint/transition imbalance invariant failed")

    homogeneous = coords.aa_min * counts.as_follower_as_leader + coords.hh_min * counts.hs_follower_hs_leader
    mixed_symmetric = 0.5 * coords.mixed_sum_min * mixed_count
    mixed_orientation = 0.5 * coords.mixed_difference_min * imbalance
    total = homogeneous + mixed_symmetric + mixed_orientation
    endpoint_class = ("A" if seq[0] == AS else "H") + ("A" if seq[-1] == AS else "H")
    return SequenceMixedDecomposition(
        leader_to_trailer=seq,
        endpoint_class=endpoint_class,
        mixed_transition_count=mixed_count,
        mixed_transition_imbalance=imbalance,
        homogeneous_component_min=homogeneous,
        mixed_symmetric_component_min=mixed_symmetric,
        mixed_orientation_component_min=mixed_orientation,
        total_interval_min=total,
    )


def reversal_total_difference(
    leader_to_trailer: Iterable[str], *, matrix: SourcePairIntervalMatrix
) -> float:
    """Return T(sequence) - T(reversed sequence) exactly from the S5 theorem."""
    dec = decompose_sequence_total(leader_to_trailer, matrix=matrix)
    return mixed_cell_coordinates(matrix).mixed_difference_min * dec.mixed_transition_imbalance


def markov_endpoint_marginals(*, n: int, model: BinaryMarkovModel) -> MarkovEndpointMarginals:
    """Exact finite initial/final AS marginals and expected cross imbalance."""
    _validate_n(n)
    p_as = model.initial_as_probability
    current_as = p_as
    for _ in range(n - 1):
        current_as = current_as * model.p_as_to_as + (1.0 - current_as) * model.p_hs_to_as
    expected_imbalance = current_as - p_as
    return MarkovEndpointMarginals(
        n=n,
        initial_as_probability=p_as,
        final_as_probability=current_as,
        expected_mixed_transition_imbalance=expected_imbalance,
    )


def identify_stationary_mixed_sum(
    *,
    p_as: float,
    rho: float,
    aa_interval_min: float,
    hh_interval_min: float,
    expected_edge_interval_min: float,
) -> StationaryMixedSumIdentification:
    """Identify S=b+c from a stationary mean when cross flux is excited."""
    if not isfinite(p_as) or not 0.0 < p_as < 1.0:
        raise ValueError("p_as must be finite and strictly within (0, 1)")
    if not isfinite(rho):
        raise ValueError("rho must be finite")
    q = 1.0 - p_as
    rho_min = -min(p_as, q) / max(p_as, q)
    if rho < rho_min - 1e-12 or rho > 1.0 + 1e-12:
        raise ValueError(f"rho must be within [{rho_min}, 1]")
    for value, field in (
        (aa_interval_min, "aa_interval_min"),
        (hh_interval_min, "hh_interval_min"),
        (expected_edge_interval_min, "expected_edge_interval_min"),
    ):
        if not isfinite(value):
            raise ValueError(f"{field} must be finite")
    cross_flux = p_as * q * (1.0 - rho)
    if cross_flux <= _EPS:
        raise ValueError("stationary cross flux must be positive to identify the mixed sum")
    s = aa_interval_min + hh_interval_min + (
        expected_edge_interval_min - p_as * aa_interval_min - q * hh_interval_min
    ) / cross_flux
    return StationaryMixedSumIdentification(
        p_as=p_as,
        rho=rho,
        cross_flux=cross_flux,
        expected_edge_interval_min=expected_edge_interval_min,
        identified_mixed_sum_min=s,
    )


def _endpoint_class(start: str, end: str) -> str:
    return ("A" if start == AS else "H") + ("A" if end == AS else "H")


def markov_endpoint_conditioned_distributions(
    *, n: int, matrix: SourcePairIntervalMatrix, model: BinaryMarkovModel
) -> tuple[EndpointConditionedDistribution, ...]:
    """Exact finite total-interval laws conditioned on physical endpoint class.

    The DP retains start/end type and canonical source pair counts. Endpoint
    classes with zero probability are omitted.
    """
    _validate_n(n)
    # key=(start,last,N_AA,N_AH,N_HA,N_HH)
    current: dict[tuple[str, str, int, int, int, int], float] = {}
    for state in _STATES:
        p = model.initial_probability(state)
        if p > 0.0:
            current[(state, state, 0, 0, 0, 0)] = p

    for _ in range(n - 1):
        nxt_map: dict[tuple[str, str, int, int, int, int], float] = {}
        for (start, last, aa, ah, ha, hh), prob in current.items():
            for nxt in _STATES:
                tp = model.transition_probability(current=last, next_state=nxt)
                if tp == 0.0:
                    continue
                naa, nah, nha, nhh = aa, ah, ha, hh
                if last == AS and nxt == AS:
                    naa += 1
                elif last == AS and nxt == HS:
                    # physical AS->HS == source HA
                    nha += 1
                elif last == HS and nxt == AS:
                    # physical HS->AS == source AH
                    nah += 1
                else:
                    nhh += 1
                key = (start, nxt, naa, nah, nha, nhh)
                nxt_map[key] = nxt_map.get(key, 0.0) + prob * tp
        current = nxt_map

    a, b, c, d = _abcd(matrix)
    raw: dict[str, dict[float, tuple[float, float]]] = {}
    endpoint_probs: dict[str, float] = {}
    for (start, end, aa, ah, ha, hh), prob in current.items():
        ep = _endpoint_class(start, end)
        endpoint_probs[ep] = endpoint_probs.get(ep, 0.0) + prob
        total = aa * a + ah * b + ha * c + hh * d
        key = _support_key(total)
        bucket = raw.setdefault(ep, {})
        if key in bucket:
            representative, old = bucket[key]
            if abs(representative - total) > 1e-9:
                raise AssertionError("endpoint support rounding collision exceeded tolerance")
            bucket[key] = (representative, old + prob)
        else:
            bucket[key] = (total, prob)

    if abs(sum(endpoint_probs.values()) - 1.0) > 1e-10:
        raise AssertionError("endpoint probabilities do not sum to one")

    order = ("AA", "AH", "HA", "HH")
    result: list[EndpointConditionedDistribution] = []
    for ep in order:
        ep_prob = endpoint_probs.get(ep, 0.0)
        if ep_prob <= 0.0:
            continue
        support = tuple(
            EndpointSupportMass(total_interval_min=total, conditional_probability=mass / ep_prob)
            for _, (total, mass) in sorted(raw[ep].items())
        )
        if abs(sum(row.conditional_probability for row in support) - 1.0) > 1e-10:
            raise AssertionError("conditional endpoint probability mass does not sum to one")
        mean = sum(row.conditional_probability * row.total_interval_min for row in support)
        result.append(
            EndpointConditionedDistribution(
                endpoint_class=ep,
                unconditional_endpoint_probability=ep_prob,
                support=support,
                expected_total_interval_min=mean,
            )
        )
    return tuple(result)


def identify_mixed_difference_from_endpoint_means(
    *, mean_total_ha_min: float, mean_total_ah_min: float
) -> EndpointMixedDifferenceIdentification:
    """Identify D=b-c from endpoint-conditioned total means."""
    if not isfinite(mean_total_ha_min) or not isfinite(mean_total_ah_min):
        raise ValueError("endpoint-conditioned means must be finite")
    return EndpointMixedDifferenceIdentification(
        mean_total_ha_min=mean_total_ha_min,
        mean_total_ah_min=mean_total_ah_min,
        identified_mixed_difference_min=mean_total_ha_min - mean_total_ah_min,
    )


def reconstruct_mixed_cells(*, mixed_sum_min: float, mixed_difference_min: float) -> ReconstructedMixedCells:
    if not isfinite(mixed_sum_min) or not isfinite(mixed_difference_min):
        raise ValueError("mixed_sum_min and mixed_difference_min must be finite")
    b = 0.5 * (mixed_sum_min + mixed_difference_min)
    c = 0.5 * (mixed_sum_min - mixed_difference_min)
    return ReconstructedMixedCells(
        mixed_sum_min=mixed_sum_min,
        mixed_difference_min=mixed_difference_min,
        ah_min=b,
        ha_min=c,
    )


def identifiability_summary() -> IdentifiabilitySummary:
    return IdentifiabilitySummary(
        aggregate_stationary_mean_identifies_mixed_sum=True,
        aggregate_stationary_mean_identifies_mixed_difference=False,
        stationary_unlabelled_distribution_identifies_difference_sign=False,
        endpoint_direction_conditioned_totals_identify_mixed_difference=True,
        combined_sum_and_difference_identify_directed_mixed_cells=True,
    )
