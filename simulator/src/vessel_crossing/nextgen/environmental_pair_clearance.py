"""Continuous pair-transition clearance under uniform environmental drift.

A85 is a physical safety primitive only.  It combines two exact A84 one-vessel
candidate transitions and conservatively bounds centre separation and, optionally,
continuous oriented ship-domain margin over the common transition interval.  It does
not interpret COLREG intent, Rule-13 progress, rank candidates, arbitrate, or execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..kinematics import norm, subtract
from .domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from .environmental_candidate_transition import (
    EnvironmentalCandidateTransition,
    EnvironmentalCommandTrajectory,
)
from .ship_domain import ShipDomain

_EPS = 1e-10


class EnvironmentalPairTransitionUnavailable(RuntimeError):
    """Raised when continuous pair clearance is requested for an unreachable command."""


@dataclass(frozen=True, slots=True)
class EnvironmentalPairTransitionConfig:
    """Physical horizon and numerical tolerances for one environmental pair."""

    evaluation_horizon_min: float
    required_min_separation_nm: float
    transition_time_tolerance_min: float = 0.01
    transition_distance_tolerance_nm: float = 1e-4
    max_transition_subdivisions: int = 20

    def __post_init__(self) -> None:
        values = (
            self.evaluation_horizon_min,
            self.required_min_separation_nm,
            self.transition_time_tolerance_min,
            self.transition_distance_tolerance_nm,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("environmental pair-transition config must be finite")
        if self.evaluation_horizon_min <= 0.0:
            raise ValueError("evaluation_horizon_min must be positive")
        if self.required_min_separation_nm < 0.0:
            raise ValueError("required_min_separation_nm must be non-negative")
        if self.transition_time_tolerance_min <= 0.0:
            raise ValueError("transition_time_tolerance_min must be positive")
        if self.transition_distance_tolerance_nm < 0.0:
            raise ValueError("transition_distance_tolerance_nm must be non-negative")
        if self.max_transition_subdivisions < 1:
            raise ValueError("max_transition_subdivisions must be at least 1")


@dataclass(frozen=True, slots=True)
class EnvironmentalCentreSeparationCertification:
    evaluated_duration_min: float
    first_transition_duration_min: float
    second_transition_duration_min: float
    sampled_min_separation_nm: float
    conservative_lower_bound_nm: float
    state_evaluations: int
    both_completed_within_horizon: bool
    required_min_separation_nm: float

    @property
    def certified_clear(self) -> bool:
        return self.conservative_lower_bound_nm + _EPS >= self.required_min_separation_nm


@dataclass(frozen=True, slots=True)
class EnvironmentalPairTransitionClearance:
    first: EnvironmentalCommandTrajectory
    second: EnvironmentalCommandTrajectory
    centre: EnvironmentalCentreSeparationCertification
    domain: DomainClearanceCertification | None

    @property
    def certified_clear(self) -> bool:
        if self.domain is not None:
            return self.domain.certified_clear
        return self.centre.certified_clear


@dataclass(frozen=True, slots=True)
class _DomainPose:
    position: tuple[float, float]
    heading_deg: float


def _state_at(transition: EnvironmentalCommandTrajectory, elapsed_min: float):
    """Query the trajectory contract, including provider-defined continuation."""

    return transition.state_at_or_continue(elapsed_min)


def _domain_pose_at(transition: EnvironmentalCommandTrajectory, elapsed_min: float) -> _DomainPose:
    """Build one domain pose from a single deterministic trajectory query."""

    state = _state_at(transition, elapsed_min)
    return _DomainPose(state.position, state.through_water_heading_deg)


def _centre_clearance(
    first: EnvironmentalCommandTrajectory,
    second: EnvironmentalCommandTrajectory,
    config: EnvironmentalPairTransitionConfig,
) -> EnvironmentalCentreSeparationCertification:
    transition_end = max(first.transition_duration_min, second.transition_duration_min)
    duration = min(transition_end, config.evaluation_horizon_min)

    def separation(time_min: float) -> float:
        first_at = _state_at(first, time_min)
        second_at = _state_at(second, time_min)
        return norm(subtract(second_at.position, first_at.position))

    if duration <= _EPS:
        distance = separation(0.0)
        return EnvironmentalCentreSeparationCertification(
            evaluated_duration_min=0.0,
            first_transition_duration_min=first.transition_duration_min,
            second_transition_duration_min=second.transition_duration_min,
            sampled_min_separation_nm=distance,
            conservative_lower_bound_nm=distance,
            state_evaluations=2,
            both_completed_within_horizon=True,
            required_min_separation_nm=config.required_min_separation_nm,
        )

    cache: dict[float, float] = {}

    def distance(time_min: float) -> float:
        key = round(time_min, 14)
        if key not in cache:
            cache[key] = separation(time_min)
        return cache[key]

    relative_speed_bound = first.ground_speed_bound + second.ground_speed_bound
    best_sample = [min(distance(0.0), distance(duration))]

    def recurse(a: float, b: float, depth: int) -> float:
        width = b - a
        mid = 0.5 * (a + b)
        da = distance(a)
        dm = distance(mid)
        db = distance(b)
        best_sample[0] = min(best_sample[0], da, dm, db)
        lower_bound = max(
            0.0,
            da - relative_speed_bound * width,
            db - relative_speed_bound * width,
            dm - relative_speed_bound * width * 0.5,
        )
        if lower_bound >= best_sample[0] - config.transition_distance_tolerance_nm:
            return lower_bound
        if width <= config.transition_time_tolerance_min:
            return lower_bound
        if depth >= config.max_transition_subdivisions:
            return lower_bound
        return min(recurse(a, mid, depth + 1), recurse(mid, b, depth + 1))

    lower = recurse(0.0, duration, 0)
    return EnvironmentalCentreSeparationCertification(
        evaluated_duration_min=duration,
        first_transition_duration_min=first.transition_duration_min,
        second_transition_duration_min=second.transition_duration_min,
        sampled_min_separation_nm=best_sample[0],
        conservative_lower_bound_nm=lower,
        state_evaluations=len(cache),
        both_completed_within_horizon=transition_end <= config.evaluation_horizon_min + _EPS,
        required_min_separation_nm=config.required_min_separation_nm,
    )


def evaluate_environmental_pair_transition_clearance(
    first: EnvironmentalCommandTrajectory,
    second: EnvironmentalCommandTrajectory,
    config: EnvironmentalPairTransitionConfig,
    *,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> EnvironmentalPairTransitionClearance:
    """Certify one pair over the bounded-command transition interval.

    Domain screening is optional but symmetric: either both domains are supplied or
    neither.  When domains are present, the result's overall ``certified_clear`` follows
    the stronger oriented-domain certificate; the centre-separation certificate remains
    available as independent evidence.
    """

    if not isinstance(first, EnvironmentalCommandTrajectory) or not isinstance(
        second, EnvironmentalCommandTrajectory
    ):
        raise TypeError("first and second must implement EnvironmentalCommandTrajectory")
    if not isinstance(config, EnvironmentalPairTransitionConfig):
        raise TypeError("config must be EnvironmentalPairTransitionConfig")
    if first.initial_state.vessel_id == second.initial_state.vessel_id:
        raise ValueError("pair transitions must belong to distinct vessels")
    if (first_domain is None) != (second_domain is None):
        raise ValueError("first_domain and second_domain must be supplied together")
    if not first.reachable or not second.reachable:
        raise EnvironmentalPairTransitionUnavailable(
            "pair clearance cannot certify a physically unreachable candidate transition"
        )

    centre = _centre_clearance(first, second, config)
    domain: DomainClearanceCertification | None = None
    if first_domain is not None and second_domain is not None:
        duration = centre.evaluated_duration_min
        domain = certify_pair_domain_clearance(
            first_state_at=lambda t: _domain_pose_at(first, t),
            second_state_at=lambda t: _domain_pose_at(second, t),
            first_domain=first_domain,
            second_domain=second_domain,
            duration_min=duration,
            first_speed_bound_nm_per_min=first.ground_speed_bound,
            second_speed_bound_nm_per_min=second.ground_speed_bound,
            first_turn_rate_bound_deg_per_min=first.turn_rate_bound_deg_per_min,
            second_turn_rate_bound_deg_per_min=second.turn_rate_bound_deg_per_min,
            time_tolerance_min=config.transition_time_tolerance_min,
            margin_tolerance_nm=config.transition_distance_tolerance_nm,
            max_subdivisions=config.max_transition_subdivisions,
        )

    return EnvironmentalPairTransitionClearance(first, second, centre, domain)
