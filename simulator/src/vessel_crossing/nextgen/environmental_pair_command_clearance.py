"""Full-horizon physical clearance for two explicit environmental commands.

This is the reusable pair-level physical contract shared by A104 JOINT screening and
opt-in source future-ingress screening.  It composes the existing A84 one-vessel
candidate transitions with A85 continuous transition clearance and then continues both
final commands on constant ground motion to the configured evaluation horizon.

No COLREG semantics, ownership, ranking, or source-admission policy lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cpa import ClosestApproach, closest_approach
from .domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from .environmental_candidate_transition import (
    EnvironmentalCandidateTransition,
    EnvironmentalCommandTrajectory,
)
from .environmental_evidence import environmental_vessel_motion_evidence
from .environmental_pair_clearance import (
    EnvironmentalPairTransitionClearance,
    EnvironmentalPairTransitionConfig,
    evaluate_environmental_pair_transition_clearance,
)
from .ship_domain import ShipDomain

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class EnvironmentalPairCommandClearance:
    """Complete bounded-command pair-clearance evidence through one fixed horizon."""

    transition: EnvironmentalPairTransitionClearance
    post_transition_cpa: ClosestApproach | None
    post_transition_domain_clearance: DomainClearanceCertification | None
    predicted_min_separation_nm: float
    transition_certified_clear: bool
    post_transition_certified_clear: bool

    @property
    def certified_clear(self) -> bool:
        return self.transition_certified_clear and self.post_transition_certified_clear

    @property
    def minimum_certified_domain_margin_nm(self) -> float | None:
        margins: list[float] = []
        if self.transition.domain is not None:
            margins.append(self.transition.domain.conservative_lower_bound_nm)
        if self.post_transition_domain_clearance is not None:
            margins.append(self.post_transition_domain_clearance.conservative_lower_bound_nm)
        return min(margins) if margins else None


@dataclass(frozen=True, slots=True)
class _DomainPose:
    position: tuple[float, float]
    heading_deg: float


def evaluate_environmental_pair_command_clearance(
    first: EnvironmentalCommandTrajectory,
    second: EnvironmentalCommandTrajectory,
    config: EnvironmentalPairTransitionConfig,
    *,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> EnvironmentalPairCommandClearance:
    """Certify two explicit commands over transition *and* residual horizon.

    The semantics intentionally match the historical A104 implementation exactly:
    continuous A85 transition clearance first, then constant final ground-motion CPA or
    oriented-domain certification until ``evaluation_horizon_min``.
    """

    if not isinstance(first, EnvironmentalCommandTrajectory) or not isinstance(
        second, EnvironmentalCommandTrajectory
    ):
        raise TypeError("first and second must implement EnvironmentalCommandTrajectory")
    if not isinstance(config, EnvironmentalPairTransitionConfig):
        raise TypeError("config must be EnvironmentalPairTransitionConfig")
    if (first_domain is None) != (second_domain is None):
        raise ValueError("first_domain and second_domain must be supplied together")

    transition = evaluate_environmental_pair_transition_clearance(
        first,
        second,
        config,
        first_domain=first_domain,
        second_domain=second_domain,
    )
    predicted_min = transition.centre.conservative_lower_bound_nm
    post_cpa: ClosestApproach | None = None
    post_domain: DomainClearanceCertification | None = None
    post_clear = True

    transition_end = max(first.transition_duration_min, second.transition_duration_min)
    horizon = config.evaluation_horizon_min
    if transition_end < horizon - _EPS:
        first_final = first.state_at_or_continue(transition_end)
        second_final = second.state_at_or_continue(transition_end)
        first_motion = environmental_vessel_motion_evidence(
            first_final, first.environmental_drift_velocity
        )
        second_motion = environmental_vessel_motion_evidence(
            second_final, second.environmental_drift_velocity
        )
        remaining = horizon - transition_end
        post_cpa = closest_approach(
            first_motion.to_ground_motion_state(),
            second_motion.to_ground_motion_state(),
            horizon=remaining,
        )
        predicted_min = min(predicted_min, post_cpa.dcpa)

        if first_domain is not None and second_domain is not None:

            def first_post(t: float) -> _DomainPose:
                vx, vy = first_motion.ground_velocity
                return _DomainPose(
                    (first_motion.position[0] + vx * t, first_motion.position[1] + vy * t),
                    first_motion.hull_heading_deg,
                )

            def second_post(t: float) -> _DomainPose:
                vx, vy = second_motion.ground_velocity
                return _DomainPose(
                    (second_motion.position[0] + vx * t, second_motion.position[1] + vy * t),
                    second_motion.hull_heading_deg,
                )

            post_domain = certify_pair_domain_clearance(
                first_state_at=first_post,
                second_state_at=second_post,
                first_domain=first_domain,
                second_domain=second_domain,
                duration_min=remaining,
                first_speed_bound_nm_per_min=first_motion.speed_over_ground,
                second_speed_bound_nm_per_min=second_motion.speed_over_ground,
                first_turn_rate_bound_deg_per_min=0.0,
                second_turn_rate_bound_deg_per_min=0.0,
                time_tolerance_min=config.transition_time_tolerance_min,
                margin_tolerance_nm=config.transition_distance_tolerance_nm,
                max_subdivisions=config.max_transition_subdivisions,
            )
            post_clear = post_domain.certified_clear
        else:
            post_clear = post_cpa.dcpa + _EPS >= config.required_min_separation_nm

    return EnvironmentalPairCommandClearance(
        transition=transition,
        post_transition_cpa=post_cpa,
        post_transition_domain_clearance=post_domain,
        predicted_min_separation_nm=predicted_min,
        transition_certified_clear=transition.certified_clear,
        post_transition_certified_clear=post_clear,
    )
