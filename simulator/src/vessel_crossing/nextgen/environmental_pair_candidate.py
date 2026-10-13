"""Explicit environmental pair-command semantic and physical evaluation.

A86 consumes one already-chosen water-relative command for each vessel.  It does not
create a candidate grid, rank alternatives, arbitrate, or execute commands.  Declarative
pair intents are checked in the hull/control frame; physical safety is checked on exact
a84 ground trajectories with a85 continuous transition clearance and constant-ground-
velocity post-transition evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from ..cpa import ClosestApproach, closest_approach
from .domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from .dynamics import normalize_heading, signed_heading_delta
from .encounter import absolute_bearing_deg
from .environmental_candidate_transition import (
    EnvironmentalCandidateTransition,
    EnvironmentalCommandTrajectory,
    build_environmental_candidate_transition,
)
from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_evidence import environmental_vessel_motion_evidence
from .environmental_pair_clearance import (
    EnvironmentalPairTransitionClearance,
    EnvironmentalPairTransitionConfig,
    evaluate_environmental_pair_transition_clearance,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .fleet_supervision import FleetSupervisorBackend
from .maneuver_intent import ManeuverIntent, ManeuverIntentKind, PairManeuverIntents

_EPS = 1e-10


class EnvironmentalPairCandidateRejection(str, Enum):
    HOLD_REQUIRED = "hold_required"
    NO_EFFECTIVE_CHANGE = "no_effective_change"
    WRONG_COURSE_DIRECTION = "wrong_course_direction"
    PORT_ALTERATION_RESTRICTED = "port_alteration_restricted"
    NOT_SUBSTANTIAL = "not_substantial"
    SPEED_ENVELOPE = "speed_envelope"
    DOMAIN_EVIDENCE_INCOMPLETE = "domain_evidence_incomplete"
    PHYSICALLY_UNREACHABLE = "physically_unreachable"
    TRANSITION_CLEARANCE = "transition_clearance"
    POST_TRANSITION_CLEARANCE = "post_transition_clearance"
    CROSSES_AHEAD = "crosses_ahead"
    SAME_ROUTE_FOLLOWING_SPEED = "same_route_following_speed"


@dataclass(frozen=True, slots=True)
class EnvironmentalPairCandidateConfig:
    pair_clearance: EnvironmentalPairTransitionConfig
    substantial_course_change_deg: float = 0.0
    substantial_speed_change_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.substantial_course_change_deg < 0.0:
            raise ValueError("substantial_course_change_deg must be non-negative")
        if self.substantial_speed_change_fraction < 0.0:
            raise ValueError("substantial_speed_change_fraction must be non-negative")


@dataclass(frozen=True, slots=True)
class EnvironmentalPairCandidateEvaluation:
    first_intent: ManeuverIntent
    second_intent: ManeuverIntent
    first_command: WaterRelativeManeuverCommand
    second_command: WaterRelativeManeuverCommand
    first_transition: EnvironmentalCommandTrajectory
    second_transition: EnvironmentalCommandTrajectory
    first_rejection_reasons: tuple[EnvironmentalPairCandidateRejection, ...]
    second_rejection_reasons: tuple[EnvironmentalPairCandidateRejection, ...]
    pair_rejection_reasons: tuple[EnvironmentalPairCandidateRejection, ...]
    transition_clearance: EnvironmentalPairTransitionClearance | None
    post_transition_cpa: ClosestApproach | None
    post_transition_domain_clearance: DomainClearanceCertification | None
    predicted_min_separation_nm: float | None
    first_crosses_ahead: bool
    second_crosses_ahead: bool

    @property
    def accepted(self) -> bool:
        return not (
            self.first_rejection_reasons
            or self.second_rejection_reasons
            or self.pair_rejection_reasons
        )


@dataclass(frozen=True, slots=True)
class _DomainPose:
    position: tuple[float, float]
    heading_deg: float


def _relative_bearing(position, hull_heading_deg: float, target_position) -> float:
    return normalize_heading(absolute_bearing_deg(position, target_position) - hull_heading_deg)


def _speed_fraction(current: float, target: float) -> float:
    if current <= _EPS:
        return 1.0 if target <= _EPS else float("inf")
    return target / current


def _semantic_rejections(
    *,
    own,
    counterpart,
    command: WaterRelativeManeuverCommand,
    intent: ManeuverIntent,
    config: EnvironmentalPairCandidateConfig,
    following_constraint=None,
) -> list[EnvironmentalPairCandidateRejection]:
    reasons: list[EnvironmentalPairCandidateRejection] = []
    delta = signed_heading_delta(own.state.through_water_heading_deg, command.target_through_water_heading_deg)
    factor = _speed_fraction(own.state.through_water_speed, command.target_through_water_speed)
    changes_course = abs(delta) > _EPS
    changes_speed = abs(command.target_through_water_speed - own.state.through_water_speed) > _EPS
    hold = not changes_course and not changes_speed

    following_override = False
    if following_constraint is not None:
        ceiling = following_constraint.max_follower_speed_nm_per_min
        if command.target_through_water_speed > ceiling + _EPS:
            reasons.append(EnvironmentalPairCandidateRejection.SAME_ROUTE_FOLLOWING_SPEED)
        following_override = (
            not changes_course
            and changes_speed
            and command.target_through_water_speed <= ceiling + _EPS
            and own.state.through_water_speed > ceiling + _EPS
        )

    if (
        intent.maintain_course_and_speed
        and not intent.action_permitted
        and not hold
        and not following_override
    ):
        reasons.append(EnvironmentalPairCandidateRejection.HOLD_REQUIRED)

    requires_change = intent.kind in {
        ManeuverIntentKind.KEEP_WELL_CLEAR,
        ManeuverIntentKind.ALTER_COURSE_STARBOARD,
        ManeuverIntentKind.STAND_ON_MUST_TAKE_AVOIDING_ACTION,
    }
    if requires_change and hold:
        reasons.append(EnvironmentalPairCandidateRejection.NO_EFFECTIVE_CHANGE)

    if intent.required_course_direction == "starboard" and delta <= _EPS:
        reasons.append(EnvironmentalPairCandidateRejection.WRONG_COURSE_DIRECTION)
    elif intent.required_course_direction == "port" and delta >= -_EPS:
        reasons.append(EnvironmentalPairCandidateRejection.WRONG_COURSE_DIRECTION)

    if intent.avoid_port_alteration_for_target_on_port:
        bearing = _relative_bearing(
            own.state.position,
            own.state.through_water_heading_deg,
            counterpart.state.position,
        )
        if 180.0 + _EPS < bearing < 360.0 - _EPS and delta < -_EPS:
            reasons.append(EnvironmentalPairCandidateRejection.PORT_ALTERATION_RESTRICTED)

    if intent.early_and_substantial and not hold:
        course_ok = changes_course and abs(delta) + _EPS >= config.substantial_course_change_deg
        speed_ok = changes_speed and (
            (not isfinite(factor))
            or abs(factor - 1.0) + _EPS >= config.substantial_speed_change_fraction
        )
        if not (course_ok or speed_ok):
            reasons.append(EnvironmentalPairCandidateRejection.NOT_SUBSTANTIAL)
    elif intent.early_and_substantial and hold:
        reasons.append(EnvironmentalPairCandidateRejection.NOT_SUBSTANTIAL)

    if own.profile is not None and own.profile.speed_envelope is not None:
        if not own.profile.speed_envelope.permits(command.target_through_water_speed):
            reasons.append(EnvironmentalPairCandidateRejection.SPEED_ENVELOPE)
    return reasons


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _crosses_ahead(first_motion, second_motion, horizon_min: float) -> bool:
    """Whether first crosses second's straight ground track before second reaches it."""
    v1 = first_motion.ground_velocity
    v2 = second_motion.ground_velocity
    denominator = _cross(v1, v2)
    if abs(denominator) <= _EPS:
        return False
    relative = (
        second_motion.position[0] - first_motion.position[0],
        second_motion.position[1] - first_motion.position[1],
    )
    first_time = _cross(relative, v2) / denominator
    second_time = _cross(relative, v1) / denominator
    if first_time < -_EPS or second_time < -_EPS:
        return False
    if first_time > horizon_min + _EPS or second_time > horizon_min + _EPS:
        return False
    return first_time + _EPS < second_time


def evaluate_environmental_pair_candidate(
    context: EnvironmentalAvoidancePlanningContext,
    intents: PairManeuverIntents,
    *,
    first_command: WaterRelativeManeuverCommand,
    second_command: WaterRelativeManeuverCommand,
    config: EnvironmentalPairCandidateConfig,
) -> EnvironmentalPairCandidateEvaluation:
    """Evaluate one explicit pair of water-relative commands without selecting among alternatives."""
    if context.backend is not FleetSupervisorBackend.PAIR or len(context.vessel_ids) != 2:
        raise ValueError("environmental pair candidate evaluation requires a two-vessel PAIR context")
    first_id, second_id = context.vessel_ids
    first_intent = intents.for_vessel(first_id)
    second_intent = intents.for_vessel(second_id)
    if first_intent.counterpart_id != second_id or second_intent.counterpart_id != first_id:
        raise ValueError("pair intents do not match planning-context vessel IDs")

    first = context.for_vessel(first_id)
    second = context.for_vessel(second_id)
    first_reasons = _semantic_rejections(
        own=first, counterpart=second, command=first_command, intent=first_intent, config=config,
        following_constraint=context.following_constraint_for(first_id),
    )
    second_reasons = _semantic_rejections(
        own=second, counterpart=first, command=second_command, intent=second_intent, config=config,
        following_constraint=context.following_constraint_for(second_id),
    )
    pair_reasons: list[EnvironmentalPairCandidateRejection] = []

    domains = (first.domain, second.domain)
    if (domains[0] is None) != (domains[1] is None):
        pair_reasons.append(EnvironmentalPairCandidateRejection.DOMAIN_EVIDENCE_INCOMPLETE)

    first_transition = build_environmental_candidate_transition(
        first.state,
        first.environmental_drift_velocity,
        first_command,
        first.maneuver_limits,
        motion_backend=context.motion_backend,
        motion_state=first.motion_state,
        evaluation_horizon_min=config.pair_clearance.evaluation_horizon_min,
    )
    second_transition = build_environmental_candidate_transition(
        second.state,
        second.environmental_drift_velocity,
        second_command,
        second.maneuver_limits,
        motion_backend=context.motion_backend,
        motion_state=second.motion_state,
        evaluation_horizon_min=config.pair_clearance.evaluation_horizon_min,
    )
    if not first_transition.reachable:
        first_reasons.append(EnvironmentalPairCandidateRejection.PHYSICALLY_UNREACHABLE)
    if not second_transition.reachable:
        second_reasons.append(EnvironmentalPairCandidateRejection.PHYSICALLY_UNREACHABLE)

    transition_clearance = None
    post_cpa = None
    post_domain = None
    predicted_min = None
    first_crosses_ahead = False
    second_crosses_ahead = False

    if first_transition.reachable and second_transition.reachable:
        transition_clearance = evaluate_environmental_pair_transition_clearance(
            first_transition,
            second_transition,
            config.pair_clearance,
            first_domain=domains[0] if domains[0] is not None and domains[1] is not None else None,
            second_domain=domains[1] if domains[0] is not None and domains[1] is not None else None,
        )
        predicted_min = transition_clearance.centre.conservative_lower_bound_nm
        if not transition_clearance.certified_clear:
            pair_reasons.append(EnvironmentalPairCandidateRejection.TRANSITION_CLEARANCE)

        transition_end = max(
            first_transition.transition_duration_min,
            second_transition.transition_duration_min,
        )
        horizon = config.pair_clearance.evaluation_horizon_min
        if transition_end < horizon - _EPS:
            # ``transition_end`` is the later of the two command-completion times.
            # The earlier-completing vessel must continue on its reached target command
            # until that common boundary; ``EnvironmentalCandidateTransition.state_at``
            # intentionally stops at its own completion time and therefore cannot be
            # used for this continuation interval.
            first_final_state = first_transition.state_at_or_continue(transition_end)
            second_final_state = second_transition.state_at_or_continue(transition_end)
            first_motion = environmental_vessel_motion_evidence(
                first_final_state, first.environmental_drift_velocity
            )
            second_motion = environmental_vessel_motion_evidence(
                second_final_state, second.environmental_drift_velocity
            )
            remaining = horizon - transition_end
            post_cpa = closest_approach(
                first_motion.to_ground_motion_state(),
                second_motion.to_ground_motion_state(),
                horizon=remaining,
            )
            predicted_min = min(predicted_min, post_cpa.dcpa)

            if domains[0] is not None and domains[1] is not None:
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
                    first_domain=domains[0],
                    second_domain=domains[1],
                    duration_min=remaining,
                    first_speed_bound_nm_per_min=first_motion.speed_over_ground,
                    second_speed_bound_nm_per_min=second_motion.speed_over_ground,
                    first_turn_rate_bound_deg_per_min=0.0,
                    second_turn_rate_bound_deg_per_min=0.0,
                    time_tolerance_min=config.pair_clearance.transition_time_tolerance_min,
                    margin_tolerance_nm=config.pair_clearance.transition_distance_tolerance_nm,
                    max_subdivisions=config.pair_clearance.max_transition_subdivisions,
                )
                if not post_domain.certified_clear:
                    pair_reasons.append(EnvironmentalPairCandidateRejection.POST_TRANSITION_CLEARANCE)
            elif post_cpa.dcpa + _EPS < config.pair_clearance.required_min_separation_nm:
                pair_reasons.append(EnvironmentalPairCandidateRejection.POST_TRANSITION_CLEARANCE)

            if first_intent.avoid_crossing_ahead:
                first_crosses_ahead = _crosses_ahead(first_motion, second_motion, remaining)
                if first_crosses_ahead:
                    first_reasons.append(EnvironmentalPairCandidateRejection.CROSSES_AHEAD)
            if second_intent.avoid_crossing_ahead:
                second_crosses_ahead = _crosses_ahead(second_motion, first_motion, remaining)
                if second_crosses_ahead:
                    second_reasons.append(EnvironmentalPairCandidateRejection.CROSSES_AHEAD)

    return EnvironmentalPairCandidateEvaluation(
        first_intent=first_intent,
        second_intent=second_intent,
        first_command=first_command,
        second_command=second_command,
        first_transition=first_transition,
        second_transition=second_transition,
        first_rejection_reasons=tuple(first_reasons),
        second_rejection_reasons=tuple(second_reasons),
        pair_rejection_reasons=tuple(pair_reasons),
        transition_clearance=transition_clearance,
        post_transition_cpa=post_cpa,
        post_transition_domain_clearance=post_domain,
        predicted_min_separation_nm=predicted_min,
        first_crosses_ahead=first_crosses_ahead,
        second_crosses_ahead=second_crosses_ahead,
    )
