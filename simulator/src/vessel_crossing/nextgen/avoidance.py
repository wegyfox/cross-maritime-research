"""Physical avoidance-candidate generation and conservative screening.

This layer translates a declarative :class:`ManeuverIntent` into candidate
``ManeuverCommand`` objects, but it does **not** execute the selected command in
``World2D``.  Candidate safety is screened over a finite horizon while respecting
bounded turn/acceleration dynamics for the acting vessel and assuming the counterpart
maintains its current course and speed.

The bounded maneuver transition is not replaced by an instantaneous velocity jump.
Minimum separation during that transition is bounded conservatively using exact state
queries from ``advance_toward_command`` plus a relative-speed Lipschitz bound.  The
remaining post-transition interval uses the already validated analytical CPA primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from ..cpa import ClosestApproach
from ..kinematics import norm, subtract
from .domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from .dynamics import (
    ManeuverCommand,
    ManeuverLimits,
    Vessel2DState,
    advance_toward_command,
    command_transition_duration,
    normalize_heading,
)
from .encounter import predict_encounter, relative_bearing_deg
from .maneuver_intent import ManeuverIntent, ManeuverIntentKind
from .ship_domain import ShipDomain

_EPS = 1e-10


class AvoidanceRejection(str, Enum):
    NO_EFFECTIVE_CHANGE = "no_effective_change"
    WRONG_COURSE_DIRECTION = "wrong_course_direction"
    PORT_ALTERATION_RESTRICTED = "port_alteration_restricted"
    NOT_SUBSTANTIAL = "not_substantial"
    PHYSICALLY_UNREACHABLE = "physically_unreachable"
    TRANSITION_CLEARANCE = "transition_clearance"
    POST_TRANSITION_CLEARANCE = "post_transition_clearance"
    CROSSES_AHEAD = "crosses_ahead"


class NoSafeAvoidanceCandidate(RuntimeError):
    """Raised when a caller requires a selected command but none passed screening."""


@dataclass(frozen=True, slots=True)
class AvoidanceCandidateConfig:
    """Explicit candidate grid, policy thresholds, and numerical tolerances.

    Course changes are signed marine-heading changes in degrees: positive is
    starboard, negative is port.  Speed fractions multiply the current acting-vessel
    speed.  The grid and safety/substantialness thresholds are simulation parameters,
    not regulatory constants.
    """

    course_deltas_deg: tuple[float, ...]
    speed_fractions: tuple[float, ...]
    evaluation_horizon_min: float
    required_min_separation_nm: float
    substantial_course_change_deg: float = 0.0
    substantial_speed_change_fraction: float = 0.0
    transition_time_tolerance_min: float = 0.01
    transition_distance_tolerance_nm: float = 1e-4
    max_transition_subdivisions: int = 20

    def __post_init__(self) -> None:
        if not self.course_deltas_deg:
            raise ValueError("course_deltas_deg must be non-empty")
        if not self.speed_fractions:
            raise ValueError("speed_fractions must be non-empty")
        if any(delta <= -180.0 or delta >= 180.0 for delta in self.course_deltas_deg):
            raise ValueError("course deltas must lie strictly between -180 and 180 degrees")
        if any(factor < 0.0 for factor in self.speed_fractions):
            raise ValueError("speed fractions must be non-negative")
        if self.evaluation_horizon_min <= 0.0:
            raise ValueError("evaluation_horizon_min must be positive")
        if self.required_min_separation_nm < 0.0:
            raise ValueError("required_min_separation_nm must be non-negative")
        if self.substantial_course_change_deg < 0.0:
            raise ValueError("substantial_course_change_deg must be non-negative")
        if self.substantial_speed_change_fraction < 0.0:
            raise ValueError("substantial_speed_change_fraction must be non-negative")
        if self.transition_time_tolerance_min <= 0.0:
            raise ValueError("transition_time_tolerance_min must be positive")
        if self.transition_distance_tolerance_nm < 0.0:
            raise ValueError("transition_distance_tolerance_nm must be non-negative")
        if self.max_transition_subdivisions < 1:
            raise ValueError("max_transition_subdivisions must be at least 1")


@dataclass(frozen=True, slots=True)
class AvoidanceCandidate:
    command: ManeuverCommand
    course_delta_deg: float
    speed_fraction: float

    @property
    def changes_course(self) -> bool:
        return abs(self.course_delta_deg) > _EPS

    @property
    def changes_speed(self) -> bool:
        return abs(self.speed_fraction - 1.0) > _EPS

    @property
    def is_hold(self) -> bool:
        return not self.changes_course and not self.changes_speed


@dataclass(frozen=True, slots=True)
class TransitionClearance:
    evaluated_duration_min: float
    command_transition_duration_min: float
    sampled_min_separation_nm: float
    conservative_lower_bound_nm: float
    state_evaluations: int
    completed_within_horizon: bool


@dataclass(frozen=True, slots=True)
class AvoidanceCandidateEvaluation:
    candidate: AvoidanceCandidate
    accepted: bool
    rejection_reasons: tuple[AvoidanceRejection, ...]
    transition: TransitionClearance | None
    post_transition_cpa: ClosestApproach | None
    post_transition_tcpa_from_now_min: float | None
    predicted_min_separation_nm: float | None
    crosses_ahead: bool
    transition_domain_clearance: DomainClearanceCertification | None = None
    post_transition_domain_clearance: DomainClearanceCertification | None = None


@dataclass(frozen=True, slots=True)
class AvoidanceSelection:
    intent: ManeuverIntent
    evaluations: tuple[AvoidanceCandidateEvaluation, ...]
    selected: AvoidanceCandidateEvaluation | None

    @property
    def has_safe_candidate(self) -> bool:
        return self.selected is not None

    def require_selected(self) -> AvoidanceCandidateEvaluation:
        if self.selected is None:
            raise NoSafeAvoidanceCandidate(
                f"no candidate satisfied avoidance intent for vessel {self.intent.vessel_id!r}"
            )
        return self.selected


def _counterpart_at(counterpart: Vessel2DState, time_min: float) -> Vessel2DState:
    vx, vy = counterpart.velocity
    return Vessel2DState(
        vessel_id=counterpart.vessel_id,
        position=(
            counterpart.position[0] + vx * time_min,
            counterpart.position[1] + vy * time_min,
        ),
        heading_deg=counterpart.heading_deg,
        speed=counterpart.speed,
    )


def _separation_at(
    own: Vessel2DState,
    counterpart: Vessel2DState,
    command: ManeuverCommand,
    limits: ManeuverLimits,
    time_min: float,
) -> float:
    own_at = advance_toward_command(own, command, time_min, limits).state
    counterpart_at = _counterpart_at(counterpart, time_min)
    return norm(subtract(counterpart_at.position, own_at.position))


def _transition_clearance(
    own: Vessel2DState,
    counterpart: Vessel2DState,
    command: ManeuverCommand,
    limits: ManeuverLimits,
    config: AvoidanceCandidateConfig,
    transition_duration: float,
) -> TransitionClearance:
    duration = min(transition_duration, config.evaluation_horizon_min)
    if duration <= _EPS:
        separation = norm(subtract(counterpart.position, own.position))
        return TransitionClearance(
            evaluated_duration_min=0.0,
            command_transition_duration_min=transition_duration,
            sampled_min_separation_nm=separation,
            conservative_lower_bound_nm=separation,
            state_evaluations=1,
            completed_within_horizon=True,
        )

    cache: dict[float, float] = {}

    def distance(time_min: float) -> float:
        key = round(time_min, 14)
        if key not in cache:
            cache[key] = _separation_at(own, counterpart, command, limits, time_min)
        return cache[key]

    # Own speed changes monotonically toward target without overshoot, so this is a
    # valid speed bound for the entire transition.  Distance between two moving points
    # is therefore Lipschitz with constant <= sum of their speed magnitudes.
    relative_speed_bound = max(own.speed, command.target_speed) + counterpart.speed
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

        # If this interval cannot improve the sampled global minimum by more than the
        # configured distance tolerance, its valid lower bound is already sufficient.
        if lower_bound >= best_sample[0] - config.transition_distance_tolerance_nm:
            return lower_bound
        if width <= config.transition_time_tolerance_min:
            return lower_bound
        if depth >= config.max_transition_subdivisions:
            return lower_bound

        return min(recurse(a, mid, depth + 1), recurse(mid, b, depth + 1))

    conservative_lower = recurse(0.0, duration, 0)
    return TransitionClearance(
        evaluated_duration_min=duration,
        command_transition_duration_min=transition_duration,
        sampled_min_separation_nm=best_sample[0],
        conservative_lower_bound_nm=conservative_lower,
        state_evaluations=len(cache),
        completed_within_horizon=transition_duration <= config.evaluation_horizon_min + _EPS,
    )


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _crosses_ahead_on_post_transition_tracks(
    own: Vessel2DState,
    counterpart: Vessel2DState,
    horizon_min: float,
) -> bool:
    """Return whether straight post-transition rays cross with own arriving first.

    This is an intentionally explicit screening approximation for Rule-15's
    ``avoid crossing ahead`` constraint.  It considers the constant-course/speed rays
    after the acting vessel has completed the candidate transition.  Full curved-path
    COLREG supervision remains a later layer.
    """
    if horizon_min <= _EPS or own.speed <= _EPS or counterpart.speed <= _EPS:
        return False

    own_v = own.velocity
    other_v = counterpart.velocity
    denominator = _cross(own_v, other_v)
    if abs(denominator) <= _EPS:
        return False

    relative = subtract(counterpart.position, own.position)
    own_time = _cross(relative, other_v) / denominator
    other_time = _cross(relative, own_v) / denominator
    if own_time < -_EPS or other_time < -_EPS:
        return False
    if own_time > horizon_min + _EPS or other_time > horizon_min + _EPS:
        return False
    return own_time + _EPS < other_time


def _substantial(candidate: AvoidanceCandidate, config: AvoidanceCandidateConfig) -> bool:
    if candidate.is_hold:
        return False
    course_substantial = (
        candidate.changes_course
        and abs(candidate.course_delta_deg) + _EPS >= config.substantial_course_change_deg
    )
    speed_substantial = (
        candidate.changes_speed
        and abs(candidate.speed_fraction - 1.0) + _EPS
        >= config.substantial_speed_change_fraction
    )
    return course_substantial or speed_substantial


def _candidate_semantic_rejections(
    candidate: AvoidanceCandidate,
    own: Vessel2DState,
    counterpart: Vessel2DState,
    intent: ManeuverIntent,
    config: AvoidanceCandidateConfig,
) -> list[AvoidanceRejection]:
    reasons: list[AvoidanceRejection] = []

    requires_effective_change = intent.kind in {
        ManeuverIntentKind.KEEP_WELL_CLEAR,
        ManeuverIntentKind.ALTER_COURSE_STARBOARD,
        ManeuverIntentKind.STAND_ON_MUST_TAKE_AVOIDING_ACTION,
    }
    if requires_effective_change and candidate.is_hold:
        reasons.append(AvoidanceRejection.NO_EFFECTIVE_CHANGE)

    if intent.required_course_direction == "starboard" and candidate.course_delta_deg <= _EPS:
        reasons.append(AvoidanceRejection.WRONG_COURSE_DIRECTION)
    elif intent.required_course_direction == "port" and candidate.course_delta_deg >= -_EPS:
        reasons.append(AvoidanceRejection.WRONG_COURSE_DIRECTION)

    if intent.avoid_port_alteration_for_target_on_port:
        target_bearing = relative_bearing_deg(own, counterpart)
        target_on_port = 180.0 + _EPS < target_bearing < 360.0 - _EPS
        if target_on_port and candidate.course_delta_deg < -_EPS:
            reasons.append(AvoidanceRejection.PORT_ALTERATION_RESTRICTED)

    if intent.early_and_substantial and not _substantial(candidate, config):
        reasons.append(AvoidanceRejection.NOT_SUBSTANTIAL)

    return reasons


def generate_avoidance_candidates(
    own: Vessel2DState,
    counterpart: Vessel2DState,
    intent: ManeuverIntent,
    config: AvoidanceCandidateConfig,
) -> tuple[tuple[AvoidanceCandidate, tuple[AvoidanceRejection, ...]], ...]:
    """Generate a deterministic candidate grid and annotate semantic rejections."""
    if own.vessel_id != intent.vessel_id:
        raise ValueError("own state does not match maneuver intent vessel_id")
    if counterpart.vessel_id != intent.counterpart_id:
        raise ValueError("counterpart state does not match maneuver intent counterpart_id")

    if intent.kind is ManeuverIntentKind.NONE:
        return ()

    # Normal Rule-17 stand-on behavior is deliberately immutable at this layer.
    if intent.maintain_course_and_speed and not intent.action_permitted:
        candidate = AvoidanceCandidate(
            command=ManeuverCommand(own.heading_deg, own.speed),
            course_delta_deg=0.0,
            speed_fraction=1.0,
        )
        return ((candidate, ()),)

    unique: dict[tuple[float, float], AvoidanceCandidate] = {}

    # Rule-17(a)(ii) is permissive.  Holding course/speed must remain an available
    # option even when the caller's avoidance grid contains only actual alterations.
    if intent.kind is ManeuverIntentKind.STAND_ON_MAY_TAKE_AVOIDING_ACTION:
        unique[(round(own.heading_deg, 12), round(own.speed, 12))] = AvoidanceCandidate(
            command=ManeuverCommand(own.heading_deg, own.speed),
            course_delta_deg=0.0,
            speed_fraction=1.0,
        )

    for delta in config.course_deltas_deg:
        for factor in config.speed_fractions:
            target_heading = normalize_heading(own.heading_deg + delta)
            target_speed = own.speed * factor
            key = (round(target_heading, 12), round(target_speed, 12))
            unique.setdefault(
                key,
                AvoidanceCandidate(
                    command=ManeuverCommand(target_heading, target_speed),
                    course_delta_deg=delta,
                    speed_fraction=factor,
                ),
            )

    annotated = []
    for candidate in unique.values():
        reasons = tuple(_candidate_semantic_rejections(candidate, own, counterpart, intent, config))
        annotated.append((candidate, reasons))
    return tuple(annotated)


def evaluate_avoidance_candidates(
    own: Vessel2DState,
    counterpart: Vessel2DState,
    limits: ManeuverLimits,
    intent: ManeuverIntent,
    config: AvoidanceCandidateConfig,
    *,
    own_domain: ShipDomain | None = None,
    counterpart_domain: ShipDomain | None = None,
) -> AvoidanceSelection:
    """Generate, physically screen, and deterministically select an avoidance candidate.

    When both ship domains are supplied, transition and post-transition safety are
    certified against oriented domain margin instead of the scalar separation threshold.
    Both domains must be supplied together; omitting both preserves the historical a12
    circular/scalar screening semantics.
    """
    if (own_domain is None) != (counterpart_domain is None):
        raise ValueError("own_domain and counterpart_domain must be supplied together")
    generated = generate_avoidance_candidates(own, counterpart, intent, config)
    evaluations: list[AvoidanceCandidateEvaluation] = []

    for candidate, semantic_rejections in generated:
        reasons = list(semantic_rejections)
        transition: TransitionClearance | None = None
        post_cpa: ClosestApproach | None = None
        post_tcpa_from_now: float | None = None
        predicted_min: float | None = None
        crosses_ahead = False
        transition_domain: DomainClearanceCertification | None = None
        post_domain: DomainClearanceCertification | None = None

        transition_duration = command_transition_duration(own, candidate.command, limits)
        if not isfinite(transition_duration):
            reasons.append(AvoidanceRejection.PHYSICALLY_UNREACHABLE)
        else:
            transition = _transition_clearance(
                own,
                counterpart,
                candidate.command,
                limits,
                config,
                transition_duration,
            )
            predicted_min = transition.conservative_lower_bound_nm
            if own_domain is not None and counterpart_domain is not None:
                transition_domain = certify_pair_domain_clearance(
                    first_state_at=lambda t: advance_toward_command(
                        own, candidate.command, t, limits
                    ).state,
                    second_state_at=lambda t: _counterpart_at(counterpart, t),
                    first_domain=own_domain,
                    second_domain=counterpart_domain,
                    duration_min=min(transition_duration, config.evaluation_horizon_min),
                    first_speed_bound_nm_per_min=max(own.speed, candidate.command.target_speed),
                    second_speed_bound_nm_per_min=counterpart.speed,
                    first_turn_rate_bound_deg_per_min=limits.max_turn_rate_deg_per_min,
                    second_turn_rate_bound_deg_per_min=0.0,
                    time_tolerance_min=config.transition_time_tolerance_min,
                    margin_tolerance_nm=config.transition_distance_tolerance_nm,
                    max_subdivisions=config.max_transition_subdivisions,
                )
                if not transition_domain.certified_clear:
                    reasons.append(AvoidanceRejection.TRANSITION_CLEARANCE)
            elif transition.conservative_lower_bound_nm + _EPS < config.required_min_separation_nm:
                reasons.append(AvoidanceRejection.TRANSITION_CLEARANCE)

            if transition_duration < config.evaluation_horizon_min - _EPS:
                own_final = advance_toward_command(own, candidate.command, transition_duration, limits).state
                counterpart_final = _counterpart_at(counterpart, transition_duration)
                remaining = config.evaluation_horizon_min - transition_duration
                post_prediction = predict_encounter(own_final, counterpart_final, horizon=remaining)
                post_cpa = post_prediction.cpa
                post_tcpa_from_now = transition_duration + post_cpa.tcpa
                predicted_min = min(predicted_min, post_cpa.dcpa)
                if own_domain is not None and counterpart_domain is not None:
                    def own_post(t: float) -> Vessel2DState:
                        vx, vy = own_final.velocity
                        return Vessel2DState(
                            vessel_id=own_final.vessel_id,
                            position=(own_final.position[0] + vx * t, own_final.position[1] + vy * t),
                            heading_deg=own_final.heading_deg,
                            speed=own_final.speed,
                        )

                    def counterpart_post(t: float) -> Vessel2DState:
                        vx, vy = counterpart_final.velocity
                        return Vessel2DState(
                            vessel_id=counterpart_final.vessel_id,
                            position=(
                                counterpart_final.position[0] + vx * t,
                                counterpart_final.position[1] + vy * t,
                            ),
                            heading_deg=counterpart_final.heading_deg,
                            speed=counterpart_final.speed,
                        )

                    post_domain = certify_pair_domain_clearance(
                        first_state_at=own_post,
                        second_state_at=counterpart_post,
                        first_domain=own_domain,
                        second_domain=counterpart_domain,
                        duration_min=remaining,
                        first_speed_bound_nm_per_min=own_final.speed,
                        second_speed_bound_nm_per_min=counterpart_final.speed,
                        first_turn_rate_bound_deg_per_min=0.0,
                        second_turn_rate_bound_deg_per_min=0.0,
                        time_tolerance_min=config.transition_time_tolerance_min,
                        margin_tolerance_nm=config.transition_distance_tolerance_nm,
                        max_subdivisions=config.max_transition_subdivisions,
                    )
                    if not post_domain.certified_clear:
                        reasons.append(AvoidanceRejection.POST_TRANSITION_CLEARANCE)
                elif post_cpa.dcpa + _EPS < config.required_min_separation_nm:
                    reasons.append(AvoidanceRejection.POST_TRANSITION_CLEARANCE)

                if intent.avoid_crossing_ahead:
                    crosses_ahead = _crosses_ahead_on_post_transition_tracks(
                        own_final,
                        counterpart_final,
                        remaining,
                    )
                    if crosses_ahead:
                        reasons.append(AvoidanceRejection.CROSSES_AHEAD)

        # Preserve deterministic reason ordering while removing duplicates.
        deduped = tuple(dict.fromkeys(reasons))
        evaluations.append(
            AvoidanceCandidateEvaluation(
                candidate=candidate,
                accepted=not deduped,
                rejection_reasons=deduped,
                transition=transition,
                post_transition_cpa=post_cpa,
                post_transition_tcpa_from_now_min=post_tcpa_from_now,
                predicted_min_separation_nm=predicted_min,
                crosses_ahead=crosses_ahead,
                transition_domain_clearance=transition_domain,
                post_transition_domain_clearance=post_domain,
            )
        )

    accepted = [evaluation for evaluation in evaluations if evaluation.accepted]
    selected: AvoidanceCandidateEvaluation | None = None
    if accepted:
        hold = next((item for item in accepted if item.candidate.is_hold), None)
        if intent.kind is ManeuverIntentKind.STAND_ON_MAY_TAKE_AVOIDING_ACTION and hold is not None:
            # Rule-17(a)(ii) grants permission rather than imposing action.  Preserve
            # course/speed if the hold candidate already satisfies the configured
            # safety screen.
            selected = hold
        elif intent.kind is ManeuverIntentKind.STAND_ON_KEEP_COURSE_SPEED:
            selected = hold
        else:
            # The candidate grid is ordered input.  Choosing the first safe candidate
            # keeps maneuver preference explicit in caller data instead of hiding a
            # weighted or "maximize DCPA" objective inside this safety screen.
            selected = accepted[0]

    return AvoidanceSelection(intent=intent, evaluations=tuple(evaluations), selected=selected)
