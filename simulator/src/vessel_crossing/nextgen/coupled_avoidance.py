"""Coupled bounded-maneuver screening for a two-vessel encounter.

``avoidance.py`` screens one acting vessel while the counterpart holds its current
course and speed.  That assumption is correct for ordinary give-way / stand-on
candidate screening, but it is not sufficient once both vessels may act at the same
time (Rule 14 mutual action or Rule 17 escalation).

This module therefore evaluates *pairs* of already-declarative maneuver candidates on
one common time axis.  Both vessels follow their own bounded ``ManeuverCommand`` with
their own ``ManeuverLimits``.  Minimum separation during either transition is bounded
conservatively using exact state queries plus a relative-speed Lipschitz bound.  Once
both transitions are complete, the remaining interval is evaluated with analytical
CPA/TCPA.

The module deliberately does not optimize or automatically select a pair.  It returns
all pair evaluations in the deterministic Cartesian-product order induced by the two
caller-supplied candidate grids.  A later arbitration layer may apply explicit policy or
cost criteria without hiding them inside the safety screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from ..cpa import ClosestApproach
from ..kinematics import norm, subtract
from .avoidance import (
    AvoidanceCandidate,
    AvoidanceCandidateConfig,
    AvoidanceRejection,
    generate_avoidance_candidates,
)
from .domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from .dynamics import (
    ManeuverLimits,
    Vessel2DState,
    advance_toward_command,
    command_transition_duration,
)
from .encounter import predict_encounter
from .maneuver_intent import ManeuverIntent
from .overtaking_progress import Rule13ProgressAssessment, assess_rule13_past_progress
from .ship_domain import ShipDomain

_EPS = 1e-10




def _relax_progress_capable_rule13_continuation(
    *,
    vessel_id: str,
    candidate: AvoidanceCandidate,
    intent: ManeuverIntent,
    reasons: list[AvoidanceRejection],
    progress: Rule13ProgressAssessment | None,
    progress_screen_enabled: bool,
) -> bool:
    """Treat a proven continuing Rule-13 action as action, not a fresh no-op.

    The unilateral semantic screen correctly rejects an exact hold for a *new*
    action-required give-way intent.  During a persisted Rule-13 episode, however, a
    later review may find that the currently commanded state already provides positive
    overtaking progress.  When the explicit a55 progress contract is enabled, that
    measurable physical evidence may satisfy the narrow continuation case without
    demanding another artificial course/speed increment.

    Only the two fresh-action diagnostics produced by an exact hold are relaxable.
    Any direction, reachability, or other semantic failure remains fail-closed.
    """

    if not progress_screen_enabled or progress is None or not progress.capable:
        return False
    if progress.overtaking_vessel_id != vessel_id or not intent.rule13_past_progress_required:
        return False
    if not candidate.is_hold:
        return False
    relaxable = {AvoidanceRejection.NO_EFFECTIVE_CHANGE, AvoidanceRejection.NOT_SUBSTANTIAL}
    if not reasons or any(reason not in relaxable for reason in reasons):
        return False
    reasons.clear()
    return True

class CoupledAvoidanceRejection(str, Enum):
    TRANSITION_CLEARANCE = "transition_clearance"
    POST_TRANSITION_CLEARANCE = "post_transition_clearance"
    FIRST_CROSSES_AHEAD = "first_crosses_ahead"
    SECOND_CROSSES_AHEAD = "second_crosses_ahead"
    RULE13_NO_PAST_PROGRESS = "rule13_no_past_progress"


@dataclass(frozen=True, slots=True)
class CoupledAvoidanceConfig:
    """Pair-level safety horizon and numerical tolerances.

    These values are simulation parameters, not COLREG constants.  Candidate-grid and
    substantialness semantics remain in each vessel's ``AvoidanceCandidateConfig``;
    this object governs only the common two-vessel physical safety evaluation.
    """

    evaluation_horizon_min: float
    required_min_separation_nm: float
    transition_time_tolerance_min: float = 0.01
    transition_distance_tolerance_nm: float = 1e-4
    max_transition_subdivisions: int = 20
    rule13_min_past_progress_rate_nm_per_min: float | None = None

    def __post_init__(self) -> None:
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
        if self.rule13_min_past_progress_rate_nm_per_min is not None:
            value = self.rule13_min_past_progress_rate_nm_per_min
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    "rule13_min_past_progress_rate_nm_per_min must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class CoupledTransitionClearance:
    evaluated_duration_min: float
    first_transition_duration_min: float
    second_transition_duration_min: float
    sampled_min_separation_nm: float
    conservative_lower_bound_nm: float
    state_evaluations: int
    both_completed_within_horizon: bool


@dataclass(frozen=True, slots=True)
class CoupledAvoidanceEvaluation:
    first_candidate: AvoidanceCandidate
    second_candidate: AvoidanceCandidate
    accepted: bool
    first_rejection_reasons: tuple[AvoidanceRejection, ...]
    second_rejection_reasons: tuple[AvoidanceRejection, ...]
    coupled_rejection_reasons: tuple[CoupledAvoidanceRejection, ...]
    transition: CoupledTransitionClearance | None
    post_transition_cpa: ClosestApproach | None
    post_transition_tcpa_from_now_min: float | None
    predicted_min_separation_nm: float | None
    first_crosses_ahead: bool
    second_crosses_ahead: bool
    transition_domain_clearance: DomainClearanceCertification | None = None
    post_transition_domain_clearance: DomainClearanceCertification | None = None
    rule13_progress: Rule13ProgressAssessment | None = None

    @property
    def minimum_certified_domain_margin_nm(self) -> float | None:
        values = [
            item.conservative_lower_bound_nm
            for item in (self.transition_domain_clearance, self.post_transition_domain_clearance)
            if item is not None
        ]
        return min(values) if values else None


@dataclass(frozen=True, slots=True)
class CoupledAvoidanceGrid:
    first_intent: ManeuverIntent
    second_intent: ManeuverIntent
    evaluations: tuple[CoupledAvoidanceEvaluation, ...]

    @property
    def safe_pairs(self) -> tuple[CoupledAvoidanceEvaluation, ...]:
        return tuple(item for item in self.evaluations if item.accepted)

    @property
    def has_safe_pair(self) -> bool:
        return any(item.accepted for item in self.evaluations)


def _separation_at(
    first: Vessel2DState,
    second: Vessel2DState,
    first_candidate: AvoidanceCandidate,
    second_candidate: AvoidanceCandidate,
    first_limits: ManeuverLimits,
    second_limits: ManeuverLimits,
    time_min: float,
) -> float:
    first_at = advance_toward_command(
        first,
        first_candidate.command,
        time_min,
        first_limits,
    ).state
    second_at = advance_toward_command(
        second,
        second_candidate.command,
        time_min,
        second_limits,
    ).state
    return norm(subtract(second_at.position, first_at.position))


def _coupled_transition_clearance(
    first: Vessel2DState,
    second: Vessel2DState,
    first_candidate: AvoidanceCandidate,
    second_candidate: AvoidanceCandidate,
    first_limits: ManeuverLimits,
    second_limits: ManeuverLimits,
    config: CoupledAvoidanceConfig,
    first_transition_duration: float,
    second_transition_duration: float,
) -> CoupledTransitionClearance:
    transition_end = max(first_transition_duration, second_transition_duration)
    duration = min(transition_end, config.evaluation_horizon_min)

    if duration <= _EPS:
        separation = norm(subtract(second.position, first.position))
        return CoupledTransitionClearance(
            evaluated_duration_min=0.0,
            first_transition_duration_min=first_transition_duration,
            second_transition_duration_min=second_transition_duration,
            sampled_min_separation_nm=separation,
            conservative_lower_bound_nm=separation,
            state_evaluations=1,
            both_completed_within_horizon=True,
        )

    cache: dict[float, float] = {}

    def distance(time_min: float) -> float:
        key = round(time_min, 14)
        if key not in cache:
            cache[key] = _separation_at(
                first,
                second,
                first_candidate,
                second_candidate,
                first_limits,
                second_limits,
                time_min,
            )
        return cache[key]

    # Each vessel's speed changes monotonically toward its target without overshoot.
    # The sum of their maximum speed magnitudes is therefore a valid global Lipschitz
    # bound for pair separation over the bounded transition interval.
    relative_speed_bound = max(first.speed, first_candidate.command.target_speed) + max(
        second.speed,
        second_candidate.command.target_speed,
    )
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

    conservative_lower = recurse(0.0, duration, 0)
    return CoupledTransitionClearance(
        evaluated_duration_min=duration,
        first_transition_duration_min=first_transition_duration,
        second_transition_duration_min=second_transition_duration,
        sampled_min_separation_nm=best_sample[0],
        conservative_lower_bound_nm=conservative_lower,
        state_evaluations=len(cache),
        both_completed_within_horizon=transition_end <= config.evaluation_horizon_min + _EPS,
    )


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _crosses_ahead_on_constant_tracks(
    own: Vessel2DState,
    counterpart: Vessel2DState,
    horizon_min: float,
) -> bool:
    """Return whether constant post-transition rays cross with own arriving first."""
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


def evaluate_coupled_avoidance_candidates(
    first: Vessel2DState,
    second: Vessel2DState,
    first_limits: ManeuverLimits,
    second_limits: ManeuverLimits,
    first_intent: ManeuverIntent,
    second_intent: ManeuverIntent,
    first_candidate_config: AvoidanceCandidateConfig,
    second_candidate_config: AvoidanceCandidateConfig,
    coupled_config: CoupledAvoidanceConfig,
    *,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> CoupledAvoidanceGrid:
    """Evaluate the deterministic Cartesian product of two candidate grids.

    No pair is automatically selected.  ``safe_pairs`` preserves the generated order
    so a later arbiter can apply an explicit preference/cost rule.
    """
    if first.vessel_id != first_intent.vessel_id:
        raise ValueError("first state does not match first_intent vessel_id")
    if second.vessel_id != second_intent.vessel_id:
        raise ValueError("second state does not match second_intent vessel_id")
    if first_intent.counterpart_id != second.vessel_id:
        raise ValueError("first_intent counterpart_id does not match second state")
    if second_intent.counterpart_id != first.vessel_id:
        raise ValueError("second_intent counterpart_id does not match first state")
    if (first_domain is None) != (second_domain is None):
        raise ValueError("first_domain and second_domain must be supplied together")

    first_generated = generate_avoidance_candidates(
        first,
        second,
        first_intent,
        first_candidate_config,
    )
    second_generated = generate_avoidance_candidates(
        second,
        first,
        second_intent,
        second_candidate_config,
    )

    evaluations: list[CoupledAvoidanceEvaluation] = []

    for first_candidate, first_semantic in first_generated:
        for second_candidate, second_semantic in second_generated:
            first_reasons = list(first_semantic)
            second_reasons = list(second_semantic)
            coupled_reasons: list[CoupledAvoidanceRejection] = []
            transition: CoupledTransitionClearance | None = None
            post_cpa: ClosestApproach | None = None
            post_tcpa_from_now: float | None = None
            predicted_min: float | None = None
            first_crosses_ahead = False
            second_crosses_ahead = False
            transition_domain: DomainClearanceCertification | None = None
            post_domain: DomainClearanceCertification | None = None
            rule13_progress: Rule13ProgressAssessment | None = None

            first_duration = command_transition_duration(first, first_candidate.command, first_limits)
            second_duration = command_transition_duration(second, second_candidate.command, second_limits)
            if not isfinite(first_duration):
                first_reasons.append(AvoidanceRejection.PHYSICALLY_UNREACHABLE)
            if not isfinite(second_duration):
                second_reasons.append(AvoidanceRejection.PHYSICALLY_UNREACHABLE)

            if isfinite(first_duration) and isfinite(second_duration):
                transition = _coupled_transition_clearance(
                    first,
                    second,
                    first_candidate,
                    second_candidate,
                    first_limits,
                    second_limits,
                    coupled_config,
                    first_duration,
                    second_duration,
                )
                predicted_min = transition.conservative_lower_bound_nm
                if first_domain is not None and second_domain is not None:
                    transition_domain = certify_pair_domain_clearance(
                        first_state_at=lambda t: advance_toward_command(
                            first, first_candidate.command, t, first_limits
                        ).state,
                        second_state_at=lambda t: advance_toward_command(
                            second, second_candidate.command, t, second_limits
                        ).state,
                        first_domain=first_domain,
                        second_domain=second_domain,
                        duration_min=min(
                            max(first_duration, second_duration),
                            coupled_config.evaluation_horizon_min,
                        ),
                        first_speed_bound_nm_per_min=max(
                            first.speed, first_candidate.command.target_speed
                        ),
                        second_speed_bound_nm_per_min=max(
                            second.speed, second_candidate.command.target_speed
                        ),
                        first_turn_rate_bound_deg_per_min=first_limits.max_turn_rate_deg_per_min,
                        second_turn_rate_bound_deg_per_min=second_limits.max_turn_rate_deg_per_min,
                        time_tolerance_min=coupled_config.transition_time_tolerance_min,
                        margin_tolerance_nm=coupled_config.transition_distance_tolerance_nm,
                        max_subdivisions=coupled_config.max_transition_subdivisions,
                    )
                    if not transition_domain.certified_clear:
                        coupled_reasons.append(CoupledAvoidanceRejection.TRANSITION_CLEARANCE)
                elif (
                    transition.conservative_lower_bound_nm + _EPS
                    < coupled_config.required_min_separation_nm
                ):
                    coupled_reasons.append(CoupledAvoidanceRejection.TRANSITION_CLEARANCE)

                transition_end = max(first_duration, second_duration)
                if coupled_config.rule13_min_past_progress_rate_nm_per_min is not None:
                    first_target_state = advance_toward_command(
                        first, first_candidate.command, transition_end, first_limits
                    ).state
                    second_target_state = advance_toward_command(
                        second, second_candidate.command, transition_end, second_limits
                    ).state
                    rule13_progress = assess_rule13_past_progress(
                        first_target_state,
                        second_target_state,
                        first_intent,
                        second_intent,
                        minimum_required_rate_nm_per_min=(
                            coupled_config.rule13_min_past_progress_rate_nm_per_min
                        ),
                    )
                    if rule13_progress is not None and not rule13_progress.capable:
                        coupled_reasons.append(CoupledAvoidanceRejection.RULE13_NO_PAST_PROGRESS)
                    else:
                        _relax_progress_capable_rule13_continuation(
                            vessel_id=first.vessel_id,
                            candidate=first_candidate,
                            intent=first_intent,
                            reasons=first_reasons,
                            progress=rule13_progress,
                            progress_screen_enabled=True,
                        )
                        _relax_progress_capable_rule13_continuation(
                            vessel_id=second.vessel_id,
                            candidate=second_candidate,
                            intent=second_intent,
                            reasons=second_reasons,
                            progress=rule13_progress,
                            progress_screen_enabled=True,
                        )

                if transition_end < coupled_config.evaluation_horizon_min - _EPS:
                    first_final = advance_toward_command(
                        first,
                        first_candidate.command,
                        transition_end,
                        first_limits,
                    ).state
                    second_final = advance_toward_command(
                        second,
                        second_candidate.command,
                        transition_end,
                        second_limits,
                    ).state
                    remaining = coupled_config.evaluation_horizon_min - transition_end
                    post_prediction = predict_encounter(first_final, second_final, horizon=remaining)
                    post_cpa = post_prediction.cpa
                    post_tcpa_from_now = transition_end + post_cpa.tcpa
                    predicted_min = min(predicted_min, post_cpa.dcpa)
                    if first_domain is not None and second_domain is not None:
                        def first_post(t: float) -> Vessel2DState:
                            vx, vy = first_final.velocity
                            return Vessel2DState(
                                vessel_id=first_final.vessel_id,
                                position=(first_final.position[0] + vx * t, first_final.position[1] + vy * t),
                                heading_deg=first_final.heading_deg,
                                speed=first_final.speed,
                            )

                        def second_post(t: float) -> Vessel2DState:
                            vx, vy = second_final.velocity
                            return Vessel2DState(
                                vessel_id=second_final.vessel_id,
                                position=(second_final.position[0] + vx * t, second_final.position[1] + vy * t),
                                heading_deg=second_final.heading_deg,
                                speed=second_final.speed,
                            )

                        post_domain = certify_pair_domain_clearance(
                            first_state_at=first_post,
                            second_state_at=second_post,
                            first_domain=first_domain,
                            second_domain=second_domain,
                            duration_min=remaining,
                            first_speed_bound_nm_per_min=first_final.speed,
                            second_speed_bound_nm_per_min=second_final.speed,
                            first_turn_rate_bound_deg_per_min=0.0,
                            second_turn_rate_bound_deg_per_min=0.0,
                            time_tolerance_min=coupled_config.transition_time_tolerance_min,
                            margin_tolerance_nm=coupled_config.transition_distance_tolerance_nm,
                            max_subdivisions=coupled_config.max_transition_subdivisions,
                        )
                        if not post_domain.certified_clear:
                            coupled_reasons.append(CoupledAvoidanceRejection.POST_TRANSITION_CLEARANCE)
                    elif post_cpa.dcpa + _EPS < coupled_config.required_min_separation_nm:
                        coupled_reasons.append(CoupledAvoidanceRejection.POST_TRANSITION_CLEARANCE)

                    if first_intent.avoid_crossing_ahead:
                        first_crosses_ahead = _crosses_ahead_on_constant_tracks(
                            first_final,
                            second_final,
                            remaining,
                        )
                        if first_crosses_ahead:
                            coupled_reasons.append(CoupledAvoidanceRejection.FIRST_CROSSES_AHEAD)

                    if second_intent.avoid_crossing_ahead:
                        second_crosses_ahead = _crosses_ahead_on_constant_tracks(
                            second_final,
                            first_final,
                            remaining,
                        )
                        if second_crosses_ahead:
                            coupled_reasons.append(CoupledAvoidanceRejection.SECOND_CROSSES_AHEAD)

            first_deduped = tuple(dict.fromkeys(first_reasons))
            second_deduped = tuple(dict.fromkeys(second_reasons))
            coupled_deduped = tuple(dict.fromkeys(coupled_reasons))
            evaluations.append(
                CoupledAvoidanceEvaluation(
                    first_candidate=first_candidate,
                    second_candidate=second_candidate,
                    accepted=not first_deduped and not second_deduped and not coupled_deduped,
                    first_rejection_reasons=first_deduped,
                    second_rejection_reasons=second_deduped,
                    coupled_rejection_reasons=coupled_deduped,
                    transition=transition,
                    post_transition_cpa=post_cpa,
                    post_transition_tcpa_from_now_min=post_tcpa_from_now,
                    predicted_min_separation_nm=predicted_min,
                    first_crosses_ahead=first_crosses_ahead,
                    second_crosses_ahead=second_crosses_ahead,
                    transition_domain_clearance=transition_domain,
                    post_transition_domain_clearance=post_domain,
                    rule13_progress=rule13_progress,
                )
            )

    return CoupledAvoidanceGrid(
        first_intent=first_intent,
        second_intent=second_intent,
        evaluations=tuple(evaluations),
    )
