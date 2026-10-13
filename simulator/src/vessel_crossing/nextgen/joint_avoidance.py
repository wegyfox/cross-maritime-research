"""Joint bounded-dynamics screening for a compatible multi-vessel encounter component.

Pairwise COLREG responsibilities can share vessels.  Once a26 has shown that the
resulting declarative maneuver intents are mutually compatible, this layer evaluates
explicit caller-supplied physical command tuples on one common time axis.

The evaluator deliberately performs no optimization.  It preserves the deterministic
Cartesian-product order induced by ``ComponentIntentAggregation.vessels`` and each
vessel's ordered command options.  Every unordered vessel pair in the component is
screened, not merely the pairs that currently carry an active COLREG responsibility;
a maneuver chosen to solve A-B and A-C must not create a new unsafe B-C encounter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
from math import isfinite
from typing import Mapping

from ..cpa import ClosestApproach
from ..kinematics import norm, subtract
from .domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from .dynamics import (
    ManeuverCommand,
    ManeuverLimits,
    Vessel2DState,
    advance_toward_command,
    command_transition_duration,
    signed_heading_delta,
)
from .encounter import predict_encounter, relative_bearing_deg
from .maneuver_intent import ManeuverIntent, ManeuverIntentKind
from .multi_intent import ComponentIntentAggregation, VesselIntentAggregate
from .overtaking_progress import Rule13ProgressAssessment, assess_rule13_past_progress
from .ship_domain import ShipDomain
from .rule13_progress_coordination import Rule13ProgressPlan

_EPS = 1e-10


class JointAvoidanceError(RuntimeError):
    """Base error for invalid joint-avoidance requests."""


class IncompatibleJointIntents(JointAvoidanceError):
    """Raised when physical search is attempted for an a26-incompatible intent set."""


class JointAvoidanceUnavailable(JointAvoidanceError):
    """Raised when an explicit joint candidate product exceeds configured resources."""


class JointVesselRejection(str, Enum):
    MAINTAIN_COURSE_SPEED = "maintain_course_speed"
    NO_EFFECTIVE_CHANGE = "no_effective_change"
    WRONG_COURSE_DIRECTION = "wrong_course_direction"
    PORT_ALTERATION_RESTRICTED = "port_alteration_restricted"
    NOT_SUBSTANTIAL = "not_substantial"
    PHYSICALLY_UNREACHABLE = "physically_unreachable"


class JointPairRejection(str, Enum):
    TRANSITION_CLEARANCE = "transition_clearance"
    POST_TRANSITION_CLEARANCE = "post_transition_clearance"
    FIRST_CROSSES_AHEAD = "first_crosses_ahead"
    SECOND_CROSSES_AHEAD = "second_crosses_ahead"
    RULE13_NO_PAST_PROGRESS = "rule13_no_past_progress"


@dataclass(frozen=True, slots=True)
class JointAvoidanceConfig:
    """Common physical horizon, clearance threshold, and explicit model tolerances."""

    evaluation_horizon_min: float
    required_min_separation_nm: float
    substantial_course_change_deg: float = 0.0
    substantial_speed_change_fraction: float = 0.0
    transition_time_tolerance_min: float = 0.01
    transition_distance_tolerance_nm: float = 1e-4
    max_transition_subdivisions: int = 20
    max_combinations: int = 10000
    rule13_min_past_progress_rate_nm_per_min: float | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.evaluation_horizon_min,
            self.required_min_separation_nm,
            self.substantial_course_change_deg,
            self.substantial_speed_change_fraction,
            self.transition_time_tolerance_min,
            self.transition_distance_tolerance_nm,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("joint avoidance configuration values must be finite")
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
        if not isinstance(self.max_combinations, int) or isinstance(self.max_combinations, bool):
            raise ValueError("max_combinations must be an integer")
        if self.max_combinations < 1:
            raise ValueError("max_combinations must be at least 1")
        if self.rule13_min_past_progress_rate_nm_per_min is not None:
            value = self.rule13_min_past_progress_rate_nm_per_min
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    "rule13_min_past_progress_rate_nm_per_min must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class JointVesselCommandEvaluation:
    vessel_id: str
    command: ManeuverCommand
    course_delta_deg: float
    speed_fraction: float | None
    transition_duration_min: float
    rejection_reasons: tuple[JointVesselRejection, ...]

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class JointPairClearance:
    first_id: str
    second_id: str
    evaluated_transition_duration_min: float
    first_transition_duration_min: float
    second_transition_duration_min: float
    sampled_min_separation_nm: float
    conservative_transition_lower_bound_nm: float
    state_evaluations: int
    both_completed_within_horizon: bool
    post_transition_cpa: ClosestApproach | None
    post_transition_tcpa_from_now_min: float | None
    predicted_min_separation_nm: float
    first_crosses_ahead: bool
    second_crosses_ahead: bool
    rejection_reasons: tuple[JointPairRejection, ...]
    transition_domain_clearance: DomainClearanceCertification | None = None
    post_transition_domain_clearance: DomainClearanceCertification | None = None
    rule13_progress: Rule13ProgressAssessment | None = None
    rule13_progress_deferred: bool = False

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons

    @property
    def minimum_certified_domain_margin_nm(self) -> float | None:
        values = [
            item.conservative_lower_bound_nm
            for item in (self.transition_domain_clearance, self.post_transition_domain_clearance)
            if item is not None
        ]
        return min(values) if values else None


@dataclass(frozen=True, slots=True)
class JointAvoidanceEvaluation:
    command_index: int
    vessel_commands: tuple[JointVesselCommandEvaluation, ...]
    pair_clearances: tuple[JointPairClearance, ...]
    accepted: bool
    predicted_min_separation_nm: float | None

    def command_for(self, vessel_id: str) -> ManeuverCommand:
        for item in self.vessel_commands:
            if item.vessel_id == vessel_id:
                return item.command
        raise KeyError(vessel_id)

    def vessel_evaluation(self, vessel_id: str) -> JointVesselCommandEvaluation:
        for item in self.vessel_commands:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)

    def pair_clearance(self, first_id: str, second_id: str) -> JointPairClearance:
        key = tuple(sorted((first_id, second_id)))
        for item in self.pair_clearances:
            if (item.first_id, item.second_id) == key:
                return item
        raise KeyError(key)

    @property
    def minimum_certified_domain_margin_nm(self) -> float | None:
        if not self.pair_clearances:
            return None
        margins = tuple(item.minimum_certified_domain_margin_nm for item in self.pair_clearances)
        if any(value is None for value in margins):
            return None
        return min(value for value in margins if value is not None)


@dataclass(frozen=True, slots=True)
class JointAvoidanceGrid:
    component_id: str
    vessel_ids: tuple[str, ...]
    evaluations: tuple[JointAvoidanceEvaluation, ...]

    @property
    def safe_tuples(self) -> tuple[JointAvoidanceEvaluation, ...]:
        return tuple(item for item in self.evaluations if item.accepted)

    @property
    def has_safe_tuple(self) -> bool:
        return any(item.accepted for item in self.evaluations)


def _effective_change(state: Vessel2DState, command: ManeuverCommand) -> bool:
    return (
        abs(signed_heading_delta(state.heading_deg, command.target_heading_deg)) > _EPS
        or abs(command.target_speed - state.speed) > _EPS
    )


def _substantial_change(
    state: Vessel2DState,
    command: ManeuverCommand,
    config: JointAvoidanceConfig,
) -> bool:
    course_delta = abs(signed_heading_delta(state.heading_deg, command.target_heading_deg))
    speed_delta = abs(command.target_speed - state.speed)
    course_ok = course_delta > _EPS and course_delta + _EPS >= config.substantial_course_change_deg
    if state.speed > _EPS:
        speed_fraction_delta = speed_delta / state.speed
        speed_ok = (
            speed_delta > _EPS
            and speed_fraction_delta + _EPS >= config.substantial_speed_change_fraction
        )
    else:
        # For an initially stopped vessel, any positive target speed is a real speed
        # change, but no meaningful multiplicative fraction exists.  A positive
        # configured fractional threshold therefore cannot be satisfied by definition.
        speed_ok = speed_delta > _EPS and config.substantial_speed_change_fraction <= _EPS
    return course_ok or speed_ok


def _intent_requires_effective_change(intent: ManeuverIntent) -> bool:
    return intent.action_required and intent.kind in {
        ManeuverIntentKind.KEEP_WELL_CLEAR,
        ManeuverIntentKind.ALTER_COURSE_STARBOARD,
        ManeuverIntentKind.STAND_ON_MUST_TAKE_AVOIDING_ACTION,
    }




def _is_provisional_rule13_continuation(
    aggregate: VesselIntentAggregate,
    state: Vessel2DState,
    command: ManeuverCommand,
    reasons: tuple[JointVesselRejection, ...],
    *,
    progress_screen_enabled: bool,
) -> bool:
    """Return whether only persisted Rule-13 fresh-action diagnostics block hold.

    This is provisional because capability is pair-relative and can only be certified
    after the complete joint command tuple is known.  A vessel with any non-Rule-13
    action-required intent is never admitted through this path.
    """

    if not progress_screen_enabled or _effective_change(state, command):
        return False
    relaxable = {JointVesselRejection.NO_EFFECTIVE_CHANGE, JointVesselRejection.NOT_SUBSTANTIAL}
    if not reasons or any(reason not in relaxable for reason in reasons):
        return False
    effective_change_intents = tuple(
        intent for intent in aggregate.intents if _intent_requires_effective_change(intent)
    )
    return bool(effective_change_intents) and all(
        intent.rule13_past_progress_required for intent in effective_change_intents
    )


def _rule13_continuation_certified_for_vessel(
    vessel_id: str,
    aggregation: ComponentIntentAggregation,
    pair_clearances: tuple[JointPairClearance, ...],
    *,
    progress_plan: Rule13ProgressPlan | None,
) -> bool:
    """Certify exact-command continuation against active/deferred Rule-13 dependencies."""

    all_required = {
        intent.counterpart_id
        for intent in aggregation.for_vessel(vessel_id).intents
        if intent.rule13_past_progress_required
    }
    if not all_required:
        return False

    if progress_plan is None:
        active_counterparts = set(all_required)
        deferred_counterparts: set[str] = set()
    else:
        active_counterparts = set(progress_plan.active_counterparts_for(vessel_id))
        deferred_counterparts = set(progress_plan.deferred_counterparts_for(vessel_id))
        if all_required != active_counterparts | deferred_counterparts:
            return False

    evidenced: set[str] = set()
    for clearance in pair_clearances:
        if vessel_id not in (clearance.first_id, clearance.second_id):
            continue
        progress = clearance.rule13_progress
        if progress is None or progress.overtaking_vessel_id != vessel_id or not progress.capable:
            continue
        counterpart = clearance.second_id if clearance.first_id == vessel_id else clearance.first_id
        evidenced.add(counterpart)
    # Deferred edges remain full keep-clear duties and are still pair-safety screened.
    # Only the additional positive-progress-now contract is postponed.
    return active_counterparts <= evidenced

def _semantic_rejections(
    aggregate: VesselIntentAggregate,
    state: Vessel2DState,
    command: ManeuverCommand,
    states: Mapping[str, Vessel2DState],
    config: JointAvoidanceConfig,
) -> tuple[JointVesselRejection, ...]:
    reasons: list[JointVesselRejection] = []
    course_delta = signed_heading_delta(state.heading_deg, command.target_heading_deg)
    speed_changed = abs(command.target_speed - state.speed) > _EPS
    course_changed = abs(course_delta) > _EPS

    for intent in aggregate.intents:
        if intent.maintain_course_and_speed and (course_changed or speed_changed):
            reasons.append(JointVesselRejection.MAINTAIN_COURSE_SPEED)

        if _intent_requires_effective_change(intent) and not _effective_change(state, command):
            reasons.append(JointVesselRejection.NO_EFFECTIVE_CHANGE)

        if intent.required_course_direction == "starboard" and course_delta <= _EPS:
            reasons.append(JointVesselRejection.WRONG_COURSE_DIRECTION)
        elif intent.required_course_direction == "port" and course_delta >= -_EPS:
            reasons.append(JointVesselRejection.WRONG_COURSE_DIRECTION)

        if intent.avoid_port_alteration_for_target_on_port and course_delta < -_EPS:
            counterpart = states[intent.counterpart_id]
            target_bearing = relative_bearing_deg(state, counterpart)
            target_on_port = 180.0 + _EPS < target_bearing < 360.0 - _EPS
            if target_on_port:
                reasons.append(JointVesselRejection.PORT_ALTERATION_RESTRICTED)

        if intent.early_and_substantial and not _substantial_change(state, command, config):
            reasons.append(JointVesselRejection.NOT_SUBSTANTIAL)

    return tuple(dict.fromkeys(reasons))


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _crosses_ahead_on_constant_tracks(
    own: Vessel2DState,
    counterpart: Vessel2DState,
    horizon_min: float,
) -> bool:
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


def _pair_transition_clearance(
    first: Vessel2DState,
    second: Vessel2DState,
    first_command: ManeuverCommand,
    second_command: ManeuverCommand,
    first_limits: ManeuverLimits,
    second_limits: ManeuverLimits,
    first_duration: float,
    second_duration: float,
    config: JointAvoidanceConfig,
) -> tuple[float, float, int, float, bool]:
    transition_end = max(first_duration, second_duration)
    duration = min(transition_end, config.evaluation_horizon_min)

    def state_at(state, command, limits, t):
        return advance_toward_command(state, command, t, limits).state

    if duration <= _EPS:
        separation = norm(subtract(second.position, first.position))
        return separation, separation, 1, 0.0, True

    cache: dict[float, float] = {}

    def distance(time_min: float) -> float:
        key = round(time_min, 14)
        if key not in cache:
            first_at = state_at(first, first_command, first_limits, time_min)
            second_at = state_at(second, second_command, second_limits, time_min)
            cache[key] = norm(subtract(second_at.position, first_at.position))
        return cache[key]

    relative_speed_bound = max(first.speed, first_command.target_speed) + max(
        second.speed,
        second_command.target_speed,
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

    lower = recurse(0.0, duration, 0)
    return (
        best_sample[0],
        lower,
        len(cache),
        duration,
        transition_end <= config.evaluation_horizon_min + _EPS,
    )


def _active_pair_intents(
    aggregation: ComponentIntentAggregation, first_id: str, second_id: str
):
    key = tuple(sorted((first_id, second_id)))
    for item in aggregation.pair_intents:
        if item.key == key:
            return item.intents
    return None


def _pair_clearance(
    first_id: str,
    second_id: str,
    states: Mapping[str, Vessel2DState],
    limits: Mapping[str, ManeuverLimits],
    commands: Mapping[str, ManeuverCommand],
    transition_durations: Mapping[str, float],
    aggregation: ComponentIntentAggregation,
    config: JointAvoidanceConfig,
    domains: Mapping[str, ShipDomain] | None,
    rule13_progress_plan: Rule13ProgressPlan | None,
) -> JointPairClearance:
    first = states[first_id]
    second = states[second_id]
    first_command = commands[first_id]
    second_command = commands[second_id]
    first_duration = transition_durations[first_id]
    second_duration = transition_durations[second_id]

    sampled, lower, state_evals, evaluated_duration, both_completed = _pair_transition_clearance(
        first,
        second,
        first_command,
        second_command,
        limits[first_id],
        limits[second_id],
        first_duration,
        second_duration,
        config,
    )
    reasons: list[JointPairRejection] = []
    predicted_min = lower
    transition_domain: DomainClearanceCertification | None = None
    post_domain: DomainClearanceCertification | None = None
    rule13_progress: Rule13ProgressAssessment | None = None
    pair_key = tuple(sorted((first_id, second_id)))
    rule13_progress_deferred = (
        rule13_progress_plan is not None
        and rule13_progress_plan.is_progress_deferred(pair_key)
    )
    pair_has_domains = (
        domains is not None and first_id in domains and second_id in domains
    )
    if pair_has_domains:
        assert domains is not None
        transition_domain = certify_pair_domain_clearance(
            first_state_at=lambda t: advance_toward_command(
                first, first_command, t, limits[first_id]
            ).state,
            second_state_at=lambda t: advance_toward_command(
                second, second_command, t, limits[second_id]
            ).state,
            first_domain=domains[first_id],
            second_domain=domains[second_id],
            duration_min=min(
                max(first_duration, second_duration),
                config.evaluation_horizon_min,
            ),
            first_speed_bound_nm_per_min=max(first.speed, first_command.target_speed),
            second_speed_bound_nm_per_min=max(second.speed, second_command.target_speed),
            first_turn_rate_bound_deg_per_min=limits[first_id].max_turn_rate_deg_per_min,
            second_turn_rate_bound_deg_per_min=limits[second_id].max_turn_rate_deg_per_min,
            time_tolerance_min=config.transition_time_tolerance_min,
            margin_tolerance_nm=config.transition_distance_tolerance_nm,
            max_subdivisions=config.max_transition_subdivisions,
        )
        if not transition_domain.certified_clear:
            reasons.append(JointPairRejection.TRANSITION_CLEARANCE)
    elif lower + _EPS < config.required_min_separation_nm:
        reasons.append(JointPairRejection.TRANSITION_CLEARANCE)

    post_cpa: ClosestApproach | None = None
    post_tcpa_from_now: float | None = None
    first_crosses_ahead = False
    second_crosses_ahead = False
    transition_end = max(first_duration, second_duration)
    progress_contract_active = (
        rule13_progress_plan is None
        or rule13_progress_plan.is_progress_active(pair_key)
    )
    if (
        config.rule13_min_past_progress_rate_nm_per_min is not None
        and progress_contract_active
    ):
        pair_intents = _active_pair_intents(aggregation, first_id, second_id)
        if pair_intents is not None:
            first_target_state = advance_toward_command(
                first, first_command, transition_end, limits[first_id]
            ).state
            second_target_state = advance_toward_command(
                second, second_command, transition_end, limits[second_id]
            ).state
            rule13_progress = assess_rule13_past_progress(
                first_target_state,
                second_target_state,
                pair_intents.for_vessel(first_id),
                pair_intents.for_vessel(second_id),
                minimum_required_rate_nm_per_min=(
                    config.rule13_min_past_progress_rate_nm_per_min
                ),
            )
            if rule13_progress is not None and not rule13_progress.capable:
                reasons.append(JointPairRejection.RULE13_NO_PAST_PROGRESS)

    if transition_end < config.evaluation_horizon_min - _EPS:
        first_final = advance_toward_command(
            first, first_command, transition_end, limits[first_id]
        ).state
        second_final = advance_toward_command(
            second, second_command, transition_end, limits[second_id]
        ).state
        remaining = config.evaluation_horizon_min - transition_end
        post = predict_encounter(first_final, second_final, horizon=remaining)
        post_cpa = post.cpa
        post_tcpa_from_now = transition_end + post_cpa.tcpa
        predicted_min = min(predicted_min, post_cpa.dcpa)
        if pair_has_domains:
            assert domains is not None
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
                first_domain=domains[first_id],
                second_domain=domains[second_id],
                duration_min=remaining,
                first_speed_bound_nm_per_min=first_final.speed,
                second_speed_bound_nm_per_min=second_final.speed,
                first_turn_rate_bound_deg_per_min=0.0,
                second_turn_rate_bound_deg_per_min=0.0,
                time_tolerance_min=config.transition_time_tolerance_min,
                margin_tolerance_nm=config.transition_distance_tolerance_nm,
                max_subdivisions=config.max_transition_subdivisions,
            )
            if not post_domain.certified_clear:
                reasons.append(JointPairRejection.POST_TRANSITION_CLEARANCE)
        elif post_cpa.dcpa + _EPS < config.required_min_separation_nm:
            reasons.append(JointPairRejection.POST_TRANSITION_CLEARANCE)

        first_aggregate = aggregation.for_vessel(first_id)
        second_aggregate = aggregation.for_vessel(second_id)
        if second_id in first_aggregate.avoid_crossing_ahead_targets:
            first_crosses_ahead = _crosses_ahead_on_constant_tracks(
                first_final, second_final, remaining
            )
            if first_crosses_ahead:
                reasons.append(JointPairRejection.FIRST_CROSSES_AHEAD)
        if first_id in second_aggregate.avoid_crossing_ahead_targets:
            second_crosses_ahead = _crosses_ahead_on_constant_tracks(
                second_final, first_final, remaining
            )
            if second_crosses_ahead:
                reasons.append(JointPairRejection.SECOND_CROSSES_AHEAD)

    return JointPairClearance(
        first_id=first_id,
        second_id=second_id,
        evaluated_transition_duration_min=evaluated_duration,
        first_transition_duration_min=first_duration,
        second_transition_duration_min=second_duration,
        sampled_min_separation_nm=sampled,
        conservative_transition_lower_bound_nm=lower,
        state_evaluations=state_evals,
        both_completed_within_horizon=both_completed,
        post_transition_cpa=post_cpa,
        post_transition_tcpa_from_now_min=post_tcpa_from_now,
        predicted_min_separation_nm=predicted_min,
        first_crosses_ahead=first_crosses_ahead,
        second_crosses_ahead=second_crosses_ahead,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        transition_domain_clearance=transition_domain,
        post_transition_domain_clearance=post_domain,
        rule13_progress=rule13_progress,
        rule13_progress_deferred=rule13_progress_deferred,
    )


def _require_exact_keys(name: str, actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys must match component vessels; missing={missing}, extra={extra}")


def evaluate_joint_avoidance_candidates(
    aggregation: ComponentIntentAggregation,
    *,
    states: Mapping[str, Vessel2DState],
    limits: Mapping[str, ManeuverLimits],
    command_options: Mapping[str, tuple[ManeuverCommand, ...]],
    config: JointAvoidanceConfig,
    domains: Mapping[str, ShipDomain] | None = None,
    rule13_progress_plan: Rule13ProgressPlan | None = None,
) -> JointAvoidanceGrid:
    """Evaluate the explicit Cartesian product of commands for a compatible component."""

    if not aggregation.compatible:
        raise IncompatibleJointIntents(
            f"component {aggregation.component_id!r} has declaratively conflicting intents"
        )
    if (
        rule13_progress_plan is not None
        and rule13_progress_plan.component_id != aggregation.component_id
    ):
        raise ValueError("Rule-13 progress plan does not match intent aggregation component")

    vessel_ids = tuple(item.vessel_id for item in aggregation.vessels)
    if len(vessel_ids) < 2:
        raise ValueError("joint avoidance requires at least two vessels")
    if len(set(vessel_ids)) != len(vessel_ids):
        raise ValueError("aggregation contains duplicate vessel IDs")
    expected = set(vessel_ids)
    _require_exact_keys("states", set(states), expected)
    _require_exact_keys("limits", set(limits), expected)
    _require_exact_keys("command_options", set(command_options), expected)
    if domains is not None:
        extra_domains = set(domains) - expected
        if extra_domains:
            raise ValueError(
                f"domains keys must be component vessels; extra={sorted(extra_domains)}"
            )

    for vessel_id in vessel_ids:
        if states[vessel_id].vessel_id != vessel_id:
            raise ValueError(f"state mapping key {vessel_id!r} does not match state.vessel_id")
        if not command_options[vessel_id]:
            raise ValueError(f"command_options for {vessel_id!r} must be non-empty")

    evaluations: list[JointAvoidanceEvaluation] = []
    ordered_option_lists = tuple(command_options[vessel_id] for vessel_id in vessel_ids)
    combination_count = 1
    for options in ordered_option_lists:
        combination_count *= len(options)
        if combination_count > config.max_combinations:
            raise JointAvoidanceUnavailable(
                f"joint candidate product has {combination_count} combinations, "
                f"exceeding max_combinations={config.max_combinations}"
            )

    for index, command_tuple in enumerate(product(*ordered_option_lists)):
        commands = dict(zip(vessel_ids, command_tuple, strict=True))
        vessel_evaluations: list[JointVesselCommandEvaluation] = []
        transition_durations: dict[str, float] = {}
        provisional_rule13_continuations: set[str] = set()

        for vessel_id in vessel_ids:
            state = states[vessel_id]
            command = commands[vessel_id]
            aggregate = aggregation.for_vessel(vessel_id)
            reasons = list(_semantic_rejections(aggregate, state, command, states, config))
            duration = command_transition_duration(state, command, limits[vessel_id])
            transition_durations[vessel_id] = duration
            if not isfinite(duration):
                reasons.append(JointVesselRejection.PHYSICALLY_UNREACHABLE)
            deduped_reasons = tuple(dict.fromkeys(reasons))
            if _is_provisional_rule13_continuation(
                aggregate,
                state,
                command,
                deduped_reasons,
                progress_screen_enabled=(
                    config.rule13_min_past_progress_rate_nm_per_min is not None
                ),
            ):
                provisional_rule13_continuations.add(vessel_id)
            course_delta = signed_heading_delta(state.heading_deg, command.target_heading_deg)
            speed_fraction = command.target_speed / state.speed if state.speed > _EPS else None
            vessel_evaluations.append(
                JointVesselCommandEvaluation(
                    vessel_id=vessel_id,
                    command=command,
                    course_delta_deg=course_delta,
                    speed_fraction=speed_fraction,
                    transition_duration_min=duration,
                    rejection_reasons=deduped_reasons,
                )
            )

        pair_clearances: list[JointPairClearance] = []
        can_screen_pairs = all(
            item.accepted or item.vessel_id in provisional_rule13_continuations
            for item in vessel_evaluations
        )
        if can_screen_pairs:
            for first_id, second_id in combinations(vessel_ids, 2):
                pair_clearances.append(
                    _pair_clearance(
                        first_id,
                        second_id,
                        states,
                        limits,
                        commands,
                        transition_durations,
                        aggregation,
                        config,
                        domains,
                        rule13_progress_plan,
                    )
                )

        if pair_clearances and provisional_rule13_continuations:
            screened = tuple(pair_clearances)
            vessel_evaluations = [
                JointVesselCommandEvaluation(
                    vessel_id=item.vessel_id,
                    command=item.command,
                    course_delta_deg=item.course_delta_deg,
                    speed_fraction=item.speed_fraction,
                    transition_duration_min=item.transition_duration_min,
                    rejection_reasons=(
                        ()
                        if item.vessel_id in provisional_rule13_continuations
                        and _rule13_continuation_certified_for_vessel(
                            item.vessel_id,
                            aggregation,
                            screened,
                            progress_plan=rule13_progress_plan,
                        )
                        else item.rejection_reasons
                    ),
                )
                for item in vessel_evaluations
            ]

        accepted = (
            all(item.accepted for item in vessel_evaluations)
            and len(pair_clearances) == len(vessel_ids) * (len(vessel_ids) - 1) // 2
            and all(item.accepted for item in pair_clearances)
        )
        predicted_min = (
            min(item.predicted_min_separation_nm for item in pair_clearances)
            if pair_clearances
            else None
        )
        evaluations.append(
            JointAvoidanceEvaluation(
                command_index=index,
                vessel_commands=tuple(vessel_evaluations),
                pair_clearances=tuple(pair_clearances),
                accepted=accepted,
                predicted_min_separation_nm=predicted_min,
            )
        )

    return JointAvoidanceGrid(
        component_id=aggregation.component_id,
        vessel_ids=vessel_ids,
        evaluations=tuple(evaluations),
    )
