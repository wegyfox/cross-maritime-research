"""Deterministic water-relative pair command-grid generation for environmental planning.

A87 reuses the explicit ``IntentDrivenCandidatePolicy`` grid but applies it to the
water-relative control frame captured by a83.  It performs only declarative filtering
and class through-water speed-envelope filtering.  Physical safety belongs to a86;
Rule-13 progress capability, ranking, arbitration and execution remain downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_generation import IntentDrivenCandidatePolicy
from .dynamics import normalize_heading, signed_heading_delta
from .encounter import absolute_bearing_deg
from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .fleet_supervision import FleetSupervisorBackend
from .maneuver_intent import ManeuverIntent, ManeuverIntentKind, PairManeuverIntents

_EPS = 1e-10


class EnvironmentalCandidateGenerationUnavailable(RuntimeError):
    """Raised when the configured policy cannot express a required pair action."""


@dataclass(frozen=True, slots=True)
class EnvironmentalGeneratedCommand:
    command: WaterRelativeManeuverCommand
    course_delta_deg: float
    speed_fraction: float

    @property
    def is_hold(self) -> bool:
        return abs(self.course_delta_deg) <= _EPS and abs(self.speed_fraction - 1.0) <= _EPS


@dataclass(frozen=True, slots=True)
class EnvironmentalPairCommandGrid:
    first_intent: ManeuverIntent
    second_intent: ManeuverIntent
    first_commands: tuple[EnvironmentalGeneratedCommand, ...]
    second_commands: tuple[EnvironmentalGeneratedCommand, ...]

    @property
    def command_pairs(self) -> tuple[tuple[EnvironmentalGeneratedCommand, EnvironmentalGeneratedCommand], ...]:
        return tuple((first, second) for first in self.first_commands for second in self.second_commands)


def _dedupe(values: tuple[float, ...]) -> tuple[float, ...]:
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(value, 12)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _target_on_port(own, counterpart) -> bool:
    absolute = absolute_bearing_deg(own.state.position, counterpart.state.position)
    relative = normalize_heading(absolute - own.state.through_water_heading_deg)
    return 180.0 + _EPS < relative < 360.0 - _EPS


def _requires_effective_change(intent: ManeuverIntent) -> bool:
    return intent.action_required and intent.kind in {
        ManeuverIntentKind.KEEP_WELL_CLEAR,
        ManeuverIntentKind.ALTER_COURSE_STARBOARD,
        ManeuverIntentKind.STAND_ON_MUST_TAKE_AVOIDING_ACTION,
    }


def _hold_permitted(intent: ManeuverIntent) -> bool:
    return (
        intent.maintain_course_and_speed
        or (
            intent.kind is ManeuverIntentKind.STAND_ON_MAY_TAKE_AVOIDING_ACTION
            and intent.action_permitted
            and not intent.action_required
        )
    )


def _commands_for_intent(
    context: EnvironmentalAvoidancePlanningContext,
    intent: ManeuverIntent,
    policy: IntentDrivenCandidatePolicy,
) -> tuple[EnvironmentalGeneratedCommand, ...]:
    own = context.for_vessel(intent.vessel_id)
    counterpart = context.for_vessel(intent.counterpart_id)
    state = own.state
    following = context.following_constraint_for(intent.vessel_id)
    following_ceiling = None if following is None else following.max_follower_speed_nm_per_min

    if intent.kind is ManeuverIntentKind.NONE:
        return ()
    if intent.maintain_course_and_speed and not intent.action_permitted:
        speed = state.through_water_speed
        if following_ceiling is not None:
            speed = min(speed, following_ceiling)
        factor = 1.0 if state.through_water_speed <= _EPS else speed / state.through_water_speed
        return (
            EnvironmentalGeneratedCommand(
                WaterRelativeManeuverCommand(
                    state.through_water_heading_deg,
                    speed,
                ),
                0.0,
                factor,
            ),
        )

    deltas = list(policy.course_deltas_deg)
    if intent.rule13_past_progress_required:
        for offset in policy.rule13_progress_course_offsets_from_counterpart_deg:
            target = normalize_heading(counterpart.state.through_water_heading_deg + offset)
            deltas.append(signed_heading_delta(state.through_water_heading_deg, target))
    deltas = list(_dedupe(tuple(deltas)))

    if intent.required_course_direction == "starboard":
        deltas = [delta for delta in deltas if delta > _EPS]
    elif intent.required_course_direction == "port":
        deltas = [delta for delta in deltas if delta < -_EPS]
    if intent.avoid_port_alteration_for_target_on_port and _target_on_port(own, counterpart):
        deltas = [delta for delta in deltas if delta >= -_EPS]

    if not deltas:
        if intent.required_course_direction is not None:
            raise EnvironmentalCandidateGenerationUnavailable(
                f"candidate grid has no {intent.required_course_direction} course change for {intent.vessel_id!r}"
            )
        zero = tuple(delta for delta in policy.course_deltas_deg if abs(delta) <= _EPS)
        deltas = list(zero)
    if not deltas:
        raise EnvironmentalCandidateGenerationUnavailable(
            f"candidate grid has no declaratively permitted course option for {intent.vessel_id!r}"
        )

    factors = list(policy.speed_fractions)
    if intent.rule13_past_progress_required:
        factors.extend(policy.rule13_progress_speed_fractions)
    factors = list(_dedupe(tuple(factors)))
    if following_ceiling is not None and state.through_water_speed > _EPS:
        ceiling_factor = max(0.0, following_ceiling / state.through_water_speed)
        factors.append(ceiling_factor)
        factors = list(_dedupe(tuple(factors)))
        factors = [
            factor for factor in factors
            if state.through_water_speed * factor <= following_ceiling + _EPS
        ]
    if own.profile is not None and own.profile.speed_envelope is not None:
        if (
            intent.rule13_past_progress_required
            and policy.rule13_progress_include_speed_envelope_ceiling
            and state.through_water_speed > _EPS
            and own.profile.speed_envelope.maximum_speed > state.through_water_speed + _EPS
        ):
            factors.append(own.profile.speed_envelope.maximum_speed / state.through_water_speed)
            factors = list(_dedupe(tuple(factors)))
        factors = [
            factor
            for factor in factors
            if own.profile.speed_envelope.permits(state.through_water_speed * factor)
        ]
    if not factors:
        raise EnvironmentalCandidateGenerationUnavailable(
            f"candidate speed grid exceeds class through-water speed envelope for {intent.vessel_id!r}"
        )

    required_change = _requires_effective_change(intent)
    if required_change and not any(
        abs(delta) > _EPS or abs(factor - 1.0) > _EPS
        for delta in deltas
        for factor in factors
    ):
        raise EnvironmentalCandidateGenerationUnavailable(
            f"candidate grid cannot express an effective required action for {intent.vessel_id!r}"
        )

    commands: list[EnvironmentalGeneratedCommand] = []
    seen: set[tuple[float, float]] = set()

    def append(delta: float, factor: float) -> None:
        heading = normalize_heading(state.through_water_heading_deg + delta)
        speed = state.through_water_speed * factor
        key = (round(heading, 12), round(speed, 12))
        if key not in seen:
            seen.add(key)
            commands.append(
                EnvironmentalGeneratedCommand(
                    WaterRelativeManeuverCommand(heading, speed),
                    delta,
                    factor,
                )
            )

    if _hold_permitted(intent) and not required_change:
        append(0.0, 1.0)
    for delta in deltas:
        for factor in factors:
            append(delta, factor)
    return tuple(commands)


def generate_environmental_pair_command_grid(
    context: EnvironmentalAvoidancePlanningContext,
    intents: PairManeuverIntents,
    *,
    policy: IntentDrivenCandidatePolicy,
) -> EnvironmentalPairCommandGrid:
    """Generate a deterministic Cartesian command grid for one environmental PAIR context."""
    if context.backend is not FleetSupervisorBackend.PAIR or len(context.vessel_ids) != 2:
        raise EnvironmentalCandidateGenerationUnavailable(
            "environmental pair command generation requires a two-vessel PAIR context"
        )
    first_id, second_id = context.vessel_ids
    first_intent = intents.for_vessel(first_id)
    second_intent = intents.for_vessel(second_id)
    if first_intent.counterpart_id != second_id or second_intent.counterpart_id != first_id:
        raise EnvironmentalCandidateGenerationUnavailable(
            "pair intents do not match planning-context vessel IDs"
        )
    first_commands = _commands_for_intent(context, first_intent, policy)
    second_commands = _commands_for_intent(context, second_intent, policy)
    return EnvironmentalPairCommandGrid(
        first_intent,
        second_intent,
        first_commands,
        second_commands,
    )
