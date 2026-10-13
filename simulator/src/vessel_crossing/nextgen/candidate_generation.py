"""Deterministic intent-driven maneuver option generation.

The generator reduces caller plumbing without becoming a safety authority.  It uses the
same declarative a11/a26 intents to remove obviously incompatible grid directions and to
preserve mandatory hold options, then returns explicit deterministic candidate configs
or joint command tuples.  a42 semantic/physical screening remains authoritative and a43
arbitration still ranks only accepted rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping

from .avoidance import AvoidanceCandidateConfig
from .dynamics import ManeuverCommand, normalize_heading, signed_heading_delta
from .encounter import relative_bearing_deg
from .encounter_registry import PairKey
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .maneuver_intent import ManeuverIntent, ManeuverIntentKind, StandOnStage, select_pair_maneuver_intents
from .multi_intent import ComponentIntentAggregation, VesselIntentAggregate, aggregate_component_intents
from .planning_context import AvoidancePlanningContext
from .rule13_progress_coordination import (
    Rule13ProgressCoordinationPolicy,
    Rule13ProgressPlan,
    build_rule13_progress_plan,
)

_EPS = 1e-10


class CandidateGenerationError(RuntimeError):
    """Base error for deterministic intent-driven candidate generation."""


class CandidateGenerationUnavailable(CandidateGenerationError):
    """Raised when the configured grid cannot express a required declarative action."""


@dataclass(frozen=True, slots=True)
class IntentDrivenCandidatePolicy:
    """One explicit ordered course/speed grid reused by pair and joint generators.

    ``rule13_progress_speed_fractions`` is an explicit opt-in expansion used only by a
    Rule-13 give-way intent that still needs to become modeled past.  Values are
    supplemental acceleration factors and therefore must be strictly greater than 1.

    ``rule13_progress_course_offsets_from_counterpart_deg`` is a separate opt-in
    expansion for the same progress duty.  Each value is an offset around the current
    heading of the saved overtaken counterpart, not around the overtaker's own heading.
    This lets the grid express longitudinal overtaking progress after earlier
    multi-encounter maneuvers have left the two courses far apart.

    ``rule13_progress_include_speed_envelope_ceiling`` is a third, default-off
    expansion.  When Rule-13 progress is active and the vessel class has remaining
    speed headroom, the exact class maximum is exposed as an additional factor.  This
    prevents a multiplicative progress ladder from losing the physically admissible
    boundary merely because its next factor overshoots the class envelope.

    The physical a55 progress/safety screen remains authoritative; this generator
    merely exposes additional caller-visible action space.
    """

    template: AvoidanceCandidateConfig
    rule13_progress_speed_fractions: tuple[float, ...] = ()
    rule13_progress_course_offsets_from_counterpart_deg: tuple[float, ...] = ()
    rule13_progress_include_speed_envelope_ceiling: bool = False

    def __post_init__(self) -> None:
        if any(
            (not isfinite(value)) or value <= 1.0 + _EPS
            for value in self.rule13_progress_speed_fractions
        ):
            raise ValueError(
                "rule13_progress_speed_fractions must contain finite factors strictly greater than 1"
            )
        if not isinstance(self.rule13_progress_include_speed_envelope_ceiling, bool):
            raise TypeError("rule13_progress_include_speed_envelope_ceiling must be bool")
        if any(
            (not isfinite(value)) or abs(value) > 180.0 + _EPS
            for value in self.rule13_progress_course_offsets_from_counterpart_deg
        ):
            raise ValueError(
                "rule13_progress_course_offsets_from_counterpart_deg must contain finite offsets in [-180, 180]"
            )

    @property
    def course_deltas_deg(self) -> tuple[float, ...]:
        return self.template.course_deltas_deg

    @property
    def speed_fractions(self) -> tuple[float, ...]:
        return self.template.speed_fractions


def _speed_fractions_for_intents(
    context: AvoidancePlanningContext,
    vessel_id: str,
    intents: tuple[ManeuverIntent, ...],
    policy: IntentDrivenCandidatePolicy,
    *,
    rule13_progress_required: bool | None = None,
) -> tuple[float, ...]:
    factors = list(policy.speed_fractions)
    needs_rule13_progress = (
        any(item.rule13_past_progress_required for item in intents)
        if rule13_progress_required is None
        else rule13_progress_required
    )
    if needs_rule13_progress:
        factors.extend(policy.rule13_progress_speed_fractions)
    # Preserve caller order while making the eventual Cartesian grid deterministic.
    factors = list(dict.fromkeys(factors))
    profile = context.profiles[vessel_id]
    if profile is not None and profile.speed_envelope is not None:
        current_speed = context.states[vessel_id].speed
        if (
            needs_rule13_progress
            and policy.rule13_progress_include_speed_envelope_ceiling
            and current_speed > _EPS
            and profile.speed_envelope.maximum_speed > current_speed + _EPS
        ):
            factors.append(profile.speed_envelope.maximum_speed / current_speed)
            factors = list(dict.fromkeys(factors))
        factors = [
            factor
            for factor in factors
            if profile.speed_envelope.permits(current_speed * factor)
        ]
    return tuple(factors)



def _rule13_progress_course_deltas(
    context: AvoidancePlanningContext,
    vessel_id: str,
    intents: tuple[ManeuverIntent, ...],
    policy: IntentDrivenCandidatePolicy,
    *,
    active_counterparts: set[str] | None = None,
) -> tuple[float, ...]:
    """Return supplemental own-relative deltas aimed around active Rule-13 counterparts.

    The configured offsets are counterpart-heading-relative, while the lower-level
    ``AvoidanceCandidateConfig`` consumes deltas relative to the acting vessel's current
    heading.  Conversion is snapshot-local and deterministic.
    """

    if not policy.rule13_progress_course_offsets_from_counterpart_deg:
        return ()
    own = context.states[vessel_id]
    result: list[float] = []
    seen: set[float] = set()
    for intent in intents:
        if not intent.rule13_past_progress_required:
            continue
        if active_counterparts is not None and intent.counterpart_id not in active_counterparts:
            continue
        counterpart = context.states[intent.counterpart_id]
        for offset in policy.rule13_progress_course_offsets_from_counterpart_deg:
            target_heading = normalize_heading(counterpart.heading_deg + offset)
            delta = signed_heading_delta(own.heading_deg, target_heading)
            key = round(delta, 12)
            if key not in seen:
                seen.add(key)
                result.append(delta)
    return tuple(result)


def _append_unique(values: tuple[float, ...], extra: tuple[float, ...]) -> tuple[float, ...]:
    result = list(values)
    seen = {round(value, 12) for value in result}
    for value in extra:
        key = round(value, 12)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)

def _target_on_port(context: AvoidancePlanningContext, intent: ManeuverIntent) -> bool:
    own = context.states[intent.vessel_id]
    counterpart = context.states[intent.counterpart_id]
    bearing = relative_bearing_deg(own, counterpart)
    return 180.0 + _EPS < bearing < 360.0 - _EPS


def _filter_course_deltas(
    context: AvoidancePlanningContext,
    intents: tuple[ManeuverIntent, ...],
    deltas: tuple[float, ...],
) -> tuple[float, ...]:
    required = {item.required_course_direction for item in intents if item.required_course_direction}
    if len(required) > 1:
        raise CandidateGenerationUnavailable("conflicting required course directions")
    direction = next(iter(required), None)

    port_restricted = any(
        item.avoid_port_alteration_for_target_on_port and _target_on_port(context, item)
        for item in intents
    )

    result: list[float] = []
    for delta in deltas:
        if direction == "starboard" and delta <= _EPS:
            continue
        if direction == "port" and delta >= -_EPS:
            continue
        if port_restricted and delta < -_EPS:
            continue
        result.append(delta)
    return tuple(result)


def _requires_effective_change(intents: tuple[ManeuverIntent, ...]) -> bool:
    return any(
        item.action_required
        and item.kind
        in {
            ManeuverIntentKind.KEEP_WELL_CLEAR,
            ManeuverIntentKind.ALTER_COURSE_STARBOARD,
            ManeuverIntentKind.STAND_ON_MUST_TAKE_AVOIDING_ACTION,
        }
        for item in intents
    )


def _hold_permitted(intents: tuple[ManeuverIntent, ...]) -> bool:
    if any(item.maintain_course_and_speed for item in intents):
        return True
    return any(
        item.kind is ManeuverIntentKind.STAND_ON_MAY_TAKE_AVOIDING_ACTION
        and item.action_permitted
        and not item.action_required
        for item in intents
    )


def _effective_grid_exists(deltas: tuple[float, ...], factors: tuple[float, ...]) -> bool:
    return any(abs(delta) > _EPS or abs(factor - 1.0) > _EPS for delta in deltas for factor in factors)


def pair_candidate_configs_from_intents(
    context: AvoidancePlanningContext,
    owner: FleetCommandOwner,
    *,
    policy: IntentDrivenCandidatePolicy,
    stand_on_stage: StandOnStage = StandOnStage.NORMAL,
) -> dict[str, AvoidanceCandidateConfig]:
    """Build exact-owner pair configs with declaratively impossible directions removed."""

    if owner.backend is not FleetSupervisorBackend.PAIR or context.backend is not FleetSupervisorBackend.PAIR:
        raise CandidateGenerationError("pair candidate generation requires PAIR owner/context")
    if owner.component_id != context.component_id or owner.vessel_ids != context.vessel_ids:
        raise CandidateGenerationError("pair owner does not match planning context")
    if len(owner.component.decisions) != 1:
        raise CandidateGenerationError("pair owner must contain exactly one active pair decision")

    pair_intents = select_pair_maneuver_intents(
        owner.component.decisions[0].entry.decision,
        stand_on_stage=stand_on_stage,
    )
    result: dict[str, AvoidanceCandidateConfig] = {}
    for vessel_id in owner.vessel_ids:
        intent = pair_intents.for_vessel(vessel_id)
        # The a12 generator replaces normal stand-on grids with one exact hold command,
        # so the template remains valid and is never treated as permission to alter.
        if intent.maintain_course_and_speed and not intent.action_permitted:
            result[vessel_id] = policy.template
            continue

        course_deltas = _append_unique(
            policy.course_deltas_deg,
            _rule13_progress_course_deltas(
                context,
                vessel_id,
                (intent,),
                policy,
            ),
        )
        deltas = _filter_course_deltas(context, (intent,), course_deltas)
        if not deltas:
            # A pure speed action can still satisfy KEEP_WELL_CLEAR/MUST_ACTION when no
            # directional course requirement exists.  Preserve a zero delta when the
            # caller explicitly provided it; otherwise an empty directional grid means
            # this policy cannot express the required action.
            if intent.required_course_direction is not None:
                raise CandidateGenerationUnavailable(
                    f"candidate grid has no {intent.required_course_direction} course change for {vessel_id!r}"
                )
            deltas = tuple(delta for delta in policy.course_deltas_deg if abs(delta) <= _EPS)
        if not deltas:
            raise CandidateGenerationUnavailable(
                f"candidate grid has no declaratively permitted course option for {vessel_id!r}"
            )
        speed_fractions = _speed_fractions_for_intents(context, vessel_id, (intent,), policy)
        if not speed_fractions:
            raise CandidateGenerationUnavailable(
                f"candidate speed grid exceeds class speed envelope for {vessel_id!r}"
            )
        if _requires_effective_change((intent,)) and not _effective_grid_exists(
            deltas, speed_fractions
        ):
            raise CandidateGenerationUnavailable(
                f"candidate grid cannot express an effective required action for {vessel_id!r}"
            )
        result[vessel_id] = replace(
            policy.template,
            course_deltas_deg=deltas,
            speed_fractions=speed_fractions,
        )
    return result


def commands_for_intent_aggregate(
    context: AvoidancePlanningContext,
    aggregate: VesselIntentAggregate,
    *,
    policy: IntentDrivenCandidatePolicy,
    rule13_progress_plan: Rule13ProgressPlan | None = None,
) -> tuple[ManeuverCommand, ...]:
    """Generate one deterministic command list for a compatible aggregated vessel intent."""

    if not aggregate.compatible:
        raise CandidateGenerationUnavailable(
            f"cannot generate commands for conflicting aggregate {aggregate.vessel_id!r}"
        )
    state = context.states[aggregate.vessel_id]
    intents = aggregate.intents

    progress_active_counterparts: set[str] = set()
    progress_deferred_counterparts: set[str] = set()
    if rule13_progress_plan is not None:
        progress_active_counterparts = set(
            rule13_progress_plan.active_counterparts_for(aggregate.vessel_id)
        )
        progress_deferred_counterparts = set(
            rule13_progress_plan.deferred_counterparts_for(aggregate.vessel_id)
        )

    if aggregate.maintain_course_and_speed:
        return (ManeuverCommand(state.heading_deg, state.speed),)

    progress_course_deltas = _rule13_progress_course_deltas(
        context,
        aggregate.vessel_id,
        intents,
        policy,
        active_counterparts=(
            progress_active_counterparts if rule13_progress_plan is not None else None
        ),
    ) if rule13_progress_plan is not None else _rule13_progress_course_deltas(
        context,
        aggregate.vessel_id,
        intents,
        policy,
    )
    deltas = _filter_course_deltas(
        context,
        intents,
        _append_unique(policy.course_deltas_deg, progress_course_deltas),
    )
    required_change = _requires_effective_change(intents)
    hold_permitted = _hold_permitted(intents)

    effective_change_intents = tuple(
        item
        for item in intents
        if item.action_required
        and item.kind
        in {
            ManeuverIntentKind.KEEP_WELL_CLEAR,
            ManeuverIntentKind.ALTER_COURSE_STARBOARD,
            ManeuverIntentKind.STAND_ON_MUST_TAKE_AVOIDING_ACTION,
        }
    )
    rule13_continuation_hold = bool(effective_change_intents) and all(
        item.rule13_past_progress_required
        and item.counterpart_id in (progress_active_counterparts | progress_deferred_counterparts)
        for item in effective_change_intents
    )

    commands: list[ManeuverCommand] = []
    seen: set[tuple[float, float]] = set()

    def append(command: ManeuverCommand) -> None:
        key = (round(command.target_heading_deg, 12), round(command.target_speed, 12))
        if key not in seen:
            seen.add(key)
            commands.append(command)

    if (hold_permitted and not required_change) or rule13_continuation_hold:
        append(ManeuverCommand(state.heading_deg, state.speed))

    speed_fractions = _speed_fractions_for_intents(
        context,
        aggregate.vessel_id,
        intents,
        policy,
        rule13_progress_required=(
            bool(progress_active_counterparts)
            if rule13_progress_plan is not None
            else None
        ),
    )
    for delta in deltas:
        for factor in speed_fractions:
            is_hold = abs(delta) <= _EPS and abs(factor - 1.0) <= _EPS
            if required_change and is_hold:
                continue
            append(
                ManeuverCommand(
                    normalize_heading(state.heading_deg + delta),
                    state.speed * factor,
                )
            )

    if not commands:
        raise CandidateGenerationUnavailable(
            f"candidate grid produced no declaratively permitted command for {aggregate.vessel_id!r}"
        )
    return tuple(commands)


def joint_command_options_from_intents(
    context: AvoidancePlanningContext,
    owner: FleetCommandOwner,
    *,
    policy: IntentDrivenCandidatePolicy,
    stand_on_stages: Mapping[PairKey, StandOnStage] | None = None,
    rule13_progress_coordination: Rule13ProgressCoordinationPolicy | None = None,
) -> dict[str, tuple[ManeuverCommand, ...]]:
    """Build exact-owner joint command options from the component intent aggregation."""

    if owner.backend is not FleetSupervisorBackend.JOINT or context.backend is not FleetSupervisorBackend.JOINT:
        raise CandidateGenerationError("joint candidate generation requires JOINT owner/context")
    if owner.component_id != context.component_id or owner.vessel_ids != context.vessel_ids:
        raise CandidateGenerationError("joint owner does not match planning context")

    aggregation: ComponentIntentAggregation = aggregate_component_intents(
        owner.component,
        stand_on_stages=stand_on_stages,
    )
    if not aggregation.compatible:
        raise CandidateGenerationUnavailable("joint component has conflicting declarative intents")
    progress_plan = (
        None
        if rule13_progress_coordination is None
        else build_rule13_progress_plan(
            aggregation, policy=rule13_progress_coordination
        )
    )
    return {
        vessel_id: commands_for_intent_aggregate(
            context,
            aggregation.for_vessel(vessel_id),
            policy=policy,
            rule13_progress_plan=progress_plan,
        )
        for vessel_id in owner.vessel_ids
    }


def make_pair_candidate_provider(
    policy: IntentDrivenCandidatePolicy,
    *,
    stand_on_stage: StandOnStage = StandOnStage.NORMAL,
):
    """Return an a47-compatible pair candidate provider closure."""

    return lambda context, owner: pair_candidate_configs_from_intents(
        context, owner, policy=policy, stand_on_stage=stand_on_stage
    )


def make_joint_command_options_provider(
    policy: IntentDrivenCandidatePolicy,
    *,
    stand_on_stages: Mapping[PairKey, StandOnStage] | None = None,
):
    """Return an a47-compatible joint command-options provider closure."""

    return lambda context, owner: joint_command_options_from_intents(
        context, owner, policy=policy, stand_on_stages=stand_on_stages
    )
