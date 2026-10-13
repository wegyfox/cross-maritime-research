"""Environmental JOINT semantic option generation and Rule-13 coordination.

A105 sits directly above the policy-free a104 physical backend.  It translates one
already-projected compatible component intent aggregation into deterministic
water-relative command options, lets a104 screen the complete Cartesian product for
all-pairs physical safety, and then applies pair semantics that cannot be established
by option generation alone:

- Rule-15 ``avoid crossing ahead`` is evaluated on post-transition **ground tracks**;
- the optional Rule-13 positive-past-progress contract is evaluated only on the
  dependencies marked active by an explicit ``Rule13ProgressPlan``;
- deferred Rule-13 dependencies retain their keep-clear duty and a104 physical safety
  screening, but the additional progress-now check is postponed.

This module does not rank tuples, apply commands, own execution, or mutate the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping

from .candidate_generation import IntentDrivenCandidatePolicy
from .dynamics import normalize_heading, signed_heading_delta
from .encounter import absolute_bearing_deg
from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_evidence import EnvironmentalVesselMotionEvidence, environmental_vessel_motion_evidence
from .environmental_joint_physics import (
    EnvironmentalJointPhysicsConfig,
    EnvironmentalJointPhysicsGrid,
    EnvironmentalJointTupleEvaluation,
    evaluate_environmental_joint_command_options,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .environmental_rule13_progress import assess_environmental_rule13_past_progress
from .fleet_supervision import FleetSupervisorBackend
from .maneuver_intent import ManeuverIntent, ManeuverIntentKind, PairManeuverIntents
from .multi_intent import ComponentIntentAggregation, VesselIntentAggregate
from .overtaking_progress import Rule13ProgressAssessment
from .rule13_progress_coordination import (
    Rule13ProgressCoordinationPolicy,
    Rule13ProgressPlan,
    build_rule13_progress_plan,
)

_EPS = 1e-10


class EnvironmentalJointSemanticError(RuntimeError):
    """Base error for invalid environmental JOINT semantic planning."""


class EnvironmentalJointSemanticUnavailable(EnvironmentalJointSemanticError):
    """Raised when declarative constraints cannot be expressed by the configured grid."""


class EnvironmentalJointPairSemanticRejection(str, Enum):
    FIRST_CROSSES_AHEAD = "first_crosses_ahead"
    SECOND_CROSSES_AHEAD = "second_crosses_ahead"
    RULE13_NO_PAST_PROGRESS = "rule13_no_past_progress"


@dataclass(frozen=True, slots=True)
class EnvironmentalJointSemanticConfig:
    candidate_policy: IntentDrivenCandidatePolicy
    physics: EnvironmentalJointPhysicsConfig
    rule13_progress_coordination: Rule13ProgressCoordinationPolicy = (
        Rule13ProgressCoordinationPolicy.STRICT_ALL_DUTIES
    )
    rule13_min_past_progress_rate_nm_per_min: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_policy, IntentDrivenCandidatePolicy):
            raise TypeError("candidate_policy must be IntentDrivenCandidatePolicy")
        if not isinstance(self.physics, EnvironmentalJointPhysicsConfig):
            raise TypeError("physics must be EnvironmentalJointPhysicsConfig")
        if not isinstance(self.rule13_progress_coordination, Rule13ProgressCoordinationPolicy):
            raise TypeError("rule13_progress_coordination must be Rule13ProgressCoordinationPolicy")
        if (
            not isfinite(self.rule13_min_past_progress_rate_nm_per_min)
            or self.rule13_min_past_progress_rate_nm_per_min < 0.0
        ):
            raise ValueError(
                "rule13_min_past_progress_rate_nm_per_min must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class EnvironmentalJointPairSemanticEvidence:
    first_id: str
    second_id: str
    pair_intents: PairManeuverIntents | None
    first_crosses_ahead: bool
    second_crosses_ahead: bool
    rule13_progress: Rule13ProgressAssessment | None
    rule13_progress_deferred: bool
    rejection_reasons: tuple[EnvironmentalJointPairSemanticRejection, ...]

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class EnvironmentalJointSemanticTuple:
    source_evaluation: EnvironmentalJointTupleEvaluation
    pair_semantics: tuple[EnvironmentalJointPairSemanticEvidence, ...]
    accepted: bool

    @property
    def command_index(self) -> int:
        return self.source_evaluation.command_index

    @property
    def predicted_min_separation_nm(self) -> float | None:
        return self.source_evaluation.predicted_min_separation_nm

    def pair_semantic(self, first_id: str, second_id: str) -> EnvironmentalJointPairSemanticEvidence:
        key = tuple(sorted((first_id, second_id)))
        for item in self.pair_semantics:
            if (item.first_id, item.second_id) == key:
                return item
        raise KeyError(key)


@dataclass(frozen=True, slots=True)
class EnvironmentalJointSemanticGrid:
    aggregation: ComponentIntentAggregation
    progress_plan: Rule13ProgressPlan
    command_options: Mapping[str, tuple[WaterRelativeManeuverCommand, ...]]
    physics_grid: EnvironmentalJointPhysicsGrid
    evaluations: tuple[EnvironmentalJointSemanticTuple, ...]

    @property
    def safe_tuples(self) -> tuple[EnvironmentalJointSemanticTuple, ...]:
        return tuple(item for item in self.evaluations if item.accepted)

    @property
    def has_safe_tuple(self) -> bool:
        return any(item.accepted for item in self.evaluations)


def _dedupe(values: tuple[float, ...]) -> tuple[float, ...]:
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(value, 12)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _append_unique(values: tuple[float, ...], extra: tuple[float, ...]) -> tuple[float, ...]:
    return _dedupe(values + extra)


def _target_on_port(context: EnvironmentalAvoidancePlanningContext, intent: ManeuverIntent) -> bool:
    own = context.for_vessel(intent.vessel_id)
    counterpart = context.for_vessel(intent.counterpart_id)
    absolute = absolute_bearing_deg(own.state.position, counterpart.state.position)
    relative = normalize_heading(absolute - own.state.through_water_heading_deg)
    return 180.0 + _EPS < relative < 360.0 - _EPS


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


def _substantial(
    delta: float,
    factor: float,
    *,
    policy: IntentDrivenCandidatePolicy,
) -> bool:
    if abs(delta) <= _EPS and abs(factor - 1.0) <= _EPS:
        return False
    template = policy.template
    course_ok = abs(delta) > _EPS and abs(delta) + _EPS >= template.substantial_course_change_deg
    speed_ok = (
        abs(factor - 1.0) > _EPS
        and abs(factor - 1.0) + _EPS >= template.substantial_speed_change_fraction
    )
    return course_ok or speed_ok


def _progress_course_deltas(
    context: EnvironmentalAvoidancePlanningContext,
    aggregate: VesselIntentAggregate,
    policy: IntentDrivenCandidatePolicy,
    progress_plan: Rule13ProgressPlan,
) -> tuple[float, ...]:
    active = set(progress_plan.active_counterparts_for(aggregate.vessel_id))
    if not active or not policy.rule13_progress_course_offsets_from_counterpart_deg:
        return ()
    own = context.for_vessel(aggregate.vessel_id).state
    result: list[float] = []
    for intent in aggregate.intents:
        if not intent.rule13_past_progress_required or intent.counterpart_id not in active:
            continue
        counterpart = context.for_vessel(intent.counterpart_id).state
        for offset in policy.rule13_progress_course_offsets_from_counterpart_deg:
            target = normalize_heading(counterpart.through_water_heading_deg + offset)
            result.append(signed_heading_delta(own.through_water_heading_deg, target))
    return _dedupe(tuple(result))


def _course_deltas_for_aggregate(
    context: EnvironmentalAvoidancePlanningContext,
    aggregate: VesselIntentAggregate,
    policy: IntentDrivenCandidatePolicy,
    progress_plan: Rule13ProgressPlan,
) -> tuple[float, ...]:
    deltas = _append_unique(
        policy.course_deltas_deg,
        _progress_course_deltas(context, aggregate, policy, progress_plan),
    )
    directions = {
        item.required_course_direction
        for item in aggregate.intents
        if item.required_course_direction is not None
    }
    if len(directions) > 1:
        raise EnvironmentalJointSemanticUnavailable(
            f"conflicting required course directions for {aggregate.vessel_id!r}"
        )
    direction = next(iter(directions), None)
    result: list[float] = []
    for delta in deltas:
        if direction == "starboard" and delta <= _EPS:
            continue
        if direction == "port" and delta >= -_EPS:
            continue
        if delta < -_EPS and any(
            intent.avoid_port_alteration_for_target_on_port and _target_on_port(context, intent)
            for intent in aggregate.intents
        ):
            continue
        result.append(delta)
    if not result:
        if direction is not None:
            raise EnvironmentalJointSemanticUnavailable(
                f"candidate grid has no {direction} course change for {aggregate.vessel_id!r}"
            )
        result = [delta for delta in policy.course_deltas_deg if abs(delta) <= _EPS]
    if not result:
        raise EnvironmentalJointSemanticUnavailable(
            f"candidate grid has no declaratively permitted course option for {aggregate.vessel_id!r}"
        )
    return _dedupe(tuple(result))


def _speed_factors_for_aggregate(
    context: EnvironmentalAvoidancePlanningContext,
    aggregate: VesselIntentAggregate,
    policy: IntentDrivenCandidatePolicy,
    progress_plan: Rule13ProgressPlan,
) -> tuple[float, ...]:
    factors = list(policy.speed_fractions)
    if progress_plan.active_counterparts_for(aggregate.vessel_id):
        factors.extend(policy.rule13_progress_speed_fractions)
    factors = list(_dedupe(tuple(factors)))
    own = context.for_vessel(aggregate.vessel_id)
    following = context.following_constraint_for(aggregate.vessel_id)
    if following is not None and own.state.through_water_speed > _EPS:
        ceiling_factor = max(0.0, following.max_follower_speed_nm_per_min / own.state.through_water_speed)
        factors.append(ceiling_factor)
        factors = list(_dedupe(tuple(factors)))
        factors = [
            factor for factor in factors
            if own.state.through_water_speed * factor <= following.max_follower_speed_nm_per_min + _EPS
        ]
    if own.profile is not None and own.profile.speed_envelope is not None:
        if (
            progress_plan.active_counterparts_for(aggregate.vessel_id)
            and policy.rule13_progress_include_speed_envelope_ceiling
            and own.state.through_water_speed > _EPS
            and own.profile.speed_envelope.maximum_speed > own.state.through_water_speed + _EPS
        ):
            factors.append(
                own.profile.speed_envelope.maximum_speed / own.state.through_water_speed
            )
            factors = list(_dedupe(tuple(factors)))
        factors = [
            factor
            for factor in factors
            if own.profile.speed_envelope.permits(own.state.through_water_speed * factor)
        ]
    if not factors:
        raise EnvironmentalJointSemanticUnavailable(
            f"candidate speed grid exceeds class through-water speed envelope for {aggregate.vessel_id!r}"
        )
    return tuple(factors)


def _commands_for_aggregate(
    context: EnvironmentalAvoidancePlanningContext,
    aggregate: VesselIntentAggregate,
    *,
    policy: IntentDrivenCandidatePolicy,
    progress_plan: Rule13ProgressPlan,
) -> tuple[WaterRelativeManeuverCommand, ...]:
    if not aggregate.compatible:
        raise EnvironmentalJointSemanticUnavailable(
            f"cannot generate commands for conflicting aggregate {aggregate.vessel_id!r}"
        )
    own = context.for_vessel(aggregate.vessel_id)
    state = own.state
    intents = aggregate.intents

    if aggregate.maintain_course_and_speed:
        following = context.following_constraint_for(aggregate.vessel_id)
        speed = state.through_water_speed
        if following is not None:
            speed = min(speed, following.max_follower_speed_nm_per_min)
        return (
            WaterRelativeManeuverCommand(
                state.through_water_heading_deg,
                speed,
            ),
        )

    deltas = _course_deltas_for_aggregate(context, aggregate, policy, progress_plan)
    factors = _speed_factors_for_aggregate(context, aggregate, policy, progress_plan)
    required_change = _requires_effective_change(intents)
    early_substantial = any(item.early_and_substantial for item in intents)

    active = set(progress_plan.active_counterparts_for(aggregate.vessel_id))
    deferred = set(progress_plan.deferred_counterparts_for(aggregate.vessel_id))
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
        item.rule13_past_progress_required and item.counterpart_id in (active | deferred)
        for item in effective_change_intents
    )

    commands: list[WaterRelativeManeuverCommand] = []
    seen: set[tuple[float, float]] = set()

    def append(delta: float, factor: float, *, allow_rule13_hold: bool = False) -> None:
        is_hold = abs(delta) <= _EPS and abs(factor - 1.0) <= _EPS
        if required_change and is_hold and not allow_rule13_hold:
            return
        if early_substantial and not is_hold and not _substantial(delta, factor, policy=policy):
            return
        heading = normalize_heading(state.through_water_heading_deg + delta)
        speed = state.through_water_speed * factor
        key = (round(heading, 12), round(speed, 12))
        if key not in seen:
            seen.add(key)
            commands.append(WaterRelativeManeuverCommand(heading, speed))

    if (_hold_permitted(intents) and not required_change) or rule13_continuation_hold:
        append(0.0, 1.0, allow_rule13_hold=rule13_continuation_hold)
    for delta in deltas:
        for factor in factors:
            append(delta, factor)

    if not commands:
        raise EnvironmentalJointSemanticUnavailable(
            f"candidate grid produced no declaratively permitted command for {aggregate.vessel_id!r}"
        )
    return tuple(commands)


def generate_environmental_joint_command_options(
    context: EnvironmentalAvoidancePlanningContext,
    aggregation: ComponentIntentAggregation,
    *,
    policy: IntentDrivenCandidatePolicy,
    progress_plan: Rule13ProgressPlan,
) -> dict[str, tuple[WaterRelativeManeuverCommand, ...]]:
    """Generate deterministic water-relative options from one compatible JOINT aggregate."""

    if context.backend is not FleetSupervisorBackend.JOINT or len(context.vessel_ids) < 3:
        raise EnvironmentalJointSemanticUnavailable(
            "environmental JOINT semantic generation requires a JOINT context"
        )
    if aggregation.component_id != context.component_id:
        raise EnvironmentalJointSemanticUnavailable(
            "intent aggregation component does not match environmental planning context"
        )
    aggregate_ids = tuple(item.vessel_id for item in aggregation.vessels)
    if aggregate_ids != context.vessel_ids:
        raise EnvironmentalJointSemanticUnavailable(
            "intent aggregation vessel order does not match environmental planning context"
        )
    if not aggregation.compatible:
        raise EnvironmentalJointSemanticUnavailable(
            "JOINT component has conflicting declarative intents"
        )
    if progress_plan.component_id != aggregation.component_id:
        raise EnvironmentalJointSemanticUnavailable(
            "Rule-13 progress plan does not match intent aggregation component"
        )
    return {
        vessel_id: _commands_for_aggregate(
            context,
            aggregation.for_vessel(vessel_id),
            policy=policy,
            progress_plan=progress_plan,
        )
        for vessel_id in context.vessel_ids
    }


def _pair_intents(
    aggregation: ComponentIntentAggregation,
    first_id: str,
    second_id: str,
) -> PairManeuverIntents | None:
    key = tuple(sorted((first_id, second_id)))
    for item in aggregation.pair_intents:
        if item.key == key:
            return item.intents
    return None


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _crosses_ahead(
    first: EnvironmentalVesselMotionEvidence,
    second: EnvironmentalVesselMotionEvidence,
    horizon_min: float,
) -> bool:
    if horizon_min <= _EPS:
        return False
    v1 = first.ground_velocity
    v2 = second.ground_velocity
    denominator = _cross(v1, v2)
    if abs(denominator) <= _EPS:
        return False
    relative = (second.position[0] - first.position[0], second.position[1] - first.position[1])
    first_time = _cross(relative, v2) / denominator
    second_time = _cross(relative, v1) / denominator
    if first_time < -_EPS or second_time < -_EPS:
        return False
    if first_time > horizon_min + _EPS or second_time > horizon_min + _EPS:
        return False
    return first_time + _EPS < second_time


def _pair_transition_motions(
    context: EnvironmentalAvoidancePlanningContext,
    row: EnvironmentalJointTupleEvaluation,
    first_id: str,
    second_id: str,
) -> tuple[float, EnvironmentalVesselMotionEvidence, EnvironmentalVesselMotionEvidence]:
    first_eval = row.vessel_evaluation(first_id)
    second_eval = row.vessel_evaluation(second_id)
    transition_end = max(
        first_eval.transition.transition_duration_min,
        second_eval.transition.transition_duration_min,
    )
    first_inputs = context.for_vessel(first_id)
    second_inputs = context.for_vessel(second_id)
    first_state = first_eval.transition.state_at_or_continue(transition_end)
    second_state = second_eval.transition.state_at_or_continue(transition_end)
    return (
        transition_end,
        environmental_vessel_motion_evidence(first_state, first_inputs.environmental_drift_velocity),
        environmental_vessel_motion_evidence(second_state, second_inputs.environmental_drift_velocity),
    )


def _semantic_pair_evidence(
    context: EnvironmentalAvoidancePlanningContext,
    aggregation: ComponentIntentAggregation,
    progress_plan: Rule13ProgressPlan,
    row: EnvironmentalJointTupleEvaluation,
    first_id: str,
    second_id: str,
    config: EnvironmentalJointSemanticConfig,
) -> EnvironmentalJointPairSemanticEvidence:
    intents = _pair_intents(aggregation, first_id, second_id)
    deferred = progress_plan.is_progress_deferred(tuple(sorted((first_id, second_id))))
    reasons: list[EnvironmentalJointPairSemanticRejection] = []
    first_crosses = False
    second_crosses = False
    assessment = None

    transition_end, first_motion, second_motion = _pair_transition_motions(
        context, row, first_id, second_id
    )
    remaining = config.physics.pair_clearance.evaluation_horizon_min - transition_end

    if intents is not None and remaining > _EPS:
        first_intent = intents.for_vessel(first_id)
        second_intent = intents.for_vessel(second_id)
        if first_intent.avoid_crossing_ahead:
            first_crosses = _crosses_ahead(first_motion, second_motion, remaining)
            if first_crosses:
                reasons.append(EnvironmentalJointPairSemanticRejection.FIRST_CROSSES_AHEAD)
        if second_intent.avoid_crossing_ahead:
            second_crosses = _crosses_ahead(second_motion, first_motion, remaining)
            if second_crosses:
                reasons.append(EnvironmentalJointPairSemanticRejection.SECOND_CROSSES_AHEAD)

    pair_key = tuple(sorted((first_id, second_id)))
    if intents is not None and progress_plan.is_progress_active(pair_key):
        assessment = assess_environmental_rule13_past_progress(
            first_motion,
            second_motion,
            intents.for_vessel(first_id),
            intents.for_vessel(second_id),
            minimum_required_rate_nm_per_min=config.rule13_min_past_progress_rate_nm_per_min,
        )
        if assessment is not None and not assessment.capable:
            reasons.append(EnvironmentalJointPairSemanticRejection.RULE13_NO_PAST_PROGRESS)

    return EnvironmentalJointPairSemanticEvidence(
        first_id,
        second_id,
        intents,
        first_crosses,
        second_crosses,
        assessment,
        deferred,
        tuple(dict.fromkeys(reasons)),
    )


def evaluate_environmental_joint_semantics(
    context: EnvironmentalAvoidancePlanningContext,
    aggregation: ComponentIntentAggregation,
    *,
    config: EnvironmentalJointSemanticConfig,
) -> EnvironmentalJointSemanticGrid:
    """Generate, physically screen and semantically screen one JOINT command product."""

    if not isinstance(context, EnvironmentalAvoidancePlanningContext):
        raise TypeError("context must be EnvironmentalAvoidancePlanningContext")
    if not isinstance(aggregation, ComponentIntentAggregation):
        raise TypeError("aggregation must be ComponentIntentAggregation")
    if not isinstance(config, EnvironmentalJointSemanticConfig):
        raise TypeError("config must be EnvironmentalJointSemanticConfig")
    if not aggregation.compatible:
        raise EnvironmentalJointSemanticUnavailable(
            "JOINT component has conflicting declarative intents"
        )

    progress_plan = build_rule13_progress_plan(
        aggregation,
        policy=config.rule13_progress_coordination,
    )
    command_options = generate_environmental_joint_command_options(
        context,
        aggregation,
        policy=config.candidate_policy,
        progress_plan=progress_plan,
    )
    physics = evaluate_environmental_joint_command_options(
        context,
        command_options=command_options,
        config=config.physics,
    )

    # Pair semantics depend only on the two commands selected for that pair plus the
    # immutable context/aggregation/progress plan/config.  A full Cartesian JOINT grid
    # repeats the same pair-command combination across many tuples, so memoize the
    # evidence exactly as a104 memoizes pair physics.  This preserves the complete
    # Cartesian product and tuple ordering; it only avoids duplicate semantic work.
    semantic_pair_cache: dict[
        tuple[
            str,
            str,
            WaterRelativeManeuverCommand,
            WaterRelativeManeuverCommand,
        ],
        EnvironmentalJointPairSemanticEvidence,
    ] = {}

    evaluations: list[EnvironmentalJointSemanticTuple] = []
    for source in physics.evaluations:
        pair_semantics: list[EnvironmentalJointPairSemanticEvidence] = []
        if source.accepted:
            for pair in source.pair_clearances:
                first_command = source.command_for(pair.first_id)
                second_command = source.command_for(pair.second_id)
                key = (pair.first_id, pair.second_id, first_command, second_command)
                evidence = semantic_pair_cache.get(key)
                if evidence is None:
                    evidence = _semantic_pair_evidence(
                        context,
                        aggregation,
                        progress_plan,
                        source,
                        pair.first_id,
                        pair.second_id,
                        config,
                    )
                    semantic_pair_cache[key] = evidence
                pair_semantics.append(evidence)
        accepted = source.accepted and all(item.accepted for item in pair_semantics)
        evaluations.append(
            EnvironmentalJointSemanticTuple(source, tuple(pair_semantics), accepted)
        )

    return EnvironmentalJointSemanticGrid(
        aggregation,
        progress_plan,
        command_options,
        physics,
        tuple(evaluations),
    )
