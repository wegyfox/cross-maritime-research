"""Explicit lexicographic ranking over a105 environmental JOINT survivors.

A106 reuses the historical ``JointArbitrationConfig`` criterion vocabulary.  It ranks
only tuples that already passed a104 all-pairs physics and a105 Rule-15/Rule-13
semantics.  Intervention is measured in the water-relative control frame captured by
the environmental planning context.  The arbiter does not mutate the world, bind a
plan, or apply commands.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dynamics import signed_heading_delta
from .environmental_joint_semantics import EnvironmentalJointSemanticGrid, EnvironmentalJointSemanticTuple
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .joint_arbiter import JointArbitrationConfig, JointArbitrationCriterion


class NoEnvironmentalJointTuple(RuntimeError):
    """Raised when environmental JOINT arbitration has no a105 survivor."""


@dataclass(frozen=True, slots=True)
class EnvironmentalJointArbitrationScore:
    command_index: int
    criterion_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RankedEnvironmentalJointTuple:
    evaluation: EnvironmentalJointSemanticTuple
    score: EnvironmentalJointArbitrationScore


@dataclass(frozen=True, slots=True)
class EnvironmentalJointArbitrationResult:
    config: JointArbitrationConfig
    ranked_safe_tuples: tuple[RankedEnvironmentalJointTuple, ...]
    selected: RankedEnvironmentalJointTuple


def _course_changes(
    context: EnvironmentalAvoidancePlanningContext,
    evaluation: EnvironmentalJointSemanticTuple,
) -> tuple[float, ...]:
    source = evaluation.source_evaluation
    return tuple(
        abs(
            signed_heading_delta(
                context.for_vessel(vessel_id).state.through_water_heading_deg,
                source.command_for(vessel_id).target_through_water_heading_deg,
            )
        )
        for vessel_id in context.vessel_ids
    )


def _speed_changes(
    context: EnvironmentalAvoidancePlanningContext,
    evaluation: EnvironmentalJointSemanticTuple,
) -> tuple[float, ...]:
    source = evaluation.source_evaluation
    values: list[float] = []
    for vessel_id in context.vessel_ids:
        initial = context.for_vessel(vessel_id).state.through_water_speed
        target = source.command_for(vessel_id).target_through_water_speed
        if initial == 0.0:
            values.append(abs(target))
        else:
            values.append(abs(target / initial - 1.0))
    return tuple(values)


def _criterion_value(
    criterion: JointArbitrationCriterion,
    context: EnvironmentalAvoidancePlanningContext,
    evaluation: EnvironmentalJointSemanticTuple,
) -> float:
    course = _course_changes(context, evaluation)
    speed = _speed_changes(context, evaluation)
    source = evaluation.source_evaluation
    if criterion is JointArbitrationCriterion.MIN_TOTAL_COURSE_CHANGE:
        return sum(course)
    if criterion is JointArbitrationCriterion.MIN_TOTAL_SPEED_CHANGE:
        return sum(speed)
    if criterion is JointArbitrationCriterion.MIN_MAX_COURSE_CHANGE:
        return max(course, default=0.0)
    if criterion is JointArbitrationCriterion.MIN_MAX_SPEED_CHANGE:
        return max(speed, default=0.0)
    if criterion is JointArbitrationCriterion.MAX_PREDICTED_MIN_SEPARATION:
        if source.predicted_min_separation_nm is None:
            raise ValueError("safe environmental JOINT tuple lacks predicted minimum separation")
        return -source.predicted_min_separation_nm
    if criterion is JointArbitrationCriterion.MAX_MIN_CERTIFIED_DOMAIN_MARGIN:
        margin = source.minimum_certified_domain_margin_nm
        if margin is None:
            raise ValueError(
                "MAX_MIN_CERTIFIED_DOMAIN_MARGIN requires domain clearance for every environmental JOINT pair"
            )
        return -margin
    if criterion is JointArbitrationCriterion.GRID_ORDER:
        return float(source.command_index)
    raise AssertionError(f"unhandled arbitration criterion {criterion!r}")


def arbitrate_environmental_joint_tuples(
    context: EnvironmentalAvoidancePlanningContext,
    grid: EnvironmentalJointSemanticGrid,
    *,
    config: JointArbitrationConfig,
) -> EnvironmentalJointArbitrationResult:
    """Rank only a105 survivors using one explicit historical JOINT policy."""

    if not isinstance(context, EnvironmentalAvoidancePlanningContext):
        raise TypeError("context must be EnvironmentalAvoidancePlanningContext")
    if not isinstance(grid, EnvironmentalJointSemanticGrid):
        raise TypeError("grid must be EnvironmentalJointSemanticGrid")
    if not isinstance(config, JointArbitrationConfig):
        raise TypeError("config must be JointArbitrationConfig")
    if grid.physics_grid.component_id != context.component_id:
        raise ValueError("environmental JOINT grid component does not match planning context")
    if grid.physics_grid.vessel_ids != context.vessel_ids:
        raise ValueError("environmental JOINT grid vessel order does not match planning context")

    survivors = tuple(item for item in grid.evaluations if item.accepted)
    if not survivors:
        raise NoEnvironmentalJointTuple("environmental JOINT semantic grid contains no safe tuple")

    ranked: list[RankedEnvironmentalJointTuple] = []
    for evaluation in survivors:
        values = tuple(_criterion_value(criterion, context, evaluation) for criterion in config.criteria)
        ranked.append(
            RankedEnvironmentalJointTuple(
                evaluation,
                EnvironmentalJointArbitrationScore(
                    evaluation.source_evaluation.command_index,
                    values,
                ),
            )
        )
    ranked.sort(key=lambda item: item.score.criterion_values)
    return EnvironmentalJointArbitrationResult(config, tuple(ranked), ranked[0])
