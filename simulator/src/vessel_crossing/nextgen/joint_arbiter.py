"""Explicit lexicographic arbitration over already-safe a27 joint command tuples.

The a27 evaluator returns every caller-supplied Cartesian-product row and marks the
safe subset.  This module ranks only that safe subset.  It never re-runs or weakens
safety checks and it does not use weighted objectives.  The caller must provide an
ordered list of named lexicographic criteria; ``GRID_ORDER`` is required as the final
explicit deterministic tie-break.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .joint_avoidance import JointAvoidanceEvaluation, JointAvoidanceGrid


class JointArbitrationCriterion(str, Enum):
    MIN_TOTAL_COURSE_CHANGE = "min_total_course_change"
    MIN_TOTAL_SPEED_CHANGE = "min_total_speed_change"
    MIN_MAX_COURSE_CHANGE = "min_max_course_change"
    MIN_MAX_SPEED_CHANGE = "min_max_speed_change"
    MAX_PREDICTED_MIN_SEPARATION = "max_predicted_min_separation"
    MAX_MIN_CERTIFIED_DOMAIN_MARGIN = "max_min_certified_domain_margin"
    GRID_ORDER = "grid_order"


class NoSafeJointTuple(RuntimeError):
    """Raised when arbitration is requested but a27 contains no safe joint tuple."""


@dataclass(frozen=True, slots=True)
class JointArbitrationConfig:
    criteria: tuple[JointArbitrationCriterion, ...]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("at least one arbitration criterion is required")
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("arbitration criteria must be unique")
        if self.criteria[-1] is not JointArbitrationCriterion.GRID_ORDER:
            raise ValueError("GRID_ORDER must be the final explicit deterministic tie-break criterion")

    @classmethod
    def caller_order_only(cls) -> "JointArbitrationConfig":
        return cls((JointArbitrationCriterion.GRID_ORDER,))

    @classmethod
    def least_intervention(cls) -> "JointArbitrationConfig":
        """Transparent model preset; not a COLREG priority rule."""
        return cls(
            (
                JointArbitrationCriterion.MIN_TOTAL_COURSE_CHANGE,
                JointArbitrationCriterion.MIN_TOTAL_SPEED_CHANGE,
                JointArbitrationCriterion.MIN_MAX_COURSE_CHANGE,
                JointArbitrationCriterion.MIN_MAX_SPEED_CHANGE,
                JointArbitrationCriterion.MAX_PREDICTED_MIN_SEPARATION,
                JointArbitrationCriterion.GRID_ORDER,
            )
        )


@dataclass(frozen=True, slots=True)
class JointArbitrationScore:
    evaluation_index: int
    criterion_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RankedJointTuple:
    evaluation: JointAvoidanceEvaluation
    score: JointArbitrationScore


@dataclass(frozen=True, slots=True)
class JointArbitrationResult:
    config: JointArbitrationConfig
    ranked_safe_tuples: tuple[RankedJointTuple, ...]
    selected: RankedJointTuple


def _course_changes(evaluation: JointAvoidanceEvaluation) -> tuple[float, ...]:
    return tuple(abs(item.course_delta_deg) for item in evaluation.vessel_commands)


def _speed_changes(evaluation: JointAvoidanceEvaluation) -> tuple[float, ...]:
    values = []
    for item in evaluation.vessel_commands:
        if item.speed_fraction is None:
            # Initial speed zero: the ratio is undefined.  Use absolute internal speed
            # change rather than inventing an infinite/zero fraction.
            values.append(abs(item.command.target_speed))
        else:
            values.append(abs(item.speed_fraction - 1.0))
    return tuple(values)


def _criterion_value(
    criterion: JointArbitrationCriterion,
    evaluation: JointAvoidanceEvaluation,
    evaluation_index: int,
) -> float:
    course = _course_changes(evaluation)
    speed = _speed_changes(evaluation)
    if criterion is JointArbitrationCriterion.MIN_TOTAL_COURSE_CHANGE:
        return sum(course)
    if criterion is JointArbitrationCriterion.MIN_TOTAL_SPEED_CHANGE:
        return sum(speed)
    if criterion is JointArbitrationCriterion.MIN_MAX_COURSE_CHANGE:
        return max(course, default=0.0)
    if criterion is JointArbitrationCriterion.MIN_MAX_SPEED_CHANGE:
        return max(speed, default=0.0)
    if criterion is JointArbitrationCriterion.MAX_PREDICTED_MIN_SEPARATION:
        if evaluation.predicted_min_separation_nm is None:
            raise ValueError("safe joint tuple lacks predicted minimum separation")
        return -evaluation.predicted_min_separation_nm
    if criterion is JointArbitrationCriterion.MAX_MIN_CERTIFIED_DOMAIN_MARGIN:
        margin = evaluation.minimum_certified_domain_margin_nm
        if margin is None:
            raise ValueError(
                "MAX_MIN_CERTIFIED_DOMAIN_MARGIN requires domain clearance for every joint pair"
            )
        return -margin
    if criterion is JointArbitrationCriterion.GRID_ORDER:
        return float(evaluation_index)
    raise AssertionError(f"unhandled arbitration criterion {criterion!r}")


def arbitrate_joint_tuples(
    grid: JointAvoidanceGrid,
    *,
    config: JointArbitrationConfig,
) -> JointArbitrationResult:
    """Rank only a27-safe tuples using the caller's explicit lexicographic policy."""
    indexed_safe = [
        (index, evaluation)
        for index, evaluation in enumerate(grid.evaluations)
        if evaluation.accepted
    ]
    if not indexed_safe:
        raise NoSafeJointTuple("joint avoidance grid contains no safe tuple")

    ranked: list[RankedJointTuple] = []
    for index, evaluation in indexed_safe:
        values = tuple(
            _criterion_value(criterion, evaluation, index)
            for criterion in config.criteria
        )
        ranked.append(
            RankedJointTuple(
                evaluation=evaluation,
                score=JointArbitrationScore(
                    evaluation_index=index,
                    criterion_values=values,
                ),
            )
        )

    ranked.sort(key=lambda item: item.score.criterion_values)
    return JointArbitrationResult(
        config=config,
        ranked_safe_tuples=tuple(ranked),
        selected=ranked[0],
    )
