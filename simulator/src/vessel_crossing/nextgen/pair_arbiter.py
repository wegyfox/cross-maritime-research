"""Explicit lexicographic arbitration over already-safe coupled avoidance pairs.

A13 deliberately returns all safe pairs and performs no optimization.  A14 accepts an
externally chosen pair.  This module fills that gap without hiding a weighted objective:
the caller must provide an ordered list of named lexicographic criteria, and the final
criterion must explicitly be ``GRID_ORDER`` for deterministic tie-breaking.

The arbiter never relaxes safety.  Rejected a13 evaluations are excluded before scoring.
Route deviation, energy, comfort and vessel-specific economics are not inferred here;
they can be added later as explicit criteria once those models exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .avoidance_supervisor import PairExecutionPlan
from .coupled_avoidance import CoupledAvoidanceEvaluation, CoupledAvoidanceGrid
from .responsibility import ResponsibilityRole


class PairArbitrationCriterion(str, Enum):
    MIN_STAND_ON_COURSE_CHANGE = "min_stand_on_course_change"
    MIN_STAND_ON_SPEED_CHANGE = "min_stand_on_speed_change"
    MIN_TOTAL_COURSE_CHANGE = "min_total_course_change"
    MIN_TOTAL_SPEED_CHANGE = "min_total_speed_change"
    MIN_MAX_COURSE_CHANGE = "min_max_course_change"
    MIN_MAX_SPEED_CHANGE = "min_max_speed_change"
    MAX_PREDICTED_MIN_SEPARATION = "max_predicted_min_separation"
    MAX_MIN_CERTIFIED_DOMAIN_MARGIN = "max_min_certified_domain_margin"
    GRID_ORDER = "grid_order"


class NoSafeCoupledPair(RuntimeError):
    """Raised when arbitration is requested but the coupled grid has no safe pair."""


@dataclass(frozen=True, slots=True)
class PairArbitrationConfig:
    criteria: tuple[PairArbitrationCriterion, ...]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("at least one arbitration criterion is required")
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("arbitration criteria must be unique")
        if self.criteria[-1] is not PairArbitrationCriterion.GRID_ORDER:
            raise ValueError("GRID_ORDER must be the final explicit deterministic tie-break criterion")

    @classmethod
    def caller_order_only(cls) -> "PairArbitrationConfig":
        return cls(criteria=(PairArbitrationCriterion.GRID_ORDER,))

    @classmethod
    def least_intervention(cls) -> "PairArbitrationConfig":
        """Named, inspectable preset; not a regulatory claim.

        Prefer minimum stand-on intervention first, then minimum total course/speed
        change, then larger modeled minimum separation, then caller grid order.
        """
        return cls(
            criteria=(
                PairArbitrationCriterion.MIN_STAND_ON_COURSE_CHANGE,
                PairArbitrationCriterion.MIN_STAND_ON_SPEED_CHANGE,
                PairArbitrationCriterion.MIN_TOTAL_COURSE_CHANGE,
                PairArbitrationCriterion.MIN_TOTAL_SPEED_CHANGE,
                PairArbitrationCriterion.MAX_PREDICTED_MIN_SEPARATION,
                PairArbitrationCriterion.GRID_ORDER,
            )
        )


@dataclass(frozen=True, slots=True)
class PairArbitrationScore:
    evaluation_index: int
    criterion_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RankedCoupledPair:
    evaluation: CoupledAvoidanceEvaluation
    score: PairArbitrationScore


@dataclass(frozen=True, slots=True)
class PairArbitrationResult:
    config: PairArbitrationConfig
    ranked_safe_pairs: tuple[RankedCoupledPair, ...]
    selected: RankedCoupledPair


def _course_change(evaluation: CoupledAvoidanceEvaluation, first: bool) -> float:
    candidate = evaluation.first_candidate if first else evaluation.second_candidate
    return abs(candidate.course_delta_deg)


def _speed_change(evaluation: CoupledAvoidanceEvaluation, first: bool) -> float:
    candidate = evaluation.first_candidate if first else evaluation.second_candidate
    return abs(candidate.speed_fraction - 1.0)


def _stand_on_metric(
    grid: CoupledAvoidanceGrid,
    evaluation: CoupledAvoidanceEvaluation,
    *,
    speed: bool,
) -> float:
    total = 0.0
    if grid.first_intent.role is ResponsibilityRole.STAND_ON:
        total += _speed_change(evaluation, True) if speed else _course_change(evaluation, True)
    if grid.second_intent.role is ResponsibilityRole.STAND_ON:
        total += _speed_change(evaluation, False) if speed else _course_change(evaluation, False)
    return total


def _criterion_value(
    criterion: PairArbitrationCriterion,
    grid: CoupledAvoidanceGrid,
    evaluation: CoupledAvoidanceEvaluation,
    evaluation_index: int,
) -> float:
    first_course = _course_change(evaluation, True)
    second_course = _course_change(evaluation, False)
    first_speed = _speed_change(evaluation, True)
    second_speed = _speed_change(evaluation, False)

    if criterion is PairArbitrationCriterion.MIN_STAND_ON_COURSE_CHANGE:
        return _stand_on_metric(grid, evaluation, speed=False)
    if criterion is PairArbitrationCriterion.MIN_STAND_ON_SPEED_CHANGE:
        return _stand_on_metric(grid, evaluation, speed=True)
    if criterion is PairArbitrationCriterion.MIN_TOTAL_COURSE_CHANGE:
        return first_course + second_course
    if criterion is PairArbitrationCriterion.MIN_TOTAL_SPEED_CHANGE:
        return first_speed + second_speed
    if criterion is PairArbitrationCriterion.MIN_MAX_COURSE_CHANGE:
        return max(first_course, second_course)
    if criterion is PairArbitrationCriterion.MIN_MAX_SPEED_CHANGE:
        return max(first_speed, second_speed)
    if criterion is PairArbitrationCriterion.MAX_PREDICTED_MIN_SEPARATION:
        # Accepted a13 pairs always have a physical separation prediction.  Keep an
        # explicit guard in case a future evaluator introduces non-trajectory safe rows.
        if evaluation.predicted_min_separation_nm is None:
            raise ValueError("safe coupled pair lacks predicted minimum separation")
        return -evaluation.predicted_min_separation_nm
    if criterion is PairArbitrationCriterion.MAX_MIN_CERTIFIED_DOMAIN_MARGIN:
        margin = evaluation.minimum_certified_domain_margin_nm
        if margin is None:
            raise ValueError(
                "MAX_MIN_CERTIFIED_DOMAIN_MARGIN requires shaped-domain-screened safe pairs"
            )
        return -margin
    if criterion is PairArbitrationCriterion.GRID_ORDER:
        return float(evaluation_index)
    raise AssertionError(f"unhandled arbitration criterion {criterion!r}")


def arbitrate_coupled_pairs(
    grid: CoupledAvoidanceGrid,
    *,
    config: PairArbitrationConfig,
) -> PairArbitrationResult:
    """Rank only safe a13 pairs using the caller's explicit lexicographic policy."""
    indexed_safe = [
        (index, evaluation)
        for index, evaluation in enumerate(grid.evaluations)
        if evaluation.accepted
    ]
    if not indexed_safe:
        raise NoSafeCoupledPair("coupled avoidance grid contains no safe pair")

    ranked = []
    for index, evaluation in indexed_safe:
        values = tuple(
            _criterion_value(criterion, grid, evaluation, index)
            for criterion in config.criteria
        )
        ranked.append(
            RankedCoupledPair(
                evaluation=evaluation,
                score=PairArbitrationScore(
                    evaluation_index=index,
                    criterion_values=values,
                ),
            )
        )

    ranked.sort(key=lambda item: item.score.criterion_values)
    return PairArbitrationResult(
        config=config,
        ranked_safe_pairs=tuple(ranked),
        selected=ranked[0],
    )


def build_pair_execution_plan(
    grid: CoupledAvoidanceGrid,
    *,
    planned_at_time: float,
    config: PairArbitrationConfig,
) -> PairExecutionPlan:
    """Arbitrate a safe pair and bind it to ordered IDs/time for the a14 supervisor."""
    result = arbitrate_coupled_pairs(grid, config=config)
    return PairExecutionPlan(
        first_id=grid.first_intent.vessel_id,
        second_id=grid.second_intent.vessel_id,
        planned_at_time=planned_at_time,
        evaluation=result.selected.evaluation,
    )
