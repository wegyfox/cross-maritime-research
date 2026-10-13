"""Explicit lexicographic ranking over a89 environmental pair-row survivors.

A90 reuses the historical pair-arbitration criterion vocabulary instead of inventing a
second objective system.  It consumes only rows that are already a88-safe and, when
Rule-13 progress is required, a89 progress-capable.  Ranking cannot weaken either
filter and has no command-ownership or execution side effect.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environmental_rule13_progress import (
    EnvironmentalRule13ProgressBatch,
    EnvironmentalRule13ProgressRow,
)
from .pair_arbiter import PairArbitrationConfig, PairArbitrationCriterion
from .responsibility import ResponsibilityRole


class NoEnvironmentalPairCandidate(RuntimeError):
    """Raised when ranking is requested but no a89 survivor exists."""


@dataclass(frozen=True, slots=True)
class EnvironmentalPairArbitrationScore:
    row_index: int
    criterion_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RankedEnvironmentalPairRow:
    row: EnvironmentalRule13ProgressRow
    score: EnvironmentalPairArbitrationScore


@dataclass(frozen=True, slots=True)
class EnvironmentalPairArbitrationResult:
    config: PairArbitrationConfig
    ranked_rows: tuple[RankedEnvironmentalPairRow, ...]
    selected: RankedEnvironmentalPairRow


def _course_change(row: EnvironmentalRule13ProgressRow, first: bool) -> float:
    generated = (
        row.source_row.first_generated_command
        if first
        else row.source_row.second_generated_command
    )
    return abs(generated.course_delta_deg)


def _speed_change(row: EnvironmentalRule13ProgressRow, first: bool) -> float:
    generated = (
        row.source_row.first_generated_command
        if first
        else row.source_row.second_generated_command
    )
    return abs(generated.speed_fraction - 1.0)


def _stand_on_metric(
    batch: EnvironmentalRule13ProgressBatch,
    row: EnvironmentalRule13ProgressRow,
    *,
    speed: bool,
) -> float:
    total = 0.0
    grid = batch.source_batch.grid
    if grid.first_intent.role is ResponsibilityRole.STAND_ON:
        total += _speed_change(row, True) if speed else _course_change(row, True)
    if grid.second_intent.role is ResponsibilityRole.STAND_ON:
        total += _speed_change(row, False) if speed else _course_change(row, False)
    return total


def _minimum_certified_domain_margin(row: EnvironmentalRule13ProgressRow) -> float | None:
    evaluation = row.source_row.evaluation
    values: list[float] = []
    if (
        evaluation.transition_clearance is not None
        and evaluation.transition_clearance.domain is not None
    ):
        values.append(evaluation.transition_clearance.domain.conservative_lower_bound_nm)
    if evaluation.post_transition_domain_clearance is not None:
        values.append(
            evaluation.post_transition_domain_clearance.conservative_lower_bound_nm
        )
    return min(values) if values else None


def _criterion_value(
    criterion: PairArbitrationCriterion,
    batch: EnvironmentalRule13ProgressBatch,
    row: EnvironmentalRule13ProgressRow,
) -> float:
    first_course = _course_change(row, True)
    second_course = _course_change(row, False)
    first_speed = _speed_change(row, True)
    second_speed = _speed_change(row, False)

    if criterion is PairArbitrationCriterion.MIN_STAND_ON_COURSE_CHANGE:
        return _stand_on_metric(batch, row, speed=False)
    if criterion is PairArbitrationCriterion.MIN_STAND_ON_SPEED_CHANGE:
        return _stand_on_metric(batch, row, speed=True)
    if criterion is PairArbitrationCriterion.MIN_TOTAL_COURSE_CHANGE:
        return first_course + second_course
    if criterion is PairArbitrationCriterion.MIN_TOTAL_SPEED_CHANGE:
        return first_speed + second_speed
    if criterion is PairArbitrationCriterion.MIN_MAX_COURSE_CHANGE:
        return max(first_course, second_course)
    if criterion is PairArbitrationCriterion.MIN_MAX_SPEED_CHANGE:
        return max(first_speed, second_speed)
    if criterion is PairArbitrationCriterion.MAX_PREDICTED_MIN_SEPARATION:
        predicted = row.source_row.evaluation.predicted_min_separation_nm
        if predicted is None:
            raise ValueError("safe environmental pair row lacks predicted minimum separation")
        return -predicted
    if criterion is PairArbitrationCriterion.MAX_MIN_CERTIFIED_DOMAIN_MARGIN:
        margin = _minimum_certified_domain_margin(row)
        if margin is None:
            raise ValueError(
                "MAX_MIN_CERTIFIED_DOMAIN_MARGIN requires shaped-domain-screened environmental rows"
            )
        return -margin
    if criterion is PairArbitrationCriterion.GRID_ORDER:
        return float(row.row_index)
    raise AssertionError(f"unhandled arbitration criterion {criterion!r}")


def arbitrate_environmental_pair_rows(
    batch: EnvironmentalRule13ProgressBatch,
    *,
    config: PairArbitrationConfig,
) -> EnvironmentalPairArbitrationResult:
    """Rank only a89 survivors under an explicit historical pair-arbitration policy."""

    survivors = [row for row in batch.rows if row.passes]
    if not survivors:
        raise NoEnvironmentalPairCandidate(
            "environmental pair batch contains no safe progress-capable row"
        )

    ranked: list[RankedEnvironmentalPairRow] = []
    for row in survivors:
        values = tuple(
            _criterion_value(criterion, batch, row)
            for criterion in config.criteria
        )
        ranked.append(
            RankedEnvironmentalPairRow(
                row=row,
                score=EnvironmentalPairArbitrationScore(
                    row_index=row.row_index,
                    criterion_values=values,
                ),
            )
        )
    ranked.sort(key=lambda item: item.score.criterion_values)
    return EnvironmentalPairArbitrationResult(
        config=config,
        ranked_rows=tuple(ranked),
        selected=ranked[0],
    )
