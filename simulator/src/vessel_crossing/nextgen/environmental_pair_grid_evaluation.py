"""Ordered batch evaluation for one deterministic environmental pair-command grid.

A88 is deliberately orchestration-only.  It preserves the exact Cartesian row order
produced by a87 and evaluates every row through the unchanged a86 explicit pair
candidate evaluator.  It does not add Rule-13 progress filtering, ranking, early exit,
arbitration, ownership mutation, or execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environmental_candidate_generation import (
    EnvironmentalGeneratedCommand,
    EnvironmentalPairCommandGrid,
)
from .environmental_pair_candidate import (
    EnvironmentalPairCandidateConfig,
    EnvironmentalPairCandidateEvaluation,
    evaluate_environmental_pair_candidate,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .maneuver_intent import PairManeuverIntents


@dataclass(frozen=True, slots=True)
class EnvironmentalPairGridEvaluationRow:
    """One immutable a87 Cartesian row and its complete a86 evaluation evidence."""

    row_index: int
    first_generated_command: EnvironmentalGeneratedCommand
    second_generated_command: EnvironmentalGeneratedCommand
    evaluation: EnvironmentalPairCandidateEvaluation

    @property
    def accepted(self) -> bool:
        return self.evaluation.accepted


@dataclass(frozen=True, slots=True)
class EnvironmentalPairGridEvaluationBatch:
    """All a86 evaluations in the exact deterministic order of one a87 grid."""

    grid: EnvironmentalPairCommandGrid
    rows: tuple[EnvironmentalPairGridEvaluationRow, ...]

    @property
    def safe_rows(self) -> tuple[EnvironmentalPairGridEvaluationRow, ...]:
        return tuple(row for row in self.rows if row.accepted)

    @property
    def has_safe_row(self) -> bool:
        return any(row.accepted for row in self.rows)


def evaluate_environmental_pair_command_grid(
    context: EnvironmentalAvoidancePlanningContext,
    grid: EnvironmentalPairCommandGrid,
    *,
    config: EnvironmentalPairCandidateConfig,
) -> EnvironmentalPairGridEvaluationBatch:
    """Evaluate every deterministic grid row without ranking or short-circuiting."""

    first_id, second_id = context.vessel_ids if len(context.vessel_ids) == 2 else (None, None)
    if first_id is None or second_id is None:
        raise ValueError("environmental pair grid evaluation requires exactly two context vessels")
    if grid.first_intent.vessel_id != first_id or grid.second_intent.vessel_id != second_id:
        raise ValueError("pair command grid intent order does not match planning-context vessel order")
    if grid.first_intent.counterpart_id != second_id or grid.second_intent.counterpart_id != first_id:
        raise ValueError("pair command grid intents do not match planning-context pair")
    if not grid.first_commands or not grid.second_commands:
        raise ValueError("pair command grid must contain at least one command for each vessel")

    intents = PairManeuverIntents(grid.first_intent, grid.second_intent)
    rows: list[EnvironmentalPairGridEvaluationRow] = []
    for row_index, (first_generated, second_generated) in enumerate(grid.command_pairs):
        evaluation = evaluate_environmental_pair_candidate(
            context,
            intents,
            first_command=first_generated.command,
            second_command=second_generated.command,
            config=config,
        )
        rows.append(
            EnvironmentalPairGridEvaluationRow(
                row_index=row_index,
                first_generated_command=first_generated,
                second_generated_command=second_generated,
                evaluation=evaluation,
            )
        )
    return EnvironmentalPairGridEvaluationBatch(grid=grid, rows=tuple(rows))
