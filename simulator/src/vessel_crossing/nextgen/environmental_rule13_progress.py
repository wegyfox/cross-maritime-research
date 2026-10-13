"""Environmental Rule-13 past-progress capability evidence for safe pair-grid rows.

A89 deliberately sits *after* a88 physical/semantic safety evaluation.  It does not
change Rule-13 responsibility, candidate generation, physical clearance, ranking,
arbitration, ownership or execution.  Its only job is to apply the already established
a55 engineering liveness contract to the environmental velocity-frame model:

- the longitudinal axis is the overtaken vessel's **hull heading** because Rule-13
  ``past`` geometry is hull/aspect-relative;
- relative progress uses **ground velocities** because actual Earth-fixed motion
  determines whether the overtaker is moving toward becoming past;
- the assessment is taken at the common post-transition boundary, after both bounded
  water-relative commands have reached their targets (the earlier finisher continues
  on its reached command to that shared timestamp).

The result remains a capability screen, not a guarantee that finite-time Rule-13
past-and-clear release will occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .dynamics import heading_unit_vector
from .environmental_evidence import (
    EnvironmentalVesselMotionEvidence,
    environmental_vessel_motion_evidence,
)
from .environmental_pair_grid_evaluation import (
    EnvironmentalPairGridEvaluationBatch,
    EnvironmentalPairGridEvaluationRow,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .maneuver_intent import ManeuverIntent
from .overtaking_progress import Rule13ProgressAssessment
from .responsibility import ResponsibilityBasis, ResponsibilityRole

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class EnvironmentalRule13ProgressEvidence:
    """Auditable environmental evidence at one a88 row's common transition boundary."""

    source_row_index: int
    transition_end_min: float
    overtaking_motion: EnvironmentalVesselMotionEvidence | None
    overtaken_motion: EnvironmentalVesselMotionEvidence | None
    assessment: Rule13ProgressAssessment | None

    @property
    def progress_required(self) -> bool:
        return self.assessment is not None

    @property
    def passes(self) -> bool:
        return self.assessment is None or self.assessment.capable


@dataclass(frozen=True, slots=True)
class EnvironmentalRule13ProgressRow:
    """One a88-safe row plus its optional Rule-13 progress evidence."""

    source_row: EnvironmentalPairGridEvaluationRow
    evidence: EnvironmentalRule13ProgressEvidence

    @property
    def row_index(self) -> int:
        return self.source_row.row_index

    @property
    def passes(self) -> bool:
        return self.evidence.passes


@dataclass(frozen=True, slots=True)
class EnvironmentalRule13ProgressBatch:
    """Stable-order progress screen over the a88-safe subset only."""

    source_batch: EnvironmentalPairGridEvaluationBatch
    rows: tuple[EnvironmentalRule13ProgressRow, ...]

    @property
    def surviving_rows(self) -> tuple[EnvironmentalPairGridEvaluationRow, ...]:
        return tuple(row.source_row for row in self.rows if row.passes)

    @property
    def rejected_for_progress(self) -> tuple[EnvironmentalRule13ProgressRow, ...]:
        return tuple(row for row in self.rows if not row.passes)

    @property
    def has_survivor(self) -> bool:
        return any(row.passes for row in self.rows)


def _required_rule13_intent(
    first_intent: ManeuverIntent,
    second_intent: ManeuverIntent,
) -> ManeuverIntent | None:
    required = tuple(
        intent
        for intent in (first_intent, second_intent)
        if intent.rule13_past_progress_required
    )
    if not required:
        return None
    if len(required) != 1:
        raise ValueError("exactly one pair intent may require Rule-13 past progress")
    intent = required[0]
    if intent.basis is not ResponsibilityBasis.RULE_13_OVERTAKING:
        raise ValueError("Rule-13 past-progress requirement is only valid for Rule-13 intent")
    if intent.role is not ResponsibilityRole.GIVE_WAY:
        raise ValueError("Rule-13 past-progress requirement must belong to give-way overtaker")
    return intent


def assess_environmental_rule13_past_progress(
    first_motion: EnvironmentalVesselMotionEvidence,
    second_motion: EnvironmentalVesselMotionEvidence,
    first_intent: ManeuverIntent,
    second_intent: ManeuverIntent,
    *,
    minimum_required_rate_nm_per_min: float = 0.0,
) -> Rule13ProgressAssessment | None:
    """Environmental equivalent of ``assess_rule13_past_progress``.

    Hull heading defines the overtaken vessel's longitudinal axis while relative
    velocity is ground-referenced.  ``0.0`` still means strictly positive progress,
    matching the historical a55 contract exactly.
    """

    if (
        not isfinite(minimum_required_rate_nm_per_min)
        or minimum_required_rate_nm_per_min < 0.0
    ):
        raise ValueError("minimum_required_rate_nm_per_min must be finite and non-negative")
    state_by_id = {first_motion.vessel_id: first_motion, second_motion.vessel_id: second_motion}
    if len(state_by_id) != 2:
        raise ValueError("Rule-13 progress assessment requires two distinct vessel IDs")

    intent = _required_rule13_intent(first_intent, second_intent)
    if intent is None:
        return None
    if intent.vessel_id not in state_by_id or intent.counterpart_id not in state_by_id:
        raise ValueError("Rule-13 intent vessel IDs do not match supplied pair motions")

    overtaker = state_by_id[intent.vessel_id]
    overtaken = state_by_id[intent.counterpart_id]
    ux, uy = heading_unit_vector(overtaken.hull_heading_deg)
    ovx, ovy = overtaker.ground_velocity
    lvx, lvy = overtaken.ground_velocity
    rate = (ovx - lvx) * ux + (ovy - lvy) * uy
    capable = rate > _EPS and rate + _EPS >= minimum_required_rate_nm_per_min
    return Rule13ProgressAssessment(
        overtaking_vessel_id=overtaker.vessel_id,
        overtaken_vessel_id=overtaken.vessel_id,
        relative_forward_progress_rate_nm_per_min=rate,
        minimum_required_rate_nm_per_min=minimum_required_rate_nm_per_min,
        capable=capable,
    )


def _common_transition_motions(
    context: EnvironmentalAvoidancePlanningContext,
    row: EnvironmentalPairGridEvaluationRow,
) -> tuple[float, EnvironmentalVesselMotionEvidence, EnvironmentalVesselMotionEvidence]:
    first_id, second_id = context.vessel_ids
    first_inputs = context.for_vessel(first_id)
    second_inputs = context.for_vessel(second_id)
    evaluation = row.evaluation
    transition_end = max(
        evaluation.first_transition.transition_duration_min,
        evaluation.second_transition.transition_duration_min,
    )
    first_state = evaluation.first_transition.state_at_or_continue(transition_end)
    second_state = evaluation.second_transition.state_at_or_continue(transition_end)
    return (
        transition_end,
        environmental_vessel_motion_evidence(
            first_state, first_inputs.environmental_drift_velocity
        ),
        environmental_vessel_motion_evidence(
            second_state, second_inputs.environmental_drift_velocity
        ),
    )


def filter_environmental_rule13_progress(
    context: EnvironmentalAvoidancePlanningContext,
    batch: EnvironmentalPairGridEvaluationBatch,
    *,
    minimum_required_rate_nm_per_min: float = 0.0,
) -> EnvironmentalRule13ProgressBatch:
    """Screen only a88-safe rows for the optional Rule-13 progress capability contract.

    Rejected a88 rows are intentionally absent: a89 cannot rehabilitate physically or
    semantically invalid commands.  Non-Rule-13 safe rows pass unchanged with an
    explicit ``assessment=None``.  Survivors preserve a88 row order and are not ranked.
    """

    if (
        not isfinite(minimum_required_rate_nm_per_min)
        or minimum_required_rate_nm_per_min < 0.0
    ):
        raise ValueError("minimum_required_rate_nm_per_min must be finite and non-negative")
    if len(context.vessel_ids) != 2:
        raise ValueError("environmental Rule-13 progress filtering requires exactly two context vessels")
    first_id, second_id = context.vessel_ids
    if (
        batch.grid.first_intent.vessel_id != first_id
        or batch.grid.second_intent.vessel_id != second_id
        or batch.grid.first_intent.counterpart_id != second_id
        or batch.grid.second_intent.counterpart_id != first_id
    ):
        raise ValueError("pair-grid intents do not match planning-context vessel order")

    rows: list[EnvironmentalRule13ProgressRow] = []
    for source_row in batch.safe_rows:
        transition_end, first_motion, second_motion = _common_transition_motions(
            context, source_row
        )
        assessment = assess_environmental_rule13_past_progress(
            first_motion,
            second_motion,
            source_row.evaluation.first_intent,
            source_row.evaluation.second_intent,
            minimum_required_rate_nm_per_min=minimum_required_rate_nm_per_min,
        )
        rows.append(
            EnvironmentalRule13ProgressRow(
                source_row=source_row,
                evidence=EnvironmentalRule13ProgressEvidence(
                    source_row_index=source_row.row_index,
                    transition_end_min=transition_end,
                    overtaking_motion=(
                        None
                        if assessment is None
                        else first_motion
                        if assessment.overtaking_vessel_id == first_motion.vessel_id
                        else second_motion
                    ),
                    overtaken_motion=(
                        None
                        if assessment is None
                        else first_motion
                        if assessment.overtaken_vessel_id == first_motion.vessel_id
                        else second_motion
                    ),
                    assessment=assessment,
                ),
            )
        )
    return EnvironmentalRule13ProgressBatch(batch, tuple(rows))
