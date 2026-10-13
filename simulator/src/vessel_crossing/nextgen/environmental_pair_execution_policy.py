"""Pure execution-policy decision over one a94 boundary record.

A95 turns already-observed execution + lifecycle evidence into a named control decision,
but deliberately performs none of the actions itself.  It is boundary-local: callers
must provide one real a94 record whose lifecycle decision was assimilated at the same
physical timestamp as the execution observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .environmental_pair_execution_session import EnvironmentalPairExecutionObservationReason
from .environmental_pair_lifecycle_execution import EnvironmentalPairLifecycleExecutionRecord


class EnvironmentalPairExecutionPolicyAction(str, Enum):
    CONTINUE_SELECTED_COMMAND = "continue_selected_command"
    REPLAN_REQUIRED = "replan_required"
    HANDOFF_ELIGIBLE = "handoff_eligible"


class EnvironmentalPairExecutionPolicyReason(str, Enum):
    COMMAND_OVERRIDDEN = "command_overridden"
    RESPONSIBILITY_CLEAR = "responsibility_clear"
    RESPONSIBILITY_ACTIVE = "responsibility_active"
    SELECTED_TARGETS_COMPLETE = "selected_targets_complete"
    SELECTED_TARGETS_IN_PROGRESS = "selected_targets_in_progress"


@dataclass(frozen=True, slots=True)
class EnvironmentalPairExecutionPolicyDecision:
    source: EnvironmentalPairLifecycleExecutionRecord
    action: EnvironmentalPairExecutionPolicyAction
    reasons: tuple[EnvironmentalPairExecutionPolicyReason, ...]

    @property
    def time_min(self) -> float:
        return self.source.execution.time_min


def decide_environmental_pair_execution_policy(
    record: EnvironmentalPairLifecycleExecutionRecord,
) -> EnvironmentalPairExecutionPolicyDecision:
    """Map one synchronized a94 boundary record to a named, side-effect-free action."""

    if not isinstance(record, EnvironmentalPairLifecycleExecutionRecord):
        raise TypeError("record must be EnvironmentalPairLifecycleExecutionRecord")
    if abs(record.execution.time_min - record.decision.observation.time_min) > 1e-10:
        raise ValueError("execution and lifecycle evidence must share one physical timestamp")

    execution = record.execution
    decision = record.decision
    if EnvironmentalPairExecutionObservationReason.COMMAND_OVERRIDDEN in execution.reasons:
        return EnvironmentalPairExecutionPolicyDecision(
            record,
            EnvironmentalPairExecutionPolicyAction.REPLAN_REQUIRED,
            (EnvironmentalPairExecutionPolicyReason.COMMAND_OVERRIDDEN,),
        )

    if not decision.responsibility_active:
        return EnvironmentalPairExecutionPolicyDecision(
            record,
            EnvironmentalPairExecutionPolicyAction.HANDOFF_ELIGIBLE,
            (EnvironmentalPairExecutionPolicyReason.RESPONSIBILITY_CLEAR,),
        )

    complete = execution.first_target_reached and execution.second_target_reached
    if complete:
        return EnvironmentalPairExecutionPolicyDecision(
            record,
            EnvironmentalPairExecutionPolicyAction.REPLAN_REQUIRED,
            (
                EnvironmentalPairExecutionPolicyReason.RESPONSIBILITY_ACTIVE,
                EnvironmentalPairExecutionPolicyReason.SELECTED_TARGETS_COMPLETE,
            ),
        )

    return EnvironmentalPairExecutionPolicyDecision(
        record,
        EnvironmentalPairExecutionPolicyAction.CONTINUE_SELECTED_COMMAND,
        (
            EnvironmentalPairExecutionPolicyReason.RESPONSIBILITY_ACTIVE,
            EnvironmentalPairExecutionPolicyReason.SELECTED_TARGETS_IN_PROGRESS,
        ),
    )
