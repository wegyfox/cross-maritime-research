"""Pure execution-policy decision over one synchronized a109 JOINT boundary record.

A110 is intentionally side-effect free.  It consumes physical execution evidence from
a108 after all original semantic pair lifecycles have been assimilated by a109 at the
same timestamp, and returns only a named local-control action.

Topology discovery/recapture remains outside this module.  Therefore
``HANDOFF_ELIGIBLE`` means only that all *original component pair* responsibilities are
clear; a fleet owner must still verify current topology before route ownership is
resumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .environmental_joint_execution_session import EnvironmentalJointExecutionObservationReason
from .environmental_joint_lifecycle_execution import EnvironmentalJointLifecycleExecutionRecord


class EnvironmentalJointExecutionPolicyAction(str, Enum):
    CONTINUE_SELECTED_COMMAND = "continue_selected_command"
    REPLAN_REQUIRED = "replan_required"
    HANDOFF_ELIGIBLE = "handoff_eligible"


class EnvironmentalJointExecutionPolicyReason(str, Enum):
    COMMAND_OVERRIDDEN = "command_overridden"
    ALL_COMPONENT_RESPONSIBILITY_CLEAR = "all_component_responsibility_clear"
    COMPONENT_RESPONSIBILITY_ACTIVE = "component_responsibility_active"
    SELECTED_TARGETS_COMPLETE = "selected_targets_complete"
    SELECTED_TARGETS_IN_PROGRESS = "selected_targets_in_progress"


@dataclass(frozen=True, slots=True)
class EnvironmentalJointExecutionPolicyDecision:
    source: EnvironmentalJointLifecycleExecutionRecord
    action: EnvironmentalJointExecutionPolicyAction
    reasons: tuple[EnvironmentalJointExecutionPolicyReason, ...]

    @property
    def time_min(self) -> float:
        return self.source.execution.time_min

    @property
    def responsibility_active_pair_keys(self):
        return self.source.responsibility_active_pair_keys


def decide_environmental_joint_execution_policy(
    record: EnvironmentalJointLifecycleExecutionRecord,
) -> EnvironmentalJointExecutionPolicyDecision:
    """Map one synchronized a109 boundary to a named action without mutation."""

    if not isinstance(record, EnvironmentalJointLifecycleExecutionRecord):
        raise TypeError("record must be EnvironmentalJointLifecycleExecutionRecord")
    if not record.pair_decisions:
        raise ValueError("JOINT execution-policy record requires component pair decisions")
    for item in record.pair_decisions:
        if abs(item.decision.observation.time_min - record.execution.time_min) > 1e-10:
            raise ValueError("execution and every pair lifecycle decision must share one timestamp")

    execution = record.execution
    if EnvironmentalJointExecutionObservationReason.COMMAND_OVERRIDDEN in execution.reasons:
        return EnvironmentalJointExecutionPolicyDecision(
            record,
            EnvironmentalJointExecutionPolicyAction.REPLAN_REQUIRED,
            (EnvironmentalJointExecutionPolicyReason.COMMAND_OVERRIDDEN,),
        )

    if record.all_responsibility_clear:
        return EnvironmentalJointExecutionPolicyDecision(
            record,
            EnvironmentalJointExecutionPolicyAction.HANDOFF_ELIGIBLE,
            (EnvironmentalJointExecutionPolicyReason.ALL_COMPONENT_RESPONSIBILITY_CLEAR,),
        )

    all_targets_complete = len(execution.target_reached_ids) == len(execution.vessels)
    if all_targets_complete:
        return EnvironmentalJointExecutionPolicyDecision(
            record,
            EnvironmentalJointExecutionPolicyAction.REPLAN_REQUIRED,
            (
                EnvironmentalJointExecutionPolicyReason.COMPONENT_RESPONSIBILITY_ACTIVE,
                EnvironmentalJointExecutionPolicyReason.SELECTED_TARGETS_COMPLETE,
            ),
        )

    return EnvironmentalJointExecutionPolicyDecision(
        record,
        EnvironmentalJointExecutionPolicyAction.CONTINUE_SELECTED_COMMAND,
        (
            EnvironmentalJointExecutionPolicyReason.COMPONENT_RESPONSIBILITY_ACTIVE,
            EnvironmentalJointExecutionPolicyReason.SELECTED_TARGETS_IN_PROGRESS,
        ),
    )
