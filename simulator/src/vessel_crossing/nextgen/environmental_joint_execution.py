"""Atomic same-timestamp application of one stale-bound environmental JOINT plan."""

from __future__ import annotations

from dataclasses import dataclass

from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_joint_plan import (
    EnvironmentalJointExecutionPlan,
    validate_environmental_joint_execution_plan,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-12


class EnvironmentalJointExecutionError(RuntimeError):
    """Base error for environmental JOINT apply failures."""


class PendingEnvironmentalJointCommandEvent(EnvironmentalJointExecutionError):
    """Raised when an unprocessed command already exists at the application time."""


class EnvironmentalJointApplyInvariantViolation(EnvironmentalJointExecutionError):
    """Raised if same-time JOINT application changes physical state/time or misses commands."""


@dataclass(frozen=True, slots=True)
class AppliedEnvironmentalJointPlan:
    plan: EnvironmentalJointExecutionPlan
    applied_at_time: float
    snapshot_before: EnvironmentalWorldSnapshot
    snapshot_after: EnvironmentalWorldSnapshot
    commands_before: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    commands_after: tuple[tuple[str, WaterRelativeManeuverCommand], ...]


def apply_environmental_joint_execution_plan(
    world: EnvironmentalWorld2D,
    plan: EnvironmentalJointExecutionPlan,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> AppliedEnvironmentalJointPlan:
    """Validate then atomically install every selected JOINT command at ``world.time``."""

    validate_environmental_joint_execution_plan(plan, world, profile_catalog=profile_catalog)
    pending = world.next_command_time()
    if pending is not None and pending <= world.time + _EPS:
        raise PendingEnvironmentalJointCommandEvent(
            "cannot apply environmental JOINT plan while a command event is pending at current time"
        )

    before = world.snapshot()
    before_commands = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in plan.vessel_ids)
    applied_at = world.time
    world.schedule_commands_atomic(at=applied_at, commands=plan.selected_commands)
    world.run_until(applied_at)

    after = world.snapshot()
    after_commands = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in plan.vessel_ids)
    if after != before or world.time != applied_at:
        raise EnvironmentalJointApplyInvariantViolation(
            "same-timestamp environmental JOINT application changed physical state or time"
        )
    if after_commands != plan.selected_commands:
        raise EnvironmentalJointApplyInvariantViolation(
            "same-timestamp environmental JOINT application did not install every selected command"
        )
    return AppliedEnvironmentalJointPlan(
        plan,
        applied_at,
        before,
        after,
        before_commands,
        after_commands,
    )
