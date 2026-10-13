"""Atomic application of one snapshot-bound environmental pair plan.

A92 is deliberately a narrow execution boundary.  It does not review encounters,
replan, own a mission clock or schedule future avoidance policy.  It validates one a91
plan against the current environmental world, refuses ambiguous same-timestamp pending
command state, atomically queues the selected pair commands at the current timestamp,
and processes that timestamp without advancing physical time.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_pair_plan import (
    EnvironmentalPairExecutionPlan,
    validate_environmental_pair_execution_plan,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-12


class EnvironmentalPairExecutionError(RuntimeError):
    """Base error for fail-closed a92 pair-plan application failures."""


class PendingEnvironmentalCommandEvent(EnvironmentalPairExecutionError):
    """Raised when an unprocessed command already exists at the application time."""


class EnvironmentalPairApplyInvariantViolation(EnvironmentalPairExecutionError):
    """Raised if the policy-free world violates the same-time apply contract."""


@dataclass(frozen=True, slots=True)
class AppliedEnvironmentalPairPlan:
    plan: EnvironmentalPairExecutionPlan
    applied_at_time: float
    snapshot_before: EnvironmentalWorldSnapshot
    snapshot_after: EnvironmentalWorldSnapshot
    commands_before: tuple[WaterRelativeManeuverCommand, WaterRelativeManeuverCommand]
    commands_after: tuple[WaterRelativeManeuverCommand, WaterRelativeManeuverCommand]


def apply_environmental_pair_execution_plan(
    world: EnvironmentalWorld2D,
    plan: EnvironmentalPairExecutionPlan,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> AppliedEnvironmentalPairPlan:
    """Validate then atomically apply one a91 pair plan at ``world.time``.

    Existing unprocessed command events at the same physical timestamp make the
    control state ambiguous relative to a91's captured ``command_for`` evidence and
    therefore fail closed before any mutation.
    """

    validate_environmental_pair_execution_plan(plan, world, profile_catalog=profile_catalog)

    pending = world.next_command_time()
    if pending is not None and pending <= world.time + _EPS:
        raise PendingEnvironmentalCommandEvent(
            "cannot apply environmental pair plan while a command event is pending at current time"
        )

    before = world.snapshot()
    before_commands = (world.command_for(plan.first_id), world.command_for(plan.second_id))
    applied_at = world.time

    world.schedule_commands_atomic(
        at=applied_at,
        commands=(
            (plan.first_id, plan.first_command),
            (plan.second_id, plan.second_command),
        ),
    )
    world.run_until(applied_at)

    after = world.snapshot()
    after_commands = (world.command_for(plan.first_id), world.command_for(plan.second_id))
    if after != before or world.time != applied_at:
        raise EnvironmentalPairApplyInvariantViolation(
            "same-timestamp environmental pair application changed physical state or time"
        )
    expected = (plan.first_command, plan.second_command)
    if after_commands != expected:
        raise EnvironmentalPairApplyInvariantViolation(
            "same-timestamp environmental pair application did not install selected commands"
        )

    return AppliedEnvironmentalPairPlan(
        plan=plan,
        applied_at_time=applied_at,
        snapshot_before=before,
        snapshot_after=after,
        commands_before=before_commands,
        commands_after=after_commands,
    )
