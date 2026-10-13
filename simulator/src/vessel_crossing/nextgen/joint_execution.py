"""Snapshot-bound execution plans for selected safe joint command tuples.

A27 establishes physical safety for one explicit world state; a28 chooses among the
safe tuples.  This module binds that chosen tuple to the exact full ``WorldSnapshot``
used at planning time and applies all component commands only if the world still
matches that snapshot exactly.

The full-snapshot check is intentional.  A same-time vessel arrival/removal can change
encounter topology without advancing the clock, so validating only ``world.time`` or
only the controlled component would be insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

from .joint_arbiter import JointArbitrationConfig, JointArbitrationResult, arbitrate_joint_tuples
from .joint_avoidance import JointAvoidanceEvaluation, JointAvoidanceGrid
from .world import World2D, WorldSnapshot


class JointExecutionError(RuntimeError):
    """Base class for fail-closed joint execution errors."""


class StaleJointExecutionPlan(JointExecutionError):
    """Raised when the current world no longer equals the planning snapshot."""


class UnsafeJointExecutionPlan(JointExecutionError):
    """Raised when a plan contains a joint evaluation that was not accepted by a27."""


@dataclass(frozen=True, slots=True)
class JointExecutionPlan:
    component_id: str
    vessel_ids: tuple[str, ...]
    planning_snapshot: WorldSnapshot
    evaluation: JointAvoidanceEvaluation

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        if len(self.vessel_ids) < 2 or any(not vessel_id for vessel_id in self.vessel_ids):
            raise ValueError("joint execution plan requires at least two non-empty vessel ids")
        if len(set(self.vessel_ids)) != len(self.vessel_ids):
            raise ValueError("joint execution plan vessel ids must be unique")
        evaluation_ids = tuple(item.vessel_id for item in self.evaluation.vessel_commands)
        if evaluation_ids != self.vessel_ids:
            raise ValueError("execution plan vessel order must exactly match evaluation vessel order")
        snapshot_ids = {item.vessel_id for item in self.planning_snapshot.vessels}
        missing = set(self.vessel_ids) - snapshot_ids
        if missing:
            raise ValueError(f"planning snapshot is missing component vessels: {sorted(missing)}")
        if not self.evaluation.accepted:
            raise UnsafeJointExecutionPlan("joint avoidance evaluation is not accepted")

    @property
    def planned_at_time(self) -> float:
        return self.planning_snapshot.time

    def command_for(self, vessel_id: str):
        return self.evaluation.command_for(vessel_id)


@dataclass(frozen=True, slots=True)
class JointExecutionReceipt:
    plan: JointExecutionPlan
    snapshot_before: WorldSnapshot
    snapshot_after: WorldSnapshot


@dataclass(frozen=True, slots=True)
class JointExecutionPlanningResult:
    arbitration: JointArbitrationResult
    plan: JointExecutionPlan


def build_joint_execution_plan(
    grid: JointAvoidanceGrid,
    *,
    planning_snapshot: WorldSnapshot,
    config: JointArbitrationConfig,
) -> JointExecutionPlanningResult:
    """Arbitrate an a27 grid and bind the selected safe tuple to a full snapshot."""
    snapshot_ids = {item.vessel_id for item in planning_snapshot.vessels}
    missing = set(grid.vessel_ids) - snapshot_ids
    if missing:
        raise ValueError(f"planning snapshot is missing grid vessels: {sorted(missing)}")
    arbitration = arbitrate_joint_tuples(grid, config=config)
    plan = JointExecutionPlan(
        component_id=grid.component_id,
        vessel_ids=grid.vessel_ids,
        planning_snapshot=planning_snapshot,
        evaluation=arbitration.selected.evaluation,
    )
    return JointExecutionPlanningResult(arbitration=arbitration, plan=plan)


def apply_joint_execution_plan(world: World2D, plan: JointExecutionPlan) -> JointExecutionReceipt:
    """Apply every component command at the current timestamp after exact snapshot validation."""
    current = world.snapshot()
    if current != plan.planning_snapshot:
        raise StaleJointExecutionPlan(
            "current world snapshot differs from the snapshot used to build the joint plan"
        )
    if not plan.evaluation.accepted:
        # Defensive guard even though the frozen dataclass constructor already checks.
        raise UnsafeJointExecutionPlan("joint avoidance evaluation is not accepted")

    # Validate all controlled IDs before scheduling any event, so a missing vessel
    # cannot leave a partially scheduled command set.
    for vessel_id in plan.vessel_ids:
        current.by_id(vessel_id)

    for vessel_id in plan.vessel_ids:
        world.schedule_command(
            at=world.time,
            vessel_id=vessel_id,
            command=plan.command_for(vessel_id),
        )
    # Process the complete same-time command batch. No physical time elapses.
    world.run_until(world.time)
    return JointExecutionReceipt(
        plan=plan,
        snapshot_before=current,
        snapshot_after=world.snapshot(),
    )
