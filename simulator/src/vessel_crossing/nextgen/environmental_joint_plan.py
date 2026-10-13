"""Snapshot/control-bound execution plan for one selected environmental JOINT tuple.

A107 binds an a106 selection to the exact a83 planning evidence and to the current
water-relative commands present in ``EnvironmentalWorld2D``.  The artifact is immutable
and has no mutation authority; validation fails closed if physical/planning evidence or
same-timestamp control state changes before application.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_joint_arbiter import EnvironmentalJointArbitrationResult
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .environmental_world import EnvironmentalWorld2D
from .fleet_supervision import FleetSupervisorBackend
from .planning_context import StalePlanningContext
from .vessel_classes import VesselProfileCatalog


class EnvironmentalJointPlanError(RuntimeError):
    """Base error for invalid/stale environmental JOINT plans."""


class StaleEnvironmentalJointPlan(EnvironmentalJointPlanError):
    """Raised when world/control evidence changed after JOINT plan binding."""


class UnsafeEnvironmentalJointPlan(EnvironmentalJointPlanError):
    """Raised when the selected a106 tuple is not a valid a105 survivor."""


@dataclass(frozen=True, slots=True)
class EnvironmentalJointExecutionPlan:
    planning_context: EnvironmentalAvoidancePlanningContext
    arbitration: EnvironmentalJointArbitrationResult
    commands_before: tuple[tuple[str, WaterRelativeManeuverCommand], ...]

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return self.planning_context.vessel_ids

    @property
    def component_id(self) -> str:
        return self.planning_context.component_id

    @property
    def planned_at_time(self) -> float:
        return self.planning_context.time_min

    @property
    def selected_command_index(self) -> int:
        return self.arbitration.selected.evaluation.source_evaluation.command_index

    @property
    def selected_commands(self) -> tuple[tuple[str, WaterRelativeManeuverCommand], ...]:
        source = self.arbitration.selected.evaluation.source_evaluation
        return tuple((vessel_id, source.command_for(vessel_id)) for vessel_id in self.vessel_ids)

    def command_for(self, vessel_id: str) -> WaterRelativeManeuverCommand:
        for candidate_id, command in self.selected_commands:
            if candidate_id == vessel_id:
                return command
        raise KeyError(vessel_id)


def _validate_selected_binding(
    context: EnvironmentalAvoidancePlanningContext,
    arbitration: EnvironmentalJointArbitrationResult,
) -> None:
    if context.backend is not FleetSupervisorBackend.JOINT or len(context.vessel_ids) < 3:
        raise EnvironmentalJointPlanError("environmental JOINT plan requires a JOINT context")
    selected = arbitration.selected.evaluation
    if not selected.accepted:
        raise UnsafeEnvironmentalJointPlan("selected a106 tuple is not an a105 survivor")
    source = selected.source_evaluation
    if not source.accepted:
        raise UnsafeEnvironmentalJointPlan("selected a106 tuple is not a104-safe")
    if tuple(item.vessel_id for item in source.vessel_commands) != context.vessel_ids:
        raise EnvironmentalJointPlanError("selected tuple vessel order does not match planning context")

    for vessel_id in context.vessel_ids:
        inputs = context.for_vessel(vessel_id)
        transition = source.vessel_evaluation(vessel_id).transition
        if (
            transition.initial_state != inputs.state
            or transition.environmental_drift_velocity != inputs.environmental_drift_velocity
            or transition.limits != inputs.maneuver_limits
        ):
            raise EnvironmentalJointPlanError(
                f"selected transition for {vessel_id!r} is not bound to planning-context evidence"
            )


def build_environmental_joint_execution_plan(
    world: EnvironmentalWorld2D,
    context: EnvironmentalAvoidancePlanningContext,
    arbitration: EnvironmentalJointArbitrationResult,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalJointExecutionPlan:
    """Bind one a106 selection to exact physical/control evidence without mutation."""

    context.validate(world, profile_catalog=profile_catalog)
    _validate_selected_binding(context, arbitration)
    return EnvironmentalJointExecutionPlan(
        context,
        arbitration,
        tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in context.vessel_ids),
    )


def validate_environmental_joint_execution_plan(
    plan: EnvironmentalJointExecutionPlan,
    world: EnvironmentalWorld2D,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> None:
    """Fail closed unless planning evidence and all current commands remain unchanged."""

    if not isinstance(plan, EnvironmentalJointExecutionPlan):
        raise TypeError("plan must be EnvironmentalJointExecutionPlan")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    try:
        plan.planning_context.validate(world, profile_catalog=profile_catalog)
    except (StalePlanningContext, KeyError) as exc:
        raise StaleEnvironmentalJointPlan(str(exc)) from exc
    _validate_selected_binding(plan.planning_context, plan.arbitration)
    try:
        current = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in plan.vessel_ids)
    except KeyError as exc:
        raise StaleEnvironmentalJointPlan("planned vessel is no longer active") from exc
    if current != plan.commands_before:
        raise StaleEnvironmentalJointPlan(
            "current water-relative command changed after environmental JOINT plan binding"
        )
