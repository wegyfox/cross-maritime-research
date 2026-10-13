"""Snapshot- and command-bound execution-plan artifact for an a90 selected pair row.

A91 still does not mutate ``EnvironmentalWorld2D``.  It binds one selected a90 row to
the exact a83 planning evidence *and* to the current water-relative commands present in
the world when the plan is built.  The command binding matters because a same-timestamp
world command event can change control state without changing the physical snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_pair_arbiter import EnvironmentalPairArbitrationResult
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .environmental_world import EnvironmentalWorld2D
from .planning_context import StalePlanningContext
from .fleet_supervision import FleetSupervisorBackend
from .vessel_classes import VesselProfileCatalog


class EnvironmentalPairPlanError(RuntimeError):
    """Base error for invalid/stale environmental pair plans."""


class StaleEnvironmentalPairPlan(EnvironmentalPairPlanError):
    """Raised when world/control evidence has changed since plan binding."""


class UnsafeEnvironmentalPairPlan(EnvironmentalPairPlanError):
    """Raised when an arbitration result is not a valid safe/progress survivor."""


@dataclass(frozen=True, slots=True)
class EnvironmentalPairExecutionPlan:
    planning_context: EnvironmentalAvoidancePlanningContext
    arbitration: EnvironmentalPairArbitrationResult
    first_command_before: WaterRelativeManeuverCommand
    second_command_before: WaterRelativeManeuverCommand

    @property
    def first_id(self) -> str:
        return self.planning_context.vessel_ids[0]

    @property
    def second_id(self) -> str:
        return self.planning_context.vessel_ids[1]

    @property
    def planned_at_time(self) -> float:
        return self.planning_context.time_min

    @property
    def selected_row_index(self) -> int:
        return self.arbitration.selected.row.row_index

    @property
    def first_command(self) -> WaterRelativeManeuverCommand:
        return self.arbitration.selected.row.source_row.evaluation.first_command

    @property
    def second_command(self) -> WaterRelativeManeuverCommand:
        return self.arbitration.selected.row.source_row.evaluation.second_command


def _validate_selected_row_binding(
    context: EnvironmentalAvoidancePlanningContext,
    arbitration: EnvironmentalPairArbitrationResult,
) -> None:
    selected = arbitration.selected.row
    if not selected.passes:
        raise UnsafeEnvironmentalPairPlan("selected a90 row does not pass a89 progress screen")
    source = selected.source_row
    if not source.evaluation.accepted:
        raise UnsafeEnvironmentalPairPlan("selected a90 row is not a88-safe")

    if context.backend is not FleetSupervisorBackend.PAIR or len(context.vessel_ids) != 2:
        raise EnvironmentalPairPlanError("environmental pair plan requires a two-vessel PAIR context")
    first_id, second_id = context.vessel_ids
    if source.evaluation.first_intent.vessel_id != first_id or source.evaluation.second_intent.vessel_id != second_id:
        raise EnvironmentalPairPlanError("selected row intent order does not match planning context")
    if source.evaluation.first_intent.counterpart_id != second_id or source.evaluation.second_intent.counterpart_id != first_id:
        raise EnvironmentalPairPlanError("selected row pair does not match planning context")

    first = context.for_vessel(first_id)
    second = context.for_vessel(second_id)
    first_transition = source.evaluation.first_transition
    second_transition = source.evaluation.second_transition
    if (
        first_transition.initial_state != first.state
        or first_transition.environmental_drift_velocity != first.environmental_drift_velocity
        or first_transition.limits != first.maneuver_limits
    ):
        raise EnvironmentalPairPlanError("selected first transition is not bound to planning-context evidence")
    if (
        second_transition.initial_state != second.state
        or second_transition.environmental_drift_velocity != second.environmental_drift_velocity
        or second_transition.limits != second.maneuver_limits
    ):
        raise EnvironmentalPairPlanError("selected second transition is not bound to planning-context evidence")


def build_environmental_pair_execution_plan(
    world: EnvironmentalWorld2D,
    context: EnvironmentalAvoidancePlanningContext,
    arbitration: EnvironmentalPairArbitrationResult,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalPairExecutionPlan:
    """Bind one a90 selection to exact physical/control evidence without executing it."""

    context.validate(world, profile_catalog=profile_catalog)
    _validate_selected_row_binding(context, arbitration)
    first_id, second_id = context.vessel_ids
    return EnvironmentalPairExecutionPlan(
        planning_context=context,
        arbitration=arbitration,
        first_command_before=world.command_for(first_id),
        second_command_before=world.command_for(second_id),
    )


def validate_environmental_pair_execution_plan(
    plan: EnvironmentalPairExecutionPlan,
    world: EnvironmentalWorld2D,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> None:
    """Fail closed unless both planning evidence and same-time control state are unchanged."""

    if not isinstance(plan, EnvironmentalPairExecutionPlan):
        raise TypeError("plan must be EnvironmentalPairExecutionPlan")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    try:
        plan.planning_context.validate(world, profile_catalog=profile_catalog)
    except StalePlanningContext as exc:
        # Preserve a91-specific stale-plan vocabulary while retaining the original
        # context validation as the causal exception.
        raise StaleEnvironmentalPairPlan(str(exc)) from exc
    _validate_selected_row_binding(plan.planning_context, plan.arbitration)
    try:
        first_now = world.command_for(plan.first_id)
        second_now = world.command_for(plan.second_id)
    except KeyError as exc:
        raise StaleEnvironmentalPairPlan("planned vessel is no longer active") from exc
    if first_now != plan.first_command_before or second_now != plan.second_command_before:
        raise StaleEnvironmentalPairPlan(
            "current water-relative command changed after environmental pair plan binding"
        )
