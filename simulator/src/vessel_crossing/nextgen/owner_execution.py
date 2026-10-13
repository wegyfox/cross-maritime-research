"""Uniform snapshot/profile-bound execution for one supervised command owner.

A45 returns an explicitly arbitrated pair or joint planning result.  This module binds
that result to the exact command owner and applies the selected command batch only if
all planning evidence is still current.  Pair and joint backends therefore share one
execution boundary: validate owner topology, validate the full captured world snapshot,
validate maneuver limits/profile bindings, then apply all owner commands at one physical
timestamp without advancing time.

The layer intentionally does not re-run safety or arbitration.  Any change to world or
profile evidence makes the already-selected plan stale and requires replanning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .dynamics import ManeuverCommand
from .encounter_registry import PairKey
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .planning_context import AvoidancePlanningContext, StalePlanningContext
from .planning_helpers import (
    JointAvoidancePlanningResult,
    PairAvoidancePlanningResult,
)
from .vessel_classes import VesselProfileCatalog
from .world import World2D, WorldSnapshot


class OwnerExecutionError(RuntimeError):
    """Base class for fail-closed owner execution errors."""


class OwnerExecutionPlanMismatch(OwnerExecutionError):
    """Raised when a planning result/plan does not belong to the supplied owner."""


class StaleOwnerExecutionPlan(OwnerExecutionError):
    """Raised when world/limits/profile evidence changed after planning."""


class OwnerExecutionCapabilityViolation(OwnerExecutionError):
    """Raised when a selected command exceeds captured vessel-class capability."""


OwnerAvoidancePlanningResult = PairAvoidancePlanningResult | JointAvoidancePlanningResult


@dataclass(frozen=True, slots=True)
class OwnerExecutionPlan:
    """One explicitly selected owner command batch bound to exact planning evidence."""

    component_id: str
    backend: FleetSupervisorBackend
    vessel_ids: tuple[str, ...]
    pair_keys: tuple[PairKey, ...]
    planning_result: OwnerAvoidancePlanningResult
    commands: tuple[tuple[str, ManeuverCommand], ...]

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        if self.component_id != "|".join(self.vessel_ids):
            raise ValueError("component_id does not match vessel_ids")
        if tuple(vessel_id for vessel_id, _ in self.commands) != self.vessel_ids:
            raise ValueError("commands must exactly follow vessel_ids order")
        if len(set(self.vessel_ids)) != len(self.vessel_ids):
            raise ValueError("duplicate vessel_id in owner execution plan")

        context = self.context
        if (
            context.component_id != self.component_id
            or context.backend is not self.backend
            or context.vessel_ids != self.vessel_ids
            or context.pair_keys != self.pair_keys
        ):
            raise ValueError("planning result context does not match execution plan owner signature")

        selected = self.planning_result.selected_commands
        if set(selected) != set(self.vessel_ids):
            raise ValueError("planning result selected commands do not exactly cover owner vessels")
        for vessel_id, command in self.commands:
            if selected[vessel_id] != command:
                raise ValueError("execution command differs from planning arbitration selection")

    @property
    def context(self) -> AvoidancePlanningContext:
        return self.planning_result.context

    @property
    def planning_snapshot(self) -> WorldSnapshot:
        return self.context.planning_snapshot

    @property
    def planned_at_time(self) -> float:
        return self.context.time_min

    def command_for(self, vessel_id: str) -> ManeuverCommand:
        for current_id, command in self.commands:
            if current_id == vessel_id:
                return command
        raise KeyError(vessel_id)


def _validate_result_backend(result: OwnerAvoidancePlanningResult) -> FleetSupervisorBackend:
    if isinstance(result, PairAvoidancePlanningResult):
        return FleetSupervisorBackend.PAIR
    if isinstance(result, JointAvoidancePlanningResult):
        return FleetSupervisorBackend.JOINT
    raise TypeError(f"unsupported owner planning result type: {type(result)!r}")


def _validate_owner_signature(
    *,
    component_id: str,
    backend: FleetSupervisorBackend,
    vessel_ids: tuple[str, ...],
    pair_keys: tuple[PairKey, ...],
    owner: FleetCommandOwner,
) -> None:
    if (
        owner.component_id != component_id
        or owner.backend is not backend
        or owner.vessel_ids != vessel_ids
        or owner.pair_keys != pair_keys
    ):
        raise OwnerExecutionPlanMismatch("execution plan does not match current command owner")




def _validate_command_capabilities(
    context: AvoidancePlanningContext,
    commands: tuple[tuple[str, ManeuverCommand], ...],
) -> None:
    for vessel_id, command in commands:
        profile = context.for_vessel(vessel_id).profile
        if (
            profile is not None
            and profile.speed_envelope is not None
            and not profile.speed_envelope.permits(command.target_speed)
        ):
            raise OwnerExecutionCapabilityViolation(
                f"selected command speed exceeds class speed envelope for {vessel_id!r}"
            )


def build_owner_execution_plan(
    planning_result: OwnerAvoidancePlanningResult,
    owner: FleetCommandOwner,
) -> OwnerExecutionPlan:
    """Bind one a45 planning result to the exact owner that was planned."""

    backend = _validate_result_backend(planning_result)
    context = planning_result.context
    _validate_owner_signature(
        component_id=context.component_id,
        backend=backend,
        vessel_ids=context.vessel_ids,
        pair_keys=context.pair_keys,
        owner=owner,
    )
    if context.backend is not backend:
        raise OwnerExecutionPlanMismatch("planning result backend disagrees with captured context")

    selected = planning_result.selected_commands
    if set(selected) != set(owner.vessel_ids):
        raise OwnerExecutionPlanMismatch("selected commands do not exactly cover owner vessels")
    ordered_commands = tuple((vessel_id, selected[vessel_id]) for vessel_id in owner.vessel_ids)
    _validate_command_capabilities(context, ordered_commands)
    return OwnerExecutionPlan(
        component_id=owner.component_id,
        backend=owner.backend,
        vessel_ids=owner.vessel_ids,
        pair_keys=owner.pair_keys,
        planning_result=planning_result,
        commands=ordered_commands,
    )


def validate_owner_execution_plan(
    world: World2D,
    plan: OwnerExecutionPlan,
    owner: FleetCommandOwner,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> None:
    """Fail closed unless owner topology and all captured planning evidence are current."""

    _validate_owner_signature(
        component_id=plan.component_id,
        backend=plan.backend,
        vessel_ids=plan.vessel_ids,
        pair_keys=plan.pair_keys,
        owner=owner,
    )
    try:
        plan.context.validate(world, profile_catalog=profile_catalog)
    except StalePlanningContext as exc:
        raise StaleOwnerExecutionPlan(str(exc)) from exc

    # Re-derive selected commands from the immutable planning result.  This catches any
    # future result type that stops preserving a one-to-one selected owner command set.
    selected = plan.planning_result.selected_commands
    if set(selected) != set(plan.vessel_ids):
        raise OwnerExecutionPlanMismatch("planning result no longer exactly covers owner vessels")
    _validate_command_capabilities(plan.context, plan.commands)
    for vessel_id, command in plan.commands:
        if selected[vessel_id] != command:
            raise OwnerExecutionPlanMismatch("execution command differs from selected planning command")
        try:
            plan.planning_snapshot.by_id(vessel_id)
        except KeyError as exc:
            raise OwnerExecutionPlanMismatch(
                f"planning snapshot does not contain owner vessel {vessel_id!r}"
            ) from exc


def apply_owner_execution_plan(
    world: World2D,
    plan: OwnerExecutionPlan,
    owner: FleetCommandOwner,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> WorldSnapshot:
    """Atomically apply one validated owner command batch at the current timestamp.

    No physical time elapses.  All fallible membership/profile/limit checks are completed
    before the first command is queued.  Because commands are immutable and all vessel
    IDs are prevalidated, the subsequent same-time batch has no expected partial-failure
    path inside ``World2D``.
    """

    validate_owner_execution_plan(
        world,
        plan,
        owner,
        profile_catalog=profile_catalog,
    )
    snapshot = world.snapshot()
    if snapshot != plan.planning_snapshot:
        raise StaleOwnerExecutionPlan("world changed after execution-plan validation")

    # Preflight every command/vessel before mutating the event queue.
    for vessel_id, _ in plan.commands:
        try:
            snapshot.by_id(vessel_id)
            world.limits_for(vessel_id)
        except KeyError as exc:
            raise StaleOwnerExecutionPlan(
                f"owner vessel {vessel_id!r} disappeared before command application"
            ) from exc

    if world.snapshot() != snapshot:
        raise StaleOwnerExecutionPlan("world changed during owner command preflight")

    for vessel_id, command in plan.commands:
        world.schedule_command(at=world.time, vessel_id=vessel_id, command=command)
    return world.run_until(world.time)
