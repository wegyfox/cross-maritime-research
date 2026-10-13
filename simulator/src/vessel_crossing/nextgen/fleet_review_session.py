"""No-time-advance fleet supervision review session.

A32 provided a complete run-to-horizon shared review scheduler.  This module extracts
its command-owner state machine into an event source/consumer that neither advances
``World2D`` nor calls ``EncounterRegistry``.  An upstream mission calendar can therefore
merge nominal events and supervision reviews, produce one registry observation for the
combined timestamp, and feed that observation into this session.

The session may receive registry updates even when no owner review is due.  That is
necessary because a nominal arrival/review can create, merge, split or release encounter
ownership before the next supervision review.  Exact surviving owners keep their
original review times; changed ownership returns ``TOPOLOGY_CHANGED`` without silently
adopting a new controller set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Iterable

from .avoidance_supervisor import PairExecutionPlan
from .encounter_registry import EncounterRegistryUpdate, PairKey
from .fleet_supervision import (
    FleetCommandOwner,
    FleetSupervisionDispatch,
    FleetSupervisorBackend,
    build_fleet_supervision_dispatch,
)
from .joint_supervisor import JointSupervisionPlan
from .owner_execution import (
    OwnerExecutionPlan,
    OwnerExecutionPlanMismatch,
    StaleOwnerExecutionPlan,
    validate_owner_execution_plan,
)
from .vessel_classes import VesselProfileCatalog
from .world import World2D, WorldSnapshot

_EPS = 1e-10


class FleetReviewError(RuntimeError):
    """Base class for fail-closed fleet-review control errors."""


class FleetReviewTopologyChanged(FleetReviewError):
    """Marker for owner topology no longer represented by the current session."""


class NoFleetReviewPlan(FleetReviewError):
    """Raised when a due/initial command owner has no accepted execution plan."""


class FleetReviewPlanMismatch(FleetReviewError):
    """Raised when a submitted plan does not belong to the owner being replanned."""


class StaleFleetReviewPlan(FleetReviewError):
    """Raised when planning changed/advanced the world before atomic plan application."""


FleetOwnerPlan = PairExecutionPlan | JointSupervisionPlan | OwnerExecutionPlan
FleetOwnerReplanner = Callable[
    [WorldSnapshot, EncounterRegistryUpdate, FleetCommandOwner],
    FleetOwnerPlan | None,
]
FleetOwnerReplanDecider = Callable[
    [EncounterRegistryUpdate, EncounterRegistryUpdate, FleetCommandOwner],
    bool,
]


@dataclass(frozen=True, slots=True)
class FleetReviewController:
    component_id: str
    backend: FleetSupervisorBackend
    review_interval_min: float
    replanner: FleetOwnerReplanner
    replan_decider: FleetOwnerReplanDecider | None = None

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id must be non-empty")
        if not isfinite(self.review_interval_min) or self.review_interval_min <= 0.0:
            raise ValueError("review_interval_min must be finite and positive")
        if not callable(self.replanner):
            raise ValueError("replanner must be callable")
        if self.replan_decider is not None and not callable(self.replan_decider):
            raise ValueError("replan_decider must be callable when supplied")


@dataclass(frozen=True, slots=True)
class FleetAppliedOwnerPlan:
    component_id: str
    backend: FleetSupervisorBackend
    vessel_ids: tuple[str, ...]
    plan: FleetOwnerPlan


class FleetReviewSessionStatus(str, Enum):
    CONTINUE = "continue"
    ALL_RELEASED = "all_released"
    TOPOLOGY_CHANGED = "topology_changed"


@dataclass(frozen=True, slots=True)
class FleetReviewSessionUpdate:
    time_min: float
    due_component_ids: tuple[str, ...]
    released_component_ids: tuple[str, ...]
    registry_update: EncounterRegistryUpdate
    dispatch_after: FleetSupervisionDispatch
    plans_applied: tuple[FleetAppliedOwnerPlan, ...]
    status: FleetReviewSessionStatus


def _owner_signature(owner: FleetCommandOwner) -> tuple[
    FleetSupervisorBackend, tuple[str, ...], tuple[PairKey, ...]
]:
    return owner.backend, owner.vessel_ids, owner.pair_keys


def _owner_map(dispatch: FleetSupervisionDispatch) -> dict[str, FleetCommandOwner]:
    result: dict[str, FleetCommandOwner] = {}
    for owner in dispatch.owners:
        if owner.component_id in result:
            raise FleetReviewError(f"duplicate component_id in dispatch: {owner.component_id!r}")
        result[owner.component_id] = owner
    return result


def _validate_pair_plan(
    plan: PairExecutionPlan,
    owner: FleetCommandOwner,
    snapshot: WorldSnapshot,
) -> None:
    if owner.backend is not FleetSupervisorBackend.PAIR:
        raise FleetReviewPlanMismatch("pair plan submitted for a non-pair owner")
    if (plan.first_id, plan.second_id) != owner.vessel_ids:
        raise FleetReviewPlanMismatch("pair plan vessel order does not match owner")
    if abs(plan.planned_at_time - snapshot.time) > _EPS:
        raise StaleFleetReviewPlan("pair plan time does not match the shared planning snapshot")
    if not plan.evaluation.accepted:
        raise FleetReviewPlanMismatch("pair plan evaluation is not accepted")
    for vessel_id in owner.vessel_ids:
        snapshot.by_id(vessel_id)


def _validate_joint_plan(
    plan: JointSupervisionPlan,
    owner: FleetCommandOwner,
    snapshot: WorldSnapshot,
) -> None:
    if owner.backend is not FleetSupervisorBackend.JOINT:
        raise FleetReviewPlanMismatch("joint plan submitted for a non-joint owner")
    execution = plan.execution_plan
    if plan.pair_keys != owner.pair_keys:
        raise FleetReviewPlanMismatch("joint plan pair topology does not match owner")
    if execution.component_id != owner.component_id:
        raise FleetReviewPlanMismatch("joint plan component_id does not match owner")
    if execution.vessel_ids != owner.vessel_ids:
        raise FleetReviewPlanMismatch("joint plan vessel order does not match owner")
    if execution.planning_snapshot != snapshot:
        raise StaleFleetReviewPlan("joint plan snapshot does not match the shared planning snapshot")
    if not execution.evaluation.accepted:
        raise FleetReviewPlanMismatch("joint plan evaluation is not accepted")


def validate_owner_plan(
    plan: FleetOwnerPlan,
    owner: FleetCommandOwner,
    snapshot: WorldSnapshot,
    *,
    world: World2D | None = None,
    profile_catalog: VesselProfileCatalog | None = None,
) -> None:
    if isinstance(plan, OwnerExecutionPlan):
        if plan.planning_snapshot != snapshot:
            raise StaleFleetReviewPlan("owner execution plan snapshot does not match shared planning snapshot")
        if world is not None:
            try:
                validate_owner_execution_plan(
                    world, plan, owner, profile_catalog=profile_catalog
                )
            except StaleOwnerExecutionPlan as exc:
                raise StaleFleetReviewPlan(str(exc)) from exc
            except OwnerExecutionPlanMismatch as exc:
                raise FleetReviewPlanMismatch(str(exc)) from exc
        else:
            if (
                plan.component_id != owner.component_id
                or plan.backend is not owner.backend
                or plan.vessel_ids != owner.vessel_ids
                or plan.pair_keys != owner.pair_keys
            ):
                raise FleetReviewPlanMismatch("owner execution plan does not match owner")
        return
    if owner.backend is FleetSupervisorBackend.PAIR:
        if not isinstance(plan, PairExecutionPlan):
            raise FleetReviewPlanMismatch("pair owner requires PairExecutionPlan")
        _validate_pair_plan(plan, owner, snapshot)
        return
    if not isinstance(plan, JointSupervisionPlan):
        raise FleetReviewPlanMismatch("joint owner requires JointSupervisionPlan")
    _validate_joint_plan(plan, owner, snapshot)


def owner_plan_commands(plan: FleetOwnerPlan, owner: FleetCommandOwner):
    if isinstance(plan, OwnerExecutionPlan):
        for vessel_id, command in plan.commands:
            yield vessel_id, command
        return
    if isinstance(plan, PairExecutionPlan):
        yield owner.vessel_ids[0], plan.evaluation.first_candidate.command
        yield owner.vessel_ids[1], plan.evaluation.second_candidate.command
        return
    execution = plan.execution_plan
    for vessel_id in owner.vessel_ids:
        yield vessel_id, execution.command_for(vessel_id)


class FleetReviewControlSession:
    """Maintain current a31 owners/review times without owning physical time or observation."""

    def __init__(
        self,
        world: World2D,
        initial_update: EncounterRegistryUpdate,
        initial_dispatch: FleetSupervisionDispatch,
        *,
        controllers: Iterable[FleetReviewController],
        profile_catalog: VesselProfileCatalog | None = None,
    ) -> None:
        self.world = world
        self.profile_catalog = profile_catalog
        if abs(initial_update.time_min - world.time) > _EPS or abs(initial_dispatch.time_min - world.time) > _EPS:
            raise ValueError("initial update/dispatch time must match current world time")
        if build_fleet_supervision_dispatch(initial_update) != initial_dispatch:
            raise ValueError("initial dispatch does not match the supplied registry update")
        self.update = initial_update
        self.dispatch = initial_dispatch
        self.owners = _owner_map(initial_dispatch)

        controller_tuple = tuple(controllers)
        self.controllers: dict[str, FleetReviewController] = {}
        for controller in controller_tuple:
            if controller.component_id in self.controllers:
                raise ValueError(f"duplicate controller for {controller.component_id!r}")
            self.controllers[controller.component_id] = controller
        if set(self.controllers) != set(self.owners):
            raise ValueError("controllers must exactly cover the initial command owners")
        for component_id, owner in self.owners.items():
            if self.controllers[component_id].backend is not owner.backend:
                raise ValueError(f"controller backend does not match owner {component_id!r}")

        self._started = False
        self._next_review: dict[str, float] = {}
        self._initial_plans: tuple[FleetAppliedOwnerPlan, ...] = ()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def initial_plans(self) -> tuple[FleetAppliedOwnerPlan, ...]:
        return self._initial_plans

    @property
    def active_component_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.owners))

    @property
    def owned_vessel_ids(self) -> tuple[str, ...]:
        return tuple(sorted(vessel_id for owner in self.owners.values() for vessel_id in owner.vessel_ids))

    @property
    def next_review_times(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self._next_review.items()))

    def next_event_time(self) -> float | None:
        if not self._started:
            raise FleetReviewError("fleet review session has not been started")
        if not self._next_review:
            return None
        value = min(self._next_review.values())
        if value < self.world.time - _EPS:
            raise FleetReviewError("fleet review event was missed by the upstream calendar")
        return value

    def _plan_batch(
        self,
        component_ids: tuple[str, ...],
        *,
        snapshot: WorldSnapshot,
        update: EncounterRegistryUpdate,
    ) -> tuple[FleetAppliedOwnerPlan, ...]:
        planned: list[FleetAppliedOwnerPlan] = []
        claimed_vessels: set[str] = set()
        for component_id in component_ids:
            owner = self.owners[component_id]
            controller = self.controllers[component_id]
            plan = controller.replanner(snapshot, update, owner)
            if plan is None:
                raise NoFleetReviewPlan(
                    f"owner {component_id!r} remains active but replanner returned no plan"
                )
            validate_owner_plan(
                plan, owner, snapshot, world=self.world, profile_catalog=self.profile_catalog
            )
            overlap = claimed_vessels.intersection(owner.vessel_ids)
            if overlap:
                raise FleetReviewError(
                    f"atomic plan batch has duplicate command ownership for {sorted(overlap)!r}"
                )
            claimed_vessels.update(owner.vessel_ids)
            planned.append(FleetAppliedOwnerPlan(component_id, owner.backend, owner.vessel_ids, plan))
        if self.world.snapshot() != snapshot:
            raise StaleFleetReviewPlan("world changed while the shared plan batch was being built")
        return tuple(planned)

    def _apply_batch(
        self,
        planned: tuple[FleetAppliedOwnerPlan, ...],
        snapshot: WorldSnapshot,
    ) -> None:
        for item in planned:
            owner = self.owners[item.component_id]
            validate_owner_plan(
                item.plan, owner, snapshot, world=self.world, profile_catalog=self.profile_catalog
            )
            for vessel_id in item.vessel_ids:
                snapshot.by_id(vessel_id)
        if self.world.snapshot() != snapshot:
            raise StaleFleetReviewPlan("world changed before the shared plan batch could be applied")
        for item in planned:
            owner = self.owners[item.component_id]
            for vessel_id, command in owner_plan_commands(item.plan, owner):
                self.world.schedule_command(at=self.world.time, vessel_id=vessel_id, command=command)
        self.world.run_until(self.world.time)

    def start(self) -> tuple[FleetAppliedOwnerPlan, ...]:
        if self._started:
            raise FleetReviewError("fleet review session is already started")
        self._started = True
        if not self.owners:
            return ()
        snapshot = self.world.snapshot()
        component_ids = tuple(sorted(self.owners))
        plans = self._plan_batch(component_ids, snapshot=snapshot, update=self.update)
        self._apply_batch(plans, snapshot)
        self._next_review = {
            component_id: self.world.time + self.controllers[component_id].review_interval_min
            for component_id in self.owners
        }
        self._initial_plans = plans
        return plans

    @staticmethod
    def _classify_dispatch_change(
        previous: dict[str, FleetCommandOwner],
        current: dict[str, FleetCommandOwner],
    ) -> tuple[tuple[str, ...], bool]:
        released = tuple(sorted(set(previous) - set(current)))
        for component_id, owner in current.items():
            old = previous.get(component_id)
            if old is None or _owner_signature(old) != _owner_signature(owner):
                return released, True
        return released, False

    def ingest_registry_update(
        self,
        update_after: EncounterRegistryUpdate,
        *,
        apply_due_plans: bool = True,
    ) -> FleetReviewSessionUpdate:
        """Consume one already-produced fleet observation at the current world time."""
        if not self._started:
            raise FleetReviewError("fleet review session must be started before ingest")
        if abs(update_after.time_min - self.world.time) > _EPS:
            raise ValueError("registry update time must match current world time")
        if update_after.time_min <= self.update.time_min + _EPS:
            raise ValueError("fleet review session requires strictly newer registry updates")
        missed = [cid for cid, time in self._next_review.items() if time < self.world.time - _EPS]
        if missed:
            raise FleetReviewError(f"upstream calendar skipped reviews for: {sorted(missed)}")
        due = tuple(
            sorted(cid for cid, time in self._next_review.items() if time <= self.world.time + _EPS)
        )

        dispatch_after = build_fleet_supervision_dispatch(update_after)
        current_owners = _owner_map(dispatch_after)
        released, topology_changed = self._classify_dispatch_change(self.owners, current_owners)
        if topology_changed:
            return FleetReviewSessionUpdate(
                self.world.time,
                due,
                released,
                update_after,
                dispatch_after,
                (),
                FleetReviewSessionStatus.TOPOLOGY_CHANGED,
            )

        previous_update = self.update
        self.owners = current_owners
        self.update = update_after
        self.dispatch = dispatch_after
        for component_id in released:
            self._next_review.pop(component_id, None)
        if not self.owners:
            return FleetReviewSessionUpdate(
                self.world.time,
                due,
                released,
                update_after,
                dispatch_after,
                (),
                FleetReviewSessionStatus.ALL_RELEASED,
            )

        due_survivors = tuple(cid for cid in due if cid in self.owners)
        plans: tuple[FleetAppliedOwnerPlan, ...] = ()
        if apply_due_plans and due_survivors:
            replan_due = tuple(
                component_id
                for component_id in due_survivors
                if self.controllers[component_id].replan_decider is None
                or self.controllers[component_id].replan_decider(
                    previous_update, update_after, self.owners[component_id]
                )
            )
            if replan_due:
                snapshot = self.world.snapshot()
                plans = self._plan_batch(replan_due, snapshot=snapshot, update=update_after)
                self._apply_batch(plans, snapshot)
            # A review remains a real observation/cadence boundary even when the
            # controller deliberately persists its already accepted absolute command.
            for component_id in due_survivors:
                self._next_review[component_id] = (
                    self.world.time + self.controllers[component_id].review_interval_min
                )

        return FleetReviewSessionUpdate(
            self.world.time,
            due,
            released,
            update_after,
            dispatch_after,
            plans,
            FleetReviewSessionStatus.CONTINUE,
        )
