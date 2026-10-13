"""Run-to-horizon compatibility wrapper for the shared fleet review control session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .encounter_registry import EncounterRegistry, EncounterRegistryUpdate, PairKey
from .fleet_review_session import (
    FleetAppliedOwnerPlan,
    FleetOwnerPlan,
    FleetOwnerReplanner,
    FleetReviewController,
    FleetReviewError,
    FleetReviewPlanMismatch,
    FleetReviewSessionStatus,
    FleetReviewTopologyChanged,
    FleetReviewControlSession,
    NoFleetReviewPlan,
    StaleFleetReviewPlan,
)
from .fleet_supervision import FleetSupervisionDispatch
from .responsibility import ResponsibilityContext
from .world import World2D, WorldSnapshot

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class FleetReviewConfig:
    max_review_events: int = 128

    def __post_init__(self) -> None:
        if self.max_review_events < 1:
            raise ValueError("max_review_events must be at least 1")


class FleetReviewStopReason(str, Enum):
    ALL_RELEASED = "all_released"
    TOPOLOGY_CHANGED = "topology_changed"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class FleetReviewEvent:
    time_min: float
    due_component_ids: tuple[str, ...]
    released_component_ids: tuple[str, ...]
    registry_update: EncounterRegistryUpdate
    dispatch_after: FleetSupervisionDispatch
    plans_applied: tuple[FleetAppliedOwnerPlan, ...]


@dataclass(frozen=True, slots=True)
class FleetReviewResult:
    snapshot: WorldSnapshot
    last_registry_update: EncounterRegistryUpdate
    last_dispatch: FleetSupervisionDispatch
    initial_plans: tuple[FleetAppliedOwnerPlan, ...]
    events: tuple[FleetReviewEvent, ...]
    active_component_ids: tuple[str, ...]
    next_review_times: tuple[tuple[str, float], ...]
    stop_reason: FleetReviewStopReason

    @property
    def handoff_ready(self) -> bool:
        return self.stop_reason is FleetReviewStopReason.ALL_RELEASED


class SharedFleetReviewScheduler:
    """Advance a34 session on its own registry-driven earliest-review clock."""

    def __init__(
        self,
        world: World2D,
        registry: EncounterRegistry,
        *,
        config: FleetReviewConfig = FleetReviewConfig(),
    ) -> None:
        self.world = world
        self.registry = registry
        self.config = config

    def run_until(
        self,
        initial_update: EncounterRegistryUpdate,
        initial_dispatch: FleetSupervisionDispatch,
        *,
        controllers: Iterable[FleetReviewController],
        end_time: float,
        uncertain_pairs: Iterable[PairKey] = (),
        responsibility_context: ResponsibilityContext | None = None,
    ) -> FleetReviewResult:
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        session = FleetReviewControlSession(
            self.world,
            initial_update,
            initial_dispatch,
            controllers=tuple(controllers),
            profile_catalog=getattr(self.registry, "profile_catalog", None),
        )
        if self.world.time >= end_time - _EPS:
            return FleetReviewResult(
                self.world.snapshot(),
                initial_update,
                initial_dispatch,
                (),
                (),
                session.active_component_ids,
                (),
                FleetReviewStopReason.ALL_RELEASED
                if not session.active_component_ids
                else FleetReviewStopReason.END_TIME,
            )

        initial_plans = session.start()
        if not session.active_component_ids:
            return FleetReviewResult(
                self.world.snapshot(),
                initial_update,
                initial_dispatch,
                initial_plans,
                (),
                (),
                (),
                FleetReviewStopReason.ALL_RELEASED,
            )

        uncertain_pairs = tuple(uncertain_pairs)
        events: list[FleetReviewEvent] = []
        last_update = initial_update
        last_dispatch = initial_dispatch

        for _ in range(self.config.max_review_events):
            next_review = session.next_event_time()
            if next_review is None:
                return FleetReviewResult(
                    self.world.snapshot(),
                    last_update,
                    last_dispatch,
                    initial_plans,
                    tuple(events),
                    session.active_component_ids,
                    session.next_review_times,
                    FleetReviewStopReason.ALL_RELEASED,
                )
            if end_time < next_review - _EPS:
                self.world.run_until(end_time)
                return FleetReviewResult(
                    self.world.snapshot(),
                    last_update,
                    last_dispatch,
                    initial_plans,
                    tuple(events),
                    session.active_component_ids,
                    session.next_review_times,
                    FleetReviewStopReason.END_TIME,
                )

            self.world.run_until(next_review)
            update_after = self.registry.update(
                self.world,
                uncertain_pairs=uncertain_pairs,
                responsibility_context=responsibility_context,
            )
            at_horizon = self.world.time >= end_time - _EPS
            item = session.ingest_registry_update(
                update_after,
                apply_due_plans=not at_horizon,
            )
            event = FleetReviewEvent(
                item.time_min,
                item.due_component_ids,
                item.released_component_ids,
                item.registry_update,
                item.dispatch_after,
                item.plans_applied,
            )
            events.append(event)
            last_update = update_after
            last_dispatch = item.dispatch_after

            if item.status is FleetReviewSessionStatus.TOPOLOGY_CHANGED:
                return FleetReviewResult(
                    self.world.snapshot(),
                    last_update,
                    last_dispatch,
                    initial_plans,
                    tuple(events),
                    tuple(owner.component_id for owner in item.dispatch_after.owners),
                    (),
                    FleetReviewStopReason.TOPOLOGY_CHANGED,
                )
            if item.status is FleetReviewSessionStatus.ALL_RELEASED:
                return FleetReviewResult(
                    self.world.snapshot(),
                    last_update,
                    last_dispatch,
                    initial_plans,
                    tuple(events),
                    (),
                    (),
                    FleetReviewStopReason.ALL_RELEASED,
                )
            if at_horizon:
                return FleetReviewResult(
                    self.world.snapshot(),
                    last_update,
                    last_dispatch,
                    initial_plans,
                    tuple(events),
                    session.active_component_ids,
                    session.next_review_times,
                    FleetReviewStopReason.END_TIME,
                )

        raise FleetReviewError("shared fleet review scheduler exceeded max_review_events")


__all__ = [
    "FleetAppliedOwnerPlan",
    "FleetOwnerPlan",
    "FleetOwnerReplanner",
    "FleetReviewConfig",
    "FleetReviewController",
    "FleetReviewError",
    "FleetReviewEvent",
    "FleetReviewPlanMismatch",
    "FleetReviewResult",
    "FleetReviewStopReason",
    "FleetReviewTopologyChanged",
    "NoFleetReviewPlan",
    "SharedFleetReviewScheduler",
    "StaleFleetReviewPlan",
]
