"""Disjoint route-recovery ownership for the unified mission calendar.

A37 allows several independent route-handoff groups to coexist on one ``World2D``
clock.  The manager owns no physical time and performs no encounter observation; it only
keeps recovery command channels disjoint and delegates each group's exact recovery state
machine to ``FleetRouteHandoffSession``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .route_handoff import (
    FleetRouteHandoffSession,
    FleetRouteHandoffUpdate,
    RouteHandoffRecoveryUnavailable,
    RouteHandoffTarget,
    RouteHandoffVesselResult,
)
from .world import World2D, WorldSnapshot


class RouteRecoveryOwnershipError(RuntimeError):
    """Raised when recovery command ownership would become ambiguous."""


def recovery_group_id(vessel_ids: Iterable[str]) -> str:
    ids = tuple(sorted(set(vessel_ids)))
    if not ids:
        raise ValueError("recovery group must contain at least one vessel")
    return "|".join(ids)


@dataclass(frozen=True, slots=True)
class RouteRecoveryGroupUpdate:
    group_id: str
    update: FleetRouteHandoffUpdate
    completed: bool

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return tuple(item.vessel_id for item in self.update.vessel_results)


@dataclass(frozen=True, slots=True)
class RouteRecoveryGroupUnavailable:
    group_id: str
    vessel_results: tuple[RouteHandoffVesselResult, ...]
    failed_vessel_id: str
    reason: str

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return tuple(item.vessel_id for item in self.vessel_results)


@dataclass(frozen=True, slots=True)
class RouteRecoveryGroupPreemption:
    group_id: str
    vessel_results: tuple[RouteHandoffVesselResult, ...]

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return tuple(item.vessel_id for item in self.vessel_results)


class FleetRouteRecoveryManager:
    """Maintain any number of disjoint no-time-advance recovery sessions."""

    def __init__(self, world: World2D) -> None:
        self.world = world
        self._groups: dict[str, FleetRouteHandoffSession] = {}

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._groups))

    @property
    def owned_vessel_ids(self) -> tuple[str, ...]:
        ids = {vessel_id for session in self._groups.values() for vessel_id in session.vessel_ids}
        return tuple(sorted(ids))

    def by_id(self, group_id: str) -> FleetRouteHandoffSession:
        return self._groups[group_id]

    def _assert_disjoint(self, vessel_ids: set[str]) -> None:
        overlap = vessel_ids.intersection(self.owned_vessel_ids)
        if overlap:
            raise RouteRecoveryOwnershipError(
                f"recovery vessel ownership overlaps existing group: {sorted(overlap)}"
            )

    def add_group(
        self,
        targets: Iterable[RouteHandoffTarget],
        *,
        source_snapshot: WorldSnapshot | None = None,
    ) -> RouteRecoveryGroupUpdate:
        items = tuple(targets)
        ids = {item.vessel_id for item in items}
        if not ids:
            raise ValueError("recovery group must contain at least one target")
        if len(ids) != len(items):
            raise ValueError("recovery group target ids must be unique")
        self._assert_disjoint(ids)
        group_id = recovery_group_id(ids)
        session = FleetRouteHandoffSession(
            self.world,
            items,
            source_snapshot=source_snapshot,
        )
        try:
            update = session.start()
        except RouteHandoffRecoveryUnavailable as exc:
            return RouteRecoveryGroupUnavailable(
                group_id,
                session.vessel_results,
                exc.vessel_id,
                exc.reason,
            )
        completed = update.nominal_ready
        if not completed:
            self._groups[group_id] = session
        return RouteRecoveryGroupUpdate(group_id, update, completed)

    def next_event_time(self) -> float | None:
        values = [session.next_event_time() for session in self._groups.values()]
        values = [value for value in values if value is not None]
        return min(values) if values else None

    def due_group_ids(self) -> tuple[str, ...]:
        due = []
        for group_id, session in self._groups.items():
            value = session.next_event_time()
            if value is not None and value <= self.world.time + 1e-10:
                due.append(group_id)
        return tuple(sorted(due))

    def process_groups_current_time(
        self,
        group_ids: Iterable[str],
    ) -> tuple[RouteRecoveryGroupUpdate | RouteRecoveryGroupUnavailable, ...]:
        results: list[RouteRecoveryGroupUpdate | RouteRecoveryGroupUnavailable] = []
        for group_id in sorted(set(group_ids)):
            session = self._groups.get(group_id)
            if session is None:
                continue
            try:
                update = session.process_current_time()
            except RouteHandoffRecoveryUnavailable as exc:
                del self._groups[group_id]
                results.append(
                    RouteRecoveryGroupUnavailable(
                        group_id,
                        session.vessel_results,
                        exc.vessel_id,
                        exc.reason,
                    )
                )
                continue
            completed = update.nominal_ready
            if completed:
                del self._groups[group_id]
            results.append(RouteRecoveryGroupUpdate(group_id, update, completed))
        return tuple(results)

    def pop_overlapping(
        self,
        vessel_ids: Iterable[str],
    ) -> tuple[RouteRecoveryGroupPreemption, ...]:
        claimed = set(vessel_ids)
        result: list[RouteRecoveryGroupPreemption] = []
        for group_id in list(sorted(self._groups)):
            session = self._groups[group_id]
            if claimed.intersection(session.vessel_ids):
                del self._groups[group_id]
                result.append(
                    RouteRecoveryGroupPreemption(group_id, session.vessel_results)
                )
        return tuple(result)
