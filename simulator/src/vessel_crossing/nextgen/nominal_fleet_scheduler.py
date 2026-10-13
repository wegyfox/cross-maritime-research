"""Shared nominal event scheduler for an arbitrary fixed active vessel set."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .follower import RouteFollowerState, RouteGuidance
from .navigator import RouteNavigator
from .route_exit import RouteEndAction, RouteEndDisposition, apply_route_end_action
from .world import World2D, WorldSnapshot

_EPS = 1e-10


class NominalFleetScheduleError(RuntimeError):
    """Base class for fail-closed fleet nominal scheduling errors."""


class NominalFleetScheduleUnavailable(NominalFleetScheduleError):
    """Raised when one unfinished route loses exact nominal review timing."""


class NominalFleetStopReason(str, Enum):
    ALL_COMPLETE = "all_complete"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class NominalFleetTarget:
    vessel_id: str
    navigator: RouteNavigator
    follower_state: RouteFollowerState = RouteFollowerState()
    end_action: RouteEndAction = RouteEndAction.CONTINUE

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")


@dataclass(frozen=True, slots=True)
class NominalFleetStep:
    time: float
    vessel_id: str
    guidance: RouteGuidance
    next_review_at: float | None


@dataclass(frozen=True, slots=True)
class NominalFleetVesselResult:
    vessel_id: str
    follower_state: RouteFollowerState
    guidance: RouteGuidance
    next_review_at: float | None
    disposition: RouteEndDisposition | None

    @property
    def completed(self) -> bool:
        return self.follower_state.completed


@dataclass(frozen=True, slots=True)
class NominalFleetRunResult:
    snapshot: WorldSnapshot
    vessels: tuple[NominalFleetVesselResult, ...]
    steps: tuple[NominalFleetStep, ...]
    stop_reason: NominalFleetStopReason

    @property
    def all_complete(self) -> bool:
        return self.stop_reason is NominalFleetStopReason.ALL_COMPLETE

    def by_id(self, vessel_id: str) -> NominalFleetVesselResult:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)


@dataclass(slots=True)
class _Cursor:
    target: NominalFleetTarget
    state: RouteFollowerState
    guidance: RouteGuidance | None = None
    next_review_at: float | None = None
    disposition: RouteEndDisposition | None = None

    def result(self) -> NominalFleetVesselResult:
        if self.guidance is None:
            raise NominalFleetScheduleUnavailable("fleet cursor has no nominal guidance")
        return NominalFleetVesselResult(
            vessel_id=self.target.vessel_id,
            follower_state=self.state,
            guidance=self.guidance,
            next_review_at=self.next_review_at,
            disposition=self.disposition,
        )


class SharedNominalFleetScheduler:
    """Run N independent nominal route followers on one deterministic event clock.

    The active target set is fixed for one run. Route-end removal may shrink the physical
    world during the run. Dynamic arrivals/spawns are intentionally a later layer.
    """

    def __init__(self, world: World2D, targets: Iterable[NominalFleetTarget]) -> None:
        self.world = world
        self.targets = tuple(targets)
        if not self.targets:
            raise ValueError("at least one nominal fleet target is required")
        ids = [target.vessel_id for target in self.targets]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate nominal fleet vessel_id")
        snapshot = world.snapshot()
        for vessel_id in ids:
            snapshot.by_id(vessel_id)

    def _compute_guidance(self, cursor: _Cursor) -> RouteGuidance:
        vessel = self.world.snapshot().by_id(cursor.target.vessel_id)
        limits = self.world.limits_for(cursor.target.vessel_id)
        guidance = cursor.target.navigator.follower.guidance(vessel, cursor.state, limits)
        if guidance.follower_state.along_track + _EPS < cursor.state.along_track:
            raise NominalFleetScheduleUnavailable("route progress regressed")
        return guidance

    def _apply_guidance(self, cursor: _Cursor, guidance: RouteGuidance, steps: list[NominalFleetStep]) -> None:
        cursor.state = guidance.follower_state
        cursor.guidance = guidance
        self.world.schedule_command(
            at=self.world.time,
            vessel_id=cursor.target.vessel_id,
            command=guidance.command,
        )

        if cursor.state.completed:
            cursor.next_review_at = None
            # Process the guidance command before any terminal action at the same timestamp.
            self.world.run_until(self.world.time)
            cursor.disposition = apply_route_end_action(
                self.world,
                vessel_id=cursor.target.vessel_id,
                guidance=guidance,
                action=cursor.target.end_action,
            )
            steps.append(NominalFleetStep(self.world.time, cursor.target.vessel_id, guidance, None))
            return

        if guidance.next_review_after is None:
            raise NominalFleetScheduleUnavailable(
                f"exact nominal review unavailable for {cursor.target.vessel_id}"
            )
        cursor.next_review_at = self.world.time + max(0.0, guidance.next_review_after)
        steps.append(
            NominalFleetStep(
                self.world.time,
                cursor.target.vessel_id,
                guidance,
                cursor.next_review_at,
            )
        )

    def _refresh_due(self, cursor: _Cursor, steps: list[NominalFleetStep]) -> None:
        if cursor.state.completed:
            return
        for _ in range(24):
            guidance = self._compute_guidance(cursor)
            self._apply_guidance(cursor, guidance, steps)
            if cursor.state.completed:
                return
            assert guidance.next_review_after is not None
            if guidance.next_review_after > _EPS:
                return
            self.world.run_until(self.world.time)
        raise NominalFleetScheduleUnavailable("too many zero-time nominal transitions")

    def run_until(self, *, end_time: float, max_events: int = 4096) -> NominalFleetRunResult:
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        if max_events < 1:
            raise ValueError("max_events must be at least 1")

        cursors = [_Cursor(target, target.follower_state) for target in self.targets]
        steps: list[NominalFleetStep] = []

        # Establish every initial command at one common timestamp before advancing time.
        initial = [(cursor, self._compute_guidance(cursor)) for cursor in cursors]
        for cursor, guidance in initial:
            self._apply_guidance(cursor, guidance, steps)
        self.world.run_until(self.world.time)

        for _ in range(max_events):
            if all(cursor.state.completed for cursor in cursors):
                return NominalFleetRunResult(
                    self.world.snapshot(),
                    tuple(cursor.result() for cursor in cursors),
                    tuple(steps),
                    NominalFleetStopReason.ALL_COMPLETE,
                )
            if self.world.time >= end_time - _EPS:
                return NominalFleetRunResult(
                    self.world.snapshot(),
                    tuple(cursor.result() for cursor in cursors),
                    tuple(steps),
                    NominalFleetStopReason.END_TIME,
                )

            reviews = [
                cursor.next_review_at
                for cursor in cursors
                if not cursor.state.completed and cursor.next_review_at is not None
            ]
            if not reviews:
                raise NominalFleetScheduleUnavailable("no future review exists for unfinished fleet")
            next_time = min(end_time, min(reviews))
            if next_time < self.world.time - _EPS:
                raise NominalFleetScheduleUnavailable("fleet nominal review time moved backwards")
            self.world.run_until(next_time)

            due = [
                cursor
                for cursor in cursors
                if not cursor.state.completed
                and cursor.next_review_at is not None
                and cursor.next_review_at <= self.world.time + _EPS
            ]
            for cursor in due:
                self._refresh_due(cursor, steps)
            self.world.run_until(self.world.time)

        raise NominalFleetScheduleUnavailable("shared nominal fleet scheduler exceeded max_events")
