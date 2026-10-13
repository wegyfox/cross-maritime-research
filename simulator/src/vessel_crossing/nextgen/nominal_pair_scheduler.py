"""Shared event-clock scheduler for two vessels under nominal route guidance.

A16 ends as soon as both vessels have been handed back to exact nominal guidance.
This module continues those independent guidance streams on one ``World2D`` clock.
It does not perform encounter detection or avoidance; if nominal exact-review timing
becomes unavailable, execution fails closed and the caller must transfer control to a
recovery/avoidance layer explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .follower import RouteFollowerState, RouteGuidance
from .route_handoff import PairRouteHandoffResult, RouteHandoffTarget
from .route_exit import RouteEndAction, RouteEndDisposition, apply_route_end_action
from .world import World2D, WorldSnapshot

_EPS = 1e-10


class NominalPairScheduleError(RuntimeError):
    """Base class for fail-closed shared nominal scheduling errors."""


class NominalPairHandoffNotReady(NominalPairScheduleError):
    """Raised when the supplied a16 handoff has not reached nominal-ready state."""


class StaleNominalPairHandoff(NominalPairScheduleError):
    """Raised when the world no longer matches the supplied handoff snapshot."""


class NominalPairScheduleUnavailable(NominalPairScheduleError):
    """Raised when exact nominal review timing is no longer available."""


class NominalPairStopReason(str, Enum):
    ALL_COMPLETE = "all_complete"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class NominalPairStep:
    time: float
    vessel_id: str
    guidance: RouteGuidance
    next_review_at: float | None


@dataclass(frozen=True, slots=True)
class NominalPairVesselResult:
    vessel_id: str
    follower_state: RouteFollowerState
    guidance: RouteGuidance
    next_review_at: float | None
    disposition: RouteEndDisposition | None = None

    @property
    def completed(self) -> bool:
        return self.follower_state.completed


@dataclass(frozen=True, slots=True)
class NominalPairRunResult:
    snapshot: WorldSnapshot
    first: NominalPairVesselResult
    second: NominalPairVesselResult
    steps: tuple[NominalPairStep, ...]
    stop_reason: NominalPairStopReason

    @property
    def all_complete(self) -> bool:
        return self.stop_reason is NominalPairStopReason.ALL_COMPLETE


@dataclass(slots=True)
class _Cursor:
    target: RouteHandoffTarget
    state: RouteFollowerState
    guidance: RouteGuidance
    next_review_at: float | None
    end_action: RouteEndAction = RouteEndAction.CONTINUE
    disposition: RouteEndDisposition | None = None

    def result(self) -> NominalPairVesselResult:
        return NominalPairVesselResult(
            vessel_id=self.target.vessel_id,
            follower_state=self.state,
            guidance=self.guidance,
            next_review_at=self.next_review_at,
            disposition=self.disposition,
        )


class SharedNominalPairScheduler:
    """Continue two nominal route followers on a single event-driven world clock.

    Route completion ends *route guidance* for that vessel.  The vessel is not removed
    from ``World2D`` and is not frozen: its last physical command remains active while
    the peer continues.  Network departure/removal is intentionally a later layer.
    """

    def __init__(
        self,
        world: World2D,
        first: RouteHandoffTarget,
        second: RouteHandoffTarget,
        *,
        first_end_action: RouteEndAction = RouteEndAction.CONTINUE,
        second_end_action: RouteEndAction = RouteEndAction.CONTINUE,
    ) -> None:
        if first.vessel_id == second.vessel_id:
            raise ValueError("shared nominal scheduler requires two distinct vessel ids")
        self.world = world
        self.first = first
        self.second = second
        self.first_end_action = first_end_action
        self.second_end_action = second_end_action
        snapshot = world.snapshot()
        snapshot.by_id(first.vessel_id)
        snapshot.by_id(second.vessel_id)

    def _validate_handoff(self, handoff: PairRouteHandoffResult) -> None:
        if not handoff.nominal_ready:
            raise NominalPairHandoffNotReady("route handoff has not reached nominal-ready state")
        if handoff.first.vessel_id != self.first.vessel_id or handoff.second.vessel_id != self.second.vessel_id:
            raise NominalPairHandoffNotReady("handoff vessel ordering does not match scheduler targets")
        if abs(handoff.snapshot.time - self.world.time) > _EPS:
            raise StaleNominalPairHandoff("world time no longer matches nominal handoff snapshot")
        current = self.world.snapshot()
        for vessel_id in (self.first.vessel_id, self.second.vessel_id):
            if current.by_id(vessel_id) != handoff.snapshot.by_id(vessel_id):
                raise StaleNominalPairHandoff(
                    f"world state for {vessel_id} no longer matches nominal handoff snapshot"
                )
        for item in (handoff.first, handoff.second):
            if item.nominal_guidance is None or item.recovery_state is not None:
                raise NominalPairHandoffNotReady("handoff vessel lacks exact nominal guidance")
            if not item.follower_state.completed and item.next_review_at is None:
                raise NominalPairHandoffNotReady("non-complete handoff vessel lacks next review")

    def _schedule_now(self, vessel_id: str, guidance: RouteGuidance) -> None:
        self.world.schedule_command(at=self.world.time, vessel_id=vessel_id, command=guidance.command)
        self.world.run_until(self.world.time)

    def _cursor_from_handoff(
        self, target: RouteHandoffTarget, item, end_action: RouteEndAction
    ) -> _Cursor:
        assert item.nominal_guidance is not None
        cursor = _Cursor(
            target, item.follower_state, item.nominal_guidance, item.next_review_at, end_action
        )
        # Idempotently re-assert the current nominal command. This makes the handoff
        # contract robust even if callers constructed a nominal-ready result without
        # relying on World2D's private current-command state.
        self._schedule_now(target.vessel_id, cursor.guidance)
        if cursor.state.completed:
            cursor.disposition = apply_route_end_action(
                self.world,
                vessel_id=target.vessel_id,
                guidance=cursor.guidance,
                action=cursor.end_action,
            )
            cursor.next_review_at = None
        return cursor

    def _refresh(self, cursor: _Cursor, steps: list[NominalPairStep]) -> None:
        if cursor.state.completed:
            cursor.next_review_at = None
            return

        for _ in range(24):
            vessel = self.world.snapshot().by_id(cursor.target.vessel_id)
            limits = self.world.limits_for(cursor.target.vessel_id)
            guidance = cursor.target.navigator.follower.guidance(vessel, cursor.state, limits)
            if guidance.follower_state.along_track + _EPS < cursor.state.along_track:
                raise NominalPairScheduleUnavailable("route progress regressed")
            cursor.state = guidance.follower_state
            cursor.guidance = guidance
            self._schedule_now(cursor.target.vessel_id, guidance)

            if cursor.state.completed:
                cursor.next_review_at = None
                cursor.disposition = apply_route_end_action(
                    self.world,
                    vessel_id=cursor.target.vessel_id,
                    guidance=guidance,
                    action=cursor.end_action,
                )
                steps.append(
                    NominalPairStep(self.world.time, cursor.target.vessel_id, guidance, None)
                )
                return

            if guidance.next_review_after is None:
                raise NominalPairScheduleUnavailable(
                    "exact nominal review is unavailable; explicit recovery/avoidance handoff is required"
                )
            cursor.next_review_at = self.world.time + max(0.0, guidance.next_review_after)
            steps.append(
                NominalPairStep(
                    self.world.time,
                    cursor.target.vessel_id,
                    guidance,
                    cursor.next_review_at,
                )
            )
            if guidance.next_review_after <= _EPS:
                continue
            return

        raise NominalPairScheduleUnavailable("too many zero-time nominal transitions")

    def run_from_handoff(
        self,
        handoff: PairRouteHandoffResult,
        *,
        end_time: float,
        max_events: int = 512,
    ) -> NominalPairRunResult:
        """Run both nominal guidance streams until both routes complete or time expires."""
        self._validate_handoff(handoff)
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        if max_events < 1:
            raise ValueError("max_events must be at least 1")

        first = self._cursor_from_handoff(
            self.first, handoff.first, self.first_end_action
        )
        second = self._cursor_from_handoff(
            self.second, handoff.second, self.second_end_action
        )
        cursors = (first, second)
        steps: list[NominalPairStep] = []

        for _ in range(max_events):
            if all(cursor.state.completed for cursor in cursors):
                return NominalPairRunResult(
                    snapshot=self.world.snapshot(),
                    first=first.result(),
                    second=second.result(),
                    steps=tuple(steps),
                    stop_reason=NominalPairStopReason.ALL_COMPLETE,
                )

            if self.world.time >= end_time - _EPS:
                return NominalPairRunResult(
                    snapshot=self.world.snapshot(),
                    first=first.result(),
                    second=second.result(),
                    steps=tuple(steps),
                    stop_reason=NominalPairStopReason.END_TIME,
                )

            reviews = [
                cursor.next_review_at
                for cursor in cursors
                if not cursor.state.completed and cursor.next_review_at is not None
            ]
            if not reviews:
                raise NominalPairScheduleUnavailable("no future review exists for unfinished route")
            next_time = min(end_time, min(reviews))
            if next_time < self.world.time - _EPS:
                raise NominalPairScheduleUnavailable("nominal review time moved backwards")
            self.world.run_until(next_time)

            due = [
                cursor
                for cursor in cursors
                if not cursor.state.completed
                and cursor.next_review_at is not None
                and cursor.next_review_at <= self.world.time + _EPS
            ]
            for cursor in due:
                self._refresh(cursor, steps)

        raise NominalPairScheduleUnavailable("shared nominal scheduler exceeded max_events")
