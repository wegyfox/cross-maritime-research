"""Event-oriented handoff from cleared avoidance back to route navigation.

The original a16 pair handoff established the key invariant: recovery for multiple
vessels must share one ``World2D`` clock.  A36 generalizes that state machine into a
no-time-advance ``FleetRouteHandoffSession`` so mission orchestration can merge route
recovery reviews with nominal arrivals and avoidance reviews on the same event calendar.
The historical ``PairRouteHandoff`` API remains as a compatibility wrapper over the
same shared engine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from typing import Iterable

from .avoidance_supervisor import AvoidanceExecutionResult
from .dynamics import ManeuverCommand, command_transition_duration
from .follower import RouteFollowerState, RouteGuidance
from .navigator import RouteNavigator
from .recovery import CrossTrackRecoveryState, RecoveryUnavailable
from .world import World2D, WorldSnapshot

_EPS = 1e-10


class RouteHandoffError(RuntimeError):
    """Base class for fail-closed route handoff errors."""


class RouteHandoffNotReady(RouteHandoffError):
    """Raised when avoidance responsibility has not actually been released."""


class StaleRouteHandoff(RouteHandoffError):
    """Raised when the world no longer matches the handoff snapshot."""


class RouteHandoffUnavailable(RouteHandoffError):
    """Raised when a vessel cannot be returned to exact route guidance."""


class RouteHandoffRecoveryUnavailable(RouteHandoffUnavailable):
    """Exact original-route recovery is physically/geometrically unavailable."""

    def __init__(self, vessel_id: str, reason: str) -> None:
        self.vessel_id = vessel_id
        self.reason = reason
        super().__init__(f"route recovery unavailable for {vessel_id!r}: {reason}")


class RouteHandoffStopReason(str, Enum):
    NOMINAL_READY = "nominal_ready"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class RouteHandoffTarget:
    vessel_id: str
    navigator: RouteNavigator
    follower_state: RouteFollowerState = RouteFollowerState()

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")


@dataclass(frozen=True, slots=True)
class RouteHandoffStep:
    time: float
    vessel_id: str
    mode: str
    command: ManeuverCommand
    next_review_at: float | None
    follower_state: RouteFollowerState
    recovery_stage: str | None = None


@dataclass(frozen=True, slots=True)
class RouteHandoffVesselResult:
    vessel_id: str
    follower_state: RouteFollowerState
    nominal_guidance: RouteGuidance | None
    recovery_state: CrossTrackRecoveryState | None
    next_review_at: float | None

    @property
    def nominal_ready(self) -> bool:
        return self.recovery_state is None and self.nominal_guidance is not None


@dataclass(frozen=True, slots=True)
class FleetRouteHandoffUpdate:
    """One same-timestamp refresh of all due route-handoff cursors."""

    time: float
    vessel_results: tuple[RouteHandoffVesselResult, ...]
    steps: tuple[RouteHandoffStep, ...]

    @property
    def nominal_ready(self) -> bool:
        return bool(self.vessel_results) and all(item.nominal_ready for item in self.vessel_results)


@dataclass(frozen=True, slots=True)
class FleetRouteHandoffResult:
    snapshot: WorldSnapshot
    vessels: tuple[RouteHandoffVesselResult, ...]
    steps: tuple[RouteHandoffStep, ...]
    stop_reason: RouteHandoffStopReason

    @property
    def nominal_ready(self) -> bool:
        return self.stop_reason is RouteHandoffStopReason.NOMINAL_READY

    def by_id(self, vessel_id: str) -> RouteHandoffVesselResult:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)


@dataclass(frozen=True, slots=True)
class PairRouteHandoffResult:
    snapshot: WorldSnapshot
    first: RouteHandoffVesselResult
    second: RouteHandoffVesselResult
    steps: tuple[RouteHandoffStep, ...]
    stop_reason: RouteHandoffStopReason

    @property
    def nominal_ready(self) -> bool:
        return self.stop_reason is RouteHandoffStopReason.NOMINAL_READY


@dataclass(slots=True)
class _Cursor:
    target: RouteHandoffTarget
    follower_state: RouteFollowerState
    recovery_state: CrossTrackRecoveryState | None = None
    nominal_guidance: RouteGuidance | None = None
    next_review_at: float | None = None
    restart_pending: bool = False

    def result(self) -> RouteHandoffVesselResult:
        return RouteHandoffVesselResult(
            vessel_id=self.target.vessel_id,
            follower_state=self.follower_state,
            nominal_guidance=self.nominal_guidance,
            recovery_state=self.recovery_state,
            next_review_at=self.next_review_at,
        )


class FleetRouteHandoffSession:
    """Shared-clock N-vessel recovery event source that never advances time itself."""

    def __init__(
        self,
        world: World2D,
        targets: Iterable[RouteHandoffTarget],
        *,
        source_snapshot: WorldSnapshot | None = None,
    ) -> None:
        self.world = world
        items = tuple(sorted(tuple(targets), key=lambda item: item.vessel_id))
        if not items:
            raise ValueError("at least one route handoff target is required")
        ids = tuple(item.vessel_id for item in items)
        if len(set(ids)) != len(ids):
            raise ValueError("route handoff target ids must be unique")
        current = world.snapshot()
        for vessel_id in ids:
            current.by_id(vessel_id)
        if source_snapshot is not None:
            if abs(source_snapshot.time - world.time) > _EPS:
                raise StaleRouteHandoff("world time no longer matches route handoff snapshot")
            for vessel_id in ids:
                if current.by_id(vessel_id) != source_snapshot.by_id(vessel_id):
                    raise StaleRouteHandoff(
                        f"world state for {vessel_id} no longer matches route handoff snapshot"
                    )
        self.targets = items
        self._cursors = {
            item.vessel_id: _Cursor(item, item.follower_state)
            for item in items
        }
        self._steps: list[RouteHandoffStep] = []
        self._started = False

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return tuple(item.vessel_id for item in self.targets)

    @property
    def vessel_results(self) -> tuple[RouteHandoffVesselResult, ...]:
        return tuple(self._cursors[vessel_id].result() for vessel_id in self.vessel_ids)

    @property
    def steps(self) -> tuple[RouteHandoffStep, ...]:
        return tuple(self._steps)

    @property
    def nominal_ready(self) -> bool:
        return self._started and all(item.nominal_ready for item in self.vessel_results)

    def _schedule_now(self, vessel_id: str, command: ManeuverCommand) -> None:
        self.world.schedule_command(at=self.world.time, vessel_id=vessel_id, command=command)
        self.world.run_until(self.world.time)

    def _start_restart(self, cursor: _Cursor, steps: list[RouteHandoffStep]) -> bool:
        vessel = self.world.snapshot().by_id(cursor.target.vessel_id)
        follower = cursor.target.navigator.follower
        if follower.cruise_speed <= _EPS:
            raise RouteHandoffUnavailable("stationary recovery requires a positive cruise speed")
        command = ManeuverCommand(vessel.heading_deg, follower.cruise_speed)
        duration = command_transition_duration(
            vessel,
            command,
            self.world.limits_for(cursor.target.vessel_id),
        )
        if not isfinite(duration) or duration <= _EPS:
            raise RouteHandoffUnavailable("stationary vessel cannot accelerate for route recovery")
        self._schedule_now(cursor.target.vessel_id, command)
        cursor.restart_pending = True
        cursor.recovery_state = None
        cursor.nominal_guidance = None
        cursor.next_review_at = self.world.time + duration
        steps.append(
            RouteHandoffStep(
                time=self.world.time,
                vessel_id=cursor.target.vessel_id,
                mode="restart",
                command=command,
                next_review_at=cursor.next_review_at,
                follower_state=cursor.follower_state,
            )
        )
        return False

    def _refresh(self, cursor: _Cursor, steps: list[RouteHandoffStep]) -> bool:
        """Refresh one due cursor; return True once it has exact nominal guidance."""
        vessel_id = cursor.target.vessel_id
        navigator = cursor.target.navigator
        limits = self.world.limits_for(vessel_id)

        for _ in range(24):
            vessel = self.world.snapshot().by_id(vessel_id)

            if cursor.restart_pending:
                cursor.restart_pending = False

            if cursor.recovery_state is not None:
                decision = navigator.recovery.guidance(
                    vessel,
                    cursor.recovery_state,
                    min_along_track=cursor.follower_state.along_track,
                    limits=limits,
                )
                cursor.recovery_state = decision.recovery_state
                self._schedule_now(vessel_id, decision.command)
                if decision.recovery_state.stage == "complete":
                    final = decision.recovery_state.final_along_track
                    if final is None:
                        raise RouteHandoffUnavailable("completed recovery has no final along-track")
                    cursor.follower_state = replace(
                        cursor.follower_state,
                        along_track=max(cursor.follower_state.along_track, final),
                        active_turn_waypoint=None,
                    )
                    cursor.recovery_state = None
                    cursor.next_review_at = None
                    steps.append(
                        RouteHandoffStep(
                            time=self.world.time,
                            vessel_id=vessel_id,
                            mode="recovery_complete",
                            command=decision.command,
                            next_review_at=None,
                            follower_state=cursor.follower_state,
                            recovery_stage="complete",
                        )
                    )
                    continue
                if decision.next_review_after is None:
                    raise RouteHandoffUnavailable("non-complete recovery has no next review")
                cursor.next_review_at = self.world.time + max(0.0, decision.next_review_after)
                steps.append(
                    RouteHandoffStep(
                        time=self.world.time,
                        vessel_id=vessel_id,
                        mode="recovery",
                        command=decision.command,
                        next_review_at=cursor.next_review_at,
                        follower_state=cursor.follower_state,
                        recovery_stage=decision.recovery_state.stage,
                    )
                )
                if decision.next_review_after <= _EPS:
                    continue
                return False

            guidance = navigator.follower.guidance(vessel, cursor.follower_state, limits)
            cursor.follower_state = guidance.follower_state
            cursor.nominal_guidance = guidance

            if guidance.next_review_after is None and not cursor.follower_state.completed:
                if vessel.speed <= _EPS:
                    return self._start_restart(cursor, steps)
                recovery = navigator.recovery.begin(
                    vessel,
                    min_along_track=cursor.follower_state.along_track,
                    limits=limits,
                )
                cursor.recovery_state = recovery.recovery_state
                cursor.nominal_guidance = None
                self._schedule_now(vessel_id, recovery.command)
                if recovery.recovery_state.stage == "complete":
                    final = recovery.recovery_state.final_along_track
                    if final is None:
                        raise RouteHandoffUnavailable("completed recovery has no final along-track")
                    cursor.follower_state = replace(
                        cursor.follower_state,
                        along_track=max(cursor.follower_state.along_track, final),
                        active_turn_waypoint=None,
                    )
                    cursor.recovery_state = None
                    steps.append(
                        RouteHandoffStep(
                            time=self.world.time,
                            vessel_id=vessel_id,
                            mode="recovery_complete",
                            command=recovery.command,
                            next_review_at=None,
                            follower_state=cursor.follower_state,
                            recovery_stage="complete",
                        )
                    )
                    continue
                if recovery.next_review_after is None:
                    raise RouteHandoffUnavailable("recovery start has no next review")
                cursor.next_review_at = self.world.time + max(0.0, recovery.next_review_after)
                steps.append(
                    RouteHandoffStep(
                        time=self.world.time,
                        vessel_id=vessel_id,
                        mode="recovery",
                        command=recovery.command,
                        next_review_at=cursor.next_review_at,
                        follower_state=cursor.follower_state,
                        recovery_stage=recovery.recovery_state.stage,
                    )
                )
                if recovery.next_review_after <= _EPS:
                    continue
                return False

            self._schedule_now(vessel_id, guidance.command)
            cursor.next_review_at = (
                None
                if guidance.next_review_after is None
                else self.world.time + max(0.0, guidance.next_review_after)
            )
            steps.append(
                RouteHandoffStep(
                    time=self.world.time,
                    vessel_id=vessel_id,
                    mode="nominal",
                    command=guidance.command,
                    next_review_at=cursor.next_review_at,
                    follower_state=cursor.follower_state,
                )
            )
            return True

        raise RouteHandoffUnavailable("too many zero-time route handoff transitions")

    def _update(self, due_ids: set[str]) -> FleetRouteHandoffUpdate:
        steps: list[RouteHandoffStep] = []
        for vessel_id in self.vessel_ids:
            if vessel_id in due_ids:
                try:
                    self._refresh(self._cursors[vessel_id], steps)
                except RecoveryUnavailable as exc:
                    raise RouteHandoffRecoveryUnavailable(vessel_id, str(exc)) from exc
        self._steps.extend(steps)
        return FleetRouteHandoffUpdate(self.world.time, self.vessel_results, tuple(steps))

    def start(self) -> FleetRouteHandoffUpdate:
        if self._started:
            raise RouteHandoffUnavailable("route handoff session already started")
        self._started = True
        return self._update(set(self.vessel_ids))

    def next_event_time(self) -> float | None:
        if not self._started:
            raise RouteHandoffUnavailable("route handoff session has not started")
        if self.nominal_ready:
            return None
        reviews = [
            cursor.next_review_at
            for cursor in self._cursors.values()
            if cursor.next_review_at is not None
        ]
        if not reviews:
            raise RouteHandoffUnavailable("no future review exists before nominal handoff")
        value = min(reviews)
        if value < self.world.time - _EPS:
            raise RouteHandoffUnavailable("route handoff review time moved backwards")
        return value

    def process_current_time(self) -> FleetRouteHandoffUpdate:
        if not self._started:
            raise RouteHandoffUnavailable("route handoff session has not started")
        if self.nominal_ready:
            raise RouteHandoffUnavailable("route handoff session is already nominal-ready")
        due = {
            vessel_id
            for vessel_id, cursor in self._cursors.items()
            if cursor.next_review_at is not None
            and cursor.next_review_at <= self.world.time + _EPS
        }
        if not due:
            raise RouteHandoffUnavailable("no route handoff review is due at current time")
        return self._update(due)

    def result(self, stop_reason: RouteHandoffStopReason) -> FleetRouteHandoffResult:
        return FleetRouteHandoffResult(
            snapshot=self.world.snapshot(),
            vessels=self.vessel_results,
            steps=self.steps,
            stop_reason=stop_reason,
        )


class PairRouteHandoff:
    """Compatibility wrapper for the original two-vessel run-to-handoff API."""

    def __init__(self, world: World2D, first: RouteHandoffTarget, second: RouteHandoffTarget) -> None:
        if first.vessel_id == second.vessel_id:
            raise ValueError("route handoff requires two distinct vessel ids")
        self.world = world
        self.first = first
        self.second = second
        snapshot = world.snapshot()
        snapshot.by_id(first.vessel_id)
        snapshot.by_id(second.vessel_id)

    def _validate_avoidance_result(self, avoidance: AvoidanceExecutionResult) -> None:
        if not avoidance.handoff_ready:
            raise RouteHandoffNotReady("avoidance lifecycle has not released responsibility")
        if abs(avoidance.snapshot.time - self.world.time) > _EPS:
            raise StaleRouteHandoff("world time no longer matches avoidance handoff snapshot")
        current = self.world.snapshot()
        for vessel_id in (self.first.vessel_id, self.second.vessel_id):
            if current.by_id(vessel_id) != avoidance.snapshot.by_id(vessel_id):
                raise StaleRouteHandoff(
                    f"world state for {vessel_id} no longer matches avoidance handoff snapshot"
                )

    def run_until_nominal(
        self,
        avoidance: AvoidanceExecutionResult,
        *,
        end_time: float,
        max_events: int = 128,
    ) -> PairRouteHandoffResult:
        self._validate_avoidance_result(avoidance)
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        if max_events < 1:
            raise ValueError("max_events must be at least 1")

        session = FleetRouteHandoffSession(
            self.world,
            (self.first, self.second),
            source_snapshot=avoidance.snapshot,
        )
        session.start()
        for _ in range(max_events):
            if session.nominal_ready:
                result = session.result(RouteHandoffStopReason.NOMINAL_READY)
                return PairRouteHandoffResult(
                    result.snapshot,
                    result.by_id(self.first.vessel_id),
                    result.by_id(self.second.vessel_id),
                    result.steps,
                    result.stop_reason,
                )
            if self.world.time >= end_time - _EPS:
                result = session.result(RouteHandoffStopReason.END_TIME)
                return PairRouteHandoffResult(
                    result.snapshot,
                    result.by_id(self.first.vessel_id),
                    result.by_id(self.second.vessel_id),
                    result.steps,
                    result.stop_reason,
                )
            next_time = session.next_event_time()
            assert next_time is not None
            self.world.run_until(min(end_time, next_time))
            if self.world.time >= end_time - _EPS and next_time > end_time + _EPS:
                result = session.result(RouteHandoffStopReason.END_TIME)
                return PairRouteHandoffResult(
                    result.snapshot,
                    result.by_id(self.first.vessel_id),
                    result.by_id(self.second.vessel_id),
                    result.steps,
                    result.stop_reason,
                )
            session.process_current_time()
        raise RouteHandoffUnavailable("route handoff exceeded max_events")
