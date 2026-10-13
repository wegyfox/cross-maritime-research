"""Persistent environmental nominal-route continuation after a98 handoff readiness.

A99 continues route ownership once a98 has returned both vessels to nominal ground
route guidance.  It intentionally does not reuse the historical exact tangent-circle
route follower under current: the latter assumes hull/through-water heading also defines
Earth-fixed translation.

The environmental session instead uses a closed-loop ground-frame route-pursuit
controller:

- route progress and cross-track error are Earth-fixed;
- a look-ahead point on the caller-owned polyline defines the desired ground vector;
- a97 velocity-frame translation converts that vector to a water-relative command;
- ``EnvironmentalWorld2D`` remains the sole bounded-dynamics/clock owner;
- fixed controller reviews keep the semantics explicit and avoid hidden integration or
  Zeno-like residual-transition events;
- route completion is a terminal route boundary, not an implicit route-end policy.

The controller can traverse interior waypoints under current, but it does not claim the
historical tangent-arc geometry is preserved.  It also does not perform encounter
replanning, ownership arbitration, or route-end disposition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import hypot, isfinite
from typing import Callable

from .environmental_dynamics import WaterRelativeManeuverCommand, water_relative_command_transition_duration
from .environmental_route_handoff import (
    EnvironmentalGroundVelocityCommand,
    EnvironmentalGroundVelocityUnavailable,
    EnvironmentalRouteHandoffBinding,
    command_for_desired_ground_velocity,
)
from .environmental_route_handoff_session import (
    EnvironmentalRouteHandoffSessionMode,
    EnvironmentalRouteHandoffSessionRunResult,
    EnvironmentalRouteHandoffSessionStopReason,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .follower import RouteFollowerState
from .route import RouteProjection, marine_bearing_between

_EPS = 1e-10


class EnvironmentalNominalRouteSessionError(RuntimeError):
    """Base error for a99 nominal environmental route continuation."""


class EnvironmentalNominalRouteNotReady(EnvironmentalNominalRouteSessionError):
    """Raised when a99 is attached before a98 reached nominal readiness."""


class StaleEnvironmentalNominalRoute(EnvironmentalNominalRouteSessionError):
    """Raised when world/control evidence changed after the a98 handoff result."""


class EnvironmentalNominalRouteUnavailable(EnvironmentalNominalRouteSessionError):
    """Raised when a desired ground-route command cannot be physically represented."""


class EnvironmentalNominalRouteMode(str, Enum):
    TRACK = "track"
    WAYPOINT_TRANSITION = "waypoint_transition"
    RECOVERY = "recovery"
    ROUTE_COMPLETE = "route_complete"


class EnvironmentalNominalRouteStopReason(str, Enum):
    ROUTE_COMPLETE = "route_complete"
    END_TIME = "end_time"
    COMMAND_OVERRIDDEN = "command_overridden"


@dataclass(frozen=True, slots=True)
class EnvironmentalNominalRouteSessionConfig:
    review_interval_min: float = 0.25
    route_lookahead_nm: float = 0.25
    cross_track_tolerance_nm: float = 0.01
    waypoint_transition_distance_nm: float = 0.15

    def __post_init__(self) -> None:
        if not isfinite(self.review_interval_min) or self.review_interval_min <= 0.0:
            raise ValueError("review_interval_min must be finite and positive")
        if not isfinite(self.route_lookahead_nm) or self.route_lookahead_nm <= 0.0:
            raise ValueError("route_lookahead_nm must be finite and positive")
        if not isfinite(self.cross_track_tolerance_nm) or self.cross_track_tolerance_nm < 0.0:
            raise ValueError("cross_track_tolerance_nm must be finite and non-negative")
        if not isfinite(self.waypoint_transition_distance_nm) or self.waypoint_transition_distance_nm < 0.0:
            raise ValueError("waypoint_transition_distance_nm must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EnvironmentalNominalRouteStep:
    time_min: float
    vessel_id: str
    mode: EnvironmentalNominalRouteMode
    projection: RouteProjection
    follower_state: RouteFollowerState
    target_along_track: float | None
    ground_target: EnvironmentalGroundVelocityCommand | None
    command: WaterRelativeManeuverCommand | None
    next_review_at: float | None


@dataclass(frozen=True, slots=True)
class EnvironmentalNominalRouteVesselResult:
    vessel_id: str
    mode: EnvironmentalNominalRouteMode
    projection: RouteProjection
    follower_state: RouteFollowerState
    ground_target: EnvironmentalGroundVelocityCommand | None
    command: WaterRelativeManeuverCommand | None
    next_review_at: float | None

    @property
    def route_complete(self) -> bool:
        return self.mode is EnvironmentalNominalRouteMode.ROUTE_COMPLETE


@dataclass(frozen=True, slots=True)
class EnvironmentalNominalRouteRunResult:
    snapshot: EnvironmentalWorldSnapshot
    vessels: tuple[EnvironmentalNominalRouteVesselResult, ...]
    steps: tuple[EnvironmentalNominalRouteStep, ...]
    stop_reason: EnvironmentalNominalRouteStopReason
    overridden_vessel_ids: tuple[str, ...] = ()

    @property
    def route_complete(self) -> bool:
        return self.stop_reason is EnvironmentalNominalRouteStopReason.ROUTE_COMPLETE

    def by_id(self, vessel_id: str) -> EnvironmentalNominalRouteVesselResult:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)


@dataclass(slots=True)
class _Cursor:
    vessel_id: str
    follower_state: RouteFollowerState
    mode: EnvironmentalNominalRouteMode | None = None
    projection: RouteProjection | None = None
    ground_target: EnvironmentalGroundVelocityCommand | None = None
    issued_command: WaterRelativeManeuverCommand | None = None
    next_review_at: float | None = None

    def result(self) -> EnvironmentalNominalRouteVesselResult:
        if self.mode is None or self.projection is None:
            raise EnvironmentalNominalRouteSessionError("nominal route cursor is not initialized")
        return EnvironmentalNominalRouteVesselResult(
            vessel_id=self.vessel_id,
            mode=self.mode,
            projection=self.projection,
            follower_state=self.follower_state,
            ground_target=self.ground_target,
            command=self.issued_command,
            next_review_at=self.next_review_at,
        )


class EnvironmentalNominalRouteSession:
    """Shared-clock environmental continuation from a98 nominal readiness to route end."""

    def __init__(
        self,
        world: EnvironmentalWorld2D,
        binding: EnvironmentalRouteHandoffBinding,
        handoff_result: EnvironmentalRouteHandoffSessionRunResult,
        *,
        config: EnvironmentalNominalRouteSessionConfig = EnvironmentalNominalRouteSessionConfig(),
        profile_catalog=None,
        before_command_write: Callable[[], None] | None = None,
        command_transform: Callable[[str, WaterRelativeManeuverCommand], WaterRelativeManeuverCommand] | None = None,
    ) -> None:
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if not isinstance(binding, EnvironmentalRouteHandoffBinding):
            raise TypeError("binding must be EnvironmentalRouteHandoffBinding")
        if not isinstance(handoff_result, EnvironmentalRouteHandoffSessionRunResult):
            raise TypeError("handoff_result must be EnvironmentalRouteHandoffSessionRunResult")
        if not isinstance(config, EnvironmentalNominalRouteSessionConfig):
            raise TypeError("config must be EnvironmentalNominalRouteSessionConfig")
        if handoff_result.stop_reason is not EnvironmentalRouteHandoffSessionStopReason.NOMINAL_READY:
            raise EnvironmentalNominalRouteNotReady("a99 requires a98 NOMINAL_READY result")
        if not handoff_result.nominal_ready:
            raise EnvironmentalNominalRouteNotReady("a98 result is not nominal-ready")
        if tuple(item.vessel_id for item in handoff_result.vessels) != binding.vessel_ids:
            raise EnvironmentalNominalRouteNotReady("a98 result pair does not match route binding")

        self.world = world
        self.binding = binding
        self.handoff_result = handoff_result
        self.config = config
        self.profile_catalog = profile_catalog
        if before_command_write is not None and not callable(before_command_write):
            raise TypeError("before_command_write must be callable or None")
        if command_transform is not None and not callable(command_transform):
            raise TypeError("command_transform must be callable or None")
        self.before_command_write = before_command_write
        self.command_transform = command_transform
        self._cursors = {
            item.vessel_id: _Cursor(item.vessel_id, item.follower_state)
            for item in handoff_result.vessels
        }
        self._steps: list[EnvironmentalNominalRouteStep] = []
        self._started = False
        self._terminal_reason: EnvironmentalNominalRouteStopReason | None = None
        self._overridden_ids: tuple[str, ...] = ()

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return self.binding.vessel_ids

    @property
    def vessel_results(self) -> tuple[EnvironmentalNominalRouteVesselResult, ...]:
        return tuple(self._cursors[vessel_id].result() for vessel_id in self.vessel_ids)

    @property
    def steps(self) -> tuple[EnvironmentalNominalRouteStep, ...]:
        return tuple(self._steps)

    @property
    def route_complete(self) -> bool:
        return self._started and all(item.route_complete for item in self.vessel_results)

    def _binding_for(self, vessel_id: str):
        return self.binding.for_vessel(vessel_id)

    def _validate_start(self) -> None:
        if abs(self.handoff_result.snapshot.time - self.world.time) > _EPS:
            raise StaleEnvironmentalNominalRoute("a98 result timestamp no longer matches world")
        current = self.world.snapshot()
        for source in self.handoff_result.vessels:
            vessel_id = source.vessel_id
            try:
                state = current.by_id(vessel_id)
                command = self.world.command_for(vessel_id)
                drift = self.world.drift_for(vessel_id)
                limits = self.world.limits_for(vessel_id)
            except KeyError as exc:
                raise StaleEnvironmentalNominalRoute(f"route vessel {vessel_id!r} is no longer active") from exc
            if state != self.handoff_result.snapshot.by_id(vessel_id):
                raise StaleEnvironmentalNominalRoute(f"state for {vessel_id!r} changed after a98 readiness")
            if source.command is not None and command != source.command:
                raise StaleEnvironmentalNominalRoute(f"active command for {vessel_id!r} changed after a98 readiness")
            bound = self._binding_for(vessel_id)
            if drift != bound.environmental_drift_velocity:
                raise StaleEnvironmentalNominalRoute(f"environmental drift for {vessel_id!r} changed after binding")
            if limits != bound.maneuver_limits:
                raise StaleEnvironmentalNominalRoute(f"maneuver limits for {vessel_id!r} changed after binding")
            if self.profile_catalog is not None:
                profile = self.profile_catalog.profile_for(vessel_id)
                if profile != bound.profile:
                    raise StaleEnvironmentalNominalRoute(f"profile for {vessel_id!r} changed after binding")
        pending = self.world.next_command_time()
        if pending is not None and pending <= self.world.time + _EPS:
            raise StaleEnvironmentalNominalRoute("pending same-timestamp command event exists before nominal route continuation")

    def _projection(self, vessel_id: str) -> tuple[RouteProjection, RouteFollowerState]:
        cursor = self._cursors[vessel_id]
        follower = self._binding_for(vessel_id).target.navigator.follower
        route = follower.route
        state = self.world.snapshot().by_id(vessel_id)
        min_s = min(cursor.follower_state.along_track, route.length)
        projection = route.project(state.position, min_along_track=min_s)
        progress = max(min_s, projection.along_track)
        # Route completion is a finish-line crossing on the final leg, not capture of
        # the endpoint as a stationary point.  Requiring Euclidean distance to the
        # endpoint makes a constant-speed bounded-turn vessel orbit the endpoint after
        # a corner.  The historical follower likewise completes at route arc-length;
        # a99 additionally requires the final-leg lateral error to remain inside the
        # explicit route corridor.
        completed = (
            progress >= route.length - follower.position_tolerance
            and abs(projection.cross_track) <= max(
                follower.position_tolerance, self.config.cross_track_tolerance_nm
            )
        )
        return projection, replace(
            cursor.follower_state,
            along_track=min(progress, route.length),
            active_turn_waypoint=None,
            completed=completed,
        )

    def _desired_for(self, vessel_id: str):
        item = self._binding_for(vessel_id)
        follower = item.target.navigator.follower
        route = follower.route
        state = self.world.snapshot().by_id(vessel_id)
        projection, follower_state = self._projection(vessel_id)
        route_tol = max(follower.position_tolerance, self.config.cross_track_tolerance_nm)

        if follower_state.completed:
            return (
                EnvironmentalNominalRouteMode.ROUTE_COMPLETE,
                projection,
                follower_state,
                None,
                None,
                None,
                None,
            )

        base_s = max(follower_state.along_track, projection.along_track)
        target_s = min(route.length, base_s + self.config.route_lookahead_nm)
        target_point = route.point_at(target_s)
        if hypot(target_point[0] - state.position[0], target_point[1] - state.position[1]) <= _EPS:
            desired_heading = route.heading_at(min(base_s, route.length))
        else:
            desired_heading = marine_bearing_between(state.position, target_point)

        next_segment_boundary = route.cumulative_lengths[min(projection.segment_index + 1, route.segment_count)]
        distance_to_boundary = max(0.0, next_segment_boundary - base_s)
        if projection.distance > route_tol:
            mode = EnvironmentalNominalRouteMode.RECOVERY
        elif projection.segment_index < route.segment_count - 1 and distance_to_boundary <= self.config.waypoint_transition_distance_nm:
            mode = EnvironmentalNominalRouteMode.WAYPOINT_TRANSITION
        else:
            mode = EnvironmentalNominalRouteMode.TRACK

        try:
            ground_target = command_for_desired_ground_velocity(
                state,
                self.world.drift_for(vessel_id),
                desired_ground_heading_deg=desired_heading,
                desired_ground_speed=follower.cruise_speed,
                profile=item.profile,
            )
        except EnvironmentalGroundVelocityUnavailable as exc:
            raise EnvironmentalNominalRouteUnavailable(
                f"nominal route target unavailable for {vessel_id!r}: {exc}"
            ) from exc

        issued_command = ground_target.command
        if self.command_transform is not None:
            issued_command = self.command_transform(vessel_id, issued_command)
            if not isinstance(issued_command, WaterRelativeManeuverCommand):
                raise TypeError("command_transform must return WaterRelativeManeuverCommand")

        duration = water_relative_command_transition_duration(
            state,
            issued_command,
            self.world.limits_for(vessel_id),
        )
        if not isfinite(duration):
            raise EnvironmentalNominalRouteUnavailable(
                f"nominal route command is unreachable for {vessel_id!r}"
            )

        return (
            mode,
            projection,
            follower_state,
            target_s,
            ground_target,
            issued_command,
            self.world.time + self.config.review_interval_min,
        )

    def _refresh(self, due_ids: set[str]) -> tuple[EnvironmentalNominalRouteStep, ...]:
        # A102 may install a fleet ownership guard. Run it before any cursor or command
        # mutation so recapture is a clean ownership-transfer boundary.
        if due_ids and self.before_command_write is not None:
            self.before_command_write()
        computed = {vessel_id: self._desired_for(vessel_id) for vessel_id in self.vessel_ids if vessel_id in due_ids}
        commands: list[tuple[str, WaterRelativeManeuverCommand]] = []
        steps: list[EnvironmentalNominalRouteStep] = []
        for vessel_id in self.vessel_ids:
            if vessel_id not in computed:
                continue
            mode, projection, follower_state, target_s, ground_target, issued_command, next_review_at = computed[vessel_id]
            cursor = self._cursors[vessel_id]
            cursor.mode = mode
            cursor.projection = projection
            cursor.follower_state = follower_state
            cursor.ground_target = ground_target
            cursor.next_review_at = next_review_at
            cursor.issued_command = issued_command
            if ground_target is not None:
                assert issued_command is not None
                commands.append((vessel_id, issued_command))
            steps.append(EnvironmentalNominalRouteStep(
                time_min=self.world.time,
                vessel_id=vessel_id,
                mode=mode,
                projection=projection,
                follower_state=follower_state,
                target_along_track=target_s,
                ground_target=ground_target,
                command=issued_command,
                next_review_at=next_review_at,
            ))
        if commands:
            self.world.schedule_commands_atomic(at=self.world.time, commands=commands)
            self.world.run_until(self.world.time)
        self._steps.extend(steps)
        return tuple(steps)

    def start(self) -> tuple[EnvironmentalNominalRouteStep, ...]:
        if self._started:
            raise EnvironmentalNominalRouteSessionError("environmental nominal route session already started")
        self._validate_start()
        self._started = True
        steps = self._refresh(set(self.vessel_ids))
        if self.route_complete:
            self._terminal_reason = EnvironmentalNominalRouteStopReason.ROUTE_COMPLETE
        return steps

    def _detect_override(self) -> tuple[str, ...]:
        overridden: list[str] = []
        for vessel_id in self.vessel_ids:
            cursor = self._cursors[vessel_id]
            if cursor.issued_command is None or cursor.next_review_at is None:
                continue
            if self.world.command_for(vessel_id) != cursor.issued_command:
                overridden.append(vessel_id)
        return tuple(overridden)

    def next_event_time(self) -> float | None:
        if not self._started:
            raise EnvironmentalNominalRouteSessionError("environmental nominal route session has not started")
        if self._terminal_reason is not None or self.route_complete:
            return None
        reviews = [cursor.next_review_at for cursor in self._cursors.values() if cursor.next_review_at is not None]
        if not reviews:
            raise EnvironmentalNominalRouteSessionError("no future nominal route review exists before route completion")
        return min(reviews)

    def result(self, reason: EnvironmentalNominalRouteStopReason) -> EnvironmentalNominalRouteRunResult:
        return EnvironmentalNominalRouteRunResult(
            snapshot=self.world.snapshot(),
            vessels=self.vessel_results,
            steps=self.steps,
            stop_reason=reason,
            overridden_vessel_ids=self._overridden_ids,
        )

    def run_until(self, end_time: float, *, max_events: int = 4096) -> EnvironmentalNominalRouteRunResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        if not self._started:
            self.start()
        if self._terminal_reason is not None:
            return self.result(self._terminal_reason)

        for _ in range(max_events):
            if self.route_complete:
                self._terminal_reason = EnvironmentalNominalRouteStopReason.ROUTE_COMPLETE
                return self.result(self._terminal_reason)
            if self.world.time >= end_time - _EPS:
                return self.result(EnvironmentalNominalRouteStopReason.END_TIME)

            next_review = self.next_event_time()
            assert next_review is not None
            next_external = self.world.next_command_time()
            boundary = min(end_time, next_review)
            if next_external is not None:
                boundary = min(boundary, next_external)
            self.world.run_until(boundary)

            overridden = self._detect_override()
            if overridden:
                self._overridden_ids = overridden
                self._terminal_reason = EnvironmentalNominalRouteStopReason.COMMAND_OVERRIDDEN
                return self.result(self._terminal_reason)

            due = {
                vessel_id for vessel_id, cursor in self._cursors.items()
                if cursor.next_review_at is not None and cursor.next_review_at <= self.world.time + _EPS
            }
            if due:
                self._refresh(due)
                if self.route_complete:
                    self._terminal_reason = EnvironmentalNominalRouteStopReason.ROUTE_COMPLETE
                    return self.result(self._terminal_reason)

            if self.world.time >= end_time - _EPS:
                return self.result(EnvironmentalNominalRouteStopReason.END_TIME)

        raise EnvironmentalNominalRouteSessionError("environmental nominal route session exceeded max_events")
