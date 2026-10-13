"""Persistent environmental route-recovery handoff to nominal ground guidance.

A98 is the first environmental route-handoff *executor*.  It deliberately does not
reuse the historical tangent-circle ``CrossTrackRecovery`` state machine under current:
that geometry assumes vessel heading/speed are also the Earth-fixed translation vector.
With environmental drift they are not.

Instead this session closes the loop on Earth-fixed route error:

- route projection and desired track are ground-referenced;
- every desired ground course/speed is translated by a97 into a water-relative command;
- bounded water-relative dynamics remain owned by ``EnvironmentalWorld2D``;
- off-track recovery uses an explicit look-ahead point on the caller-supplied route;
- reviews are explicit model events, not hidden numerical integration steps;
- nominal readiness requires both route proximity and the *actual ground velocity* to
  match the nominal route vector.

The session stops once every bound vessel is nominal-ready (or route-complete), at an
explicit end-time boundary, or after an external command override.  It does not execute
future route turns, route-end disposition, pair replanning, or fleet ownership changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import hypot, isfinite
from typing import Callable

from ..kinematics import Vector2
from .environmental_dynamics import (
    WaterRelativeManeuverCommand,
    water_relative_command_transition_duration,
)
from .environmental_evidence import environmental_vessel_motion_evidence
from .environmental_route_handoff import (
    EnvironmentalGroundVelocityCommand,
    EnvironmentalGroundVelocityUnavailable,
    EnvironmentalRouteHandoffBinding,
    StaleEnvironmentalRouteHandoff,
    command_for_desired_ground_velocity,
    validate_environmental_route_handoff_binding,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .follower import RouteFollowerState
from .route import RouteProjection, marine_bearing_between
from .velocity_frames import velocity_from_course_speed

_EPS = 1e-10


class EnvironmentalRouteHandoffSessionError(RuntimeError):
    """Base error for a98 handoff execution failures."""


class EnvironmentalRouteRecoveryUnavailable(EnvironmentalRouteHandoffSessionError):
    """Raised when the explicit ground-track recovery target cannot be executed."""


class EnvironmentalRouteHandoffSessionMode(str, Enum):
    RECOVERY = "recovery"
    NOMINAL_ALIGN = "nominal_align"
    NOMINAL_READY = "nominal_ready"
    ROUTE_COMPLETE = "route_complete"


class EnvironmentalRouteHandoffSessionStopReason(str, Enum):
    NOMINAL_READY = "nominal_ready"
    END_TIME = "end_time"
    COMMAND_OVERRIDDEN = "command_overridden"


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteHandoffSessionConfig:
    review_interval_min: float = 0.25
    recovery_lookahead_nm: float = 0.5
    cross_track_tolerance_nm: float = 0.01
    ground_velocity_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not isfinite(self.review_interval_min) or self.review_interval_min <= 0.0:
            raise ValueError("review_interval_min must be finite and positive")
        if not isfinite(self.recovery_lookahead_nm) or self.recovery_lookahead_nm <= 0.0:
            raise ValueError("recovery_lookahead_nm must be finite and positive")
        if not isfinite(self.cross_track_tolerance_nm) or self.cross_track_tolerance_nm < 0.0:
            raise ValueError("cross_track_tolerance_nm must be finite and non-negative")
        if not isfinite(self.ground_velocity_tolerance) or self.ground_velocity_tolerance < 0.0:
            raise ValueError("ground_velocity_tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteHandoffSessionStep:
    time_min: float
    vessel_id: str
    mode: EnvironmentalRouteHandoffSessionMode
    projection: RouteProjection
    follower_state: RouteFollowerState
    ground_target: EnvironmentalGroundVelocityCommand | None
    command: WaterRelativeManeuverCommand | None
    next_review_at: float | None


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteHandoffSessionVesselResult:
    vessel_id: str
    mode: EnvironmentalRouteHandoffSessionMode
    follower_state: RouteFollowerState
    projection: RouteProjection
    ground_target: EnvironmentalGroundVelocityCommand | None
    command: WaterRelativeManeuverCommand | None
    next_review_at: float | None

    @property
    def nominal_ready(self) -> bool:
        return self.mode in {
            EnvironmentalRouteHandoffSessionMode.NOMINAL_READY,
            EnvironmentalRouteHandoffSessionMode.ROUTE_COMPLETE,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteHandoffSessionRunResult:
    snapshot: EnvironmentalWorldSnapshot
    vessels: tuple[EnvironmentalRouteHandoffSessionVesselResult, ...]
    steps: tuple[EnvironmentalRouteHandoffSessionStep, ...]
    stop_reason: EnvironmentalRouteHandoffSessionStopReason
    overridden_vessel_ids: tuple[str, ...] = ()

    @property
    def nominal_ready(self) -> bool:
        return self.stop_reason is EnvironmentalRouteHandoffSessionStopReason.NOMINAL_READY

    def by_id(self, vessel_id: str) -> EnvironmentalRouteHandoffSessionVesselResult:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)


@dataclass(slots=True)
class _Cursor:
    vessel_id: str
    follower_state: RouteFollowerState
    mode: EnvironmentalRouteHandoffSessionMode | None = None
    projection: RouteProjection | None = None
    ground_target: EnvironmentalGroundVelocityCommand | None = None
    issued_command: WaterRelativeManeuverCommand | None = None
    next_review_at: float | None = None

    def result(self) -> EnvironmentalRouteHandoffSessionVesselResult:
        if self.mode is None or self.projection is None:
            raise EnvironmentalRouteHandoffSessionError("handoff cursor has not been initialized")
        return EnvironmentalRouteHandoffSessionVesselResult(
            vessel_id=self.vessel_id,
            mode=self.mode,
            follower_state=self.follower_state,
            projection=self.projection,
            ground_target=self.ground_target,
            command=self.issued_command,
            next_review_at=self.next_review_at,
        )


def _vector_error(a: Vector2, b: Vector2) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


class EnvironmentalRouteHandoffSession:
    """Shared-clock pair recovery until nominal ground-route guidance is re-established."""

    def __init__(
        self,
        world: EnvironmentalWorld2D,
        binding: EnvironmentalRouteHandoffBinding,
        *,
        config: EnvironmentalRouteHandoffSessionConfig = EnvironmentalRouteHandoffSessionConfig(),
        profile_catalog=None,
        before_command_write: Callable[[], None] | None = None,
        command_transform: Callable[[str, WaterRelativeManeuverCommand], WaterRelativeManeuverCommand] | None = None,
    ) -> None:
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if not isinstance(binding, EnvironmentalRouteHandoffBinding):
            raise TypeError("binding must be EnvironmentalRouteHandoffBinding")
        if not isinstance(config, EnvironmentalRouteHandoffSessionConfig):
            raise TypeError("config must be EnvironmentalRouteHandoffSessionConfig")
        self.world = world
        self.binding = binding
        self.config = config
        self.profile_catalog = profile_catalog
        if before_command_write is not None and not callable(before_command_write):
            raise TypeError("before_command_write must be callable or None")
        if command_transform is not None and not callable(command_transform):
            raise TypeError("command_transform must be callable or None")
        self.before_command_write = before_command_write
        self.command_transform = command_transform
        self._cursors = {
            item.vessel_id: _Cursor(item.vessel_id, item.target.follower_state)
            for item in binding.vessels
        }
        self._steps: list[EnvironmentalRouteHandoffSessionStep] = []
        self._started = False
        self._terminal_reason: EnvironmentalRouteHandoffSessionStopReason | None = None
        self._overridden_ids: tuple[str, ...] = ()

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return self.binding.vessel_ids

    @property
    def vessel_results(self) -> tuple[EnvironmentalRouteHandoffSessionVesselResult, ...]:
        return tuple(self._cursors[vessel_id].result() for vessel_id in self.vessel_ids)

    @property
    def steps(self) -> tuple[EnvironmentalRouteHandoffSessionStep, ...]:
        return tuple(self._steps)

    @property
    def nominal_ready(self) -> bool:
        return self._started and all(item.nominal_ready for item in self.vessel_results)

    def _binding_for(self, vessel_id: str):
        return self.binding.for_vessel(vessel_id)

    def _projection(self, vessel_id: str) -> tuple[RouteProjection, RouteFollowerState]:
        cursor = self._cursors[vessel_id]
        target = self._binding_for(vessel_id).target
        route = target.navigator.follower.route
        state = self.world.snapshot().by_id(vessel_id)
        min_s = min(cursor.follower_state.along_track, route.length)
        projection = route.project(state.position, min_along_track=min_s)
        progress = max(min_s, projection.along_track)
        follower_state = replace(
            cursor.follower_state,
            along_track=min(progress, route.length),
            active_turn_waypoint=None,
            completed=(progress >= route.length - target.navigator.follower.position_tolerance),
        )
        return projection, follower_state

    def _desired_for(self, vessel_id: str) -> tuple[
        EnvironmentalRouteHandoffSessionMode,
        RouteProjection,
        RouteFollowerState,
        EnvironmentalGroundVelocityCommand | None,
        WaterRelativeManeuverCommand | None,
        float | None,
    ]:
        item = self._binding_for(vessel_id)
        target = item.target
        follower = target.navigator.follower
        route = follower.route
        state = self.world.snapshot().by_id(vessel_id)
        projection, follower_state = self._projection(vessel_id)

        route_tolerance = max(follower.position_tolerance, self.config.cross_track_tolerance_nm)
        if follower_state.completed and projection.distance <= route_tolerance:
            return (
                EnvironmentalRouteHandoffSessionMode.ROUTE_COMPLETE,
                projection,
                follower_state,
                None,
                None,
                None,
            )

        if projection.distance > route_tolerance:
            if follower.cruise_speed <= _EPS:
                raise EnvironmentalRouteRecoveryUnavailable(
                    f"off-route vessel {vessel_id!r} requires positive nominal ground speed"
                )
            target_s = min(
                route.length,
                max(follower_state.along_track, projection.along_track)
                + self.config.recovery_lookahead_nm,
            )
            target_point = route.point_at(target_s)
            if hypot(target_point[0] - state.position[0], target_point[1] - state.position[1]) <= _EPS:
                desired_heading = projection.heading_deg
            else:
                desired_heading = marine_bearing_between(state.position, target_point)
            mode = EnvironmentalRouteHandoffSessionMode.RECOVERY
        else:
            desired_heading = route.heading_at(min(follower_state.along_track, route.length))
            mode = EnvironmentalRouteHandoffSessionMode.NOMINAL_ALIGN

        try:
            ground_target = command_for_desired_ground_velocity(
                state,
                self.world.drift_for(vessel_id),
                desired_ground_heading_deg=desired_heading,
                desired_ground_speed=follower.cruise_speed,
                profile=item.profile,
            )
        except EnvironmentalGroundVelocityUnavailable as exc:
            raise EnvironmentalRouteRecoveryUnavailable(
                f"route target unavailable for {vessel_id!r}: {exc}"
            ) from exc

        issued_command = ground_target.command
        if self.command_transform is not None:
            issued_command = self.command_transform(vessel_id, issued_command)
            if not isinstance(issued_command, WaterRelativeManeuverCommand):
                raise TypeError("command_transform must return WaterRelativeManeuverCommand")

        desired_ground = ground_target.desired_ground_velocity
        actual_ground = environmental_vessel_motion_evidence(
            state,
            self.world.drift_for(vessel_id),
        ).ground_velocity
        if (
            mode is EnvironmentalRouteHandoffSessionMode.NOMINAL_ALIGN
            and issued_command == ground_target.command
            and projection.distance <= route_tolerance
            and _vector_error(actual_ground, desired_ground) <= self.config.ground_velocity_tolerance
        ):
            mode = EnvironmentalRouteHandoffSessionMode.NOMINAL_READY
            return mode, projection, follower_state, ground_target, issued_command, None

        duration = water_relative_command_transition_duration(
            state,
            issued_command,
            self.world.limits_for(vessel_id),
        )
        if not isfinite(duration):
            raise EnvironmentalRouteRecoveryUnavailable(
                f"water-relative route command is unreachable for {vessel_id!r}"
            )
        # Review cadence is an explicit controller contract, not the residual
        # command-transition duration.  Using ever-smaller residual turn times with a
        # moving look-ahead target can create a Zeno-like sequence of tiny reviews.
        # Reachability is checked above; physical propagation remains exact inside the
        # environmental world between fixed review boundaries.
        return (
            mode,
            projection,
            follower_state,
            ground_target,
            issued_command,
            self.world.time + self.config.review_interval_min,
        )

    def _refresh(self, due_ids: set[str]) -> tuple[EnvironmentalRouteHandoffSessionStep, ...]:
        # A102 may install a fleet ownership guard. Run it before any cursor or command
        # mutation so recapture is a clean ownership-transfer boundary.
        if due_ids and self.before_command_write is not None:
            self.before_command_write()
        # Compute the full due set before any command mutation.
        computed = {
            vessel_id: self._desired_for(vessel_id)
            for vessel_id in self.vessel_ids
            if vessel_id in due_ids
        }
        commands: list[tuple[str, WaterRelativeManeuverCommand]] = []
        steps: list[EnvironmentalRouteHandoffSessionStep] = []
        for vessel_id in self.vessel_ids:
            if vessel_id not in computed:
                continue
            mode, projection, follower_state, ground_target, issued_command, next_review_at = computed[vessel_id]
            cursor = self._cursors[vessel_id]
            cursor.mode = mode
            cursor.projection = projection
            cursor.follower_state = follower_state
            cursor.ground_target = ground_target
            cursor.next_review_at = next_review_at
            cursor.issued_command = issued_command
            if ground_target is not None and mode not in {
                EnvironmentalRouteHandoffSessionMode.NOMINAL_READY,
                EnvironmentalRouteHandoffSessionMode.ROUTE_COMPLETE,
            }:
                assert issued_command is not None
                commands.append((vessel_id, issued_command))
            elif ground_target is not None and mode is EnvironmentalRouteHandoffSessionMode.NOMINAL_READY:
                # Publish the exact nominal command before releasing handoff ownership.
                assert issued_command is not None
                commands.append((vessel_id, issued_command))
            steps.append(
                EnvironmentalRouteHandoffSessionStep(
                    time_min=self.world.time,
                    vessel_id=vessel_id,
                    mode=mode,
                    projection=projection,
                    follower_state=follower_state,
                    ground_target=ground_target,
                    command=issued_command,
                    next_review_at=next_review_at,
                )
            )

        if commands:
            self.world.schedule_commands_atomic(at=self.world.time, commands=commands)
            self.world.run_until(self.world.time)
        self._steps.extend(steps)
        return tuple(steps)

    def start(self) -> tuple[EnvironmentalRouteHandoffSessionStep, ...]:
        if self._started:
            raise EnvironmentalRouteHandoffSessionError("environmental route handoff already started")
        validate_environmental_route_handoff_binding(
            self.binding,
            self.world,
            profile_catalog=self.profile_catalog,
        )
        pending = self.world.next_command_time()
        if pending is not None and pending <= self.world.time + _EPS:
            raise StaleEnvironmentalRouteHandoff(
                "pending same-timestamp command event exists before environmental route handoff"
            )
        self._started = True
        steps = self._refresh(set(self.vessel_ids))
        if self.nominal_ready:
            self._terminal_reason = EnvironmentalRouteHandoffSessionStopReason.NOMINAL_READY
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
            raise EnvironmentalRouteHandoffSessionError("environmental route handoff has not started")
        if self._terminal_reason is not None or self.nominal_ready:
            return None
        reviews = [
            cursor.next_review_at
            for cursor in self._cursors.values()
            if cursor.next_review_at is not None
        ]
        if not reviews:
            raise EnvironmentalRouteHandoffSessionError(
                "no future route-handoff review exists before nominal readiness"
            )
        return min(reviews)

    def result(self, reason: EnvironmentalRouteHandoffSessionStopReason) -> EnvironmentalRouteHandoffSessionRunResult:
        return EnvironmentalRouteHandoffSessionRunResult(
            snapshot=self.world.snapshot(),
            vessels=self.vessel_results,
            steps=self.steps,
            stop_reason=reason,
            overridden_vessel_ids=self._overridden_ids,
        )

    def run_until(
        self,
        end_time: float,
        *,
        max_events: int = 256,
    ) -> EnvironmentalRouteHandoffSessionRunResult:
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
            if self.nominal_ready:
                self._terminal_reason = EnvironmentalRouteHandoffSessionStopReason.NOMINAL_READY
                return self.result(self._terminal_reason)
            if self.world.time >= end_time - _EPS:
                return self.result(EnvironmentalRouteHandoffSessionStopReason.END_TIME)

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
                self._terminal_reason = EnvironmentalRouteHandoffSessionStopReason.COMMAND_OVERRIDDEN
                return self.result(self._terminal_reason)

            due = {
                vessel_id
                for vessel_id, cursor in self._cursors.items()
                if cursor.next_review_at is not None
                and cursor.next_review_at <= self.world.time + _EPS
            }
            if due:
                self._refresh(due)
                if self.nominal_ready:
                    self._terminal_reason = EnvironmentalRouteHandoffSessionStopReason.NOMINAL_READY
                    return self.result(self._terminal_reason)

            if self.world.time >= end_time - _EPS:
                return self.result(EnvironmentalRouteHandoffSessionStopReason.END_TIME)

        raise EnvironmentalRouteHandoffSessionError("environmental route handoff exceeded max_events")
