"""Event-oriented route guidance for the next-generation 2D simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf
from typing import Literal

from ..kinematics import SpeedLimits, time_to_cover_distance
from .dynamics import ManeuverCommand, ManeuverLimits, Vessel2DState, signed_heading_delta
from .route import Route2D, RouteProjection, TurnGeometry
from .world import World2D, WorldSnapshot

_EPS = 1e-10

FollowerMode = Literal["track", "turn", "complete"]


@dataclass(frozen=True, slots=True)
class RouteFollowerState:
    """Persistent monotonic routing state for one vessel."""

    along_track: float = 0.0
    active_turn_waypoint: int | None = None
    completed: bool = False

    def __post_init__(self) -> None:
        if self.along_track < 0:
            raise ValueError("along_track must be non-negative")
        if self.active_turn_waypoint is not None and self.active_turn_waypoint < 1:
            raise ValueError("active_turn_waypoint must be an interior waypoint index")


@dataclass(frozen=True, slots=True)
class RouteGuidance:
    """One deterministic guidance decision plus its next required review boundary."""

    mode: FollowerMode
    command: ManeuverCommand
    projection: RouteProjection
    follower_state: RouteFollowerState
    target_segment_index: int
    turn: TurnGeometry | None
    next_review_after: float | None
    next_review_along_track: float | None


@dataclass(frozen=True, slots=True)
class GuidedRunResult:
    snapshot: WorldSnapshot
    follower_state: RouteFollowerState
    guidance_steps: tuple[RouteGuidance, ...]


@dataclass(frozen=True, slots=True)
class RouteFollower:
    """Follow a ``Route2D`` with speed/heading commands and analytic review timing.

    Alpha-3 is intentionally a nominal-track follower, not a complete autopilot. On a
    straight leg it schedules the next guidance review exactly at the tangent-arc start
    (or route end). During a turn it schedules the review exactly when the commanded
    heading is reached. This keeps the normal route-following path event-driven rather
    than dependent on an arbitrary controller time step.

    Large off-track recovery is deliberately outside this slice. A caller can still ask
    for a corrective command, but ``next_review_after`` is ``None`` until the vessel is
    sufficiently aligned for exact along-track timing again.
    """

    route: Route2D
    cruise_speed: float
    position_tolerance: float = 1e-7
    heading_tolerance_deg: float = 1e-7

    def __post_init__(self) -> None:
        if self.cruise_speed < 0:
            raise ValueError("cruise_speed must be non-negative")
        if self.position_tolerance < 0:
            raise ValueError("position_tolerance must be non-negative")
        if self.heading_tolerance_deg < 0:
            raise ValueError("heading_tolerance_deg must be non-negative")

    @staticmethod
    def _speed_limits(limits: ManeuverLimits) -> SpeedLimits:
        return SpeedLimits(
            max_acceleration=limits.max_acceleration,
            max_deceleration=limits.max_deceleration,
        )

    def _time_to_along_boundary(
        self,
        vessel: Vessel2DState,
        projection: RouteProjection,
        target_along: float,
        limits: ManeuverLimits,
        *,
        required_heading: float,
    ) -> float | None:
        remaining = max(0.0, target_along - projection.along_track)
        if remaining <= self.position_tolerance:
            return 0.0
        if abs(projection.cross_track) > self.position_tolerance:
            return None
        if abs(signed_heading_delta(vessel.heading_deg, required_heading)) > self.heading_tolerance_deg:
            return None
        value = time_to_cover_distance(
            vessel.speed,
            self.cruise_speed,
            remaining,
            self._speed_limits(limits),
        )
        return None if value == inf else value

    def _turn_for_segment(self, segment_index: int, limits: ManeuverLimits) -> TurnGeometry | None:
        waypoint_index = segment_index + 1
        if waypoint_index >= len(self.route.waypoints) - 1:
            return None
        return self.route.turn_geometry(
            waypoint_index,
            speed=self.cruise_speed,
            max_turn_rate_deg_per_min=limits.max_turn_rate_deg_per_min,
        )

    def guidance(
        self,
        vessel: Vessel2DState,
        follower_state: RouteFollowerState,
        limits: ManeuverLimits,
    ) -> RouteGuidance:
        """Return the command to apply now and the next exact guidance-review boundary."""
        min_s = min(follower_state.along_track, self.route.length)
        projection = self.route.project(vessel.position, min_along_track=min_s)
        progress = max(min_s, projection.along_track)
        state = replace(follower_state, along_track=progress)

        # Continue a previously initiated turn until the outgoing heading is reached.
        if state.active_turn_waypoint is not None:
            turn = self.route.turn_geometry(
                state.active_turn_waypoint,
                speed=self.cruise_speed,
                max_turn_rate_deg_per_min=limits.max_turn_rate_deg_per_min,
            )
            heading_error = signed_heading_delta(vessel.heading_deg, turn.outgoing_heading_deg)
            if abs(heading_error) > self.heading_tolerance_deg:
                review_after = (
                    abs(heading_error) / limits.max_turn_rate_deg_per_min
                    if limits.max_turn_rate_deg_per_min > 0
                    else None
                )
                return RouteGuidance(
                    mode="turn",
                    command=ManeuverCommand(turn.outgoing_heading_deg, self.cruise_speed),
                    projection=projection,
                    follower_state=state,
                    target_segment_index=state.active_turn_waypoint,
                    turn=turn,
                    next_review_after=review_after,
                    next_review_along_track=turn.end_along_track,
                )

            # Heading target reached: leave turn mode and let the outgoing leg take over.
            state = replace(
                state,
                along_track=max(state.along_track, min(turn.end_along_track, self.route.length)),
                active_turn_waypoint=None,
            )
            projection = self.route.project(vessel.position, min_along_track=state.along_track)
            state = replace(state, along_track=max(state.along_track, projection.along_track))

        if state.along_track >= self.route.length - self.position_tolerance:
            final_heading = self.route.segment_heading(self.route.segment_count - 1)
            return RouteGuidance(
                mode="complete",
                command=ManeuverCommand(final_heading, self.cruise_speed),
                projection=projection,
                follower_state=replace(state, along_track=self.route.length, completed=True),
                target_segment_index=self.route.segment_count - 1,
                turn=None,
                next_review_after=None,
                next_review_along_track=None,
            )

        segment_index = self.route.segment_index_at(state.along_track)
        segment_heading = self.route.segment_heading(segment_index)
        turn = self._turn_for_segment(segment_index, limits)

        if turn is not None and state.along_track >= turn.start_along_track - self.position_tolerance:
            heading_error = signed_heading_delta(vessel.heading_deg, turn.outgoing_heading_deg)
            review_after = (
                abs(heading_error) / limits.max_turn_rate_deg_per_min
                if abs(heading_error) > self.heading_tolerance_deg and limits.max_turn_rate_deg_per_min > 0
                else 0.0
            )
            turn_state = replace(state, active_turn_waypoint=turn.waypoint_index)
            return RouteGuidance(
                mode="turn",
                command=ManeuverCommand(turn.outgoing_heading_deg, self.cruise_speed),
                projection=projection,
                follower_state=turn_state,
                target_segment_index=segment_index + 1,
                turn=turn,
                next_review_after=review_after,
                next_review_along_track=turn.end_along_track,
            )

        boundary = turn.start_along_track if turn is not None else self.route.length
        review_after = self._time_to_along_boundary(
            vessel,
            projection,
            boundary,
            limits,
            required_heading=segment_heading,
        )
        return RouteGuidance(
            mode="track",
            command=ManeuverCommand(segment_heading, self.cruise_speed),
            projection=projection,
            follower_state=state,
            target_segment_index=segment_index,
            turn=turn,
            next_review_after=review_after,
            next_review_along_track=boundary,
        )

    def run_world_until(
        self,
        world: World2D,
        *,
        vessel_id: str,
        end_time: float,
        follower_state: RouteFollowerState | None = None,
    ) -> GuidedRunResult:
        """Drive one vessel in ``World2D`` through exact normal-route guidance events."""
        if end_time < world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        state = follower_state or RouteFollowerState()
        steps: list[RouteGuidance] = []

        while world.time < end_time - _EPS and not state.completed:
            vessel = world.snapshot().by_id(vessel_id)
            limits = world.limits_for(vessel_id)
            decision = self.guidance(vessel, state, limits)
            steps.append(decision)
            state = decision.follower_state
            world.schedule_command(at=world.time, vessel_id=vessel_id, command=decision.command)

            if state.completed:
                break
            if decision.next_review_after is None:
                raise RuntimeError(
                    "exact next review is unavailable for an off-track/misaligned state; "
                    "external recovery review is required"
                )
            if decision.next_review_after <= _EPS:
                # Apply the command without advancing, then recompute once. The next
                # decision must either leave this boundary or progress into turn mode.
                world.run_until(world.time)
                continue
            world.run_until(min(end_time, world.time + decision.next_review_after))

        if world.time < end_time - _EPS and not state.completed:
            world.run_until(end_time)
        return GuidedRunResult(
            snapshot=world.snapshot(),
            follower_state=state,
            guidance_steps=tuple(steps),
        )
