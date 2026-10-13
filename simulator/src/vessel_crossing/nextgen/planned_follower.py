"""Event-oriented execution of a precomputed RouteSpeedPlan."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf
from typing import Literal

from ..kinematics import SpeedLimits, time_to_cover_distance
from .dynamics import ManeuverCommand, ManeuverLimits, Vessel2DState, signed_heading_delta
from .follower import RouteFollowerState
from .route import RouteProjection
from .speed_plan import PlannedTurn, RouteSpeedPlan
from .world import World2D, WorldSnapshot

_EPS = 1e-9
PlannedFollowerMode = Literal["cruise", "brake", "turn", "complete"]


class SpeedPlanExecutionError(RuntimeError):
    """Raised when the current vessel state cannot execute the frozen speed plan exactly."""


@dataclass(frozen=True, slots=True)
class PlannedRouteGuidance:
    mode: PlannedFollowerMode
    command: ManeuverCommand
    projection: RouteProjection
    follower_state: RouteFollowerState
    turn: PlannedTurn | None
    next_review_after: float | None
    next_review_along_track: float | None


@dataclass(frozen=True, slots=True)
class PlannedRouteRunResult:
    snapshot: WorldSnapshot
    follower_state: RouteFollowerState
    guidance_steps: tuple[PlannedRouteGuidance, ...]


@dataclass(frozen=True, slots=True)
class SpeedPlannedRouteFollower:
    plan: RouteSpeedPlan
    position_tolerance: float = 1e-7
    heading_tolerance_deg: float = 1e-7
    speed_tolerance: float = 1e-8

    @property
    def route(self):
        return self.plan.route

    @staticmethod
    def _speed_limits(limits: ManeuverLimits) -> SpeedLimits:
        return SpeedLimits(limits.max_acceleration, limits.max_deceleration)

    def _time_to_boundary(
        self,
        vessel: Vessel2DState,
        projection: RouteProjection,
        *,
        target_along: float,
        target_speed: float,
        required_heading: float,
        limits: ManeuverLimits,
    ) -> float | None:
        distance = max(0.0, target_along - projection.along_track)
        if distance <= self.position_tolerance:
            return 0.0
        if abs(projection.cross_track) > self.position_tolerance:
            return None
        if abs(signed_heading_delta(vessel.heading_deg, required_heading)) > self.heading_tolerance_deg:
            return None
        value = time_to_cover_distance(
            vessel.speed,
            target_speed,
            distance,
            self._speed_limits(limits),
        )
        return None if value == inf else value

    def _turn_for_segment(self, segment_index: int) -> PlannedTurn | None:
        waypoint_index = segment_index + 1
        try:
            return self.plan.turn_for_waypoint(waypoint_index)
        except KeyError:
            return None

    def guidance(
        self,
        vessel: Vessel2DState,
        follower_state: RouteFollowerState,
        limits: ManeuverLimits,
    ) -> PlannedRouteGuidance:
        min_s = min(follower_state.along_track, self.route.length)
        projection = self.route.project(vessel.position, min_along_track=min_s)
        state = replace(follower_state, along_track=max(min_s, projection.along_track))

        if state.along_track <= self.position_tolerance and vessel.speed > self.plan.entry_speed_ceiling + self.speed_tolerance:
            raise SpeedPlanExecutionError(
                f"entry speed {vessel.speed} exceeds route entry ceiling {self.plan.entry_speed_ceiling}"
            )

        if state.active_turn_waypoint is not None:
            turn = self.plan.turn_for_waypoint(state.active_turn_waypoint)
            outgoing = self.route.segment_heading(state.active_turn_waypoint)
            error = signed_heading_delta(vessel.heading_deg, outgoing)
            if abs(error) > self.heading_tolerance_deg:
                return PlannedRouteGuidance(
                    mode="turn",
                    command=ManeuverCommand(outgoing, turn.planned_speed),
                    projection=projection,
                    follower_state=state,
                    turn=turn,
                    next_review_after=abs(error) / limits.max_turn_rate_deg_per_min,
                    next_review_along_track=turn.end_along_track,
                )
            state = replace(
                state,
                along_track=max(state.along_track, turn.end_along_track),
                active_turn_waypoint=None,
            )
            projection = self.route.project(vessel.position, min_along_track=state.along_track)
            state = replace(state, along_track=max(state.along_track, projection.along_track))

        if state.along_track >= self.route.length - self.position_tolerance:
            final_heading = self.route.segment_heading(self.route.segment_count - 1)
            return PlannedRouteGuidance(
                mode="complete",
                command=ManeuverCommand(final_heading, self.plan.desired_speed),
                projection=projection,
                follower_state=replace(state, along_track=self.route.length, completed=True),
                turn=None,
                next_review_after=None,
                next_review_along_track=None,
            )

        segment_index = self.route.segment_index_at(state.along_track)
        heading = self.route.segment_heading(segment_index)
        turn = self._turn_for_segment(segment_index)

        if turn is None:
            target_speed = self.plan.desired_speed
            boundary = self.route.length
            mode: PlannedFollowerMode = "cruise"
        elif state.along_track >= turn.start_along_track - self.position_tolerance:
            if abs(vessel.speed - turn.planned_speed) > self.speed_tolerance:
                raise SpeedPlanExecutionError(
                    f"turn {turn.waypoint_index} reached at speed {vessel.speed}, planned {turn.planned_speed}"
                )
            outgoing = self.route.segment_heading(turn.waypoint_index)
            error = signed_heading_delta(vessel.heading_deg, outgoing)
            turn_state = replace(state, active_turn_waypoint=turn.waypoint_index)
            return PlannedRouteGuidance(
                mode="turn",
                command=ManeuverCommand(outgoing, turn.planned_speed),
                projection=projection,
                follower_state=turn_state,
                turn=turn,
                next_review_after=(
                    abs(error) / limits.max_turn_rate_deg_per_min
                    if abs(error) > self.heading_tolerance_deg
                    else 0.0
                ),
                next_review_along_track=turn.end_along_track,
            )
        elif state.along_track >= turn.braking_start_along_track - self.position_tolerance:
            target_speed = turn.planned_speed
            boundary = turn.start_along_track
            mode = "brake"
        else:
            target_speed = self.plan.desired_speed
            boundary = turn.braking_start_along_track
            mode = "cruise"

        review_after = self._time_to_boundary(
            vessel,
            projection,
            target_along=boundary,
            target_speed=target_speed,
            required_heading=heading,
            limits=limits,
        )
        return PlannedRouteGuidance(
            mode=mode,
            command=ManeuverCommand(heading, target_speed),
            projection=projection,
            follower_state=state,
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
        max_events: int = 256,
    ) -> PlannedRouteRunResult:
        state = follower_state or RouteFollowerState()
        steps: list[PlannedRouteGuidance] = []

        for _ in range(max_events):
            if state.completed or world.time >= end_time - _EPS:
                break
            vessel = world.snapshot().by_id(vessel_id)
            decision = self.guidance(vessel, state, world.limits_for(vessel_id))
            steps.append(decision)
            state = decision.follower_state
            world.schedule_command(at=world.time, vessel_id=vessel_id, command=decision.command)

            if state.completed:
                world.run_until(world.time)
                break
            if decision.next_review_after is None:
                raise SpeedPlanExecutionError("off-track/misaligned state requires recovery before speed-plan execution")
            if decision.next_review_after <= _EPS:
                world.run_until(world.time)
                continue
            world.run_until(min(end_time, world.time + decision.next_review_after))
        else:
            raise RuntimeError("speed-planned route follower exceeded max_events")

        return PlannedRouteRunResult(world.snapshot(), state, tuple(steps))
