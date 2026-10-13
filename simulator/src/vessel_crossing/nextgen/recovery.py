"""Exact event-oriented cross-track recovery for straight route legs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import acos, cos, pi, sin
from typing import Literal

from .dynamics import ManeuverCommand, ManeuverLimits, Vessel2DState, normalize_heading, signed_heading_delta
from .route import Route2D
from .world import World2D, WorldSnapshot

_EPS = 1e-10
RecoveryStage = Literal["align", "intercept", "merge", "complete"]


class RecoveryUnavailable(RuntimeError):
    """Raised when the exact straight-leg merge construction is not feasible."""


@dataclass(frozen=True, slots=True)
class CrossTrackRecoveryState:
    stage: RecoveryStage
    segment_index: int
    recovery_speed: float
    side_sign: int
    intercept_angle_deg: float
    intercept_heading_deg: float
    merge_start_cross_track: float | None = None
    final_along_track: float | None = None


@dataclass(frozen=True, slots=True)
class CrossTrackRecoveryGuidance:
    command: ManeuverCommand
    recovery_state: CrossTrackRecoveryState
    next_review_after: float | None


@dataclass(frozen=True, slots=True)
class CrossTrackRecoveryResult:
    snapshot: WorldSnapshot
    final_along_track: float
    guidance_steps: tuple[CrossTrackRecoveryGuidance, ...]


@dataclass(frozen=True, slots=True)
class CrossTrackRecovery:
    """Merge a vessel back onto one straight route leg with an exact two-turn plan.

    Recovery intentionally holds the vessel's current non-zero speed. The construction is:
    1. turn to a bounded intercept heading toward the route line;
    2. travel straight to a calculated second-turn start;
    3. turn back to the route-leg heading, ending tangent to and on the route line.

    Because speed is held constant and both turns use the bounded constant angular rate,
    every review occurs at an analytically known physical transition rather than a fixed
    controller time step.
    """

    route: Route2D
    max_intercept_angle_deg: float = 35.0
    cross_track_tolerance: float = 1e-7
    heading_tolerance_deg: float = 1e-7
    feasibility_fraction: float = 0.90

    def __post_init__(self) -> None:
        if not 0 < self.max_intercept_angle_deg < 90:
            raise ValueError("max_intercept_angle_deg must be in (0, 90)")
        if self.cross_track_tolerance < 0:
            raise ValueError("cross_track_tolerance must be non-negative")
        if self.heading_tolerance_deg < 0:
            raise ValueError("heading_tolerance_deg must be non-negative")
        if not 0 < self.feasibility_fraction < 1:
            raise ValueError("feasibility_fraction must be in (0, 1)")

    def _boundary_along(self, segment_index: int, speed: float, limits: ManeuverLimits) -> float:
        waypoint_index = segment_index + 1
        if waypoint_index >= len(self.route.waypoints) - 1:
            return self.route.length
        turn = self.route.turn_geometry(
            waypoint_index,
            speed=speed,
            max_turn_rate_deg_per_min=limits.max_turn_rate_deg_per_min,
        )
        return turn.start_along_track

    def _select_angle(self, cross_track_abs: float, speed: float, limits: ManeuverLimits) -> float:
        if speed <= _EPS:
            raise RecoveryUnavailable("stationary vessel cannot execute geometric recovery")
        if limits.max_turn_rate_deg_per_min <= _EPS:
            raise RecoveryUnavailable("zero turn-rate vessel cannot execute geometric recovery")
        radius = speed / (limits.max_turn_rate_deg_per_min * pi / 180.0)
        max_angle = self.max_intercept_angle_deg * pi / 180.0
        # Reserve some lateral distance for the straight intercept leg so the first and
        # second turns cannot collapse into one numerically ambiguous boundary.
        usable_cross = self.feasibility_fraction * cross_track_abs
        if radius <= _EPS:
            return self.max_intercept_angle_deg
        ratio = min(2.0, usable_cross / radius)
        feasible = pi if ratio >= 2.0 else acos(max(-1.0, min(1.0, 1.0 - ratio)))
        angle = min(max_angle, feasible)
        if angle <= 1e-8:
            raise RecoveryUnavailable("cross-track displacement is too small for a distinct recovery arc")
        return angle * 180.0 / pi

    def begin(
        self,
        vessel: Vessel2DState,
        *,
        min_along_track: float,
        limits: ManeuverLimits,
    ) -> CrossTrackRecoveryGuidance:
        projection = self.route.project(vessel.position, min_along_track=min_along_track)
        segment_index = projection.segment_index
        route_heading = self.route.segment_heading(segment_index)
        boundary = self._boundary_along(segment_index, vessel.speed, limits)
        if projection.along_track > boundary + self.cross_track_tolerance:
            raise RecoveryUnavailable(
                "recovery starts after the straight-leg boundary reserved for the next turn"
            )
        if abs(projection.cross_track) <= self.cross_track_tolerance and abs(
            signed_heading_delta(vessel.heading_deg, route_heading)
        ) <= self.heading_tolerance_deg:
            state = CrossTrackRecoveryState(
                stage="complete",
                segment_index=segment_index,
                recovery_speed=vessel.speed,
                side_sign=0,
                intercept_angle_deg=0.0,
                intercept_heading_deg=route_heading,
                final_along_track=projection.along_track,
            )
            return CrossTrackRecoveryGuidance(
                command=ManeuverCommand(route_heading, vessel.speed),
                recovery_state=state,
                next_review_after=None,
            )

        if abs(projection.cross_track) <= self.cross_track_tolerance:
            # A heading-only error at the exact line cannot be corrected by turning in
            # place without leaving the line. Treat a tiny signed offset in the direction
            # implied by the heading error as the recovery side, then re-plan after turn.
            side_sign = 1 if signed_heading_delta(route_heading, vessel.heading_deg) >= 0 else -1
            cross_for_angle = max(self.cross_track_tolerance * 10.0, 1e-6)
        else:
            side_sign = 1 if projection.cross_track > 0 else -1
            cross_for_angle = abs(projection.cross_track)

        angle = self._select_angle(cross_for_angle, vessel.speed, limits)
        intercept_heading = normalize_heading(route_heading + side_sign * angle)
        heading_error = signed_heading_delta(vessel.heading_deg, intercept_heading)
        review_after = (
            abs(heading_error) / limits.max_turn_rate_deg_per_min
            if abs(heading_error) > self.heading_tolerance_deg
            else 0.0
        )
        state = CrossTrackRecoveryState(
            stage="align",
            segment_index=segment_index,
            recovery_speed=vessel.speed,
            side_sign=side_sign,
            intercept_angle_deg=angle,
            intercept_heading_deg=intercept_heading,
        )
        return CrossTrackRecoveryGuidance(
            command=ManeuverCommand(intercept_heading, vessel.speed),
            recovery_state=state,
            next_review_after=review_after,
        )

    def guidance(
        self,
        vessel: Vessel2DState,
        state: CrossTrackRecoveryState,
        *,
        min_along_track: float,
        limits: ManeuverLimits,
    ) -> CrossTrackRecoveryGuidance:
        route_heading = self.route.segment_heading(state.segment_index)
        projection = self.route.project(vessel.position, min_along_track=min_along_track)

        if state.stage == "complete":
            return CrossTrackRecoveryGuidance(
                command=ManeuverCommand(route_heading, state.recovery_speed),
                recovery_state=state,
                next_review_after=None,
            )

        if state.stage == "align":
            heading_error = signed_heading_delta(vessel.heading_deg, state.intercept_heading_deg)
            if abs(heading_error) > self.heading_tolerance_deg:
                return CrossTrackRecoveryGuidance(
                    command=ManeuverCommand(state.intercept_heading_deg, state.recovery_speed),
                    recovery_state=state,
                    next_review_after=abs(heading_error) / limits.max_turn_rate_deg_per_min,
                )

            # Re-evaluate feasibility after the physical alignment turn because that turn
            # itself changes both cross-track and along-track position.
            x = projection.cross_track
            if abs(x) <= self.cross_track_tolerance:
                # We crossed the line during alignment. Start again from the new side.
                return self.begin(vessel, min_along_track=min_along_track, limits=limits)
            side_now = 1 if x > 0 else -1
            if side_now != state.side_sign:
                return self.begin(vessel, min_along_track=min_along_track, limits=limits)

            beta = state.intercept_angle_deg * pi / 180.0
            radius = state.recovery_speed / (limits.max_turn_rate_deg_per_min * pi / 180.0)
            required_cross_abs = radius * (1.0 - cos(beta))
            if abs(x) <= required_cross_abs + self.cross_track_tolerance:
                # Reduce the intercept angle and realign rather than overshooting the line.
                return self.begin(vessel, min_along_track=min_along_track, limits=limits)

            straight_distance = (abs(x) - required_cross_abs) / sin(beta)
            straight_along = straight_distance * cos(beta)
            arc_along = radius * sin(beta)
            final_along = projection.along_track + straight_along + arc_along
            boundary = self._boundary_along(state.segment_index, state.recovery_speed, limits)
            if final_along > boundary + self.cross_track_tolerance:
                raise RecoveryUnavailable("insufficient straight-leg distance before next turn/route end")

            signed_required = state.side_sign * required_cross_abs
            intercept_state = replace(
                state,
                stage="intercept",
                merge_start_cross_track=signed_required,
                final_along_track=final_along,
            )
            return CrossTrackRecoveryGuidance(
                command=ManeuverCommand(state.intercept_heading_deg, state.recovery_speed),
                recovery_state=intercept_state,
                next_review_after=straight_distance / state.recovery_speed,
            )

        if state.stage == "intercept":
            beta = state.intercept_angle_deg
            merge_state = replace(state, stage="merge")
            return CrossTrackRecoveryGuidance(
                command=ManeuverCommand(route_heading, state.recovery_speed),
                recovery_state=merge_state,
                next_review_after=beta / limits.max_turn_rate_deg_per_min,
            )

        if state.stage == "merge":
            heading_error = signed_heading_delta(vessel.heading_deg, route_heading)
            if abs(heading_error) > self.heading_tolerance_deg:
                return CrossTrackRecoveryGuidance(
                    command=ManeuverCommand(route_heading, state.recovery_speed),
                    recovery_state=state,
                    next_review_after=abs(heading_error) / limits.max_turn_rate_deg_per_min,
                )
            final_projection = self.route.project(vessel.position, min_along_track=min_along_track)
            if abs(final_projection.cross_track) > max(self.cross_track_tolerance * 20.0, 1e-8):
                raise RecoveryUnavailable(
                    f"merge geometry did not close on route line: cross_track={final_projection.cross_track}"
                )
            complete = replace(
                state,
                stage="complete",
                final_along_track=max(state.final_along_track or 0.0, final_projection.along_track),
            )
            return CrossTrackRecoveryGuidance(
                command=ManeuverCommand(route_heading, state.recovery_speed),
                recovery_state=complete,
                next_review_after=None,
            )

        raise RuntimeError(f"unknown recovery stage: {state.stage}")

    def run_world_until_recovered(
        self,
        world: World2D,
        *,
        vessel_id: str,
        min_along_track: float,
        max_events: int = 32,
    ) -> CrossTrackRecoveryResult:
        vessel = world.snapshot().by_id(vessel_id)
        limits = world.limits_for(vessel_id)
        decision = self.begin(vessel, min_along_track=min_along_track, limits=limits)
        steps = [decision]
        state = decision.recovery_state

        for _ in range(max_events):
            world.schedule_command(at=world.time, vessel_id=vessel_id, command=decision.command)
            if state.stage == "complete":
                world.run_until(world.time)
                final = state.final_along_track
                assert final is not None
                return CrossTrackRecoveryResult(world.snapshot(), final, tuple(steps))
            if decision.next_review_after is None:
                raise RuntimeError("non-complete recovery decision must schedule a review")
            world.run_until(world.time + max(0.0, decision.next_review_after))
            vessel = world.snapshot().by_id(vessel_id)
            decision = self.guidance(
                vessel,
                state,
                min_along_track=min_along_track,
                limits=limits,
            )
            steps.append(decision)
            state = decision.recovery_state

        raise RuntimeError("cross-track recovery exceeded max_events")
