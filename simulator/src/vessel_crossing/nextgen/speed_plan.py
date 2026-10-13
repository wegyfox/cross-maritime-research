"""Route-level turn-speed feasibility and braking-envelope planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, pi, sqrt, tan

from .dynamics import ManeuverLimits
from .route import Route2D

_EPS = 1e-12


class SpeedPlanInfeasible(RuntimeError):
    """Raised when a requested route speed cannot be made physically reachable."""


@dataclass(frozen=True, slots=True)
class PlannedTurn:
    waypoint_index: int
    deflection_deg: float
    desired_speed: float
    planned_speed: float
    pair_cap_before: float | None
    pair_cap_after: float | None
    endpoint_cap: float | None
    tangent_distance: float
    start_along_track: float
    end_along_track: float
    braking_start_along_track: float

    @property
    def is_speed_limited(self) -> bool:
        return self.planned_speed < self.desired_speed - 1e-12


@dataclass(frozen=True, slots=True)
class RouteSpeedPlan:
    route: Route2D
    desired_speed: float
    max_deceleration: float
    turns: tuple[PlannedTurn, ...]

    @property
    def entry_speed_ceiling(self) -> float:
        return self.speed_ceiling_at(0.0)

    def turn_for_waypoint(self, waypoint_index: int) -> PlannedTurn:
        for turn in self.turns:
            if turn.waypoint_index == waypoint_index:
                return turn
        raise KeyError(waypoint_index)

    def speed_ceiling_at(self, along_track: float) -> float:
        """Maximum speed consistent with all future planned turn-entry constraints.

        Before each turn, the ceiling is the backward braking envelope
        ``sqrt(v_turn^2 + 2*a*distance)``. Inside a tangent arc it is the planned
        turn speed. Constraints from all future turns are combined by taking the minimum.
        """
        if along_track < -_EPS or along_track > self.route.length + _EPS:
            raise ValueError("along_track is outside route")
        s = min(max(along_track, 0.0), self.route.length)
        ceiling = self.desired_speed
        for turn in self.turns:
            if s < turn.start_along_track - _EPS:
                distance = turn.start_along_track - s
                if self.max_deceleration <= _EPS:
                    candidate = turn.planned_speed
                else:
                    candidate = sqrt(
                        max(0.0, turn.planned_speed * turn.planned_speed + 2.0 * self.max_deceleration * distance)
                    )
                ceiling = min(ceiling, candidate)
            elif s <= turn.end_along_track + _EPS:
                ceiling = min(ceiling, turn.planned_speed)
        return ceiling


@dataclass(frozen=True, slots=True)
class RouteSpeedPlanner:
    """Compute a conservative local speed plan for all route corners.

    Each interior turn has tangent demand ``d_i = q_i * v_i`` with
    ``q_i = tan(|delta_i|/2)/omega``. On a leg shared by adjacent turns, a safe local
    pair cap is

    ``v_pair = L / (q_i + q_j)``.

    Applying that cap to both adjacent turns guarantees their tangent reservations do
    not overlap. This is conservative when one neighbor is already slower for another
    reason, but it is deterministic, local, and never hides an infeasible geometry.
    """

    route: Route2D

    def plan(self, *, desired_speed: float, limits: ManeuverLimits) -> RouteSpeedPlan:
        if desired_speed < 0:
            raise ValueError("desired_speed must be non-negative")
        if limits.max_turn_rate_deg_per_min <= _EPS:
            if len(self.route.waypoints) > 2:
                raise SpeedPlanInfeasible("route has turns but vessel turn rate is zero")
            return RouteSpeedPlan(self.route, desired_speed, limits.max_deceleration, ())

        interior = range(1, len(self.route.waypoints) - 1)
        omega = limits.max_turn_rate_deg_per_min * pi / 180.0
        q: dict[int, float] = {}
        deflection: dict[int, float] = {}
        for i in interior:
            delta = self.route.turn_deflection_deg(i)
            if abs(abs(delta) - 180.0) <= 1e-9:
                raise SpeedPlanInfeasible("180-degree reversal has no finite tangent-speed plan")
            deflection[i] = delta
            q[i] = tan(abs(delta) * pi / 360.0) / omega if abs(delta) > _EPS else 0.0

        pair_cap_before: dict[int, float | None] = {i: None for i in interior}
        pair_cap_after: dict[int, float | None] = {i: None for i in interior}

        # Segment j (between waypoint j and j+1) is shared by turns at j and j+1.
        for j in range(1, len(self.route.waypoints) - 2):
            qi = q[j]
            qj = q[j + 1]
            denom = qi + qj
            cap = inf if denom <= _EPS else self.route.segment_lengths[j] / denom
            pair_cap_after[j] = cap
            pair_cap_before[j + 1] = cap

        planned: list[PlannedTurn] = []
        cumulative = self.route.cumulative_lengths
        lengths = self.route.segment_lengths
        last_turn = len(self.route.waypoints) - 2

        for i in interior:
            caps = [desired_speed]
            endpoint_cap: float | None = None
            if q[i] > _EPS:
                if i == 1:
                    endpoint_cap = lengths[0] / q[i]
                    caps.append(endpoint_cap)
                if i == last_turn:
                    cap_last = lengths[-1] / q[i]
                    endpoint_cap = cap_last if endpoint_cap is None else min(endpoint_cap, cap_last)
                    caps.append(cap_last)
            if pair_cap_before[i] is not None:
                caps.append(pair_cap_before[i])
            if pair_cap_after[i] is not None:
                caps.append(pair_cap_after[i])

            speed = max(0.0, min(caps))
            tangent = q[i] * speed
            start = cumulative[i] - tangent
            end = cumulative[i] + tangent
            if start < -_EPS or end > self.route.length + _EPS:
                raise SpeedPlanInfeasible("planned tangent arc lies outside route")

            if desired_speed > speed + _EPS and limits.max_deceleration > _EPS:
                braking_distance = (desired_speed * desired_speed - speed * speed) / (2.0 * limits.max_deceleration)
            else:
                braking_distance = 0.0
            # A negative braking origin is not a route-geometry failure. It means the
            # vessel must enter the modeled route already below desired cruise speed.
            braking_start = max(0.0, start - braking_distance)

            planned.append(
                PlannedTurn(
                    waypoint_index=i,
                    deflection_deg=deflection[i],
                    desired_speed=desired_speed,
                    planned_speed=speed,
                    pair_cap_before=pair_cap_before[i],
                    pair_cap_after=pair_cap_after[i],
                    endpoint_cap=endpoint_cap,
                    tangent_distance=tangent,
                    start_along_track=max(0.0, start),
                    end_along_track=min(self.route.length, end),
                    braking_start_along_track=braking_start,
                )
            )

        # Final geometric verification uses the actual planned tangent reservations.
        for left, right in zip(planned, planned[1:]):
            shared_segment = left.waypoint_index
            if left.tangent_distance + right.tangent_distance > lengths[shared_segment] + 1e-10:
                raise RuntimeError("internal speed planner error: adjacent tangent arcs overlap")

        return RouteSpeedPlan(
            route=self.route,
            desired_speed=desired_speed,
            max_deceleration=limits.max_deceleration,
            turns=tuple(planned),
        )
