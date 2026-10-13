"""Continuous 2D vessel state and bounded command following for the next-gen engine.

Coordinate convention
---------------------
- x: east, nautical miles
- y: north, nautical miles
- heading: degrees true, 0=north, 90=east, increasing clockwise
- time: minutes
- speed: NM/min internally (knots / 60)

The integrator is event-friendly rather than fixed-step.  A command is followed with
constant maximum longitudinal acceleration/deceleration and constant maximum turn
rate until each target is reached.  If speed and heading targets are reached at
different times, the interval is split exactly at those transition times.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, pi, sin

from ..kinematics import Vector2, knots_to_nm_per_minute

_EPS = 1e-12


def normalize_heading(heading_deg: float) -> float:
    """Normalize heading to [0, 360)."""
    return heading_deg % 360.0


def signed_heading_delta(current_deg: float, target_deg: float) -> float:
    """Shortest signed target-current angle in [-180, 180). Positive is starboard."""
    current = normalize_heading(current_deg)
    target = normalize_heading(target_deg)
    return ((target - current + 180.0) % 360.0) - 180.0


def heading_unit_vector(heading_deg: float) -> Vector2:
    """Return an east/north unit vector for marine heading degrees true."""
    theta = heading_deg * pi / 180.0
    return (sin(theta), cos(theta))


@dataclass(frozen=True, slots=True)
class Vessel2DState:
    vessel_id: str
    position: Vector2
    heading_deg: float
    speed: float

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if self.speed < 0:
            raise ValueError("speed must be non-negative")
        object.__setattr__(self, "heading_deg", normalize_heading(self.heading_deg))

    @classmethod
    def from_knots(
        cls,
        vessel_id: str,
        position: Vector2,
        heading_deg: float,
        speed_knots: float,
    ) -> "Vessel2DState":
        if speed_knots < 0:
            raise ValueError("speed_knots must be non-negative")
        return cls(
            vessel_id=vessel_id,
            position=position,
            heading_deg=heading_deg,
            speed=knots_to_nm_per_minute(speed_knots),
        )

    @property
    def velocity(self) -> Vector2:
        ux, uy = heading_unit_vector(self.heading_deg)
        return (ux * self.speed, uy * self.speed)


@dataclass(frozen=True, slots=True)
class ManeuverLimits:
    """Symmetric turn-rate limit plus longitudinal speed-change limits."""

    max_acceleration: float
    max_deceleration: float
    max_turn_rate_deg_per_min: float

    def __post_init__(self) -> None:
        if self.max_acceleration < 0:
            raise ValueError("max_acceleration must be non-negative")
        if self.max_deceleration < 0:
            raise ValueError("max_deceleration must be non-negative")
        if self.max_turn_rate_deg_per_min < 0:
            raise ValueError("max_turn_rate_deg_per_min must be non-negative")

    @classmethod
    def from_knots_per_minute(
        cls,
        *,
        max_acceleration_knots_per_min: float,
        max_deceleration_knots_per_min: float,
        max_turn_rate_deg_per_min: float,
    ) -> "ManeuverLimits":
        return cls(
            max_acceleration=knots_to_nm_per_minute(max_acceleration_knots_per_min),
            max_deceleration=knots_to_nm_per_minute(max_deceleration_knots_per_min),
            max_turn_rate_deg_per_min=max_turn_rate_deg_per_min,
        )


@dataclass(frozen=True, slots=True)
class ManeuverCommand:
    target_heading_deg: float
    target_speed: float

    def __post_init__(self) -> None:
        if self.target_speed < 0:
            raise ValueError("target_speed must be non-negative")
        object.__setattr__(self, "target_heading_deg", normalize_heading(self.target_heading_deg))

    @classmethod
    def from_knots(cls, *, target_heading_deg: float, target_speed_knots: float) -> "ManeuverCommand":
        if target_speed_knots < 0:
            raise ValueError("target_speed_knots must be non-negative")
        return cls(
            target_heading_deg=target_heading_deg,
            target_speed=knots_to_nm_per_minute(target_speed_knots),
        )


@dataclass(frozen=True, slots=True)
class AdvanceResult:
    state: Vessel2DState
    duration: float
    segments: int
    distance: float


def command_transition_duration(
    state: Vessel2DState,
    command: ManeuverCommand,
    limits: ManeuverLimits,
) -> float:
    """Return minutes required to reach both command targets.

    Heading and speed changes proceed simultaneously.  ``math.inf`` is returned
    when a requested non-zero heading or speed change cannot be produced by the
    supplied maneuver limits.
    """
    speed_delta = command.target_speed - state.speed
    if abs(speed_delta) <= _EPS:
        speed_time = 0.0
    else:
        speed_rate = limits.max_acceleration if speed_delta > 0 else limits.max_deceleration
        speed_time = abs(speed_delta) / speed_rate if speed_rate > _EPS else float("inf")

    heading_delta = signed_heading_delta(state.heading_deg, command.target_heading_deg)
    if abs(heading_delta) <= _EPS:
        heading_time = 0.0
    else:
        turn_rate = limits.max_turn_rate_deg_per_min
        heading_time = abs(heading_delta) / turn_rate if turn_rate > _EPS else float("inf")

    return max(speed_time, heading_time)


def _speed_control(current: float, target: float, limits: ManeuverLimits) -> tuple[float, float]:
    """Return signed acceleration and time to target; inf represented by a very large value."""
    delta = target - current
    if abs(delta) <= _EPS:
        return (0.0, float("inf"))
    magnitude = limits.max_acceleration if delta > 0 else limits.max_deceleration
    if magnitude <= _EPS:
        return (0.0, float("inf"))
    acceleration = magnitude if delta > 0 else -magnitude
    return (acceleration, abs(delta) / magnitude)


def _heading_control(current: float, target: float, limits: ManeuverLimits) -> tuple[float, float]:
    delta = signed_heading_delta(current, target)
    if abs(delta) <= _EPS:
        return (0.0, float("inf"))
    rate = limits.max_turn_rate_deg_per_min
    if rate <= _EPS:
        return (0.0, float("inf"))
    signed_rate = rate if delta > 0 else -rate
    return (signed_rate, abs(delta) / rate)


def _integrate_segment(
    position: Vector2,
    heading_deg: float,
    speed: float,
    acceleration: float,
    turn_rate_deg_per_min: float,
    duration: float,
) -> tuple[Vector2, float, float, float]:
    """Analytically integrate one constant-acceleration/constant-turn-rate segment."""
    if duration <= 0:
        return position, normalize_heading(heading_deg), speed, 0.0

    theta0 = heading_deg * pi / 180.0
    omega = turn_rate_deg_per_min * pi / 180.0
    final_speed = max(0.0, speed + acceleration * duration)
    distance = speed * duration + 0.5 * acceleration * duration * duration

    if abs(omega) <= _EPS:
        ux, uy = heading_unit_vector(heading_deg)
        new_position = (position[0] + ux * distance, position[1] + uy * distance)
        return new_position, normalize_heading(heading_deg), final_speed, distance

    theta1 = theta0 + omega * duration

    dx = (
        speed * (cos(theta0) - cos(theta1)) / omega
        + acceleration
        * (-duration * cos(theta1) / omega + (sin(theta1) - sin(theta0)) / (omega * omega))
    )
    dy = (
        speed * (sin(theta1) - sin(theta0)) / omega
        + acceleration
        * (duration * sin(theta1) / omega + (cos(theta1) - cos(theta0)) / (omega * omega))
    )

    new_position = (position[0] + dx, position[1] + dy)
    new_heading = normalize_heading(heading_deg + turn_rate_deg_per_min * duration)
    return new_position, new_heading, final_speed, distance


def advance_toward_command(
    state: Vessel2DState,
    command: ManeuverCommand,
    duration: float,
    limits: ManeuverLimits,
) -> AdvanceResult:
    """Advance a vessel exactly over ``duration`` while following a bounded command.

    The interval is split only when the speed target or heading target is reached,
    never into arbitrary numerical time steps.
    """
    if duration < 0:
        raise ValueError("duration must be non-negative")
    if duration == 0:
        return AdvanceResult(state=state, duration=0.0, segments=0, distance=0.0)

    current = state
    remaining = duration
    total_distance = 0.0
    segments = 0

    while remaining > _EPS:
        acceleration, speed_time = _speed_control(current.speed, command.target_speed, limits)
        turn_rate, heading_time = _heading_control(current.heading_deg, command.target_heading_deg, limits)

        transition_times = [remaining]
        if speed_time != float("inf"):
            transition_times.append(speed_time)
        if heading_time != float("inf"):
            transition_times.append(heading_time)
        dt = min(t for t in transition_times if t > _EPS)
        dt = min(dt, remaining)

        position, heading, speed, distance = _integrate_segment(
            current.position,
            current.heading_deg,
            current.speed,
            acceleration,
            turn_rate,
            dt,
        )

        # Snap reached targets to remove accumulated floating-point angle/speed residue.
        if speed_time <= dt + _EPS:
            speed = command.target_speed
        if heading_time <= dt + _EPS:
            heading = command.target_heading_deg

        current = replace(current, position=position, heading_deg=heading, speed=speed)
        total_distance += distance
        remaining -= dt
        segments += 1

        if segments > 8:
            raise RuntimeError("unexpected number of maneuver transition segments")

    return AdvanceResult(state=current, duration=duration, segments=segments, distance=total_distance)
