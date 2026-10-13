"""Small deterministic kinematics primitives.

Modern simulation convention
----------------------------
Distance: nautical miles (NM)
Time: minutes
Velocity: nautical miles per minute (NM/min)

This keeps maritime input/output natural: knots are converted with
``knots / 60`` because one knot is one nautical mile per hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

Vector2 = tuple[float, float]


def knots_to_nm_per_minute(knots: float) -> float:
    """Convert knots (NM/hour) to NM/minute."""
    return knots / 60.0


def nm_per_minute_to_knots(speed: float) -> float:
    """Convert NM/minute to knots."""
    return speed * 60.0


def add(a: Vector2, b: Vector2) -> Vector2:
    return (a[0] + b[0], a[1] + b[1])


def subtract(a: Vector2, b: Vector2) -> Vector2:
    return (a[0] - b[0], a[1] - b[1])


def scale(v: Vector2, factor: float) -> Vector2:
    return (v[0] * factor, v[1] * factor)


def dot(a: Vector2, b: Vector2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(v: Vector2) -> float:
    return hypot(v[0], v[1])


@dataclass(frozen=True, slots=True)
class MotionState:
    """Point-mass vessel state for one constant-velocity segment."""

    position: Vector2
    velocity: Vector2

    def position_at(self, minutes: float) -> Vector2:
        if minutes < 0:
            raise ValueError("minutes must be non-negative")
        return add(self.position, scale(self.velocity, minutes))


def knots_per_minute_to_nm_per_minute_squared(value: float) -> float:
    """Convert a speed-change rate in knots/minute to NM/minute²."""
    return value / 60.0


@dataclass(frozen=True, slots=True)
class SpeedLimits:
    """Maximum magnitude of longitudinal speed changes.

    Values are positive magnitudes in NM/minute².
    """

    max_acceleration: float
    max_deceleration: float

    def __post_init__(self) -> None:
        if self.max_acceleration < 0:
            raise ValueError("max_acceleration must be non-negative")
        if self.max_deceleration < 0:
            raise ValueError("max_deceleration must be non-negative")


@dataclass(frozen=True, slots=True)
class SpeedTransition:
    """Integrated result of moving current speed toward a target speed."""

    initial_speed: float
    target_speed: float
    final_speed: float
    duration: float
    acceleration_duration: float
    distance: float


def integrate_speed_target(
    current_speed: float,
    target_speed: float,
    duration: float,
    limits: SpeedLimits,
) -> SpeedTransition:
    """Integrate one longitudinal speed command with bounded acceleration.

    Acceleration/deceleration is constant at the configured limit until the
    target is reached; remaining time, if any, is traveled at target speed.
    Distance is integrated analytically, avoiding a small simulation time step.
    """

    if current_speed < 0:
        raise ValueError("current_speed must be non-negative")
    if target_speed < 0:
        raise ValueError("target_speed must be non-negative")
    if duration < 0:
        raise ValueError("duration must be non-negative")

    if duration == 0 or target_speed == current_speed:
        return SpeedTransition(
            initial_speed=current_speed,
            target_speed=target_speed,
            final_speed=current_speed,
            duration=duration,
            acceleration_duration=0.0,
            distance=current_speed * duration,
        )

    increasing = target_speed > current_speed
    magnitude = limits.max_acceleration if increasing else limits.max_deceleration

    if magnitude == 0:
        return SpeedTransition(
            initial_speed=current_speed,
            target_speed=target_speed,
            final_speed=current_speed,
            duration=duration,
            acceleration_duration=0.0,
            distance=current_speed * duration,
        )

    signed_acceleration = magnitude if increasing else -magnitude
    time_to_target = abs(target_speed - current_speed) / magnitude
    acceleration_duration = min(duration, time_to_target)

    distance_during_change = (
        current_speed * acceleration_duration
        + 0.5 * signed_acceleration * acceleration_duration**2
    )
    final_speed = current_speed + signed_acceleration * acceleration_duration

    # Eliminate tiny floating point overshoot and make exact target-reaching
    # semantics explicit.
    if acceleration_duration == time_to_target:
        final_speed = target_speed

    cruise_duration = duration - acceleration_duration
    distance = distance_during_change + final_speed * cruise_duration

    return SpeedTransition(
        initial_speed=current_speed,
        target_speed=target_speed,
        final_speed=final_speed,
        duration=duration,
        acceleration_duration=acceleration_duration,
        distance=distance,
    )


def time_to_reach_speed_target(
    current_speed: float,
    target_speed: float,
    limits: SpeedLimits,
) -> float:
    """Return minutes required to reach a longitudinal speed target.

    Returns ``math.inf`` when the relevant acceleration/deceleration limit is
    zero and the current speed differs from the target.
    """

    from math import inf

    if current_speed < 0 or target_speed < 0:
        raise ValueError("speeds must be non-negative")
    if target_speed == current_speed:
        return 0.0

    increasing = target_speed > current_speed
    magnitude = limits.max_acceleration if increasing else limits.max_deceleration
    if magnitude == 0:
        return inf
    return abs(target_speed - current_speed) / magnitude


def time_to_cover_distance(
    current_speed: float,
    target_speed: float,
    distance: float,
    limits: SpeedLimits,
) -> float:
    """Return time needed to cover distance while moving toward target speed.

    Returns ``math.inf`` when the commanded motion cannot cover the requested
    distance (for example, a stationary vessel with zero acceleration, or a
    vessel commanded to stop before the requested distance and then remain
    stopped).
    """

    from math import inf, sqrt

    if current_speed < 0 or target_speed < 0:
        raise ValueError("speeds must be non-negative")
    if distance < 0:
        raise ValueError("distance must be non-negative")
    if distance == 0:
        return 0.0

    if target_speed == current_speed:
        return distance / current_speed if current_speed > 0 else inf

    increasing = target_speed > current_speed
    magnitude = limits.max_acceleration if increasing else limits.max_deceleration

    if magnitude == 0:
        return distance / current_speed if current_speed > 0 else inf

    if increasing:
        time_to_target = (target_speed - current_speed) / magnitude
        distance_to_target = (
            current_speed * time_to_target
            + 0.5 * magnitude * time_to_target**2
        )
        if distance <= distance_to_target:
            # 0.5*a*t² + v0*t - d = 0
            discriminant = current_speed**2 + 2.0 * magnitude * distance
            return (-current_speed + sqrt(discriminant)) / magnitude

        if target_speed == 0:
            return inf
        return time_to_target + (distance - distance_to_target) / target_speed

    # Deceleration: d(t) = v0*t - 0.5*a*t², where a is positive magnitude.
    time_to_target = (current_speed - target_speed) / magnitude
    distance_to_target = (
        current_speed * time_to_target
        - 0.5 * magnitude * time_to_target**2
    )

    if distance <= distance_to_target:
        discriminant = current_speed**2 - 2.0 * magnitude * distance
        # Small negative values may occur at the exact stopping-distance limit.
        discriminant = max(0.0, discriminant)
        return (current_speed - sqrt(discriminant)) / magnitude

    if target_speed == 0:
        return inf
    return time_to_target + (distance - distance_to_target) / target_speed
