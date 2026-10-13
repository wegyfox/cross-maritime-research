"""Explicit water-relative, environmental-drift and ground velocity frames.

This module introduces the frame algebra needed before environmental forcing can be
connected to ``World2D``.  It deliberately does *not* reinterpret the existing
``Vessel2DState.heading_deg`` / ``speed`` fields: through a71 those fields retain their
historical next-gen meaning as ground-referenced kinematics.

Coordinate / unit convention
----------------------------
- vector x: east
- vector y: north
- velocity: nautical miles per minute (NM/min)
- marine course: 0 deg north, 90 deg east, clockwise positive

The physical composition is explicit::

    ground_velocity = through_water_velocity + environmental_drift_velocity

where environmental drift is an Earth-fixed translation velocity (for example a
locally uniform current).  No wind/leeway or hydrodynamic response model is implied by
this algebraic foundation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite

from ..kinematics import Vector2, knots_to_nm_per_minute, nm_per_minute_to_knots
from .dynamics import Vessel2DState, heading_unit_vector, normalize_heading

_EPS = 1e-12


def _validate_vector(name: str, vector: Vector2) -> Vector2:
    if len(vector) != 2:
        raise ValueError(f"{name} must contain exactly two components")
    x = float(vector[0])
    y = float(vector[1])
    if not isfinite(x) or not isfinite(y):
        raise ValueError(f"{name} components must be finite")
    return (x, y)


def velocity_from_course_speed(*, course_deg: float, speed: float) -> Vector2:
    """Convert a marine course and non-negative speed to an east/north vector."""
    if not isfinite(course_deg):
        raise ValueError("course_deg must be finite")
    if not isfinite(speed) or speed < 0.0:
        raise ValueError("speed must be finite and non-negative")
    ux, uy = heading_unit_vector(course_deg)
    return (ux * speed, uy * speed)


def velocity_from_course_speed_knots(*, course_deg: float, speed_knots: float) -> Vector2:
    """Knots convenience wrapper for :func:`velocity_from_course_speed`."""
    if not isfinite(speed_knots) or speed_knots < 0.0:
        raise ValueError("speed_knots must be finite and non-negative")
    return velocity_from_course_speed(
        course_deg=course_deg,
        speed=knots_to_nm_per_minute(speed_knots),
    )


def speed_of_velocity(vector: Vector2) -> float:
    x, y = _validate_vector("vector", vector)
    return hypot(x, y)


def course_of_velocity(vector: Vector2) -> float | None:
    """Return marine course of a velocity vector, or ``None`` at zero speed."""
    x, y = _validate_vector("vector", vector)
    if hypot(x, y) <= _EPS:
        return None
    return normalize_heading(degrees(atan2(x, y)))


def compose_ground_velocity(
    through_water_velocity: Vector2,
    environmental_drift_velocity: Vector2,
) -> Vector2:
    """Compose water-relative vessel motion and Earth-fixed drift."""
    tw = _validate_vector("through_water_velocity", through_water_velocity)
    drift = _validate_vector("environmental_drift_velocity", environmental_drift_velocity)
    return (tw[0] + drift[0], tw[1] + drift[1])


def infer_through_water_velocity(
    ground_velocity: Vector2,
    environmental_drift_velocity: Vector2,
) -> Vector2:
    """Invert the frame composition when ground velocity and drift are known."""
    ground = _validate_vector("ground_velocity", ground_velocity)
    drift = _validate_vector("environmental_drift_velocity", environmental_drift_velocity)
    return (ground[0] - drift[0], ground[1] - drift[1])


@dataclass(frozen=True, slots=True)
class VelocityFrameState:
    """One explicit three-frame velocity snapshot.

    ``through_water_velocity`` and ``environmental_drift_velocity`` are authoritative
    inputs. ``ground_velocity`` is derived exactly from their vector sum.
    """

    through_water_velocity: Vector2
    environmental_drift_velocity: Vector2 = (0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "through_water_velocity",
            _validate_vector("through_water_velocity", self.through_water_velocity),
        )
        object.__setattr__(
            self,
            "environmental_drift_velocity",
            _validate_vector("environmental_drift_velocity", self.environmental_drift_velocity),
        )

    @classmethod
    def from_through_water_course_speed(
        cls,
        *,
        course_deg: float,
        speed: float,
        environmental_drift_velocity: Vector2 = (0.0, 0.0),
    ) -> "VelocityFrameState":
        return cls(
            through_water_velocity=velocity_from_course_speed(
                course_deg=course_deg,
                speed=speed,
            ),
            environmental_drift_velocity=environmental_drift_velocity,
        )

    @classmethod
    def from_through_water_course_speed_knots(
        cls,
        *,
        course_deg: float,
        speed_knots: float,
        environmental_drift_velocity: Vector2 = (0.0, 0.0),
    ) -> "VelocityFrameState":
        return cls(
            through_water_velocity=velocity_from_course_speed_knots(
                course_deg=course_deg,
                speed_knots=speed_knots,
            ),
            environmental_drift_velocity=environmental_drift_velocity,
        )

    @classmethod
    def from_ground_and_drift(
        cls,
        *,
        ground_velocity: Vector2,
        environmental_drift_velocity: Vector2,
    ) -> "VelocityFrameState":
        return cls(
            through_water_velocity=infer_through_water_velocity(
                ground_velocity,
                environmental_drift_velocity,
            ),
            environmental_drift_velocity=environmental_drift_velocity,
        )

    @property
    def ground_velocity(self) -> Vector2:
        return compose_ground_velocity(
            self.through_water_velocity,
            self.environmental_drift_velocity,
        )

    @property
    def through_water_speed(self) -> float:
        return speed_of_velocity(self.through_water_velocity)

    @property
    def through_water_course_deg(self) -> float | None:
        return course_of_velocity(self.through_water_velocity)

    @property
    def ground_speed(self) -> float:
        return speed_of_velocity(self.ground_velocity)

    @property
    def ground_speed_knots(self) -> float:
        return nm_per_minute_to_knots(self.ground_speed)

    @property
    def ground_course_deg(self) -> float | None:
        return course_of_velocity(self.ground_velocity)


def legacy_ground_velocity(state: Vessel2DState) -> Vector2:
    """Return the pre-environment ground velocity without reinterpreting the state.

    This compatibility function makes the a71 migration boundary executable: until a
    later version introduces an environment-aware controlled state, existing
    ``Vessel2DState.velocity`` remains the ground-motion truth used by World2D/CPA.
    """
    if not isinstance(state, Vessel2DState):
        raise TypeError("state must be a Vessel2DState")
    return state.velocity
