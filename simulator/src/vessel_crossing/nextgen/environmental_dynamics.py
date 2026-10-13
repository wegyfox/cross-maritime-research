"""Bounded water-relative vessel dynamics under a uniform Earth-fixed drift.

A72 is an integration bridge, not a World2D migration.  Existing ``Vessel2DState`` and
``ManeuverCommand`` retain their ground-referenced a1-a71 semantics.  This module uses
new types whose heading/speed and command targets are explicitly *through-water*.

For a constant environmental drift vector ``d`` over an interval, motion is composed
exactly as::

    position_ground(t) = position_water_relative(t) + d * t

The water-relative trajectory reuses the validated bounded dynamics integrator, so the
zero-drift case is an exact compatibility identity rather than a second implementation
of turn/acceleration physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..kinematics import Vector2, knots_to_nm_per_minute
from .dynamics import (
    AdvanceResult,
    ManeuverCommand,
    ManeuverLimits,
    Vessel2DState,
    advance_toward_command,
    command_transition_duration,
    heading_unit_vector,
    normalize_heading,
)
from .velocity_frames import VelocityFrameState


def _finite_drift(vector: Vector2) -> Vector2:
    if len(vector) != 2:
        raise ValueError("environmental_drift_velocity must contain exactly two components")
    x, y = float(vector[0]), float(vector[1])
    if not isfinite(x) or not isfinite(y):
        raise ValueError("environmental_drift_velocity components must be finite")
    return (x, y)


@dataclass(frozen=True, slots=True)
class WaterRelativeVesselState:
    vessel_id: str
    position: Vector2
    through_water_heading_deg: float
    through_water_speed: float

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if not isfinite(self.position[0]) or not isfinite(self.position[1]):
            raise ValueError("position components must be finite")
        if not isfinite(self.through_water_heading_deg):
            raise ValueError("through_water_heading_deg must be finite")
        if not isfinite(self.through_water_speed) or self.through_water_speed < 0.0:
            raise ValueError("through_water_speed must be finite and non-negative")
        object.__setattr__(
            self,
            "through_water_heading_deg",
            normalize_heading(self.through_water_heading_deg),
        )

    @classmethod
    def from_knots(
        cls,
        vessel_id: str,
        position: Vector2,
        through_water_heading_deg: float,
        through_water_speed_knots: float,
    ) -> "WaterRelativeVesselState":
        if not isfinite(through_water_speed_knots) or through_water_speed_knots < 0.0:
            raise ValueError("through_water_speed_knots must be finite and non-negative")
        return cls(
            vessel_id=vessel_id,
            position=position,
            through_water_heading_deg=through_water_heading_deg,
            through_water_speed=knots_to_nm_per_minute(through_water_speed_knots),
        )

    @property
    def through_water_velocity(self) -> Vector2:
        ux, uy = heading_unit_vector(self.through_water_heading_deg)
        return (ux * self.through_water_speed, uy * self.through_water_speed)

    def velocity_frames(self, environmental_drift_velocity: Vector2) -> VelocityFrameState:
        return VelocityFrameState(
            through_water_velocity=self.through_water_velocity,
            environmental_drift_velocity=_finite_drift(environmental_drift_velocity),
        )


@dataclass(frozen=True, slots=True)
class WaterRelativeManeuverCommand:
    target_through_water_heading_deg: float
    target_through_water_speed: float

    def __post_init__(self) -> None:
        if not isfinite(self.target_through_water_heading_deg):
            raise ValueError("target_through_water_heading_deg must be finite")
        if not isfinite(self.target_through_water_speed) or self.target_through_water_speed < 0.0:
            raise ValueError("target_through_water_speed must be finite and non-negative")
        object.__setattr__(
            self,
            "target_through_water_heading_deg",
            normalize_heading(self.target_through_water_heading_deg),
        )

    @classmethod
    def from_knots(
        cls,
        *,
        target_through_water_heading_deg: float,
        target_through_water_speed_knots: float,
    ) -> "WaterRelativeManeuverCommand":
        if not isfinite(target_through_water_speed_knots) or target_through_water_speed_knots < 0.0:
            raise ValueError("target_through_water_speed_knots must be finite and non-negative")
        return cls(
            target_through_water_heading_deg=target_through_water_heading_deg,
            target_through_water_speed=knots_to_nm_per_minute(target_through_water_speed_knots),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentalAdvanceResult:
    state: WaterRelativeVesselState
    duration: float
    segments: int
    through_water_distance: float
    environmental_displacement: Vector2
    ground_displacement: Vector2


def _shadow_state(state: WaterRelativeVesselState) -> Vessel2DState:
    return Vessel2DState(
        vessel_id=state.vessel_id,
        position=state.position,
        heading_deg=state.through_water_heading_deg,
        speed=state.through_water_speed,
    )


def _shadow_command(command: WaterRelativeManeuverCommand) -> ManeuverCommand:
    return ManeuverCommand(
        target_heading_deg=command.target_through_water_heading_deg,
        target_speed=command.target_through_water_speed,
    )



def water_relative_command_transition_duration(
    state: WaterRelativeVesselState,
    command: WaterRelativeManeuverCommand,
    limits: ManeuverLimits,
) -> float:
    """Return time required to reach water-relative heading and speed targets.

    This is an exact compatibility wrapper over the validated bounded-dynamics
    transition-duration contract. Environmental drift does not enter turn-rate or
    longitudinal control timing.
    """

    if not isinstance(state, WaterRelativeVesselState):
        raise TypeError("state must be a WaterRelativeVesselState")
    if not isinstance(command, WaterRelativeManeuverCommand):
        raise TypeError("command must be a WaterRelativeManeuverCommand")
    if not isinstance(limits, ManeuverLimits):
        raise TypeError("limits must be ManeuverLimits")
    return command_transition_duration(
        _shadow_state(state),
        _shadow_command(command),
        limits,
    )

def advance_with_uniform_drift(
    state: WaterRelativeVesselState,
    command: WaterRelativeManeuverCommand,
    duration: float,
    limits: ManeuverLimits,
    *,
    environmental_drift_velocity: Vector2 = (0.0, 0.0),
) -> EnvironmentalAdvanceResult:
    """Advance water-relative command following under constant environmental drift.

    The drift is not fed into acceleration or turn-rate control.  It contributes only
    an Earth-fixed translational displacement over the interval.  This is the exact
    model for a spatially/time-uniform current acting as additive velocity; richer
    forcing remains outside a72.
    """
    if not isinstance(state, WaterRelativeVesselState):
        raise TypeError("state must be a WaterRelativeVesselState")
    if not isinstance(command, WaterRelativeManeuverCommand):
        raise TypeError("command must be a WaterRelativeManeuverCommand")
    if not isinstance(limits, ManeuverLimits):
        raise TypeError("limits must be ManeuverLimits")
    drift = _finite_drift(environmental_drift_velocity)

    water_result: AdvanceResult = advance_toward_command(
        _shadow_state(state),
        _shadow_command(command),
        duration,
        limits,
    )
    drift_displacement = (drift[0] * duration, drift[1] * duration)
    final_position = (
        water_result.state.position[0] + drift_displacement[0],
        water_result.state.position[1] + drift_displacement[1],
    )
    ground_displacement = (
        final_position[0] - state.position[0],
        final_position[1] - state.position[1],
    )
    final_state = WaterRelativeVesselState(
        vessel_id=state.vessel_id,
        position=final_position,
        through_water_heading_deg=water_result.state.heading_deg,
        through_water_speed=water_result.state.speed,
    )
    return EnvironmentalAdvanceResult(
        state=final_state,
        duration=water_result.duration,
        segments=water_result.segments,
        through_water_distance=water_result.distance,
        environmental_displacement=drift_displacement,
        ground_displacement=ground_displacement,
    )
