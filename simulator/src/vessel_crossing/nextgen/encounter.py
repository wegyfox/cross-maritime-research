"""Encounter prediction bridge for the next-generation 2D vessel state."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

from ..cpa import ClosestApproach, closest_approach
from ..kinematics import MotionState
from .dynamics import Vessel2DState, normalize_heading


def absolute_bearing_deg(origin: tuple[float, float], target: tuple[float, float]) -> float:
    """Marine bearing from origin to target: 0=north, 90=east."""
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    return normalize_heading(degrees(atan2(dx, dy)))


def relative_bearing_deg(observer: Vessel2DState, target: Vessel2DState) -> float:
    """Relative bearing in [0, 360), clockwise from observer's bow."""
    return normalize_heading(absolute_bearing_deg(observer.position, target.position) - observer.heading_deg)


@dataclass(frozen=True, slots=True)
class EncounterPrediction:
    first_id: str
    second_id: str
    first_relative_bearing_deg: float
    second_relative_bearing_deg: float
    cpa: ClosestApproach


def predict_encounter(
    first: Vessel2DState,
    second: Vessel2DState,
    *,
    horizon: float | None = None,
) -> EncounterPrediction:
    """Predict constant-current-velocity CPA/TCPA from arbitrary 2D headings."""
    first_motion = MotionState(position=first.position, velocity=first.velocity)
    second_motion = MotionState(position=second.position, velocity=second.velocity)
    return EncounterPrediction(
        first_id=first.vessel_id,
        second_id=second.vessel_id,
        first_relative_bearing_deg=relative_bearing_deg(first, second),
        second_relative_bearing_deg=relative_bearing_deg(second, first),
        cpa=closest_approach(first_motion, second_motion, horizon=horizon),
    )
