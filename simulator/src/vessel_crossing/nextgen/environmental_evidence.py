"""Neutral ground-motion evidence for environment-aware vessel states.

A74 bridges a71-a73 physical states to analytical evidence without coercing them into
``Vessel2DState``.  That coercion would lose information when environmental drift makes
course over ground differ from the vessel/hull heading.

- CPA uses position + **ground velocity**.
- instantaneous shaped-domain orientation uses **hull heading** (the explicit
  water-relative heading in this no-leeway model).
- no COLREG responsibility, risk activation or command decision is implied.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cpa import ClosestApproach, closest_approach
from ..kinematics import MotionState, Vector2
from .environmental_dynamics import WaterRelativeVesselState
from .ship_domain import PairShipDomainClearance, ShipDomain, evaluate_pair_ship_domain
from .velocity_frames import VelocityFrameState


@dataclass(frozen=True, slots=True)
class EnvironmentalVesselMotionEvidence:
    vessel_id: str
    position: Vector2
    hull_heading_deg: float
    velocity_frames: VelocityFrameState

    @property
    def through_water_velocity(self) -> Vector2:
        return self.velocity_frames.through_water_velocity

    @property
    def environmental_drift_velocity(self) -> Vector2:
        return self.velocity_frames.environmental_drift_velocity

    @property
    def ground_velocity(self) -> Vector2:
        return self.velocity_frames.ground_velocity

    @property
    def course_over_ground_deg(self) -> float | None:
        return self.velocity_frames.ground_course_deg

    @property
    def speed_over_ground(self) -> float:
        return self.velocity_frames.ground_speed

    def to_ground_motion_state(self) -> MotionState:
        return MotionState(position=self.position, velocity=self.ground_velocity)


@dataclass(frozen=True, slots=True)
class EnvironmentalPairMotionEvidence:
    first: EnvironmentalVesselMotionEvidence
    second: EnvironmentalVesselMotionEvidence
    cpa: ClosestApproach
    current_domain_clearance: PairShipDomainClearance | None = None


def environmental_vessel_motion_evidence(
    state: WaterRelativeVesselState,
    environmental_drift_velocity: Vector2,
) -> EnvironmentalVesselMotionEvidence:
    if not isinstance(state, WaterRelativeVesselState):
        raise TypeError("state must be a WaterRelativeVesselState")
    frames = state.velocity_frames(environmental_drift_velocity)
    return EnvironmentalVesselMotionEvidence(
        vessel_id=state.vessel_id,
        position=state.position,
        hull_heading_deg=state.through_water_heading_deg,
        velocity_frames=frames,
    )


def predict_environmental_pair_motion(
    first_state: WaterRelativeVesselState,
    first_drift_velocity: Vector2,
    second_state: WaterRelativeVesselState,
    second_drift_velocity: Vector2,
    *,
    horizon: float | None = None,
    first_domain: ShipDomain | None = None,
    second_domain: ShipDomain | None = None,
) -> EnvironmentalPairMotionEvidence:
    """Build neutral current-position/domain plus constant-ground-velocity CPA evidence.

    Domains, when supplied, are evaluated only at the current snapshot and use hull
    headings. The CPA projection uses ground velocities. Supplying exactly one domain is
    invalid because pair-domain clearance is a two-vessel quantity.
    """
    if (first_domain is None) != (second_domain is None):
        raise ValueError("first_domain and second_domain must be supplied together")
    first = environmental_vessel_motion_evidence(first_state, first_drift_velocity)
    second = environmental_vessel_motion_evidence(second_state, second_drift_velocity)
    cpa = closest_approach(
        first.to_ground_motion_state(),
        second.to_ground_motion_state(),
        horizon=horizon,
    )
    clearance = None
    if first_domain is not None and second_domain is not None:
        relative_position = (
            second.position[0] - first.position[0],
            second.position[1] - first.position[1],
        )
        clearance = evaluate_pair_ship_domain(
            first_heading_deg=first.hull_heading_deg,
            second_heading_deg=second.hull_heading_deg,
            relative_position=relative_position,
            first_domain=first_domain,
            second_domain=second_domain,
        )
    return EnvironmentalPairMotionEvidence(
        first=first,
        second=second,
        cpa=cpa,
        current_domain_clearance=clearance,
    )
