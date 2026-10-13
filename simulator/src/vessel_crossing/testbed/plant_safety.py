"""Continuous oriented-hull safety certification for richer C16 plant trajectories.

The legacy/nextgen safety metric is tied to A162 bounded commands.  MANEUVERING_REALISM
needs the same conservative domain certificate to consume arbitrary deterministic plant
states, including body-frame sway and delayed yaw.  This module provides that bridge
without feeding safety outcomes back into the plant or controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from typing import Callable

from vessel_crossing.nextgen.domain_clearance import DomainClearanceCertification, certify_pair_domain_clearance
from vessel_crossing.nextgen.safety_metrics import modeled_hull_domain
from vessel_crossing.nextgen.ship_domain import ShipDomain
from vessel_crossing.nextgen.vessel_classes import VesselDimensions

from .plant_fidelity import PlantDynamicState


@dataclass(frozen=True, slots=True)
class PlantTrajectoryBounds:
    """Conservative physical bounds over one certification interval."""

    speed_bound_nm_per_min: float
    turn_rate_bound_deg_per_min: float

    def __post_init__(self) -> None:
        if not isfinite(self.speed_bound_nm_per_min) or self.speed_bound_nm_per_min < 0.0:
            raise ValueError("speed_bound_nm_per_min must be finite and non-negative")
        if not isfinite(self.turn_rate_bound_deg_per_min) or self.turn_rate_bound_deg_per_min < 0.0:
            raise ValueError("turn_rate_bound_deg_per_min must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PlantSweptSafetyConfig:
    time_tolerance_min: float = 1e-4
    margin_tolerance_nm: float = 1e-5
    max_subdivisions: int = 20

    def __post_init__(self) -> None:
        if not isfinite(self.time_tolerance_min) or self.time_tolerance_min <= 0.0:
            raise ValueError("time_tolerance_min must be finite and positive")
        if not isfinite(self.margin_tolerance_nm) or self.margin_tolerance_nm < 0.0:
            raise ValueError("margin_tolerance_nm must be finite and non-negative")
        if self.max_subdivisions < 1:
            raise ValueError("max_subdivisions must be at least 1")


@dataclass(frozen=True, slots=True)
class HullReferencePoints:
    """Reference-point approximation of the oriented hull at one instant.

    The current modeled hull is a symmetric ellipse.  These principal points are
    audit/visualization evidence; collision certification still uses the complete
    oriented ellipse through ``modeled_hull_domain``.
    """

    center: tuple[float, float]
    bow: tuple[float, float]
    stern: tuple[float, float]
    port_midship: tuple[float, float]
    starboard_midship: tuple[float, float]


@dataclass(frozen=True, slots=True)
class RigidBodyHullKinematics:
    """Instantaneous velocities of principal hull points for audit/evidence.

    C16 uses +sway to port and positive yaw to starboard (clockwise).  With those
    conventions a point at forward offset ``x`` has local transverse velocity
    ``v - r*x``.  The instantaneous pivot along the centreline is therefore
    ``x_pivot = v/r`` when yaw rate is non-zero.
    """

    center_velocity_nm_per_min: tuple[float, float]
    bow_velocity_nm_per_min: tuple[float, float]
    stern_velocity_nm_per_min: tuple[float, float]
    pivot_longitudinal_offset_nm: float | None


@dataclass(frozen=True, slots=True)
class PlantSweptPairSafety:
    hull_clearance: DomainClearanceCertification
    safety_domain_clearance: DomainClearanceCertification | None

    @property
    def hull_certified_clear(self) -> bool:
        return self.hull_clearance.certified_clear

    @property
    def safety_domain_certified_clear(self) -> bool | None:
        if self.safety_domain_clearance is None:
            return None
        return self.safety_domain_clearance.certified_clear


def hull_reference_points(state: PlantDynamicState, dimensions: VesselDimensions) -> HullReferencePoints:
    center = state.vessel.position
    theta = state.vessel.heading_deg * pi / 180.0
    forward = (sin(theta), cos(theta))
    starboard = (cos(theta), -sin(theta))
    half_length = 0.5 * dimensions.length_nm
    half_beam = 0.5 * dimensions.beam_nm
    return HullReferencePoints(
        center=center,
        bow=(center[0] + forward[0] * half_length, center[1] + forward[1] * half_length),
        stern=(center[0] - forward[0] * half_length, center[1] - forward[1] * half_length),
        port_midship=(center[0] - starboard[0] * half_beam, center[1] - starboard[1] * half_beam),
        starboard_midship=(center[0] + starboard[0] * half_beam, center[1] + starboard[1] * half_beam),
    )


def rigid_body_hull_kinematics(
    state: PlantDynamicState,
    dimensions: VesselDimensions,
) -> RigidBodyHullKinematics:
    """Return centre/bow/stern velocities implied by surge/sway/yaw state."""

    if state.has_body_velocity:
        u = state.surge_velocity_nm_per_min
        v = state.sway_velocity_nm_per_min
        assert u is not None and v is not None
    else:
        u = state.vessel.speed
        v = 0.0
    r = state.yaw_rate_deg_per_min * pi / 180.0  # clockwise rad/min
    half_length = 0.5 * dimensions.length_nm

    def earth_velocity_at(longitudinal_offset_nm: float) -> tuple[float, float]:
        point_u = u
        point_v = v - r * longitudinal_offset_nm
        theta = state.vessel.heading_deg * pi / 180.0
        return (
            point_u * sin(theta) - point_v * cos(theta),
            point_u * cos(theta) + point_v * sin(theta),
        )

    pivot = None if abs(r) <= 1e-12 else v / r
    return RigidBodyHullKinematics(
        center_velocity_nm_per_min=state.earth_velocity_nm_per_min,
        bow_velocity_nm_per_min=earth_velocity_at(half_length),
        stern_velocity_nm_per_min=earth_velocity_at(-half_length),
        pivot_longitudinal_offset_nm=pivot,
    )


def _certify(
    *,
    first_state_at: Callable[[float], PlantDynamicState],
    second_state_at: Callable[[float], PlantDynamicState],
    first_domain: ShipDomain,
    second_domain: ShipDomain,
    duration_min: float,
    first_bounds: PlantTrajectoryBounds,
    second_bounds: PlantTrajectoryBounds,
    config: PlantSweptSafetyConfig,
) -> DomainClearanceCertification:
    return certify_pair_domain_clearance(
        first_state_at=lambda t: first_state_at(t).vessel,
        second_state_at=lambda t: second_state_at(t).vessel,
        first_domain=first_domain,
        second_domain=second_domain,
        duration_min=duration_min,
        first_speed_bound_nm_per_min=first_bounds.speed_bound_nm_per_min,
        second_speed_bound_nm_per_min=second_bounds.speed_bound_nm_per_min,
        first_turn_rate_bound_deg_per_min=first_bounds.turn_rate_bound_deg_per_min,
        second_turn_rate_bound_deg_per_min=second_bounds.turn_rate_bound_deg_per_min,
        time_tolerance_min=config.time_tolerance_min,
        margin_tolerance_nm=config.margin_tolerance_nm,
        max_subdivisions=config.max_subdivisions,
    )


def certify_pair_plant_swept_clearance(
    *,
    first_state_at: Callable[[float], PlantDynamicState],
    second_state_at: Callable[[float], PlantDynamicState],
    first_dimensions: VesselDimensions,
    second_dimensions: VesselDimensions,
    duration_min: float,
    first_bounds: PlantTrajectoryBounds,
    second_bounds: PlantTrajectoryBounds,
    first_safety_domain: ShipDomain | None = None,
    second_safety_domain: ShipDomain | None = None,
    config: PlantSweptSafetyConfig | None = None,
) -> PlantSweptPairSafety:
    """Certify simultaneous swept oriented hull/domain clearance over an interval.

    A richer plant may translate sideways while its hull heading rotates.  The supplied
    state callbacks therefore define the actual reference-point and hull-orientation
    history; the certificate never assumes COG equals hull heading.  Bow/stern are not
    collapsed to centre CPA: the full oriented hull ellipse is evaluated at each
    adaptively certified time state.
    """

    if not isfinite(duration_min) or duration_min < 0.0:
        raise ValueError("duration_min must be finite and non-negative")
    cfg = config or PlantSweptSafetyConfig()
    hull = _certify(
        first_state_at=first_state_at,
        second_state_at=second_state_at,
        first_domain=modeled_hull_domain(first_dimensions),
        second_domain=modeled_hull_domain(second_dimensions),
        duration_min=duration_min,
        first_bounds=first_bounds,
        second_bounds=second_bounds,
        config=cfg,
    )
    if (first_safety_domain is None) != (second_safety_domain is None):
        raise ValueError("safety domains must be supplied for both vessels or neither")
    safety = None
    if first_safety_domain is not None and second_safety_domain is not None:
        safety = _certify(
            first_state_at=first_state_at,
            second_state_at=second_state_at,
            first_domain=first_safety_domain,
            second_domain=second_safety_domain,
            duration_min=duration_min,
            first_bounds=first_bounds,
            second_bounds=second_bounds,
            config=cfg,
        )
    return PlantSweptPairSafety(hull_clearance=hull, safety_domain_clearance=safety)
