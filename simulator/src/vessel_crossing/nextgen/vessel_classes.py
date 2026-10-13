"""Explicit immutable vessel-class profiles for heterogeneous next-gen traffic.

A class profile is scenario/model metadata, not an empirical claim about any named ship.
It groups physical dimensions and the maneuver envelope already supported by ``World2D``.
Safety-domain geometry and an optional class-level speed envelope remain explicit model
metadata rather than being inferred from length/beam.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ..kinematics import knots_to_nm_per_minute, nm_per_minute_to_knots
from .dynamics import ManeuverLimits
from .ship_domain import ShipDomain

_METERS_PER_NM = 1852.0


@dataclass(frozen=True, slots=True)
class VesselDimensions:
    """Horizontal principal dimensions in canonical nautical-mile units."""

    length_nm: float
    beam_nm: float

    def __post_init__(self) -> None:
        if not isfinite(self.length_nm) or self.length_nm <= 0.0:
            raise ValueError("length_nm must be finite and positive")
        if not isfinite(self.beam_nm) or self.beam_nm <= 0.0:
            raise ValueError("beam_nm must be finite and positive")
        if self.beam_nm > self.length_nm:
            raise ValueError("beam_nm must not exceed length_nm")

    @classmethod
    def from_meters(cls, *, length_m: float, beam_m: float) -> "VesselDimensions":
        if not isfinite(length_m) or length_m <= 0.0:
            raise ValueError("length_m must be finite and positive")
        if not isfinite(beam_m) or beam_m <= 0.0:
            raise ValueError("beam_m must be finite and positive")
        return cls(length_nm=length_m / _METERS_PER_NM, beam_nm=beam_m / _METERS_PER_NM)

    @property
    def length_m(self) -> float:
        return self.length_nm * _METERS_PER_NM

    @property
    def beam_m(self) -> float:
        return self.beam_nm * _METERS_PER_NM




@dataclass(frozen=True, slots=True)
class VesselSpeedEnvelope:
    """Explicit class-level maximum commanded/nominal speed in canonical units.

    This is scenario/model metadata, not an empirical claim about a named vessel.
    Acceleration/deceleration rates still live in ``ManeuverLimits``; this envelope
    bounds the target speed that those rates may approach.
    """

    maximum_speed: float

    def __post_init__(self) -> None:
        if not isfinite(self.maximum_speed) or self.maximum_speed <= 0.0:
            raise ValueError("maximum_speed must be finite and positive")

    @classmethod
    def from_knots(cls, *, maximum_speed_knots: float) -> "VesselSpeedEnvelope":
        if not isfinite(maximum_speed_knots) or maximum_speed_knots <= 0.0:
            raise ValueError("maximum_speed_knots must be finite and positive")
        return cls(knots_to_nm_per_minute(maximum_speed_knots))

    @property
    def maximum_speed_knots(self) -> float:
        return nm_per_minute_to_knots(self.maximum_speed)

    def permits(self, speed: float, *, tolerance: float = 1e-10) -> bool:
        return speed <= self.maximum_speed + tolerance


@dataclass(frozen=True, slots=True)
class VesselClassProfile:
    """Reusable vessel class with dimensions and bounded maneuvering envelope."""

    class_id: str
    dimensions: VesselDimensions
    maneuver_limits: ManeuverLimits
    safety_domain: ShipDomain | None = None
    speed_envelope: VesselSpeedEnvelope | None = None

    def __post_init__(self) -> None:
        if not self.class_id or not self.class_id.strip():
            raise ValueError("class_id must be non-empty")
        object.__setattr__(self, "class_id", self.class_id.strip())


class VesselProfileCatalog:
    """Mutable runtime assignment of active vessel IDs to immutable class profiles."""

    def __init__(self, assignments: dict[str, VesselClassProfile] | None = None) -> None:
        self._profiles: dict[str, VesselClassProfile] = {}
        for vessel_id, profile in (assignments or {}).items():
            self.assign(vessel_id, profile)

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def profile_for(self, vessel_id: str) -> VesselClassProfile | None:
        return self._profiles.get(vessel_id)

    def domain_for(self, vessel_id: str) -> ShipDomain | None:
        profile = self.profile_for(vessel_id)
        return None if profile is None else profile.safety_domain

    def assign(self, vessel_id: str, profile: VesselClassProfile) -> None:
        if not vessel_id:
            raise ValueError("vessel_id must be non-empty")
        existing = self._profiles.get(vessel_id)
        if existing is not None and existing != profile:
            raise ValueError(f"vessel profile conflict for {vessel_id}")
        self._profiles[vessel_id] = profile

    def remove(self, vessel_id: str) -> VesselClassProfile | None:
        return self._profiles.pop(vessel_id, None)
