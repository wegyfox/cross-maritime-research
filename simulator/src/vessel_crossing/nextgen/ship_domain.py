"""Transparent oriented non-circular ship-domain geometry.

The modeled domain is a four-quadrant ellipse around a vessel reference point.  Each
quadrant chooses its longitudinal extent from ahead/astern and transverse extent from
port/starboard.  Pair clearance is evaluated along the current relative-position ray by
summing the two vessels' directional extents.  This is an explicit radial-clearance
model, not a claim of an exact Minkowski sum for arbitrary ellipses.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, pi, sin, cos, sqrt

from ..kinematics import Vector2

_METERS_PER_NM = 1852.0
_EPS = 1e-15


@dataclass(frozen=True, slots=True)
class ShipDomain:
    ahead_nm: float
    astern_nm: float
    port_nm: float
    starboard_nm: float

    def __post_init__(self) -> None:
        for name in ("ahead_nm", "astern_nm", "port_nm", "starboard_nm"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def circular(cls, radius_nm: float) -> "ShipDomain":
        return cls(radius_nm, radius_nm, radius_nm, radius_nm)

    @classmethod
    def from_meters(
        cls,
        *,
        ahead_m: float,
        astern_m: float,
        port_m: float,
        starboard_m: float,
    ) -> "ShipDomain":
        values = (ahead_m, astern_m, port_m, starboard_m)
        if any(not isfinite(v) or v <= 0.0 for v in values):
            raise ValueError("ship-domain meter extents must be finite and positive")
        return cls(*(v / _METERS_PER_NM for v in values))

    def directional_extent(self, heading_deg: float, direction: Vector2) -> float:
        """Return radial boundary distance in one global direction vector."""
        norm = hypot(direction[0], direction[1])
        if norm <= _EPS:
            raise ValueError("direction vector must be non-zero")
        ux, uy = direction[0] / norm, direction[1] / norm
        theta = heading_deg * pi / 180.0
        # Marine heading unit vectors: forward=(sin h, cos h), starboard=(cos h,-sin h).
        forward = (sin(theta), cos(theta))
        starboard = (cos(theta), -sin(theta))
        local_starboard = ux * starboard[0] + uy * starboard[1]
        local_forward = ux * forward[0] + uy * forward[1]
        transverse = self.starboard_nm if local_starboard >= 0.0 else self.port_nm
        longitudinal = self.ahead_nm if local_forward >= 0.0 else self.astern_nm
        denominator = sqrt(
            (local_starboard / transverse) ** 2 + (local_forward / longitudinal) ** 2
        )
        return 1.0 / denominator


@dataclass(frozen=True, slots=True)
class PairShipDomainClearance:
    actual_distance_nm: float
    first_extent_nm: float
    second_extent_nm: float
    required_distance_nm: float
    intrudes: bool


def evaluate_pair_ship_domain(
    *,
    first_heading_deg: float,
    second_heading_deg: float,
    relative_position: Vector2,
    first_domain: ShipDomain,
    second_domain: ShipDomain,
) -> PairShipDomainClearance:
    """Evaluate radial pair-domain clearance for ``relative_position = p2 - p1``."""
    actual = hypot(relative_position[0], relative_position[1])
    if actual <= _EPS:
        # Coincident reference points necessarily intrude any positive modeled domains.
        return PairShipDomainClearance(actual, 0.0, 0.0, 0.0, True)
    first_extent = first_domain.directional_extent(first_heading_deg, relative_position)
    opposite = (-relative_position[0], -relative_position[1])
    second_extent = second_domain.directional_extent(second_heading_deg, opposite)
    required = first_extent + second_extent
    return PairShipDomainClearance(
        actual_distance_nm=actual,
        first_extent_nm=first_extent,
        second_extent_nm=second_extent,
        required_distance_nm=required,
        intrudes=actual <= required + 1e-12,
    )
