"""Geometry primitives for crossing-fairway scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True, slots=True)
class Route:
    """Straight route through the simulated area."""

    name: str
    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def vector(self) -> tuple[float, float]:
        return (self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def length(self) -> float:
        dx, dy = self.vector
        return hypot(dx, dy)

    @property
    def unit_direction(self) -> tuple[float, float]:
        dx, dy = self.vector
        length = self.length
        if length == 0:
            raise ValueError(f"route {self.name!r} has zero length")
        return (dx / length, dy / length)

    def velocity(self, speed: float) -> tuple[float, float]:
        """Return a velocity vector along the route at scalar ``speed``."""
        if speed < 0:
            raise ValueError("speed must be non-negative")
        ux, uy = self.unit_direction
        return (ux * speed, uy * speed)


@dataclass(frozen=True, slots=True)
class CrossingGeometry:
    """Named routes forming one simulation scenario."""

    routes: tuple[Route, ...]

    def by_name(self, name: str) -> Route:
        for route in self.routes:
            if route.name == name:
                return route
        raise KeyError(name)


def build_legacy_crossing_geometry() -> CrossingGeometry:
    """Reproduce the geometric intent of the diploma's final prototype.

    Coordinates are in cables.  The four 100-cable routes form two
    perpendicular two-way fairways with 10-cable separation between the
    opposing lanes:

    * A: north -> south, x = 45
    * B: south -> north, x = 55
    * C: west -> east,   y = 45
    * D: east -> west,   y = 55
    """

    return CrossingGeometry(
        routes=(
            Route("A", (45.0, 100.0), (45.0, 0.0)),
            Route("B", (55.0, 0.0), (55.0, 100.0)),
            Route("C", (0.0, 45.0), (100.0, 45.0)),
            Route("D", (100.0, 55.0), (0.0, 55.0)),
        )
    )
