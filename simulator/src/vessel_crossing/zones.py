"""Geometric measurement/control zones for the crossing."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Route


@dataclass(frozen=True, slots=True)
class RouteZoneInterval:
    """Progress interval for which a straight route lies inside one zone."""

    entry_progress: float
    exit_progress: float

    def __post_init__(self) -> None:
        if self.entry_progress < 0:
            raise ValueError("entry_progress must be non-negative")
        if self.exit_progress < self.entry_progress:
            raise ValueError("exit_progress cannot precede entry_progress")


@dataclass(frozen=True, slots=True)
class AxisAlignedConflictZone:
    """Axis-aligned rectangular crossing measurement zone in NM coordinates."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    _EPSILON = 1e-12

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must exceed x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must exceed y_min")

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        return (
            self.x_min - self._EPSILON <= x <= self.x_max + self._EPSILON
            and self.y_min - self._EPSILON <= y <= self.y_max + self._EPSILON
        )

    def route_interval(self, route: Route) -> RouteZoneInterval | None:
        """Clip a straight route segment to this rectangle.

        Liang-Barsky clipping returns the parameter interval ``t in [0, 1]``
        lying inside the rectangle. Progress is then ``t * route.length``.
        """

        x0, y0 = route.start
        dx, dy = route.vector
        lower = 0.0
        upper = 1.0

        constraints = (
            (-dx, x0 - self.x_min),
            (dx, self.x_max - x0),
            (-dy, y0 - self.y_min),
            (dy, self.y_max - y0),
        )

        for p, q in constraints:
            if abs(p) <= self._EPSILON:
                if q < -self._EPSILON:
                    return None
                continue

            ratio = q / p
            if p < 0:
                lower = max(lower, ratio)
            else:
                upper = min(upper, ratio)
            if lower - upper > self._EPSILON:
                return None

        lower = min(max(lower, 0.0), 1.0)
        upper = min(max(upper, 0.0), 1.0)
        if lower - upper > self._EPSILON:
            return None
        return RouteZoneInterval(lower * route.length, upper * route.length)


@dataclass(frozen=True, slots=True)
class ConflictZonePassage:
    """Observed entry/exit timestamps for one vessel and one crossing zone."""

    vessel_id: str
    entered_at: float | None = None
    exited_at: float | None = None

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if self.entered_at is not None and self.entered_at < 0:
            raise ValueError("entered_at must be non-negative")
        if self.exited_at is not None:
            if self.entered_at is None:
                raise ValueError("exited_at requires entered_at")
            if self.exited_at < self.entered_at:
                raise ValueError("exited_at cannot precede entered_at")

    @property
    def occupancy_time(self) -> float | None:
        if self.entered_at is None or self.exited_at is None:
            return None
        return self.exited_at - self.entered_at


def build_reference_conflict_zone(*, half_extent: float = 1.0) -> AxisAlignedConflictZone:
    """Return a square zone centered on the 5 NM x 5 NM reference crossing."""

    if half_extent <= 0:
        raise ValueError("half_extent must be positive")
    return AxisAlignedConflictZone(
        5.0 - half_extent,
        5.0 + half_extent,
        5.0 - half_extent,
        5.0 + half_extent,
    )
