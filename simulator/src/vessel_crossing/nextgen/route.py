"""Polyline route geometry for the next-generation 2D simulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, isfinite, pi, tan

from ..kinematics import Vector2
from .dynamics import normalize_heading, signed_heading_delta

_EPS = 1e-12


def _dot(a: Vector2, b: Vector2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Vector2, b: Vector2) -> Vector2:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Vector2, b: Vector2) -> Vector2:
    return (a[0] + b[0], a[1] + b[1])


def _scale(v: Vector2, factor: float) -> Vector2:
    return (v[0] * factor, v[1] * factor)


def marine_bearing_between(start: Vector2, end: Vector2) -> float:
    """Return true bearing from ``start`` to ``end`` (0=north, clockwise positive)."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if hypot(dx, dy) <= _EPS:
        raise ValueError("bearing is undefined for coincident points")
    return normalize_heading(degrees(atan2(dx, dy)))


@dataclass(frozen=True, slots=True)
class RouteProjection:
    segment_index: int
    point: Vector2
    segment_fraction: float
    along_track: float
    cross_track: float
    distance: float
    heading_deg: float
    distance_to_end: float


@dataclass(frozen=True, slots=True)
class TurnGeometry:
    waypoint_index: int
    incoming_heading_deg: float
    outgoing_heading_deg: float
    deflection_deg: float
    radius: float
    tangent_distance: float
    start_along_track: float
    end_along_track: float

    @property
    def is_starboard(self) -> bool:
        return self.deflection_deg > 0.0

    @property
    def is_port(self) -> bool:
        return self.deflection_deg < 0.0


@dataclass(frozen=True, slots=True)
class Route2D:
    """Ordered polyline route in local east/north nautical-mile coordinates."""

    waypoints: tuple[Vector2, ...]

    def __post_init__(self) -> None:
        points = tuple((float(p[0]), float(p[1])) for p in self.waypoints)
        if len(points) < 2:
            raise ValueError("route requires at least two waypoints")
        for i, (a, b) in enumerate(zip(points, points[1:])):
            if hypot(b[0] - a[0], b[1] - a[1]) <= _EPS:
                raise ValueError(f"zero-length route segment at index {i}")
        object.__setattr__(self, "waypoints", points)

    @property
    def segment_count(self) -> int:
        return len(self.waypoints) - 1

    @property
    def segment_lengths(self) -> tuple[float, ...]:
        return tuple(
            hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(self.waypoints, self.waypoints[1:])
        )

    @property
    def cumulative_lengths(self) -> tuple[float, ...]:
        values = [0.0]
        for length in self.segment_lengths:
            values.append(values[-1] + length)
        return tuple(values)

    @property
    def length(self) -> float:
        return self.cumulative_lengths[-1]

    def segment_heading(self, segment_index: int) -> float:
        if not 0 <= segment_index < self.segment_count:
            raise IndexError(segment_index)
        return marine_bearing_between(self.waypoints[segment_index], self.waypoints[segment_index + 1])

    def segment_index_at(self, along_track: float) -> int:
        if along_track < -_EPS or along_track > self.length + _EPS:
            raise ValueError("along_track is outside route")
        s = min(max(along_track, 0.0), self.length)
        cumulative = self.cumulative_lengths
        if s >= self.length - _EPS:
            return self.segment_count - 1
        # At an interior waypoint choose the outgoing segment. This is useful for
        # monotonic route progress and removes an otherwise ambiguous tie.
        for i in range(self.segment_count):
            if s < cumulative[i + 1] - _EPS:
                return i
        return self.segment_count - 1

    def point_at(self, along_track: float) -> Vector2:
        if along_track < -_EPS or along_track > self.length + _EPS:
            raise ValueError("along_track is outside route")
        s = min(max(along_track, 0.0), self.length)
        i = self.segment_index_at(s)
        cumulative = self.cumulative_lengths
        length = self.segment_lengths[i]
        if s >= self.length - _EPS:
            return self.waypoints[-1]
        fraction = (s - cumulative[i]) / length
        a = self.waypoints[i]
        b = self.waypoints[i + 1]
        return (a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction)

    def heading_at(self, along_track: float) -> float:
        return self.segment_heading(self.segment_index_at(along_track))

    def project(self, point: Vector2, *, min_along_track: float = 0.0) -> RouteProjection:
        """Project a point onto the route with optional monotonic-progress lower bound.

        ``cross_track`` is signed: positive is port/left of route direction, negative
        is starboard/right. If two segments have the same distance (e.g. exactly at a
        waypoint), the farther-along projection wins deterministically.
        """
        if min_along_track < -_EPS or min_along_track > self.length + _EPS:
            raise ValueError("min_along_track is outside route")
        min_s = min(max(min_along_track, 0.0), self.length)
        cumulative = self.cumulative_lengths
        best: tuple[float, float, int, RouteProjection] | None = None

        for i, (a, b, length) in enumerate(zip(self.waypoints, self.waypoints[1:], self.segment_lengths)):
            seg_start = cumulative[i]
            seg_end = cumulative[i + 1]
            if seg_end < min_s - _EPS:
                continue

            d = (b[0] - a[0], b[1] - a[1])
            rel = (point[0] - a[0], point[1] - a[1])
            t = _dot(rel, d) / (length * length)
            lower_t = 0.0
            if min_s > seg_start:
                lower_t = min(1.0, (min_s - seg_start) / length)
            t = min(1.0, max(lower_t, t))
            projected = _add(a, _scale(d, t))
            offset = _sub(point, projected)
            distance = hypot(offset[0], offset[1])
            along = seg_start + t * length
            cross_track = (d[0] * offset[1] - d[1] * offset[0]) / length
            projection = RouteProjection(
                segment_index=i,
                point=projected,
                segment_fraction=t,
                along_track=along,
                cross_track=cross_track,
                distance=distance,
                heading_deg=self.segment_heading(i),
                distance_to_end=self.length - along,
            )
            key = (distance, -along, -i)
            if best is None or key < (best[0], best[1], best[2]):
                best = (key[0], key[1], key[2], projection)

        if best is None:
            # Only possible when min_s is at the exact end; return the final point.
            i = self.segment_count - 1
            projected = self.waypoints[-1]
            offset = _sub(point, projected)
            length = self.segment_lengths[i]
            d = _sub(self.waypoints[-1], self.waypoints[-2])
            cross_track = (d[0] * offset[1] - d[1] * offset[0]) / length
            return RouteProjection(
                segment_index=i,
                point=projected,
                segment_fraction=1.0,
                along_track=self.length,
                cross_track=cross_track,
                distance=hypot(offset[0], offset[1]),
                heading_deg=self.segment_heading(i),
                distance_to_end=0.0,
            )
        return best[3]

    def turn_deflection_deg(self, waypoint_index: int) -> float:
        if not 1 <= waypoint_index < len(self.waypoints) - 1:
            raise IndexError("turn waypoint must be interior")
        incoming = self.segment_heading(waypoint_index - 1)
        outgoing = self.segment_heading(waypoint_index)
        return signed_heading_delta(incoming, outgoing)

    def turn_geometry(
        self,
        waypoint_index: int,
        *,
        speed: float,
        max_turn_rate_deg_per_min: float,
    ) -> TurnGeometry:
        """Return ideal constant-speed tangent-arc geometry for one route corner.

        For turn angle ``delta`` and constant turn radius ``R=v/omega``, tangent
        distance from the polyline corner is ``R*tan(|delta|/2)`` on both legs.
        A 180-degree reversal is singular and deliberately rejected.
        """
        if speed < 0:
            raise ValueError("speed must be non-negative")
        if max_turn_rate_deg_per_min <= 0:
            raise ValueError("max_turn_rate_deg_per_min must be positive")

        incoming = self.segment_heading(waypoint_index - 1)
        outgoing = self.segment_heading(waypoint_index)
        deflection = signed_heading_delta(incoming, outgoing)
        abs_delta_rad = abs(deflection) * pi / 180.0
        omega = max_turn_rate_deg_per_min * pi / 180.0

        if abs(abs(deflection) - 180.0) <= 1e-9:
            raise ValueError("180-degree reversal has no finite single tangent-arc solution")

        radius = 0.0 if speed <= _EPS else speed / omega
        tangent_distance = 0.0 if abs_delta_rad <= _EPS else radius * tan(abs_delta_rad / 2.0)
        if not isfinite(tangent_distance):
            raise ValueError("turn tangent distance is not finite")

        cumulative = self.cumulative_lengths
        waypoint_s = cumulative[waypoint_index]
        incoming_length = self.segment_lengths[waypoint_index - 1]
        outgoing_length = self.segment_lengths[waypoint_index]
        if tangent_distance > incoming_length + _EPS or tangent_distance > outgoing_length + _EPS:
            raise ValueError("route legs are too short for the requested speed/turn-rate tangent arc")

        return TurnGeometry(
            waypoint_index=waypoint_index,
            incoming_heading_deg=incoming,
            outgoing_heading_deg=outgoing,
            deflection_deg=deflection,
            radius=radius,
            tangent_distance=tangent_distance,
            start_along_track=waypoint_s - tangent_distance,
            end_along_track=waypoint_s + tangent_distance,
        )
