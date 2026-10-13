"""Generic C2 world objects and navigational-constraint geometry.

The physical A162 vessel kernel remains unchanged.  This module provides immutable,
topic-agnostic truth objects that can coexist with vessels and later feed ObservedWorld,
fault/degradation, ODD and evaluation layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import cos, hypot, isfinite, radians, sin
from typing import Iterable, Mapping, Protocol, TypeAlias

from .scope import TestbedFeature

_EPS = 1e-12


class ConsequenceType(str, Enum):
    HULL_COLLISION = "HULL_COLLISION"
    GROUNDING = "GROUNDING"
    DOMAIN_INTRUSION = "DOMAIN_INTRUSION"
    ENTANGLEMENT_PROPELLER_HAZARD = "ENTANGLEMENT_PROPELLER_HAZARD"
    PROHIBITED_AREA_ENTRY = "PROHIBITED_AREA_ENTRY"


class WorldObjectKind(str, Enum):
    BUOY_MARKER = "BUOY_MARKER"
    FISHING_GEAR = "FISHING_GEAR"
    COASTLINE = "COASTLINE"
    SHALLOW_WATER = "SHALLOW_WATER"
    CHANNEL_BOUNDARY = "CHANNEL_BOUNDARY"
    EXCLUSION_ZONE = "EXCLUSION_ZONE"
    PROTECTED_ZONE = "PROTECTED_ZONE"
    STATIC_OBSTRUCTION = "STATIC_OBSTRUCTION"
    TEMPORARY_OBSTRUCTION = "TEMPORARY_OBSTRUCTION"
    PROTECTED_MOVING_OBJECT = "PROTECTED_MOVING_OBJECT"
    EMERGENT_MOVING_OBJECT = "EMERGENT_MOVING_OBJECT"


class VesselObjectClass(str, Enum):
    POWERED_MERCHANT = "POWERED_MERCHANT"
    SMALL_CRAFT = "SMALL_CRAFT"
    FISHING_CRAFT = "FISHING_CRAFT"
    AUTONOMOUS_POLICY = "AUTONOMOUS_POLICY"
    OTHER = "OTHER"


class AppearanceMode(str, Enum):
    PRESENT_AT_START = "PRESENT_AT_START"
    SCHEDULED = "SCHEDULED"
    EMERGENT_PHYSICAL = "EMERGENT_PHYSICAL"
    INJECTED = "INJECTED"


@dataclass(frozen=True, slots=True)
class Point2D:
    x_east_nm: float
    y_north_nm: float

    def __post_init__(self) -> None:
        if not isfinite(self.x_east_nm) or not isfinite(self.y_north_nm):
            raise ValueError("point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class PointGeometry:
    point: Point2D


@dataclass(frozen=True, slots=True)
class CircleGeometry:
    center: Point2D
    radius_nm: float

    def __post_init__(self) -> None:
        if not isfinite(self.radius_nm) or self.radius_nm < 0:
            raise ValueError("radius_nm must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PolylineGeometry:
    points: tuple[Point2D, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("polyline requires at least two points")


@dataclass(frozen=True, slots=True)
class PolygonGeometry:
    vertices: tuple[Point2D, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("polygon requires at least three vertices")
        if len(set(self.vertices)) < 3:
            raise ValueError("polygon requires at least three distinct vertices")


WorldGeometry: TypeAlias = PointGeometry | CircleGeometry | PolylineGeometry | PolygonGeometry


@dataclass(frozen=True, slots=True)
class ObjectLifecycle:
    active_from_min: float = 0.0
    active_until_min: float | None = None
    appearance_mode: AppearanceMode = AppearanceMode.PRESENT_AT_START

    def __post_init__(self) -> None:
        if not isfinite(self.active_from_min) or self.active_from_min < 0:
            raise ValueError("active_from_min must be finite and non-negative")
        if self.active_until_min is not None:
            if not isfinite(self.active_until_min) or self.active_until_min <= self.active_from_min:
                raise ValueError("active_until_min must be finite and greater than active_from_min")
        if self.appearance_mode is AppearanceMode.PRESENT_AT_START and self.active_from_min > _EPS:
            raise ValueError("PRESENT_AT_START objects must activate at t=0")
        if self.appearance_mode is AppearanceMode.EMERGENT_PHYSICAL and self.active_from_min <= _EPS:
            raise ValueError("EMERGENT_PHYSICAL objects must appear after t=0")

    def active_at(self, time_min: float) -> bool:
        if not isfinite(time_min) or time_min < 0:
            raise ValueError("time_min must be finite and non-negative")
        if time_min + _EPS < self.active_from_min:
            return False
        return self.active_until_min is None or time_min < self.active_until_min - _EPS


@dataclass(frozen=True, slots=True)
class LinearMotion:
    """Constant truth motion; base geometry is defined at lifecycle activation time."""

    course_deg: float
    speed_knots: float

    def __post_init__(self) -> None:
        if not isfinite(self.course_deg) or not 0 <= self.course_deg <= 360:
            raise ValueError("course_deg must be within [0, 360]")
        if not isfinite(self.speed_knots) or self.speed_knots < 0:
            raise ValueError("speed_knots must be finite and non-negative")

    def displacement_nm(self, elapsed_min: float) -> tuple[float, float]:
        if not isfinite(elapsed_min) or elapsed_min < 0:
            raise ValueError("elapsed_min must be finite and non-negative")
        distance_nm = self.speed_knots * elapsed_min / 60.0
        angle = radians(self.course_deg)
        return distance_nm * sin(angle), distance_nm * cos(angle)


@dataclass(frozen=True, slots=True)
class ConsequenceRule:
    consequence: ConsequenceType
    clearance_nm: float = 0.0
    max_safe_draught_m: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.clearance_nm) or self.clearance_nm < 0:
            raise ValueError("clearance_nm must be finite and non-negative")
        if self.max_safe_draught_m is not None and (
            not isfinite(self.max_safe_draught_m) or self.max_safe_draught_m <= 0
        ):
            raise ValueError("max_safe_draught_m must be finite and positive when provided")
        if self.consequence is ConsequenceType.GROUNDING and self.max_safe_draught_m is None:
            raise ValueError("GROUNDING rule requires max_safe_draught_m")
        if self.consequence is not ConsequenceType.GROUNDING and self.max_safe_draught_m is not None:
            raise ValueError("max_safe_draught_m is only valid for GROUNDING")


@dataclass(frozen=True, slots=True)
class CanonicalWorldObject:
    object_id: str
    kind: WorldObjectKind
    geometry: WorldGeometry
    rules: tuple[ConsequenceRule, ...]
    lifecycle: ObjectLifecycle = ObjectLifecycle()
    motion: LinearMotion | None = None
    uncertainty_radius_nm: float = 0.0
    label: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be non-empty")
        if not self.rules:
            raise ValueError("world object requires at least one consequence rule")
        if not isfinite(self.uncertainty_radius_nm) or self.uncertainty_radius_nm < 0:
            raise ValueError("uncertainty_radius_nm must be finite and non-negative")
        if tuple(sorted(self.attributes)) != self.attributes:
            raise ValueError("world-object attributes must be sorted for canonical identity")
        if self.motion is not None and self.kind not in {
            WorldObjectKind.TEMPORARY_OBSTRUCTION,
            WorldObjectKind.PROTECTED_MOVING_OBJECT,
            WorldObjectKind.EMERGENT_MOVING_OBJECT,
        }:
            raise ValueError("motion is only supported for temporary/protected/emergent moving objects")

    def attribute_map(self) -> Mapping[str, str]:
        return dict(self.attributes)

    def active_at(self, time_min: float) -> bool:
        return self.lifecycle.active_at(time_min)

    def geometry_at(self, time_min: float) -> WorldGeometry:
        if not self.active_at(time_min):
            raise ValueError(f"object {self.object_id!r} is not active at t={time_min}")
        if self.motion is None:
            return self.geometry
        dx, dy = self.motion.displacement_nm(time_min - self.lifecycle.active_from_min)
        return _translate_geometry(self.geometry, dx, dy)


@dataclass(frozen=True, slots=True)
class NavigationQuery:
    entity_id: str
    x_east_nm: float
    y_north_nm: float
    hull_radius_nm: float = 0.0
    draught_m: float | None = None

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must be non-empty")
        if not isfinite(self.x_east_nm) or not isfinite(self.y_north_nm):
            raise ValueError("query coordinates must be finite")
        if not isfinite(self.hull_radius_nm) or self.hull_radius_nm < 0:
            raise ValueError("hull_radius_nm must be finite and non-negative")
        if self.draught_m is not None and (not isfinite(self.draught_m) or self.draught_m <= 0):
            raise ValueError("draught_m must be finite and positive when provided")

    @property
    def point(self) -> Point2D:
        return Point2D(self.x_east_nm, self.y_north_nm)


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    entity_id: str
    object_id: str
    object_kind: WorldObjectKind
    consequence: ConsequenceType
    time_min: float
    geometric_clearance_nm: float
    effective_limit_nm: float


class WorldObjectSource(Protocol):
    component_id: TestbedFeature

    def active_objects(self, time_min: float) -> tuple[CanonicalWorldObject, ...]:
        ...


@dataclass(frozen=True, slots=True)
class WorldObjectRegistry:
    objects: tuple[CanonicalWorldObject, ...] = ()
    component_id = TestbedFeature.C2_WORLD_OBJECTS_CONSTRAINTS

    def __post_init__(self) -> None:
        ids = [obj.object_id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate world object_id")

    def active_objects(self, time_min: float) -> tuple[CanonicalWorldObject, ...]:
        return tuple(obj for obj in self.objects if obj.active_at(time_min))

    def evaluate(self, query: NavigationQuery, *, time_min: float) -> tuple[ConstraintViolation, ...]:
        violations: list[ConstraintViolation] = []
        for obj in self.active_objects(time_min):
            geometry = obj.geometry_at(time_min)
            clearance = _distance_to_geometry(query.point, geometry)
            for rule in obj.rules:
                effective_limit = query.hull_radius_nm + obj.uncertainty_radius_nm + rule.clearance_nm
                if clearance > effective_limit + _EPS:
                    continue
                if rule.consequence is ConsequenceType.GROUNDING:
                    if query.draught_m is None or query.draught_m <= rule.max_safe_draught_m + _EPS:
                        continue
                violations.append(
                    ConstraintViolation(
                        entity_id=query.entity_id,
                        object_id=obj.object_id,
                        object_kind=obj.kind,
                        consequence=rule.consequence,
                        time_min=time_min,
                        geometric_clearance_nm=clearance,
                        effective_limit_nm=effective_limit,
                    )
                )
        return tuple(violations)


def _translate_point(point: Point2D, dx: float, dy: float) -> Point2D:
    return Point2D(point.x_east_nm + dx, point.y_north_nm + dy)


def _translate_geometry(geometry: WorldGeometry, dx: float, dy: float) -> WorldGeometry:
    if isinstance(geometry, PointGeometry):
        return PointGeometry(_translate_point(geometry.point, dx, dy))
    if isinstance(geometry, CircleGeometry):
        return CircleGeometry(_translate_point(geometry.center, dx, dy), geometry.radius_nm)
    if isinstance(geometry, PolylineGeometry):
        return PolylineGeometry(tuple(_translate_point(p, dx, dy) for p in geometry.points))
    if isinstance(geometry, PolygonGeometry):
        return PolygonGeometry(tuple(_translate_point(p, dx, dy) for p in geometry.vertices))
    raise TypeError(f"unsupported geometry {type(geometry)!r}")


def _distance_point_segment(point: Point2D, a: Point2D, b: Point2D) -> float:
    vx = b.x_east_nm - a.x_east_nm
    vy = b.y_north_nm - a.y_north_nm
    wx = point.x_east_nm - a.x_east_nm
    wy = point.y_north_nm - a.y_north_nm
    vv = vx * vx + vy * vy
    if vv <= _EPS:
        return hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    px = a.x_east_nm + t * vx
    py = a.y_north_nm + t * vy
    return hypot(point.x_east_nm - px, point.y_north_nm - py)


def _point_in_polygon(point: Point2D, vertices: tuple[Point2D, ...]) -> bool:
    inside = False
    x, y = point.x_east_nm, point.y_north_nm
    j = len(vertices) - 1
    for i, vi in enumerate(vertices):
        vj = vertices[j]
        if _distance_point_segment(point, vj, vi) <= _EPS:
            return True
        crosses = (vi.y_north_nm > y) != (vj.y_north_nm > y)
        if crosses:
            x_intersection = (
                (vj.x_east_nm - vi.x_east_nm)
                * (y - vi.y_north_nm)
                / (vj.y_north_nm - vi.y_north_nm)
                + vi.x_east_nm
            )
            if x < x_intersection:
                inside = not inside
        j = i
    return inside


def _distance_to_geometry(point: Point2D, geometry: WorldGeometry) -> float:
    if isinstance(geometry, PointGeometry):
        return hypot(point.x_east_nm - geometry.point.x_east_nm, point.y_north_nm - geometry.point.y_north_nm)
    if isinstance(geometry, CircleGeometry):
        center_distance = hypot(
            point.x_east_nm - geometry.center.x_east_nm,
            point.y_north_nm - geometry.center.y_north_nm,
        )
        return max(0.0, center_distance - geometry.radius_nm)
    if isinstance(geometry, PolylineGeometry):
        return min(
            _distance_point_segment(point, a, b)
            for a, b in zip(geometry.points, geometry.points[1:], strict=False)
        )
    if isinstance(geometry, PolygonGeometry):
        if _point_in_polygon(point, geometry.vertices):
            return 0.0
        closed = geometry.vertices + (geometry.vertices[0],)
        return min(
            _distance_point_segment(point, a, b)
            for a, b in zip(closed, closed[1:], strict=False)
        )
    raise TypeError(f"unsupported geometry {type(geometry)!r}")


def emergent_moving_object(
    *,
    object_id: str,
    geometry: WorldGeometry,
    appears_at_min: float,
    course_deg: float,
    speed_knots: float,
    rules: Iterable[ConsequenceRule] | None = None,
    uncertainty_radius_nm: float = 0.0,
    label: str | None = None,
    attributes: Mapping[str, str] | None = None,
) -> CanonicalWorldObject:
    """Create a truth-level object that physically appears after simulation start.

    Typical examples include a surfaced submarine, newly exposed floating obstruction,
    or another object that was genuinely absent from the modeled surface scene before
    ``appears_at_min``.  Detection timing remains the responsibility of C4/C5.
    """

    actual_rules = tuple(rules) if rules is not None else (
        ConsequenceRule(ConsequenceType.HULL_COLLISION),
    )
    return CanonicalWorldObject(
        object_id=object_id,
        kind=WorldObjectKind.EMERGENT_MOVING_OBJECT,
        geometry=geometry,
        rules=actual_rules,
        lifecycle=ObjectLifecycle(
            active_from_min=appears_at_min,
            appearance_mode=AppearanceMode.EMERGENT_PHYSICAL,
        ),
        motion=LinearMotion(course_deg=course_deg, speed_knots=speed_knots),
        uncertainty_radius_nm=uncertainty_radius_nm,
        label=label,
        attributes=tuple(sorted((str(k), str(v)) for k, v in (attributes or {}).items())),
    )
