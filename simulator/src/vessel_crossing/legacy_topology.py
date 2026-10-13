"""Recovered four-crossing topology and automatic legacy stop-slot geometry.

The 2020 diploma uses two opposing vertical lanes (A/B) and two opposing
horizontal lanes (C/D).  Their geometry contains four distinct pairwise
crossings.  Each route reaches two of them in sequence: under the simplified
Rule-15 mapping it is stand-on/privileged at the first and give-way at the
second.

The original prototype hard-coded queue-stop coordinates (``AXstop`` etc.)
and explicitly left an in-code TODO to generate them algorithmically.  This
module closes that historical TODO without adding any new navigation rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from .geometry import CrossingGeometry, Route
from .policies.rule15 import CrossingResponsibility, diploma_rule15_responsibilities
from .reference_scenarios import build_diploma_reference_geometry_nm


_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RouteIntersection:
    """One geometric intersection between two finite straight routes."""

    crossing_id: str
    first_route: str
    second_route: str
    position: tuple[float, float]
    first_progress: float
    second_progress: float

    def progress_for(self, route_name: str) -> float:
        if route_name == self.first_route:
            return self.first_progress
        if route_name == self.second_route:
            return self.second_progress
        raise KeyError(route_name)

    @property
    def route_names(self) -> frozenset[str]:
        return frozenset((self.first_route, self.second_route))


@dataclass(frozen=True, slots=True)
class RouteCrossingVisit:
    """One route's ordered visit to one crossing."""

    route_name: str
    crossing_id: str
    other_route: str
    position: tuple[float, float]
    progress: float
    give_way: bool

    @property
    def privileged(self) -> bool:
        return not self.give_way


@dataclass(frozen=True, slots=True)
class StopSlot:
    """One allowed queue-stop position upstream of a give-way crossing."""

    route_name: str
    index_from_crossing: int
    progress: float
    position: tuple[float, float]


def _canonical_crossing_id(first_route: str, second_route: str) -> str:
    return "".join(sorted((first_route, second_route)))


def intersect_routes(first: Route, second: Route) -> RouteIntersection:
    """Return the unique finite-segment intersection of two straight routes.

    Collinear/parallel routes and routes whose infinite lines meet outside one
    of the finite route segments are rejected because they are not crossing
    servers in the recovered diploma topology.
    """

    p = first.start
    r = first.vector
    q = second.start
    s = second.vector

    cross_rs = r[0] * s[1] - r[1] * s[0]
    if abs(cross_rs) <= _EPSILON:
        raise ValueError("routes do not have one unique crossing")

    q_minus_p = (q[0] - p[0], q[1] - p[1])
    t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / cross_rs
    u = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / cross_rs
    if t < -_EPSILON or t > 1.0 + _EPSILON or u < -_EPSILON or u > 1.0 + _EPSILON:
        raise ValueError("route segments do not intersect")

    t = min(1.0, max(0.0, t))
    u = min(1.0, max(0.0, u))
    position = (p[0] + t * r[0], p[1] + t * r[1])
    return RouteIntersection(
        crossing_id=_canonical_crossing_id(first.name, second.name),
        first_route=first.name,
        second_route=second.name,
        position=position,
        first_progress=t * first.length,
        second_progress=u * second.length,
    )


def build_crossing_intersections(
    geometry: CrossingGeometry,
    *,
    crossing_pairs: tuple[tuple[str, str], ...],
) -> tuple[RouteIntersection, ...]:
    """Build a deterministic set of named route intersections."""

    seen: set[frozenset[str]] = set()
    result: list[RouteIntersection] = []
    for first_name, second_name in crossing_pairs:
        pair = frozenset((first_name, second_name))
        if len(pair) != 2:
            raise ValueError("crossing pair requires two distinct route names")
        if pair in seen:
            raise ValueError("duplicate crossing pair")
        seen.add(pair)
        result.append(intersect_routes(geometry.by_name(first_name), geometry.by_name(second_name)))
    return tuple(sorted(result, key=lambda item: item.crossing_id))


def route_crossing_visits(
    geometry: CrossingGeometry,
    *,
    responsibilities: tuple[CrossingResponsibility, ...],
) -> dict[str, tuple[RouteCrossingVisit, ...]]:
    """Return ordered crossing visits and give-way role for each route."""

    intersections = build_crossing_intersections(
        geometry,
        crossing_pairs=tuple((item.first_route, item.second_route) for item in responsibilities),
    )
    rule_by_pair = {item.pair: item for item in responsibilities}

    visits: dict[str, list[RouteCrossingVisit]] = {route.name: [] for route in geometry.routes}
    for intersection in intersections:
        pair = intersection.route_names
        try:
            rule = rule_by_pair[pair]
        except KeyError as exc:
            raise ValueError("missing crossing responsibility") from exc

        for route_name, other_route in (
            (intersection.first_route, intersection.second_route),
            (intersection.second_route, intersection.first_route),
        ):
            visits[route_name].append(
                RouteCrossingVisit(
                    route_name=route_name,
                    crossing_id=intersection.crossing_id,
                    other_route=other_route,
                    position=intersection.position,
                    progress=intersection.progress_for(route_name),
                    give_way=rule.give_way_route == route_name,
                )
            )

    return {
        route_name: tuple(sorted(items, key=lambda item: (item.progress, item.crossing_id)))
        for route_name, items in visits.items()
    }


def build_diploma_route_crossing_visits_nm() -> dict[str, tuple[RouteCrossingVisit, ...]]:
    """Return the source-faithful A/B/C/D two-crossing sequence in NM."""

    return route_crossing_visits(
        build_diploma_reference_geometry_nm(),
        responsibilities=diploma_rule15_responsibilities(),
    )


def generate_stop_slots(
    route: Route,
    *,
    give_way_crossing_progress: float,
    minimum_spacing: float,
) -> tuple[StopSlot, ...]:
    """Generate historical queue slots upstream of a give-way crossing.

    The nearest slot is exactly ``minimum_spacing`` before the crossing center;
    additional slots are stepped upstream by the same spacing.  The route start
    is always included as the final fallback slot, matching the 2020 hard-coded
    arrays when the final remainder is shorter than one spacing interval.
    """

    if not isfinite(give_way_crossing_progress):
        raise ValueError("give_way_crossing_progress must be finite")
    if give_way_crossing_progress <= 0 or give_way_crossing_progress > route.length + _EPSILON:
        raise ValueError("give-way crossing progress must lie inside the route after its start")
    if not isfinite(minimum_spacing) or minimum_spacing <= 0:
        raise ValueError("minimum_spacing must be finite and positive")

    progress_values: list[float] = []
    candidate = give_way_crossing_progress - minimum_spacing
    while candidate > _EPSILON:
        progress_values.append(candidate)
        candidate -= minimum_spacing
    if not progress_values or progress_values[-1] > _EPSILON:
        progress_values.append(0.0)

    ux, uy = route.unit_direction
    return tuple(
        StopSlot(
            route_name=route.name,
            index_from_crossing=index,
            progress=progress,
            position=(route.start[0] + ux * progress, route.start[1] + uy * progress),
        )
        for index, progress in enumerate(progress_values)
    )


def build_diploma_stop_slots_nm(*, minimum_spacing_nm: float = 1.0) -> dict[str, tuple[StopSlot, ...]]:
    """Reconstruct the old ``AXstop/BXstop/CXstop/DXstop`` arrays in NM."""

    geometry = build_diploma_reference_geometry_nm()
    visits = build_diploma_route_crossing_visits_nm()
    result: dict[str, tuple[StopSlot, ...]] = {}
    for route in geometry.routes:
        give_way_visits = tuple(item for item in visits[route.name] if item.give_way)
        if len(give_way_visits) != 1:
            raise ValueError("diploma topology requires exactly one give-way crossing per route")
        result[route.name] = generate_stop_slots(
            route,
            give_way_crossing_progress=give_way_visits[0].progress,
            minimum_spacing=minimum_spacing_nm,
        )
    return result


def minimum_crossing_center_spacing(visits: dict[str, tuple[RouteCrossingVisit, ...]]) -> float:
    """Return the smallest center-to-center separation along any multi-crossing route."""

    distances: list[float] = []
    for route_visits in visits.values():
        for first, second in zip(route_visits, route_visits[1:]):
            distances.append(second.progress - first.progress)
    if not distances:
        raise ValueError("no route contains multiple crossings")
    return min(distances)
