"""Explicit directed route-network geometry and deterministic path resolution.

This module is deliberately above ``Route2D`` and below mission control.  It may
resolve a network destination into route geometry, but it does not mutate ``World2D``,
select avoidance commands, infer VTS priority, or invent navigator parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import heapq
from math import hypot, isfinite
from types import MappingProxyType
from typing import Callable, Iterable

from ..kinematics import Vector2
from .follower import RouteFollowerState
from .navigator import RouteNavigator
from .nominal_fleet_scheduler import NominalFleetTarget
from .route import Route2D
from .route_exit import RouteEndAction

_EPS = 1e-9


class RouteNetworkError(RuntimeError):
    """Base class for fail-closed route-network errors."""


class InvalidRouteNetwork(RouteNetworkError):
    """Raised when graph identity or geometry is inconsistent."""


class RoutePathUnavailable(RouteNetworkError):
    """Raised when a requested directed path cannot be resolved."""


class RouteSelectionPolicy(str, Enum):
    """Named, inspectable route-selection policies supported by this slice."""

    SHORTEST_DISTANCE = "shortest_distance"


@dataclass(frozen=True, slots=True)
class RouteNetworkNode:
    node_id: str
    position: Vector2

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        x, y = float(self.position[0]), float(self.position[1])
        if not (isfinite(x) and isfinite(y)):
            raise ValueError("route-network node coordinates must be finite")
        object.__setattr__(self, "position", (x, y))


@dataclass(frozen=True, slots=True)
class RouteNetworkEdge:
    edge_id: str
    from_node_id: str
    to_node_id: str
    route: Route2D

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise ValueError("edge_id must be non-empty")
        if not self.from_node_id or not self.to_node_id:
            raise ValueError("route-network edge endpoint IDs must be non-empty")
        if self.from_node_id == self.to_node_id:
            raise ValueError("route-network self-loop edges are outside the a63 scope")

    @property
    def distance_nm(self) -> float:
        return self.route.length


@dataclass(frozen=True, slots=True)
class RouteNetworkPath:
    origin_node_id: str
    destination_node_id: str
    edge_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    route: Route2D
    total_distance_nm: float
    selection_policy: RouteSelectionPolicy | None


@dataclass(frozen=True, slots=True)
class RouteRequest:
    origin_node_id: str
    destination_node_id: str
    policy: RouteSelectionPolicy = RouteSelectionPolicy.SHORTEST_DISTANCE

    def __post_init__(self) -> None:
        if not self.origin_node_id or not self.destination_node_id:
            raise ValueError("route request node IDs must be non-empty")
        if self.origin_node_id == self.destination_node_id:
            raise ValueError("origin and destination must differ in the a63 routing slice")


class RouteNetwork:
    """Immutable directed graph whose edges already carry validated ``Route2D`` geometry."""

    __slots__ = ("nodes", "edges", "_nodes_by_id", "_edges_by_id", "_outgoing")

    def __init__(
        self,
        nodes: Iterable[RouteNetworkNode],
        edges: Iterable[RouteNetworkEdge],
    ) -> None:
        node_items = tuple(nodes)
        edge_items = tuple(edges)
        if not node_items:
            raise InvalidRouteNetwork("route network requires at least one node")

        node_ids = [node.node_id for node in node_items]
        if len(set(node_ids)) != len(node_ids):
            raise InvalidRouteNetwork("duplicate route-network node_id")
        edge_ids = [edge.edge_id for edge in edge_items]
        if len(set(edge_ids)) != len(edge_ids):
            raise InvalidRouteNetwork("duplicate route-network edge_id")

        nodes_by_id = {node.node_id: node for node in node_items}
        edges_by_id = {edge.edge_id: edge for edge in edge_items}
        outgoing: dict[str, list[RouteNetworkEdge]] = {node_id: [] for node_id in node_ids}

        for edge in edge_items:
            try:
                start = nodes_by_id[edge.from_node_id].position
                end = nodes_by_id[edge.to_node_id].position
            except KeyError as exc:
                raise InvalidRouteNetwork(
                    f"edge {edge.edge_id} references an unknown endpoint node"
                ) from exc
            if hypot(edge.route.waypoints[0][0] - start[0], edge.route.waypoints[0][1] - start[1]) > _EPS:
                raise InvalidRouteNetwork(
                    f"edge {edge.edge_id} route start does not match from-node position"
                )
            if hypot(edge.route.waypoints[-1][0] - end[0], edge.route.waypoints[-1][1] - end[1]) > _EPS:
                raise InvalidRouteNetwork(
                    f"edge {edge.edge_id} route end does not match to-node position"
                )
            outgoing[edge.from_node_id].append(edge)

        object.__setattr__(self, "nodes", node_items)
        object.__setattr__(self, "edges", edge_items)
        object.__setattr__(self, "_nodes_by_id", MappingProxyType(nodes_by_id))
        object.__setattr__(self, "_edges_by_id", MappingProxyType(edges_by_id))
        object.__setattr__(
            self,
            "_outgoing",
            MappingProxyType(
                {node_id: tuple(sorted(items, key=lambda edge: edge.edge_id)) for node_id, items in outgoing.items()}
            ),
        )

    def node(self, node_id: str) -> RouteNetworkNode:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as exc:
            raise RoutePathUnavailable(f"unknown route-network node {node_id}") from exc

    def edge(self, edge_id: str) -> RouteNetworkEdge:
        try:
            return self._edges_by_id[edge_id]
        except KeyError as exc:
            raise RoutePathUnavailable(f"unknown route-network edge {edge_id}") from exc

    def explicit_path(self, edge_ids: Iterable[str]) -> RouteNetworkPath:
        ids = tuple(edge_ids)
        if not ids:
            raise RoutePathUnavailable("explicit route path must contain at least one edge")
        selected = tuple(self.edge(edge_id) for edge_id in ids)
        for previous, current in zip(selected, selected[1:]):
            if previous.to_node_id != current.from_node_id:
                raise RoutePathUnavailable(
                    f"explicit route path is disconnected between {previous.edge_id} and {current.edge_id}"
                )
        return self._compose(selected, selection_policy=None)

    def resolve(self, request: RouteRequest) -> RouteNetworkPath:
        self.node(request.origin_node_id)
        self.node(request.destination_node_id)
        if request.policy is RouteSelectionPolicy.SHORTEST_DISTANCE:
            edges = self._shortest_distance_edges(
                request.origin_node_id,
                request.destination_node_id,
            )
            return self._compose(edges, selection_policy=request.policy)
        raise RoutePathUnavailable(f"unsupported route selection policy {request.policy!r}")

    def _shortest_distance_edges(
        self,
        origin_node_id: str,
        destination_node_id: str,
    ) -> tuple[RouteNetworkEdge, ...]:
        # Heap key exposes the deterministic tie-break: total distance, then the complete
        # edge-ID path.  No hidden traffic/congestion/VTS objective participates.
        queue: list[tuple[float, tuple[str, ...], str]] = [(0.0, (), origin_node_id)]
        best: dict[str, tuple[float, tuple[str, ...]]] = {origin_node_id: (0.0, ())}

        while queue:
            distance, path_ids, node_id = heapq.heappop(queue)
            recorded = best.get(node_id)
            if recorded is None:
                continue
            if distance > recorded[0] + _EPS:
                continue
            if abs(distance - recorded[0]) <= _EPS and path_ids != recorded[1]:
                continue
            if node_id == destination_node_id:
                return tuple(self.edge(edge_id) for edge_id in path_ids)

            for edge in self._outgoing[node_id]:
                new_distance = distance + edge.distance_nm
                new_ids = path_ids + (edge.edge_id,)
                previous = best.get(edge.to_node_id)
                improves = previous is None or new_distance < previous[0] - _EPS
                ties_better = (
                    previous is not None
                    and abs(new_distance - previous[0]) <= _EPS
                    and new_ids < previous[1]
                )
                if improves or ties_better:
                    best[edge.to_node_id] = (new_distance, new_ids)
                    heapq.heappush(queue, (new_distance, new_ids, edge.to_node_id))

        raise RoutePathUnavailable(
            f"no directed route from {origin_node_id} to {destination_node_id}"
        )

    def _compose(
        self,
        edges: tuple[RouteNetworkEdge, ...],
        *,
        selection_policy: RouteSelectionPolicy | None,
    ) -> RouteNetworkPath:
        if not edges:
            raise RoutePathUnavailable("route path requires at least one edge")
        node_ids = [edges[0].from_node_id]
        waypoints: list[Vector2] = []
        total_distance = 0.0

        for index, edge in enumerate(edges):
            from_position = self._nodes_by_id[edge.from_node_id].position
            to_position = self._nodes_by_id[edge.to_node_id].position
            canonical = (from_position,) + edge.route.waypoints[1:-1] + (to_position,)
            if index == 0:
                waypoints.extend(canonical)
            else:
                waypoints.extend(canonical[1:])
            node_ids.append(edge.to_node_id)
            total_distance += edge.distance_nm

        try:
            route = Route2D(tuple(waypoints))
        except ValueError as exc:
            raise RoutePathUnavailable(
                "resolved edge sequence cannot be composed into one Route2D"
            ) from exc

        return RouteNetworkPath(
            origin_node_id=edges[0].from_node_id,
            destination_node_id=edges[-1].to_node_id,
            edge_ids=tuple(edge.edge_id for edge in edges),
            node_ids=tuple(node_ids),
            route=route,
            total_distance_nm=total_distance,
            selection_policy=selection_policy,
        )


NavigatorFactory = Callable[[Route2D], RouteNavigator]


def build_nominal_target_from_route_path(
    *,
    vessel_id: str,
    path: RouteNetworkPath,
    navigator_factory: NavigatorFactory,
    follower_state: RouteFollowerState = RouteFollowerState(),
    end_action: RouteEndAction = RouteEndAction.CONTINUE,
) -> NominalFleetTarget:
    """Adapt resolved geometry to the explicit a62 reroute target boundary.

    The caller still supplies the complete navigator construction policy.  Routing does
    not choose cruise speed, recovery angle, maneuver limits or route-end semantics.
    """
    navigator = navigator_factory(path.route)
    if not isinstance(navigator, RouteNavigator):
        raise TypeError("navigator_factory must return RouteNavigator")
    if navigator.follower.route != path.route or navigator.recovery.route != path.route:
        raise InvalidRouteNetwork("navigator_factory must bind the resolved Route2D exactly")
    return NominalFleetTarget(
        vessel_id=vessel_id,
        navigator=navigator,
        follower_state=follower_state,
        end_action=end_action,
    )
