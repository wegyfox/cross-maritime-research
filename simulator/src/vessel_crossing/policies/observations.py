"""Build policy snapshots from active vessel states and CPA/TCPA safety logic."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ..safety import CircularSafetyDomain, assess_encounter
from ..vessel import Vessel, VesselStatus
from .base import EncounterObservation, PolicySnapshot

RoutePair = frozenset[str]


def route_pair(first_route: str, second_route: str) -> RoutePair:
    """Create a canonical unordered route pair for encounter filtering."""
    if not first_route or not second_route:
        raise ValueError("route names must be non-empty")
    if first_route == second_route:
        raise ValueError("route pair requires two distinct routes")
    return frozenset((first_route, second_route))


@dataclass(frozen=True, slots=True)
class EncounterObserver:
    """Convert active vessel states into immutable policy observations.

    ``allowed_route_pairs`` is optional.  When supplied, only those route pairs
    are assessed.  This is important for the diploma geometry: A/B and C/D are
    opposing lanes of the same fairway, while the crossing controller should
    focus on vertical-vs-horizontal route pairs.
    """

    safety_domain: CircularSafetyDomain
    allowed_route_pairs: frozenset[RoutePair] | None = None

    def __post_init__(self) -> None:
        if self.allowed_route_pairs is not None:
            for pair in self.allowed_route_pairs:
                if len(pair) != 2:
                    raise ValueError("every allowed route pair must contain two routes")

    def build_snapshot(
        self,
        *,
        time: float,
        vessels: tuple[Vessel, ...] | list[Vessel],
    ) -> PolicySnapshot:
        active = tuple(
            sorted(
                (v for v in vessels if v.status is VesselStatus.ACTIVE),
                key=lambda v: v.vessel_id,
            )
        )

        encounters: list[EncounterObservation] = []
        for first, second in combinations(active, 2):
            if first.route.name == second.route.name:
                # Same-stream longitudinal safety is owned by the simulation
                # route controller, not by crossing policies.
                continue

            pair = route_pair(first.route.name, second.route.name)
            if self.allowed_route_pairs is not None and pair not in self.allowed_route_pairs:
                continue

            encounters.append(
                EncounterObservation(
                    first_vessel_id=first.vessel_id,
                    second_vessel_id=second.vessel_id,
                    assessment=assess_encounter(
                        first.motion_state,
                        second.motion_state,
                        self.safety_domain,
                    ),
                )
            )

        return PolicySnapshot(
            time=time,
            vessels=active,
            encounters=tuple(encounters),
        )
