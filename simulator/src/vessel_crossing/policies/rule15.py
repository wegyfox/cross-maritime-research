"""Explicitly simplified Rule-15/16 speed-only research baseline.

This module is not a complete COLREG implementation.  See
``docs/COLREG_BASELINE_ASSUMPTIONS.md`` for scope and exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import PolicySnapshot, SpeedCommand
from .observations import RoutePair, route_pair


@dataclass(frozen=True, slots=True)
class CrossingResponsibility:
    """Which route gives way for one unordered crossing route pair."""

    first_route: str
    second_route: str
    give_way_route: str

    def __post_init__(self) -> None:
        pair = route_pair(self.first_route, self.second_route)
        if self.give_way_route not in pair:
            raise ValueError("give_way_route must belong to crossing pair")

    @property
    def pair(self) -> RoutePair:
        return route_pair(self.first_route, self.second_route)


@dataclass(frozen=True, slots=True)
class SimplifiedRule15SpeedPolicy:
    """Deterministic speed-only give-way baseline for crossing research."""

    responsibilities: tuple[CrossingResponsibility, ...]
    reduced_speed_fraction: float = 2.0 / 3.0
    name: str = "simplified_rule15_speed"

    _EPSILON = 1e-12

    def __post_init__(self) -> None:
        if not 0 < self.reduced_speed_fraction < 1:
            raise ValueError("reduced_speed_fraction must lie strictly between 0 and 1")
        pairs = [rule.pair for rule in self.responsibilities]
        if len(pairs) != len(set(pairs)):
            raise ValueError("crossing responsibility pairs must be unique")

    def decide(self, snapshot: PolicySnapshot) -> tuple[SpeedCommand, ...]:
        responsibility_by_pair = {rule.pair: rule for rule in self.responsibilities}
        requested: dict[str, tuple[float, str]] = {}
        open_yield_obligations: set[str] = set()

        # Inspect every observed crossing, not only the currently dangerous ones.
        # Once a policy reduction has made an encounter safe, that reduction must
        # remain installed until closest approach has actually been reached/passed.
        # Otherwise a stateless policy oscillates 6 -> 4 -> 6 -> 4 simply because
        # its own successful action temporarily changes the CPA classification.
        for encounter in snapshot.encounters:
            first = snapshot.vessel(encounter.first_vessel_id)
            second = snapshot.vessel(encounter.second_vessel_id)
            pair = route_pair(first.route.name, second.route.name)
            rule = responsibility_by_pair.get(pair)
            if rule is None:
                continue

            give_way = first if first.route.name == rule.give_way_route else second
            own_target = (
                give_way.desired_speed
                if give_way.target_speed is None
                else give_way.target_speed
            )
            cpa = encounter.assessment.closest_approach
            past_clear = (
                not encounter.assessment.dangerous
                and cpa.raw_tcpa <= self._EPSILON
            )
            if (
                own_target < give_way.desired_speed - self._EPSILON
                and not past_clear
            ):
                open_yield_obligations.add(give_way.vessel_id)

            if not encounter.assessment.dangerous:
                continue

            open_yield_obligations.add(give_way.vessel_id)
            reduced = give_way.desired_speed * self.reduced_speed_fraction
            if own_target > reduced + self._EPSILON:
                target = reduced
                reason = (
                    f"Rule-15 crossing give-way {give_way.route.name}; "
                    "single substantial speed reduction"
                )
            else:
                target = 0.0
                reason = (
                    f"Rule-15 crossing give-way {give_way.route.name}; "
                    "persistent conflict stop fallback"
                )

            previous = requested.get(give_way.vessel_id)
            if previous is None or target < previous[0]:
                requested[give_way.vessel_id] = (target, reason)

        # A previous policy reduction is released only when no observed crossing
        # still carries an open give-way obligation.  In particular, an encounter
        # that is currently CLEAR solely because of the reduction is held through
        # closest approach instead of immediately undoing the successful action.
        for vessel in snapshot.vessels:
            own_target = vessel.desired_speed if vessel.target_speed is None else vessel.target_speed
            if (
                vessel.vessel_id not in open_yield_obligations
                and own_target < vessel.desired_speed - self._EPSILON
            ):
                requested[vessel.vessel_id] = (
                    vessel.desired_speed,
                    "crossing past and clear; restore desired speed",
                )

        return tuple(
            SpeedCommand(vessel_id, target, reason)
            for vessel_id, (target, reason) in sorted(requested.items())
        )


def diploma_rule15_responsibilities() -> tuple[CrossingResponsibility, ...]:
    """Rule-15 roles for the fixed A/B/C/D diploma route headings."""

    return (
        CrossingResponsibility("A", "C", "A"),
        CrossingResponsibility("A", "D", "D"),
        CrossingResponsibility("B", "C", "C"),
        CrossingResponsibility("B", "D", "B"),
    )
