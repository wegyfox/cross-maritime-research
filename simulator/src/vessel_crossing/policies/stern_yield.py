"""Recovered alternative rule: the vessel projected to pass astern yields.

The 2020 thesis explicitly proposed a second numerical experiment after the
simplified Rule-15 run: for an unsafe crossing, make the vessel that would pass
behind the other vessel reduce speed/stop.  The original code did not complete
that branch.

For the diploma's straight fixed routes and equal prescribed cruise speeds,
"passes astern" is represented by the later *free-flow arrival time at the
geometric route intersection*.  The key is based on activation time, route
geometry and desired speed, so a successful speed reduction does not reverse
its own precedence decision on the next policy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..legacy_topology import intersect_routes
from .base import PolicySnapshot, SpeedCommand


@dataclass(frozen=True, slots=True)
class SternPasserYieldSpeedPolicy:
    """Yield the later free-flow arrival at each geometric crossing."""

    reduced_speed_fraction: float = 2.0 / 3.0
    name: str = "stern_passer_yield_speed"

    _EPSILON = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.reduced_speed_fraction < 1.0:
            raise ValueError("reduced_speed_fraction must lie strictly between 0 and 1")

    @staticmethod
    def crossing_eta(vessel, other_route) -> float:
        if vessel.activated_at is None:
            raise ValueError("stern-yield policy requires activated_at for every active vessel")
        if vessel.desired_speed <= 0:
            raise ValueError("stern-yield policy requires positive desired speed")
        intersection = intersect_routes(vessel.route, other_route)
        progress = intersection.progress_for(vessel.route.name)
        return vessel.activated_at + progress / vessel.desired_speed

    def decide(self, snapshot: PolicySnapshot) -> tuple[SpeedCommand, ...]:
        requested: dict[str, tuple[float, str]] = {}
        open_yield_obligations: set[str] = set()

        for encounter in snapshot.encounters:
            first = snapshot.vessel(encounter.first_vessel_id)
            second = snapshot.vessel(encounter.second_vessel_id)

            try:
                first_eta = self.crossing_eta(first, second.route)
                second_eta = self.crossing_eta(second, first.route)
            except ValueError as exc:
                # Same-direction and opposing parallel lanes are not crossing
                # encounters. Longitudinal separation is handled by the
                # simulation's same-stream mechanism, not this policy.
                if "unique crossing" in str(exc) or "do not intersect" in str(exc):
                    continue
                raise
            first_key = (first_eta, first.vessel_id)
            second_key = (second_eta, second.vessel_id)
            # Earlier geometric crossing arrival passes ahead; later arrival is
            # the vessel projected to pass astern and therefore gives way.
            give_way = second if first_key < second_key else first
            stand_on = first if give_way is second else second

            own_target = give_way.desired_speed if give_way.target_speed is None else give_way.target_speed
            cpa = encounter.assessment.closest_approach
            past_clear = not encounter.assessment.dangerous and cpa.raw_tcpa <= self._EPSILON
            if own_target < give_way.desired_speed - self._EPSILON and not past_clear:
                open_yield_obligations.add(give_way.vessel_id)

            if not encounter.assessment.dangerous:
                continue

            open_yield_obligations.add(give_way.vessel_id)
            reduced = give_way.desired_speed * self.reduced_speed_fraction
            if own_target > reduced + self._EPSILON:
                target = reduced
                reason = (
                    f"projected astern of {stand_on.vessel_id} at geometric crossing; "
                    "reduce speed"
                )
            else:
                target = 0.0
                reason = (
                    f"projected astern of {stand_on.vessel_id} at geometric crossing; "
                    "persistent conflict stop fallback"
                )

            previous = requested.get(give_way.vessel_id)
            if previous is None or target < previous[0]:
                requested[give_way.vessel_id] = (target, reason)

        for vessel in snapshot.vessels:
            own_target = vessel.desired_speed if vessel.target_speed is None else vessel.target_speed
            if (
                vessel.vessel_id not in open_yield_obligations
                and own_target < vessel.desired_speed - self._EPSILON
            ):
                requested[vessel.vessel_id] = (
                    vessel.desired_speed,
                    "astern-yield crossing past and clear; restore desired speed",
                )

        return tuple(
            SpeedCommand(vessel_id, target, reason)
            for vessel_id, (target, reason) in sorted(requested.items())
        )
