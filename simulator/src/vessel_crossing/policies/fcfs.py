"""First-Come-First-Served crossing-control baseline."""

from __future__ import annotations

from dataclasses import dataclass

from .base import PolicySnapshot, SpeedCommand


@dataclass(frozen=True, slots=True)
class FCFSSpeedPolicy:
    """Give crossing precedence to the vessel admitted first.

    "First come" is defined as the earliest ``activated_at`` timestamp: the
    instant at which a vessel is actually admitted onto the modeled route, not
    the stochastic ``arrival_time`` at which it may have joined an entrance
    queue.  Equal activation times are broken deterministically by vessel ID.

    The first numerical action is deliberately comparable with the simplified
    Rule-15 baseline: a yielding vessel first reduces to a configurable
    fraction of desired speed; if the dangerous encounter persists while that
    reduced own target is already installed, the fallback command is stop.
    """

    reduced_speed_fraction: float = 2.0 / 3.0
    name: str = "fcfs_speed"

    _EPSILON = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.reduced_speed_fraction < 1.0:
            raise ValueError("reduced_speed_fraction must lie strictly between 0 and 1")

    @staticmethod
    def precedence_key(vessel) -> tuple[float, str]:
        if vessel.activated_at is None:
            raise ValueError("FCFS requires activated_at for every active vessel")
        return (vessel.activated_at, vessel.vessel_id)

    def decide(self, snapshot: PolicySnapshot) -> tuple[SpeedCommand, ...]:
        requested: dict[str, tuple[float, str]] = {}
        open_yield_obligations: set[str] = set()

        for encounter in snapshot.encounters:
            first = snapshot.vessel(encounter.first_vessel_id)
            second = snapshot.vessel(encounter.second_vessel_id)

            first_key = self.precedence_key(first)
            second_key = self.precedence_key(second)
            give_way = second if first_key < second_key else first
            stand_on = first if give_way is second else second
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
                    f"FCFS precedence to {stand_on.vessel_id}; "
                    "later-admitted vessel reduces speed"
                )
            else:
                target = 0.0
                reason = (
                    f"FCFS precedence to {stand_on.vessel_id}; "
                    "persistent conflict stop fallback"
                )

            previous = requested.get(give_way.vessel_id)
            if previous is None or target < previous[0]:
                requested[give_way.vessel_id] = (target, reason)

        # Keep a successful reduction installed through closest approach.
        # Restoring merely because the controlled trajectory is currently safe
        # would recreate the conflict and cause command oscillation.
        for vessel in snapshot.vessels:
            own_target = (
                vessel.desired_speed if vessel.target_speed is None else vessel.target_speed
            )
            if (
                vessel.vessel_id not in open_yield_obligations
                and own_target < vessel.desired_speed - self._EPSILON
            ):
                requested[vessel.vessel_id] = (
                    vessel.desired_speed,
                    "FCFS crossing past and clear; restore desired speed",
                )

        return tuple(
            SpeedCommand(vessel_id, target, reason)
            for vessel_id, (target, reason) in sorted(requested.items())
        )
