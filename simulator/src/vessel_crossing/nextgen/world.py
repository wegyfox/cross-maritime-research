"""Minimal event-driven world kernel for next-generation continuous 2D motion."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from itertools import count
from typing import Iterable

from .dynamics import AdvanceResult, ManeuverCommand, ManeuverLimits, Vessel2DState, advance_toward_command

_EPS = 1e-12


@dataclass(order=True, frozen=True, slots=True)
class _QueuedCommand:
    at: float
    sequence: int
    vessel_id: str = field(compare=False)
    command: ManeuverCommand = field(compare=False)


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    time: float
    vessels: tuple[Vessel2DState, ...]

    def by_id(self, vessel_id: str) -> Vessel2DState:
        for vessel in self.vessels:
            if vessel.vessel_id == vessel_id:
                return vessel
        raise KeyError(vessel_id)


class World2D:
    """Event-driven independent-motion kernel.

    This first vertical slice intentionally does not contain COLREG or routing logic.
    Those layers submit maneuver commands; the world owns monotonic time and physical
    propagation between command-change events.
    """

    def __init__(
        self,
        vessels: Iterable[Vessel2DState],
        *,
        limits: ManeuverLimits | dict[str, ManeuverLimits],
        start_time: float = 0.0,
    ) -> None:
        if start_time < 0:
            raise ValueError("start_time must be non-negative")
        vessel_list = tuple(vessels)
        states = {v.vessel_id: v for v in vessel_list}
        if len(states) != len(vessel_list):
            raise ValueError("duplicate vessel_id")
        self._states = states
        self._time = start_time
        self._events: list[_QueuedCommand] = []
        self._counter = count()
        self._limits = limits
        self._commands = {
            vessel_id: ManeuverCommand(target_heading_deg=v.heading_deg, target_speed=v.speed)
            for vessel_id, v in states.items()
        }

        if isinstance(limits, dict):
            missing = set(states) - set(limits)
            if missing:
                raise ValueError(f"missing maneuver limits for: {sorted(missing)}")

    @property
    def time(self) -> float:
        return self._time

    def snapshot(self) -> WorldSnapshot:
        vessels = tuple(self._states[k] for k in sorted(self._states))
        return WorldSnapshot(time=self._time, vessels=vessels)

    def limits_for(self, vessel_id: str) -> ManeuverLimits:
        """Return immutable maneuver limits for one vessel."""
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        if isinstance(self._limits, ManeuverLimits):
            return self._limits
        return self._limits[vessel_id]

    def command_for(self, vessel_id: str) -> ManeuverCommand:
        """Return the currently effective immutable target command for one vessel.

        This is a read-only inspection surface for diagnostics/metrics.  It does not
        expose or mutate the queued-event container and therefore does not move policy
        responsibility into the physics kernel.
        """
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        return self._commands[vessel_id]

    def next_command_time(self) -> float | None:
        """Return the earliest queued command timestamp, if any.

        Read-only metric recorders use this boundary to avoid pretending that one target
        command remained constant across a scheduled command change.
        """
        return None if not self._events else self._events[0].at

    def _limits_for(self, vessel_id: str) -> ManeuverLimits:
        return self.limits_for(vessel_id)

    def add_vessel(
        self,
        vessel: Vessel2DState,
        *,
        limits: ManeuverLimits | None = None,
    ) -> None:
        """Insert one vessel at the current world time without advancing the clock."""
        if vessel.vessel_id in self._states:
            raise ValueError(f"duplicate vessel_id: {vessel.vessel_id}")
        if isinstance(self._limits, dict):
            if limits is None:
                raise ValueError("per-vessel World2D requires limits for each added vessel")
            self._limits[vessel.vessel_id] = limits
        elif limits is not None and limits != self._limits:
            raise ValueError("cannot override limits in a World2D with shared maneuver limits")
        self._states[vessel.vessel_id] = vessel
        self._commands[vessel.vessel_id] = ManeuverCommand(
            target_heading_deg=vessel.heading_deg, target_speed=vessel.speed
        )

    def remove_vessel(self, vessel_id: str) -> Vessel2DState:
        """Remove one vessel at the current world time and cancel its queued commands.

        This is an explicit lifecycle operation for route-network sinks. It does not
        advance time. The removed terminal state is returned for archival/result use.
        """
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        state = self._states.pop(vessel_id)
        self._commands.pop(vessel_id, None)
        if isinstance(self._limits, dict):
            self._limits.pop(vessel_id, None)
        if self._events:
            self._events = [event for event in self._events if event.vessel_id != vessel_id]
            heapq.heapify(self._events)
        return state

    def schedule_command(self, *, at: float, vessel_id: str, command: ManeuverCommand) -> None:
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        if at < self._time - _EPS:
            raise ValueError("cannot schedule an event in the past")
        heapq.heappush(
            self._events,
            _QueuedCommand(at=at, sequence=next(self._counter), vessel_id=vessel_id, command=command),
        )

    def _advance_all(self, duration: float) -> dict[str, AdvanceResult]:
        results: dict[str, AdvanceResult] = {}
        if duration <= _EPS:
            return results
        for vessel_id in sorted(self._states):
            result = advance_toward_command(
                self._states[vessel_id],
                self._commands[vessel_id],
                duration,
                self._limits_for(vessel_id),
            )
            self._states[vessel_id] = result.state
            results[vessel_id] = result
        self._time += duration
        return results

    def run_until(self, end_time: float) -> WorldSnapshot:
        if end_time < self._time - _EPS:
            raise ValueError("end_time must not move backwards")

        while self._events and self._events[0].at <= end_time + _EPS:
            event_time = self._events[0].at
            if event_time > self._time + _EPS:
                self._advance_all(event_time - self._time)
            else:
                self._time = max(self._time, event_time)

            # Deterministic same-timestamp command application by insertion sequence.
            while self._events and abs(self._events[0].at - event_time) <= _EPS:
                event = heapq.heappop(self._events)
                self._commands[event.vessel_id] = event.command

        if end_time > self._time + _EPS:
            self._advance_all(end_time - self._time)
        else:
            self._time = end_time

        return self.snapshot()
