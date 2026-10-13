"""Policy-free event world for explicit water-relative dynamics plus uniform drift.

A73 mirrors the proven ``World2D`` event-timing contract without changing that legacy
next-gen kernel.  Each vessel has a water-relative controlled state/command and one
Earth-fixed drift vector that remains constant until this slice is extended by an
explicit environmental-event model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from itertools import count
from math import isfinite
from typing import Iterable

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .environmental_candidate_transition import (
    EnvironmentalMotionBackend,
    EnvironmentalMotionState,
    validate_environmental_motion_state,
)
from .environmental_dynamics import (
    EnvironmentalAdvanceResult,
    WaterRelativeManeuverCommand,
    WaterRelativeVesselState,
    advance_with_uniform_drift,
)

_EPS = 1e-12


def _validate_drift(vector: Vector2) -> Vector2:
    if len(vector) != 2:
        raise ValueError("drift velocity must contain exactly two components")
    x, y = float(vector[0]), float(vector[1])
    if not isfinite(x) or not isfinite(y):
        raise ValueError("drift velocity components must be finite")
    return (x, y)


@dataclass(order=True, frozen=True, slots=True)
class _QueuedWaterCommand:
    at: float
    sequence: int
    vessel_id: str = field(compare=False)
    command: WaterRelativeManeuverCommand = field(compare=False)


@dataclass(frozen=True, slots=True)
class EnvironmentalWorldSnapshot:
    time: float
    vessels: tuple[WaterRelativeVesselState, ...]

    def by_id(self, vessel_id: str) -> WaterRelativeVesselState:
        for vessel in self.vessels:
            if vessel.vessel_id == vessel_id:
                return vessel
        raise KeyError(vessel_id)


class EnvironmentalWorld2D:
    """Independent event-driven motion under fixed per-vessel environmental drift.

    This world owns monotonic time, bounded water-relative propagation and scheduled
    water-relative command changes. It contains no routing, COLREG, risk or policy
    selection logic. Drift is construction-time state in a73 and cannot change during
    a run; time-varying/spatial fields require a later explicit contract.
    """

    def __init__(
        self,
        vessels: Iterable[WaterRelativeVesselState],
        *,
        limits: ManeuverLimits | dict[str, ManeuverLimits],
        environmental_drift_velocity: Vector2 | dict[str, Vector2] = (0.0, 0.0),
        start_time: float = 0.0,
        motion_backend: EnvironmentalMotionBackend | None = None,
    ) -> None:
        if not isfinite(start_time) or start_time < 0.0:
            raise ValueError("start_time must be finite and non-negative")
        vessel_list = tuple(vessels)
        states = {v.vessel_id: v for v in vessel_list}
        if len(states) != len(vessel_list):
            raise ValueError("duplicate vessel_id")
        self._states = states
        self._time = start_time
        self._events: list[_QueuedWaterCommand] = []
        self._counter = count()
        self._limits = limits
        self._commands = {
            vessel_id: WaterRelativeManeuverCommand(
                target_through_water_heading_deg=v.through_water_heading_deg,
                target_through_water_speed=v.through_water_speed,
            )
            for vessel_id, v in states.items()
        }

        if isinstance(limits, dict):
            missing = set(states) - set(limits)
            if missing:
                raise ValueError(f"missing maneuver limits for: {sorted(missing)}")

        if isinstance(environmental_drift_velocity, dict):
            missing = set(states) - set(environmental_drift_velocity)
            if missing:
                raise ValueError(f"missing environmental drift for: {sorted(missing)}")
            self._drifts: Vector2 | dict[str, Vector2] = {
                vessel_id: _validate_drift(environmental_drift_velocity[vessel_id])
                for vessel_id in states
            }
        else:
            self._drifts = _validate_drift(environmental_drift_velocity)

        if motion_backend is not None and not isinstance(motion_backend, EnvironmentalMotionBackend):
            raise TypeError("motion_backend must implement EnvironmentalMotionBackend")
        self._motion_backend = motion_backend
        self._motion_states: dict[str, EnvironmentalMotionState | None] = {}
        if motion_backend is not None:
            for vessel_id in sorted(self._states):
                initial_motion_state = motion_backend.initialize_motion_state(
                    self._states[vessel_id],
                    self.limits_for(vessel_id),
                    self.drift_for(vessel_id),
                )
                self._motion_states[vessel_id] = validate_environmental_motion_state(
                    motion_backend,
                    initial_motion_state,
                    boundary=f"world initialize {vessel_id!r}",
                )

    @property
    def time(self) -> float:
        return self._time

    @property
    def motion_backend(self) -> EnvironmentalMotionBackend | None:
        """Explicit richer backend, or ``None`` for historical parity execution."""

        return self._motion_backend

    def motion_state_for(self, vessel_id: str) -> EnvironmentalMotionState | None:
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        if self._motion_backend is None:
            return None
        return self._motion_states[vessel_id]

    def snapshot(self) -> EnvironmentalWorldSnapshot:
        vessels = tuple(self._states[k] for k in sorted(self._states))
        return EnvironmentalWorldSnapshot(time=self._time, vessels=vessels)

    def limits_for(self, vessel_id: str) -> ManeuverLimits:
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        if isinstance(self._limits, ManeuverLimits):
            return self._limits
        return self._limits[vessel_id]

    def drift_for(self, vessel_id: str) -> Vector2:
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        if isinstance(self._drifts, dict):
            return self._drifts[vessel_id]
        return self._drifts

    def command_for(self, vessel_id: str) -> WaterRelativeManeuverCommand:
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        return self._commands[vessel_id]

    def next_command_time(self) -> float | None:
        return None if not self._events else self._events[0].at

    def schedule_command(
        self,
        *,
        at: float,
        vessel_id: str,
        command: WaterRelativeManeuverCommand,
    ) -> None:
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        if not isinstance(command, WaterRelativeManeuverCommand):
            raise TypeError("command must be WaterRelativeManeuverCommand")
        if not isfinite(at):
            raise ValueError("event time must be finite")
        if at < self._time - _EPS:
            raise ValueError("cannot schedule an event in the past")
        heapq.heappush(
            self._events,
            _QueuedWaterCommand(
                at=at,
                sequence=next(self._counter),
                vessel_id=vessel_id,
                command=command,
            ),
        )

    def schedule_commands_atomic(
        self,
        *,
        at: float,
        commands: Iterable[tuple[str, WaterRelativeManeuverCommand]],
    ) -> None:
        """Schedule one timestamped command batch only after whole-batch validation.

        This is a policy-free transaction boundary for callers that must not leave a
        partially queued multi-vessel action when a later batch item is invalid.  The
        supplied order is preserved through the existing monotonically increasing
        event sequence numbers.
        """

        if not isfinite(at):
            raise ValueError("event time must be finite")
        if at < self._time - _EPS:
            raise ValueError("cannot schedule an event in the past")
        batch = tuple(commands)
        for vessel_id, command in batch:
            if vessel_id not in self._states:
                raise KeyError(vessel_id)
            if not isinstance(command, WaterRelativeManeuverCommand):
                raise TypeError("command must be WaterRelativeManeuverCommand")

        # Build the complete replacement heap before publishing it.  Validation
        # failures above therefore cannot leave a prefix of the batch queued.
        new_events = [
            _QueuedWaterCommand(
                at=at,
                sequence=next(self._counter),
                vessel_id=vessel_id,
                command=command,
            )
            for vessel_id, command in batch
        ]
        combined = [*self._events, *new_events]
        heapq.heapify(combined)
        self._events = combined

    def apply_current_transaction_atomic(
        self,
        *,
        commands: Iterable[tuple[str, WaterRelativeManeuverCommand]] = (),
        removals: Iterable[str] = (),
    ) -> tuple[WaterRelativeVesselState, ...]:
        """Atomically apply current-time commands and vessel removals.

        This is a policy-free transaction primitive for higher-level ownership layers.
        The complete command/removal set is validated and materialized on copies before
        any live world dictionary is replaced.  A vessel cannot be both commanded and
        removed in the same transaction.  Pending command events at the current
        timestamp are rejected so the caller cannot silently reorder event ownership.

        The method does not advance physical time.  Returned states are the removed
        vessels in the caller-supplied removal order.
        """

        command_batch = tuple(commands)
        removal_ids = tuple(removals)
        command_ids = tuple(vessel_id for vessel_id, _ in command_batch)
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("current transaction command vessel ids must be unique")
        if len(removal_ids) != len(set(removal_ids)):
            raise ValueError("current transaction removal vessel ids must be unique")
        overlap = set(command_ids).intersection(removal_ids)
        if overlap:
            raise ValueError(
                f"current transaction cannot command and remove the same vessel: {sorted(overlap)!r}"
            )

        pending = self.next_command_time()
        if pending is not None and pending <= self._time + _EPS:
            raise ValueError("pending current-time command event exists before atomic transaction")

        for vessel_id, command in command_batch:
            if vessel_id not in self._states:
                raise KeyError(vessel_id)
            if not isinstance(command, WaterRelativeManeuverCommand):
                raise TypeError("command must be WaterRelativeManeuverCommand")
        for vessel_id in removal_ids:
            if vessel_id not in self._states:
                raise KeyError(vessel_id)

        states = dict(self._states)
        active_commands = dict(self._commands)
        motion_states = dict(self._motion_states)
        limits = dict(self._limits) if isinstance(self._limits, dict) else self._limits
        drifts = dict(self._drifts) if isinstance(self._drifts, dict) else self._drifts
        events = list(self._events)

        for vessel_id, command in command_batch:
            active_commands[vessel_id] = command

        removed_states: list[WaterRelativeVesselState] = []
        removal_set = set(removal_ids)
        for vessel_id in removal_ids:
            removed_states.append(states.pop(vessel_id))
            active_commands.pop(vessel_id, None)
            motion_states.pop(vessel_id, None)
            if isinstance(limits, dict):
                limits.pop(vessel_id, None)
            if isinstance(drifts, dict):
                drifts.pop(vessel_id, None)
        if removal_set:
            events = [event for event in events if event.vessel_id not in removal_set]
            heapq.heapify(events)

        self._states = states
        self._commands = active_commands
        self._motion_states = motion_states
        self._limits = limits
        self._drifts = drifts
        self._events = events
        return tuple(removed_states)

    def add_vessel(
        self,
        vessel: WaterRelativeVesselState,
        *,
        limits: ManeuverLimits | None = None,
        environmental_drift_velocity: Vector2 | None = None,
    ) -> None:
        if vessel.vessel_id in self._states:
            raise ValueError(f"duplicate vessel_id: {vessel.vessel_id}")
        if isinstance(self._limits, dict):
            if limits is None:
                raise ValueError("per-vessel world requires limits for each added vessel")
            self._limits[vessel.vessel_id] = limits
        elif limits is not None and limits != self._limits:
            raise ValueError("cannot override shared maneuver limits")

        if isinstance(self._drifts, dict):
            if environmental_drift_velocity is None:
                raise ValueError("per-vessel world requires drift for each added vessel")
            self._drifts[vessel.vessel_id] = _validate_drift(environmental_drift_velocity)
        elif environmental_drift_velocity is not None:
            if _validate_drift(environmental_drift_velocity) != self._drifts:
                raise ValueError("cannot override shared environmental drift")

        self._states[vessel.vessel_id] = vessel
        self._commands[vessel.vessel_id] = WaterRelativeManeuverCommand(
            target_through_water_heading_deg=vessel.through_water_heading_deg,
            target_through_water_speed=vessel.through_water_speed,
        )
        if self._motion_backend is not None:
            initial_motion_state = self._motion_backend.initialize_motion_state(
                vessel,
                self.limits_for(vessel.vessel_id),
                self.drift_for(vessel.vessel_id),
            )
            self._motion_states[vessel.vessel_id] = validate_environmental_motion_state(
                self._motion_backend,
                initial_motion_state,
                boundary=f"world add_vessel {vessel.vessel_id!r}",
            )

    def remove_vessel(self, vessel_id: str) -> WaterRelativeVesselState:
        if vessel_id not in self._states:
            raise KeyError(vessel_id)
        state = self._states.pop(vessel_id)
        self._commands.pop(vessel_id, None)
        self._motion_states.pop(vessel_id, None)
        if isinstance(self._limits, dict):
            self._limits.pop(vessel_id, None)
        if isinstance(self._drifts, dict):
            self._drifts.pop(vessel_id, None)
        if self._events:
            self._events = [event for event in self._events if event.vessel_id != vessel_id]
            heapq.heapify(self._events)
        return state

    def _advance_all(self, duration: float) -> dict[str, EnvironmentalAdvanceResult]:
        results: dict[str, EnvironmentalAdvanceResult] = {}
        if duration <= _EPS:
            return results
        for vessel_id in sorted(self._states):
            if self._motion_backend is None:
                result = advance_with_uniform_drift(
                    self._states[vessel_id],
                    self._commands[vessel_id],
                    duration,
                    self.limits_for(vessel_id),
                    environmental_drift_velocity=self.drift_for(vessel_id),
                )
            else:
                current_motion_state = validate_environmental_motion_state(
                    self._motion_backend,
                    self._motion_states[vessel_id],
                    boundary=f"world pre-advance {vessel_id!r}",
                )
                backend_result = self._motion_backend.advance(
                    self._states[vessel_id],
                    self._commands[vessel_id],
                    duration,
                    self.limits_for(vessel_id),
                    self.drift_for(vessel_id),
                    current_motion_state,
                )
                next_motion_state = validate_environmental_motion_state(
                    self._motion_backend,
                    backend_result.motion_state,
                    boundary=f"world post-advance {vessel_id!r}",
                )
                result = backend_result.advance
                self._motion_states[vessel_id] = next_motion_state
            self._states[vessel_id] = result.state
            results[vessel_id] = result
        self._time += duration
        return results

    def run_until(self, end_time: float) -> EnvironmentalWorldSnapshot:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self._time - _EPS:
            raise ValueError("end_time must not move backwards")

        while self._events and self._events[0].at <= end_time + _EPS:
            event_time = self._events[0].at
            if event_time > self._time + _EPS:
                self._advance_all(event_time - self._time)
            else:
                self._time = max(self._time, event_time)

            while self._events and abs(self._events[0].at - event_time) <= _EPS:
                event = heapq.heappop(self._events)
                self._commands[event.vessel_id] = event.command

        if end_time > self._time + _EPS:
            self._advance_all(end_time - self._time)
        else:
            self._time = end_time
        return self.snapshot()
