"""Persistent observation of one already-applied environmental JOINT maneuver.

A108 observes actual execution of one successful a107 ``AppliedEnvironmentalJointPlan``
receipt.  It advances the single policy-free ``EnvironmentalWorld2D`` clock only to
explicit execution boundaries and records evidence without changing COLREG policy,
lifecycles, candidate generation, arbitration or replanning.

True observation boundaries are:
- each selected vessel's exact a84 transition-completion time;
- an optional fixed periodic review cadence; and
- already queued world command-event timestamps.

An arbitrary caller ``run_until`` horizon may advance the world but does not fabricate a
persisted observation record.  Any controlled-vessel command override before JOINT
completion is terminal evidence; command events for unrelated vessels are neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import inf, isfinite

from .dynamics import signed_heading_delta
from .environmental_dynamics import WaterRelativeManeuverCommand, WaterRelativeVesselState
from .environmental_evidence import (
    EnvironmentalVesselMotionEvidence,
    environmental_vessel_motion_evidence,
)
from .environmental_joint_execution import AppliedEnvironmentalJointPlan
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot

_EPS = 1e-10
_STATE_EPS = 1e-9


class EnvironmentalJointExecutionSessionError(RuntimeError):
    """Base error for persistent environmental JOINT execution observation."""


class StaleEnvironmentalJointExecutionSession(EnvironmentalJointExecutionSessionError):
    """Raised when the world changes outside the persistent JOINT session."""


class EnvironmentalJointExecutionSessionInvariantViolation(EnvironmentalJointExecutionSessionError):
    """Raised when an accepted bounded transition does not materialize as predicted."""


class EnvironmentalJointExecutionObservationReason(str, Enum):
    APPLIED = "applied"
    PERIODIC_REVIEW = "periodic_review"
    COMMAND_EVENT = "command_event"
    TARGET_REACHED = "target_reached"
    COMMAND_OVERRIDDEN = "command_overridden"


class EnvironmentalJointExecutionStopReason(str, Enum):
    COMPLETE = "complete"
    COMMAND_OVERRIDDEN = "command_overridden"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class EnvironmentalJointExecutionSessionConfig:
    review_interval_min: float | None = None

    def __post_init__(self) -> None:
        if self.review_interval_min is not None:
            if not isfinite(self.review_interval_min) or self.review_interval_min <= 0.0:
                raise ValueError("review_interval_min must be finite and positive when supplied")


@dataclass(frozen=True, slots=True)
class EnvironmentalJointVesselExecutionObservation:
    vessel_id: str
    state: WaterRelativeVesselState
    motion: EnvironmentalVesselMotionEvidence
    command: WaterRelativeManeuverCommand
    target_reached: bool


@dataclass(frozen=True, slots=True)
class EnvironmentalJointExecutionObservation:
    time_min: float
    reasons: tuple[EnvironmentalJointExecutionObservationReason, ...]
    vessels: tuple[EnvironmentalJointVesselExecutionObservation, ...]
    target_reached_ids: tuple[str, ...]

    def for_vessel(self, vessel_id: str) -> EnvironmentalJointVesselExecutionObservation:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)


@dataclass(frozen=True, slots=True)
class EnvironmentalJointExecutionSessionResult:
    snapshot: EnvironmentalWorldSnapshot
    stop_reason: EnvironmentalJointExecutionStopReason
    records: tuple[EnvironmentalJointExecutionObservation, ...]


class EnvironmentalJointExecutionSession:
    """Advance/observe one already-applied a107 JOINT maneuver without replanning."""

    def __init__(
        self,
        world: EnvironmentalWorld2D,
        applied: AppliedEnvironmentalJointPlan,
        *,
        config: EnvironmentalJointExecutionSessionConfig | None = None,
    ) -> None:
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if not isinstance(applied, AppliedEnvironmentalJointPlan):
            raise TypeError("applied must be AppliedEnvironmentalJointPlan")
        self.world = world
        self.applied = applied
        self.config = config or EnvironmentalJointExecutionSessionConfig()
        self.vessel_ids = applied.plan.vessel_ids

        if abs(world.time - applied.applied_at_time) > _EPS:
            raise StaleEnvironmentalJointExecutionSession(
                "world time changed after environmental JOINT-plan application"
            )
        if world.snapshot() != applied.snapshot_after:
            raise StaleEnvironmentalJointExecutionSession(
                "world snapshot changed after environmental JOINT-plan application"
            )
        try:
            current_commands = tuple((v, world.command_for(v)) for v in self.vessel_ids)
        except KeyError as exc:
            raise StaleEnvironmentalJointExecutionSession(
                "planned JOINT vessel is no longer active"
            ) from exc
        if current_commands != applied.commands_after:
            raise StaleEnvironmentalJointExecutionSession(
                "JOINT commands changed after environmental JOINT-plan application"
            )
        pending = world.next_command_time()
        if pending is not None and pending <= world.time + _EPS:
            raise StaleEnvironmentalJointExecutionSession(
                "unprocessed command event already exists at JOINT session start time"
            )

        source = applied.plan.arbitration.selected.evaluation.source_evaluation
        completion_times = []
        for vessel_id in self.vessel_ids:
            transition = source.vessel_evaluation(vessel_id).transition
            duration = transition.transition_duration_min
            if not isfinite(duration):
                raise EnvironmentalJointExecutionSessionInvariantViolation(
                    f"applied accepted JOINT tuple contains unreachable transition for {vessel_id!r}"
                )
            exact_completion = bool(
                getattr(transition, "transition_duration_is_target_completion", True)
            )
            completion_times.append((
                vessel_id,
                applied.applied_at_time + duration if exact_completion else inf,
            ))
        self.target_completion_times = tuple(completion_times)
        self.completion_time = max(time for _vessel_id, time in completion_times)

        self._next_review_time = (
            None
            if self.config.review_interval_min is None
            else applied.applied_at_time + self.config.review_interval_min
        )
        self._history: list[EnvironmentalJointExecutionObservation] = []
        self._terminal_reason: EnvironmentalJointExecutionStopReason | None = None
        self._last_snapshot = world.snapshot()
        self._last_commands = current_commands

        reached = self._target_reached_ids()
        start_reasons = [EnvironmentalJointExecutionObservationReason.APPLIED]
        due_now = tuple(
            vessel_id
            for vessel_id, completion_time in self.target_completion_times
            if completion_time <= world.time + _EPS
        )
        if due_now:
            if any(vessel_id not in reached for vessel_id in due_now):
                missing = tuple(v for v in due_now if v not in reached)
                raise EnvironmentalJointExecutionSessionInvariantViolation(
                    f"selected JOINT targets were not reached at application boundary: {missing}"
                )
            start_reasons.append(EnvironmentalJointExecutionObservationReason.TARGET_REACHED)
        self._history.append(self._observation(tuple(start_reasons)))
        if len(reached) == len(self.vessel_ids) and (
            not isfinite(self.completion_time) or self.completion_time <= world.time + _EPS
        ):
            self._terminal_reason = EnvironmentalJointExecutionStopReason.COMPLETE

    @property
    def history(self) -> tuple[EnvironmentalJointExecutionObservation, ...]:
        return tuple(self._history)

    @property
    def terminal_reason(self) -> EnvironmentalJointExecutionStopReason | None:
        return self._terminal_reason

    def completion_time_for(self, vessel_id: str) -> float:
        for candidate_id, time_min in self.target_completion_times:
            if candidate_id == vessel_id:
                return time_min
        raise KeyError(vessel_id)

    def _validate_session_freshness(self) -> None:
        if abs(self.world.time - self._last_snapshot.time) > _EPS:
            raise StaleEnvironmentalJointExecutionSession(
                "world advanced outside environmental JOINT execution session"
            )
        snapshot = self.world.snapshot()
        context = self.applied.plan.planning_context
        try:
            current_commands = tuple((v, self.world.command_for(v)) for v in self.vessel_ids)
            for vessel_id in self.vessel_ids:
                if snapshot.by_id(vessel_id) != self._last_snapshot.by_id(vessel_id):
                    raise StaleEnvironmentalJointExecutionSession(
                        f"planned JOINT vessel {vessel_id!r} physical state changed outside session"
                    )
                inputs = context.for_vessel(vessel_id)
                if self.world.limits_for(vessel_id) != inputs.maneuver_limits:
                    raise StaleEnvironmentalJointExecutionSession(
                        f"maneuver limits changed during JOINT execution for {vessel_id!r}"
                    )
                if self.world.drift_for(vessel_id) != inputs.environmental_drift_velocity:
                    raise StaleEnvironmentalJointExecutionSession(
                        f"environmental drift changed during JOINT execution for {vessel_id!r}"
                    )
        except KeyError as exc:
            raise StaleEnvironmentalJointExecutionSession(
                "planned JOINT vessel is no longer active"
            ) from exc
        if current_commands != self._last_commands:
            raise StaleEnvironmentalJointExecutionSession(
                "JOINT command changed outside environmental JOINT execution session"
            )

    def _target_reached(
        self,
        state: WaterRelativeVesselState,
        command: WaterRelativeManeuverCommand,
    ) -> bool:
        return (
            abs(
                signed_heading_delta(
                    state.through_water_heading_deg,
                    command.target_through_water_heading_deg,
                )
            )
            <= _STATE_EPS
            and abs(state.through_water_speed - command.target_through_water_speed) <= _STATE_EPS
        )

    def _target_reached_ids(self) -> tuple[str, ...]:
        snapshot = self.world.snapshot()
        return tuple(
            vessel_id
            for vessel_id in self.vessel_ids
            if self._target_reached(snapshot.by_id(vessel_id), self.applied.plan.command_for(vessel_id))
        )

    def _observation(
        self,
        reasons: tuple[EnvironmentalJointExecutionObservationReason, ...],
    ) -> EnvironmentalJointExecutionObservation:
        snapshot = self.world.snapshot()
        reached = set(self._target_reached_ids())
        vessels = []
        for vessel_id in self.vessel_ids:
            state = snapshot.by_id(vessel_id)
            vessels.append(
                EnvironmentalJointVesselExecutionObservation(
                    vessel_id=vessel_id,
                    state=state,
                    motion=environmental_vessel_motion_evidence(
                        state, self.world.drift_for(vessel_id)
                    ),
                    command=self.world.command_for(vessel_id),
                    target_reached=vessel_id in reached,
                )
            )
        return EnvironmentalJointExecutionObservation(
            time_min=self.world.time,
            reasons=reasons,
            vessels=tuple(vessels),
            target_reached_ids=tuple(v for v in self.vessel_ids if v in reached),
        )

    def _advance_next_review_after(self, time_min: float) -> None:
        interval = self.config.review_interval_min
        if interval is None or self._next_review_time is None:
            return
        while self._next_review_time <= time_min + _EPS:
            self._next_review_time += interval

    def _candidate_boundary_times(self) -> tuple[float, ...]:
        now = self.world.time
        values: list[float] = []
        for _vessel_id, completion_time in self.target_completion_times:
            if isfinite(completion_time) and completion_time > now + _EPS:
                values.append(completion_time)
        if self._next_review_time is not None and self._next_review_time > now + _EPS:
            values.append(self._next_review_time)
        command_time = self.world.next_command_time()
        if command_time is not None:
            if command_time <= now + _EPS:
                raise StaleEnvironmentalJointExecutionSession(
                    "unprocessed command event exists at current JOINT execution-session time"
                )
            values.append(command_time)
        return tuple(values)

    def next_boundary_time(self) -> float | None:
        self._validate_session_freshness()
        if self._terminal_reason is not None:
            return None
        candidates = self._candidate_boundary_times()
        return min(candidates) if candidates else None

    def run_until(self, end_time: float) -> EnvironmentalJointExecutionSessionResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        self._validate_session_freshness()
        start_history_len = len(self._history)

        if self._terminal_reason is not None:
            return EnvironmentalJointExecutionSessionResult(
                snapshot=self.world.snapshot(),
                stop_reason=self._terminal_reason,
                records=(),
            )

        while self.world.time < end_time - _EPS:
            candidates = self._candidate_boundary_times()
            next_boundary = min(candidates) if candidates else None
            if next_boundary is None or next_boundary > end_time + _EPS:
                self.world.run_until(end_time)
                self._last_snapshot = self.world.snapshot()
                self._last_commands = tuple(
                    (vessel_id, self.world.command_for(vessel_id)) for vessel_id in self.vessel_ids
                )
                break

            boundary = min(next_boundary, end_time)
            command_time = self.world.next_command_time()
            command_event_due = command_time is not None and abs(command_time - boundary) <= _EPS
            review_due = (
                self._next_review_time is not None
                and abs(self._next_review_time - boundary) <= _EPS
            )
            due_ids = tuple(
                vessel_id
                for vessel_id, completion_time in self.target_completion_times
                if abs(completion_time - boundary) <= _EPS
            )

            self.world.run_until(boundary)
            reasons: list[EnvironmentalJointExecutionObservationReason] = []
            if review_due:
                reasons.append(EnvironmentalJointExecutionObservationReason.PERIODIC_REVIEW)
                self._advance_next_review_after(boundary)
            if command_event_due:
                reasons.append(EnvironmentalJointExecutionObservationReason.COMMAND_EVENT)

            reached = self._target_reached_ids()
            if due_ids:
                missing = tuple(vessel_id for vessel_id in due_ids if vessel_id not in reached)
                if missing:
                    raise EnvironmentalJointExecutionSessionInvariantViolation(
                        f"selected JOINT targets not reached at predicted completion: {missing}"
                    )
                reasons.append(EnvironmentalJointExecutionObservationReason.TARGET_REACHED)

            try:
                current_commands = tuple(
                    (vessel_id, self.world.command_for(vessel_id)) for vessel_id in self.vessel_ids
                )
            except KeyError as exc:
                raise StaleEnvironmentalJointExecutionSession(
                    "planned JOINT vessel was removed during execution"
                ) from exc
            selected_commands = self.applied.plan.selected_commands
            all_reached = len(reached) == len(self.vessel_ids)
            complete = all_reached and (
                not isfinite(self.completion_time) or boundary >= self.completion_time - _EPS
            )
            overridden = current_commands != selected_commands
            if overridden and not complete:
                reasons.append(EnvironmentalJointExecutionObservationReason.COMMAND_OVERRIDDEN)

            self._history.append(self._observation(tuple(reasons)))
            self._last_snapshot = self.world.snapshot()
            self._last_commands = current_commands

            if complete:
                self._terminal_reason = EnvironmentalJointExecutionStopReason.COMPLETE
                break
            if overridden:
                self._terminal_reason = EnvironmentalJointExecutionStopReason.COMMAND_OVERRIDDEN
                break

        stop = self._terminal_reason or EnvironmentalJointExecutionStopReason.END_TIME
        return EnvironmentalJointExecutionSessionResult(
            snapshot=self.world.snapshot(),
            stop_reason=stop,
            records=tuple(self._history[start_history_len:]),
        )
