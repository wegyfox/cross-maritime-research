"""Persistent observation of one already-applied environmental pair maneuver.

A93 is intentionally *not* a replanning supervisor.  It starts from one successful a92
``AppliedEnvironmentalPairPlan`` receipt, advances the policy-free environmental world
to explicit boundaries, and records whether the two selected water-relative target
commands are still being executed and when each target is physically reached.

Boundaries are limited to:
- the first/second a84 transition-completion times;
- an optional caller-configured periodic observation cadence;
- already queued world command-event timestamps; and
- the caller's run horizon (which advances the world but is not itself persisted as an
  observation record unless it coincides with one of the boundaries above).

No encounter lifecycle update, candidate generation, arbitration or automatic replanning
occurs here.
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
from .environmental_pair_execution import AppliedEnvironmentalPairPlan
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot

_EPS = 1e-10
_STATE_EPS = 1e-9


class EnvironmentalPairExecutionSessionError(RuntimeError):
    """Base error for a93 persistent execution-observation failures."""


class StaleEnvironmentalPairExecutionSession(EnvironmentalPairExecutionSessionError):
    """Raised when the world was mutated outside the session between calls."""


class EnvironmentalPairExecutionSessionInvariantViolation(EnvironmentalPairExecutionSessionError):
    """Raised when the applied plan's exact bounded transition fails to materialize."""


class EnvironmentalPairExecutionObservationReason(str, Enum):
    APPLIED = "applied"
    PERIODIC_REVIEW = "periodic_review"
    COMMAND_EVENT = "command_event"
    FIRST_TARGET_REACHED = "first_target_reached"
    SECOND_TARGET_REACHED = "second_target_reached"
    COMMAND_OVERRIDDEN = "command_overridden"


class EnvironmentalPairExecutionStopReason(str, Enum):
    COMPLETE = "complete"
    COMMAND_OVERRIDDEN = "command_overridden"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class EnvironmentalPairExecutionSessionConfig:
    review_interval_min: float | None = None

    def __post_init__(self) -> None:
        if self.review_interval_min is not None:
            if not isfinite(self.review_interval_min) or self.review_interval_min <= 0.0:
                raise ValueError("review_interval_min must be finite and positive when supplied")


@dataclass(frozen=True, slots=True)
class EnvironmentalPairExecutionObservation:
    time_min: float
    reasons: tuple[EnvironmentalPairExecutionObservationReason, ...]
    first_state: WaterRelativeVesselState
    second_state: WaterRelativeVesselState
    first_motion: EnvironmentalVesselMotionEvidence
    second_motion: EnvironmentalVesselMotionEvidence
    first_command: WaterRelativeManeuverCommand
    second_command: WaterRelativeManeuverCommand
    first_target_reached: bool
    second_target_reached: bool


@dataclass(frozen=True, slots=True)
class EnvironmentalPairExecutionSessionResult:
    snapshot: EnvironmentalWorldSnapshot
    stop_reason: EnvironmentalPairExecutionStopReason
    records: tuple[EnvironmentalPairExecutionObservation, ...]


class EnvironmentalPairExecutionSession:
    """Advance and observe one already-applied a92 pair maneuver without replanning."""

    def __init__(
        self,
        world: EnvironmentalWorld2D,
        applied: AppliedEnvironmentalPairPlan,
        *,
        config: EnvironmentalPairExecutionSessionConfig | None = None,
    ) -> None:
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if not isinstance(applied, AppliedEnvironmentalPairPlan):
            raise TypeError("applied must be AppliedEnvironmentalPairPlan")
        self.world = world
        self.applied = applied
        self.config = config or EnvironmentalPairExecutionSessionConfig()
        self.first_id = applied.plan.first_id
        self.second_id = applied.plan.second_id

        if abs(world.time - applied.applied_at_time) > _EPS:
            raise StaleEnvironmentalPairExecutionSession(
                "world time changed after environmental pair-plan application"
            )
        if world.snapshot() != applied.snapshot_after:
            raise StaleEnvironmentalPairExecutionSession(
                "world snapshot changed after environmental pair-plan application"
            )
        try:
            current_commands = (world.command_for(self.first_id), world.command_for(self.second_id))
        except KeyError as exc:
            raise StaleEnvironmentalPairExecutionSession("planned vessel is no longer active") from exc
        if current_commands != applied.commands_after:
            raise StaleEnvironmentalPairExecutionSession(
                "pair commands changed after environmental pair-plan application"
            )
        pending = world.next_command_time()
        if pending is not None and pending <= world.time + _EPS:
            raise StaleEnvironmentalPairExecutionSession(
                "unprocessed command event already exists at session start time"
            )

        evaluation = applied.plan.arbitration.selected.row.source_row.evaluation
        first_transition = evaluation.first_transition
        second_transition = evaluation.second_transition
        first_duration = first_transition.transition_duration_min
        second_duration = second_transition.transition_duration_min
        if not isfinite(first_duration) or not isfinite(second_duration):
            raise EnvironmentalPairExecutionSessionInvariantViolation(
                "applied accepted row contains an unreachable transition"
            )
        first_exact = bool(getattr(first_transition, "transition_duration_is_target_completion", True))
        second_exact = bool(getattr(second_transition, "transition_duration_is_target_completion", True))
        self.first_completion_time = (
            applied.applied_at_time + first_duration if first_exact else inf
        )
        self.second_completion_time = (
            applied.applied_at_time + second_duration if second_exact else inf
        )
        self.completion_time = max(self.first_completion_time, self.second_completion_time)

        self._next_review_time = (
            None
            if self.config.review_interval_min is None
            else applied.applied_at_time + self.config.review_interval_min
        )
        self._history: list[EnvironmentalPairExecutionObservation] = []
        self._terminal_reason: EnvironmentalPairExecutionStopReason | None = None
        self._last_snapshot = world.snapshot()
        self._last_commands = current_commands

        start_reasons = [EnvironmentalPairExecutionObservationReason.APPLIED]
        first_reached, second_reached = self._target_reached_flags()
        if self.first_completion_time <= world.time + _EPS and first_reached:
            start_reasons.append(EnvironmentalPairExecutionObservationReason.FIRST_TARGET_REACHED)
        if self.second_completion_time <= world.time + _EPS and second_reached:
            start_reasons.append(EnvironmentalPairExecutionObservationReason.SECOND_TARGET_REACHED)
        start = self._observation(tuple(start_reasons))
        self._history.append(start)
        if first_reached and second_reached and (
            not isfinite(self.completion_time) or self.completion_time <= world.time + _EPS
        ):
            self._terminal_reason = EnvironmentalPairExecutionStopReason.COMPLETE

    @property
    def history(self) -> tuple[EnvironmentalPairExecutionObservation, ...]:
        return tuple(self._history)

    @property
    def terminal_reason(self) -> EnvironmentalPairExecutionStopReason | None:
        return self._terminal_reason

    def _planned_inputs(self):
        context = self.applied.plan.planning_context
        return context.for_vessel(self.first_id), context.for_vessel(self.second_id)

    def _validate_session_freshness(self) -> None:
        if abs(self.world.time - self._last_snapshot.time) > _EPS:
            raise StaleEnvironmentalPairExecutionSession(
                "world advanced outside environmental pair execution session"
            )
        try:
            first_now = self.world.snapshot().by_id(self.first_id)
            second_now = self.world.snapshot().by_id(self.second_id)
            first_last = self._last_snapshot.by_id(self.first_id)
            second_last = self._last_snapshot.by_id(self.second_id)
            commands = (self.world.command_for(self.first_id), self.world.command_for(self.second_id))
            first_input, second_input = self._planned_inputs()
            if self.world.limits_for(self.first_id) != first_input.maneuver_limits:
                raise StaleEnvironmentalPairExecutionSession("first maneuver limits changed during execution")
            if self.world.limits_for(self.second_id) != second_input.maneuver_limits:
                raise StaleEnvironmentalPairExecutionSession("second maneuver limits changed during execution")
            if self.world.drift_for(self.first_id) != first_input.environmental_drift_velocity:
                raise StaleEnvironmentalPairExecutionSession("first environmental drift changed during execution")
            if self.world.drift_for(self.second_id) != second_input.environmental_drift_velocity:
                raise StaleEnvironmentalPairExecutionSession("second environmental drift changed during execution")
        except KeyError as exc:
            raise StaleEnvironmentalPairExecutionSession("planned vessel is no longer active") from exc
        if first_now != first_last or second_now != second_last:
            raise StaleEnvironmentalPairExecutionSession(
                "planned vessel physical state changed outside execution session"
            )
        if commands != self._last_commands:
            raise StaleEnvironmentalPairExecutionSession(
                "pair command changed outside execution session"
            )

    def _target_reached(self, state: WaterRelativeVesselState, command: WaterRelativeManeuverCommand) -> bool:
        return (
            abs(signed_heading_delta(state.through_water_heading_deg, command.target_through_water_heading_deg))
            <= _STATE_EPS
            and abs(state.through_water_speed - command.target_through_water_speed) <= _STATE_EPS
        )

    def _target_reached_flags(self) -> tuple[bool, bool]:
        snapshot = self.world.snapshot()
        first = snapshot.by_id(self.first_id)
        second = snapshot.by_id(self.second_id)
        return (
            self._target_reached(first, self.applied.plan.first_command),
            self._target_reached(second, self.applied.plan.second_command),
        )

    def _observation(
        self,
        reasons: tuple[EnvironmentalPairExecutionObservationReason, ...],
    ) -> EnvironmentalPairExecutionObservation:
        snapshot = self.world.snapshot()
        first = snapshot.by_id(self.first_id)
        second = snapshot.by_id(self.second_id)
        first_command = self.world.command_for(self.first_id)
        second_command = self.world.command_for(self.second_id)
        first_reached = self._target_reached(first, self.applied.plan.first_command)
        second_reached = self._target_reached(second, self.applied.plan.second_command)
        return EnvironmentalPairExecutionObservation(
            time_min=self.world.time,
            reasons=reasons,
            first_state=first,
            second_state=second,
            first_motion=environmental_vessel_motion_evidence(first, self.world.drift_for(self.first_id)),
            second_motion=environmental_vessel_motion_evidence(second, self.world.drift_for(self.second_id)),
            first_command=first_command,
            second_command=second_command,
            first_target_reached=first_reached,
            second_target_reached=second_reached,
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
        if isfinite(self.first_completion_time) and self.first_completion_time > now + _EPS:
            values.append(self.first_completion_time)
        if isfinite(self.second_completion_time) and self.second_completion_time > now + _EPS:
            values.append(self.second_completion_time)
        if self._next_review_time is not None and self._next_review_time > now + _EPS:
            values.append(self._next_review_time)
        command_time = self.world.next_command_time()
        if command_time is not None:
            if command_time <= now + _EPS:
                raise StaleEnvironmentalPairExecutionSession(
                    "unprocessed command event exists at current execution-session time"
                )
            values.append(command_time)
        return tuple(values)

    def next_boundary_time(self) -> float | None:
        """Return the next real execution-observation boundary without advancing time.

        This is an orchestration primitive for supervisors that must evaluate policy at
        every a93 boundary before the world is allowed to advance farther.  Arbitrary
        caller horizons are deliberately not reported as boundaries.
        """

        self._validate_session_freshness()
        if self._terminal_reason is not None:
            return None
        candidates = self._candidate_boundary_times()
        return min(candidates) if candidates else None

    def run_until(self, end_time: float) -> EnvironmentalPairExecutionSessionResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        self._validate_session_freshness()
        start_history_len = len(self._history)

        if self._terminal_reason is not None:
            return EnvironmentalPairExecutionSessionResult(
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
                self._last_commands = (
                    self.world.command_for(self.first_id),
                    self.world.command_for(self.second_id),
                )
                break

            boundary = min(next_boundary, end_time)
            command_time = self.world.next_command_time()
            command_event_due = command_time is not None and abs(command_time - boundary) <= _EPS
            first_due = abs(self.first_completion_time - boundary) <= _EPS
            second_due = abs(self.second_completion_time - boundary) <= _EPS
            review_due = self._next_review_time is not None and abs(self._next_review_time - boundary) <= _EPS

            self.world.run_until(boundary)
            reasons: list[EnvironmentalPairExecutionObservationReason] = []
            if review_due:
                reasons.append(EnvironmentalPairExecutionObservationReason.PERIODIC_REVIEW)
                self._advance_next_review_after(boundary)
            if command_event_due:
                reasons.append(EnvironmentalPairExecutionObservationReason.COMMAND_EVENT)

            first_reached, second_reached = self._target_reached_flags()
            if first_due:
                if not first_reached:
                    raise EnvironmentalPairExecutionSessionInvariantViolation(
                        "first selected command target was not reached at predicted transition completion"
                    )
                reasons.append(EnvironmentalPairExecutionObservationReason.FIRST_TARGET_REACHED)
            if second_due:
                if not second_reached:
                    raise EnvironmentalPairExecutionSessionInvariantViolation(
                        "second selected command target was not reached at predicted transition completion"
                    )
                reasons.append(EnvironmentalPairExecutionObservationReason.SECOND_TARGET_REACHED)

            current_commands = (
                self.world.command_for(self.first_id),
                self.world.command_for(self.second_id),
            )
            selected_commands = (self.applied.plan.first_command, self.applied.plan.second_command)
            overridden = current_commands != selected_commands
            complete = first_reached and second_reached and (
                not isfinite(self.completion_time) or boundary >= self.completion_time - _EPS
            )
            if overridden and not complete:
                reasons.append(EnvironmentalPairExecutionObservationReason.COMMAND_OVERRIDDEN)

            record = self._observation(tuple(reasons))
            self._history.append(record)
            self._last_snapshot = self.world.snapshot()
            self._last_commands = current_commands

            if complete:
                self._terminal_reason = EnvironmentalPairExecutionStopReason.COMPLETE
                break
            if overridden:
                self._terminal_reason = EnvironmentalPairExecutionStopReason.COMMAND_OVERRIDDEN
                break

        if self._terminal_reason is not None:
            stop = self._terminal_reason
        else:
            stop = EnvironmentalPairExecutionStopReason.END_TIME
        return EnvironmentalPairExecutionSessionResult(
            snapshot=self.world.snapshot(),
            stop_reason=stop,
            records=tuple(self._history[start_history_len:]),
        )
