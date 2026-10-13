'''Single-clock fleet-guarded wrapper around the a100 environmental pair cycle.

A102 turns the a101 current-time ownership decision into an actual transfer boundary.
It does not add a second simulation clock and it does not implement JOINT planning.  One
``EnvironmentalWorld2D`` remains the only physical clock/command-event owner.

The wrapper adds two fleet-level boundary classes around an otherwise unchanged a100
cycle:

- fixed fleet ownership reviews;
- caller-scheduled vessel membership arrivals/removals.

A100/a98/a99 expose an optional pre-command-write hook in a102.  The hook applies any
membership event due at the current timestamp, observes the environmental encounter
registry exactly once at that timestamp, and runs the a101 ownership gate *before* the
local PAIR writer may mutate commands.  ``RECAPTURE_REQUIRED`` therefore stops local
ownership at the boundary rather than one command later.

A102 deliberately does not choose a PAIR/JOINT replacement controller after recapture.
It returns the current canonical fleet dispatch to the higher-level dispatcher.
'''

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from math import isfinite

from .dynamics import ManeuverLimits
from .encounter_registry import EncounterRegistry
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_fleet_ownership import (
    EnvironmentalFleetOwnershipObservation,
    EnvironmentalPairOwnershipAction,
    EnvironmentalPairOwnershipDecision,
    assess_environmental_pair_cycle_ownership,
    observe_environmental_fleet_ownership,
)
from .environmental_pair_control_cycle import (
    EnvironmentalPairControlCycleResult,
    EnvironmentalPairControlSession,
    EnvironmentalPairControlStage,
    EnvironmentalPairControlStopReason,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from ..kinematics import Vector2

_EPS = 1e-10


class EnvironmentalFleetGuardedPairError(RuntimeError):
    '''Base error for a102 fleet-guarded pair orchestration failures.'''


class EnvironmentalFleetGuardedPairStopReason(str, Enum):
    END_TIME = 'end_time'
    ROUTE_COMPLETE = 'route_complete'
    COMMAND_OVERRIDDEN = 'command_overridden'
    RECAPTURE_REQUIRED = 'recapture_required'


class EnvironmentalFleetRecaptureReason(str, Enum):
    OWNERSHIP_CHANGED = 'ownership_changed'
    CONTROLLED_VESSEL_REMOVED = 'controlled_vessel_removed'


class EnvironmentalFleetMembershipEventKind(str, Enum):
    VESSEL_ARRIVAL = 'vessel_arrival'
    VESSEL_REMOVAL = 'vessel_removal'


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetGuardedPairConfig:
    review_interval_min: float
    max_boundaries_per_run: int = 8192

    def __post_init__(self) -> None:
        if not isfinite(self.review_interval_min) or self.review_interval_min <= 0.0:
            raise ValueError('review_interval_min must be finite and positive')
        if self.max_boundaries_per_run < 1:
            raise ValueError('max_boundaries_per_run must be at least 1')


@dataclass(order=True, frozen=True, slots=True)
class _MembershipEvent:
    at: float
    sequence: int
    kind: EnvironmentalFleetMembershipEventKind = field(compare=False)
    vessel: WaterRelativeVesselState | None = field(compare=False, default=None)
    vessel_id: str | None = field(compare=False, default=None)
    limits: ManeuverLimits | None = field(compare=False, default=None)
    drift: Vector2 | None = field(compare=False, default=None)


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRecapture:
    time_min: float
    reason: EnvironmentalFleetRecaptureReason
    fleet_observation: EnvironmentalFleetOwnershipObservation
    ownership_decision: EnvironmentalPairOwnershipDecision | None
    removed_controlled_vessel_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetGuardedPairResult:
    snapshot: EnvironmentalWorldSnapshot
    stop_reason: EnvironmentalFleetGuardedPairStopReason
    pair_result: EnvironmentalPairControlCycleResult
    recapture: EnvironmentalFleetRecapture | None
    fleet_observations: tuple[EnvironmentalFleetOwnershipObservation, ...]
    ownership_decisions: tuple[EnvironmentalPairOwnershipDecision, ...]

    @property
    def recapture_required(self) -> bool:
        return self.stop_reason is EnvironmentalFleetGuardedPairStopReason.RECAPTURE_REQUIRED

    @property
    def route_complete(self) -> bool:
        return self.stop_reason is EnvironmentalFleetGuardedPairStopReason.ROUTE_COMPLETE


class _RecaptureSignal(RuntimeError):
    def __init__(self, recapture: EnvironmentalFleetRecapture) -> None:
        super().__init__(recapture.reason.value)
        self.recapture = recapture


class EnvironmentalFleetGuardedPairControlSession:
    '''Run one a100 PAIR cycle under current-time a101 fleet ownership checks.'''

    def __init__(
        self,
        registry: EncounterRegistry,
        pair_cycle: EnvironmentalPairControlSession,
        *,
        config: EnvironmentalFleetGuardedPairConfig,
        initial_observation: EnvironmentalFleetOwnershipObservation | None = None,
    ) -> None:
        if not isinstance(registry, EncounterRegistry):
            raise TypeError('registry must be EncounterRegistry')
        if not isinstance(pair_cycle, EnvironmentalPairControlSession):
            raise TypeError('pair_cycle must be EnvironmentalPairControlSession')
        if not isinstance(config, EnvironmentalFleetGuardedPairConfig):
            raise TypeError('config must be EnvironmentalFleetGuardedPairConfig')
        if pair_cycle.before_command_write is not None:
            raise EnvironmentalFleetGuardedPairError(
                'pair cycle already has a command-write guard installed'
            )
        if abs(pair_cycle.world.time - pair_cycle.latest_decision.observation.time_min) > _EPS:
            raise EnvironmentalFleetGuardedPairError(
                'a102 must attach before the a100 pair cycle advances beyond its initial boundary'
            )

        self.registry = registry
        self.pair_cycle = pair_cycle
        self.world: EnvironmentalWorld2D = pair_cycle.world
        self.config = config
        self.pair_ids = tuple(item.vessel_id for item in pair_cycle.route_targets)
        if len(self.pair_ids) != 2:
            raise EnvironmentalFleetGuardedPairError('a102 requires exactly two controlled vessel ids')

        self._started = False
        self._next_review_time: float | None = None
        self._membership_events: list[_MembershipEvent] = []
        self._counter = count()
        self._fleet_observations: list[EnvironmentalFleetOwnershipObservation] = []
        self._ownership_decisions: list[EnvironmentalPairOwnershipDecision] = []
        self._recapture: EnvironmentalFleetRecapture | None = None
        self._initial_observation = initial_observation

        if initial_observation is not None:
            if not isinstance(initial_observation, EnvironmentalFleetOwnershipObservation):
                raise TypeError('initial_observation must be EnvironmentalFleetOwnershipObservation or None')
            if abs(initial_observation.time_min - self.world.time) > _EPS:
                raise EnvironmentalFleetGuardedPairError(
                    'initial fleet ownership observation must match current world time'
                )
            self._fleet_observations.append(initial_observation)

        # The a100 hook is intentionally attached only after every a102 field required by
        # the callback is initialized.
        self.pair_cycle.before_command_write = self._before_local_command_write

    @property
    def fleet_observations(self) -> tuple[EnvironmentalFleetOwnershipObservation, ...]:
        return tuple(self._fleet_observations)

    @property
    def ownership_decisions(self) -> tuple[EnvironmentalPairOwnershipDecision, ...]:
        return tuple(self._ownership_decisions)

    @property
    def recapture(self) -> EnvironmentalFleetRecapture | None:
        return self._recapture

    @property
    def next_review_time(self) -> float | None:
        return self._next_review_time

    def _validate_future_membership_time(self, at: float) -> None:
        if not isfinite(at):
            raise ValueError('membership event time must be finite')
        if at <= self.world.time + _EPS:
            raise ValueError('membership event time must be strictly after current world time')

    def schedule_arrival(
        self,
        *,
        at: float,
        vessel: WaterRelativeVesselState,
        limits: ManeuverLimits | None = None,
        environmental_drift_velocity: Vector2 | None = None,
    ) -> None:
        self._validate_future_membership_time(at)
        if not isinstance(vessel, WaterRelativeVesselState):
            raise TypeError('vessel must be WaterRelativeVesselState')
        heapq.heappush(
            self._membership_events,
            _MembershipEvent(
                float(at),
                next(self._counter),
                EnvironmentalFleetMembershipEventKind.VESSEL_ARRIVAL,
                vessel=vessel,
                limits=limits,
                drift=environmental_drift_velocity,
            ),
        )

    def schedule_removal(self, *, at: float, vessel_id: str) -> None:
        self._validate_future_membership_time(at)
        if not vessel_id:
            raise ValueError('vessel_id must be non-empty')
        heapq.heappush(
            self._membership_events,
            _MembershipEvent(
                float(at),
                next(self._counter),
                EnvironmentalFleetMembershipEventKind.VESSEL_REMOVAL,
                vessel_id=vessel_id,
            ),
        )

    def _apply_due_membership(self) -> tuple[str, ...]:
        removed_controlled: list[str] = []
        while self._membership_events and self._membership_events[0].at <= self.world.time + _EPS:
            event = heapq.heappop(self._membership_events)
            if event.at < self.world.time - _EPS:
                raise EnvironmentalFleetGuardedPairError(
                    'membership boundary was skipped by the fleet-guarded scheduler'
                )
            if event.kind is EnvironmentalFleetMembershipEventKind.VESSEL_ARRIVAL:
                assert event.vessel is not None
                self.world.add_vessel(
                    event.vessel,
                    limits=event.limits,
                    environmental_drift_velocity=event.drift,
                )
            elif event.kind is EnvironmentalFleetMembershipEventKind.VESSEL_REMOVAL:
                assert event.vessel_id is not None
                self.world.remove_vessel(event.vessel_id)
                if event.vessel_id in self.pair_ids:
                    removed_controlled.append(event.vessel_id)
            else:  # pragma: no cover
                raise AssertionError(event.kind)
        return tuple(removed_controlled)

    def _observe_current(self) -> EnvironmentalFleetOwnershipObservation:
        if self._fleet_observations and abs(self._fleet_observations[-1].time_min - self.world.time) <= _EPS:
            return self._fleet_observations[-1]
        observation = observe_environmental_fleet_ownership(self.registry, self.world)
        self._fleet_observations.append(observation)
        return observation

    def _assess_current(
        self,
        stage: EnvironmentalPairControlStage,
        *,
        requested_policy_action=None,
        removed_controlled: tuple[str, ...] = (),
    ) -> EnvironmentalPairOwnershipDecision | None:
        observation = self._observe_current()
        if removed_controlled:
            self._recapture = EnvironmentalFleetRecapture(
                self.world.time,
                EnvironmentalFleetRecaptureReason.CONTROLLED_VESSEL_REMOVED,
                observation,
                None,
                removed_controlled,
            )
            return None

        decision = assess_environmental_pair_cycle_ownership(
            stage,
            self.pair_ids[0],
            self.pair_ids[1],
            observation.dispatch,
            requested_policy_action=requested_policy_action,
        )
        self._ownership_decisions.append(decision)
        if decision.action is EnvironmentalPairOwnershipAction.RECAPTURE_REQUIRED:
            self._recapture = EnvironmentalFleetRecapture(
                self.world.time,
                EnvironmentalFleetRecaptureReason.OWNERSHIP_CHANGED,
                observation,
                decision,
            )
        return decision

    def _before_local_command_write(self, stage: EnvironmentalPairControlStage) -> None:
        removed = self._apply_due_membership()
        pending = self.pair_cycle._pending_policy  # noqa: SLF001
        requested = None if pending is None else pending.action
        decision = self._assess_current(
            stage,
            requested_policy_action=requested,
            removed_controlled=removed,
        )
        if self._recapture is not None:
            raise _RecaptureSignal(self._recapture)
        if decision is None:  # pragma: no cover - only controlled removal produces None
            raise AssertionError('missing ownership decision without recapture')

    def _advance_review_after_current(self) -> None:
        if self._next_review_time is None:
            return
        while self._next_review_time <= self.world.time + _EPS:
            self._next_review_time += self.config.review_interval_min

    def _next_membership_time(self) -> float | None:
        return self._membership_events[0].at if self._membership_events else None

    def start(self) -> EnvironmentalPairOwnershipDecision | None:
        if self._started:
            raise EnvironmentalFleetGuardedPairError('a102 session is already started')
        self._started = True
        removed = self._apply_due_membership()
        decision = self._assess_current(self.pair_cycle.stage, removed_controlled=removed)
        self._next_review_time = self.world.time + self.config.review_interval_min
        if self._recapture is not None:
            return decision
        return decision

    def _pair_result_at_boundary(self) -> EnvironmentalPairControlCycleResult:
        return self.pair_cycle.result(EnvironmentalPairControlStopReason.END_TIME)

    def _result(
        self,
        reason: EnvironmentalFleetGuardedPairStopReason,
        pair_result: EnvironmentalPairControlCycleResult,
    ) -> EnvironmentalFleetGuardedPairResult:
        return EnvironmentalFleetGuardedPairResult(
            snapshot=self.world.snapshot(),
            stop_reason=reason,
            pair_result=pair_result,
            recapture=self._recapture,
            fleet_observations=self.fleet_observations,
            ownership_decisions=self.ownership_decisions,
        )

    @staticmethod
    def _map_pair_stop(reason: EnvironmentalPairControlStopReason) -> EnvironmentalFleetGuardedPairStopReason:
        if reason is EnvironmentalPairControlStopReason.END_TIME:
            return EnvironmentalFleetGuardedPairStopReason.END_TIME
        if reason is EnvironmentalPairControlStopReason.ROUTE_COMPLETE:
            return EnvironmentalFleetGuardedPairStopReason.ROUTE_COMPLETE
        if reason is EnvironmentalPairControlStopReason.COMMAND_OVERRIDDEN:
            return EnvironmentalFleetGuardedPairStopReason.COMMAND_OVERRIDDEN
        raise AssertionError(reason)

    def run_until(self, end_time: float) -> EnvironmentalFleetGuardedPairResult:
        if not isfinite(end_time):
            raise ValueError('end_time must be finite')
        if end_time < self.world.time - _EPS:
            raise ValueError('end_time must not move backwards')
        if not self._started:
            self.start()
        if self._recapture is not None:
            return self._result(
                EnvironmentalFleetGuardedPairStopReason.RECAPTURE_REQUIRED,
                self._pair_result_at_boundary(),
            )

        for _ in range(self.config.max_boundaries_per_run):
            if self.world.time >= end_time - _EPS:
                return self._result(
                    EnvironmentalFleetGuardedPairStopReason.END_TIME,
                    self._pair_result_at_boundary(),
                )

            candidates = [end_time]
            if self._next_review_time is not None:
                candidates.append(self._next_review_time)
            membership_time = self._next_membership_time()
            if membership_time is not None:
                candidates.append(membership_time)
            horizon = min(candidates)

            try:
                pair_result = self.pair_cycle.run_until(horizon)
            except _RecaptureSignal as signal:
                self._recapture = signal.recapture
                return self._result(
                    EnvironmentalFleetGuardedPairStopReason.RECAPTURE_REQUIRED,
                    self._pair_result_at_boundary(),
                )

            # Pair terminals may occur before the fleet horizon (for example an external
            # post-avoidance override). Assimilate one fleet ownership snapshot at the
            # actual terminal timestamp when possible, then return the local terminal.
            if pair_result.stop_reason is not EnvironmentalPairControlStopReason.END_TIME:
                removed = self._apply_due_membership()
                if not self._fleet_observations or self._fleet_observations[-1].time_min < self.world.time - _EPS:
                    self._assess_current(self.pair_cycle.stage, removed_controlled=removed)
                    if self._recapture is not None:
                        return self._result(
                            EnvironmentalFleetGuardedPairStopReason.RECAPTURE_REQUIRED,
                            pair_result,
                        )
                return self._result(self._map_pair_stop(pair_result.stop_reason), pair_result)

            # A fleet review/membership boundary is caller-owned by a102. If the a100
            # command guard already observed this timestamp, the cached observation is
            # reused; EncounterRegistry therefore still advances at most once per time.
            removed = self._apply_due_membership()
            fleet_due = (
                (self._next_review_time is not None and self._next_review_time <= self.world.time + _EPS)
                or bool(removed)
                or (membership_time is not None and membership_time <= self.world.time + _EPS)
            )
            if fleet_due:
                self._assess_current(self.pair_cycle.stage, removed_controlled=removed)
                self._advance_review_after_current()
                if self._recapture is not None:
                    return self._result(
                        EnvironmentalFleetGuardedPairStopReason.RECAPTURE_REQUIRED,
                        pair_result,
                    )

            if self.world.time >= end_time - _EPS:
                return self._result(EnvironmentalFleetGuardedPairStopReason.END_TIME, pair_result)

        raise EnvironmentalFleetGuardedPairError(
            f'a102 exceeded max_boundaries_per_run={self.config.max_boundaries_per_run}'
        )
