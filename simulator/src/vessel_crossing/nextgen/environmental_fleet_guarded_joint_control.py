"""Single-clock fleet/topology guard around the a112 environmental JOINT cycle.

A113 turns current-time fleet ownership into a hard execution boundary for a112.  It
uses the same ``EnvironmentalWorld2D`` clock and adds only fleet review / membership
calendar boundaries.  No replacement backend is selected here.

The a112 ``before_policy_action`` hook is the critical pre-write/pre-handoff gate.  At
all real a110 boundaries it applies due membership changes, observes the canonical fleet
registry at most once for that physical timestamp, and checks the exact JOINT owner
before a112 may continue, replan, or enter local handoff eligibility.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from math import isfinite

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .encounter_registry import EncounterRegistry, PairKey
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_fleet_joint_ownership import (
    EnvironmentalJointOwnershipAction,
    EnvironmentalJointOwnershipDecision,
    assess_environmental_joint_cycle_ownership,
)
from .environmental_fleet_ownership import (
    EnvironmentalFleetOwnershipObservation,
    observe_environmental_fleet_ownership,
)
from .environmental_joint_control_cycle import (
    EnvironmentalJointControlCycleResult,
    EnvironmentalJointControlSession,
    EnvironmentalJointControlStopReason,
)
from .environmental_joint_execution_policy import EnvironmentalJointExecutionPolicyDecision
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot

_EPS = 1e-10


class EnvironmentalFleetGuardedJointError(RuntimeError):
    """Base error for a113 fleet-guarded JOINT orchestration failures."""


class EnvironmentalFleetGuardedJointStopReason(str, Enum):
    END_TIME = "end_time"
    HANDOFF_ELIGIBLE = "handoff_eligible"
    RECAPTURE_REQUIRED = "recapture_required"


class EnvironmentalFleetJointRecaptureReason(str, Enum):
    OWNERSHIP_CHANGED = "ownership_changed"
    CONTROLLED_VESSEL_REMOVED = "controlled_vessel_removed"


class EnvironmentalFleetJointMembershipEventKind(str, Enum):
    VESSEL_ARRIVAL = "vessel_arrival"
    VESSEL_REMOVAL = "vessel_removal"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetGuardedJointConfig:
    review_interval_min: float
    max_boundaries_per_run: int = 8192

    def __post_init__(self) -> None:
        if not isfinite(self.review_interval_min) or self.review_interval_min <= 0.0:
            raise ValueError("review_interval_min must be finite and positive")
        if self.max_boundaries_per_run < 1:
            raise ValueError("max_boundaries_per_run must be at least 1")


@dataclass(order=True, frozen=True, slots=True)
class _MembershipEvent:
    at: float
    sequence: int
    kind: EnvironmentalFleetJointMembershipEventKind = field(compare=False)
    vessel: WaterRelativeVesselState | None = field(compare=False, default=None)
    vessel_id: str | None = field(compare=False, default=None)
    limits: ManeuverLimits | None = field(compare=False, default=None)
    drift: Vector2 | None = field(compare=False, default=None)


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetJointRecapture:
    time_min: float
    reason: EnvironmentalFleetJointRecaptureReason
    fleet_observation: EnvironmentalFleetOwnershipObservation
    ownership_decision: EnvironmentalJointOwnershipDecision | None
    removed_controlled_vessel_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetGuardedJointResult:
    snapshot: EnvironmentalWorldSnapshot
    stop_reason: EnvironmentalFleetGuardedJointStopReason
    joint_result: EnvironmentalJointControlCycleResult
    recapture: EnvironmentalFleetJointRecapture | None
    fleet_observations: tuple[EnvironmentalFleetOwnershipObservation, ...]
    ownership_decisions: tuple[EnvironmentalJointOwnershipDecision, ...]

    @property
    def recapture_required(self) -> bool:
        return self.stop_reason is EnvironmentalFleetGuardedJointStopReason.RECAPTURE_REQUIRED


class _JointRecaptureSignal(RuntimeError):
    def __init__(self, recapture: EnvironmentalFleetJointRecapture) -> None:
        super().__init__(recapture.reason.value)
        self.recapture = recapture


class EnvironmentalFleetGuardedJointControlSession:
    """Run one a112 JOINT cycle under current-time canonical fleet ownership checks."""

    def __init__(
        self,
        registry: EncounterRegistry,
        joint_cycle: EnvironmentalJointControlSession,
        *,
        config: EnvironmentalFleetGuardedJointConfig,
        initial_observation: EnvironmentalFleetOwnershipObservation | None = None,
    ) -> None:
        if not isinstance(registry, EncounterRegistry):
            raise TypeError("registry must be EncounterRegistry")
        if not isinstance(joint_cycle, EnvironmentalJointControlSession):
            raise TypeError("joint_cycle must be EnvironmentalJointControlSession")
        if not isinstance(config, EnvironmentalFleetGuardedJointConfig):
            raise TypeError("config must be EnvironmentalFleetGuardedJointConfig")
        if joint_cycle.before_policy_action is not None:
            raise EnvironmentalFleetGuardedJointError(
                "JOINT cycle already has a policy-boundary guard installed"
            )

        self.registry = registry
        self.joint_cycle = joint_cycle
        self.world: EnvironmentalWorld2D = joint_cycle.world
        self.config = config
        plan = joint_cycle.current_applied.plan
        self.component_id = plan.component_id
        self.vessel_ids = plan.vessel_ids
        self.pair_keys: tuple[PairKey, ...] = tuple(plan.planning_context.pair_keys)

        self._started = False
        self._next_review_time: float | None = None
        self._membership_events: list[_MembershipEvent] = []
        self._counter = count()
        self._fleet_observations: list[EnvironmentalFleetOwnershipObservation] = []
        self._ownership_decisions: list[EnvironmentalJointOwnershipDecision] = []
        self._recapture: EnvironmentalFleetJointRecapture | None = None

        if initial_observation is not None:
            if not isinstance(initial_observation, EnvironmentalFleetOwnershipObservation):
                raise TypeError("initial_observation must be EnvironmentalFleetOwnershipObservation or None")
            if abs(initial_observation.time_min - self.world.time) > _EPS:
                raise EnvironmentalFleetGuardedJointError(
                    "initial fleet ownership observation must match current world time"
                )
            self._fleet_observations.append(initial_observation)

        joint_cycle.before_policy_action = self._before_policy_action

    @property
    def fleet_observations(self) -> tuple[EnvironmentalFleetOwnershipObservation, ...]:
        return tuple(self._fleet_observations)

    @property
    def ownership_decisions(self) -> tuple[EnvironmentalJointOwnershipDecision, ...]:
        return tuple(self._ownership_decisions)

    @property
    def recapture(self) -> EnvironmentalFleetJointRecapture | None:
        return self._recapture

    @property
    def next_review_time(self) -> float | None:
        return self._next_review_time

    def _validate_future_membership_time(self, at: float) -> None:
        if not isfinite(at):
            raise ValueError("membership event time must be finite")
        if at <= self.world.time + _EPS:
            raise ValueError("membership event time must be strictly after current world time")

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
            raise TypeError("vessel must be WaterRelativeVesselState")
        heapq.heappush(
            self._membership_events,
            _MembershipEvent(
                float(at),
                next(self._counter),
                EnvironmentalFleetJointMembershipEventKind.VESSEL_ARRIVAL,
                vessel=vessel,
                limits=limits,
                drift=environmental_drift_velocity,
            ),
        )

    def schedule_removal(self, *, at: float, vessel_id: str) -> None:
        self._validate_future_membership_time(at)
        if not vessel_id:
            raise ValueError("vessel_id must be non-empty")
        heapq.heappush(
            self._membership_events,
            _MembershipEvent(
                float(at),
                next(self._counter),
                EnvironmentalFleetJointMembershipEventKind.VESSEL_REMOVAL,
                vessel_id=vessel_id,
            ),
        )

    def _apply_due_membership(self) -> tuple[str, ...]:
        removed_controlled: list[str] = []
        while self._membership_events and self._membership_events[0].at <= self.world.time + _EPS:
            event = heapq.heappop(self._membership_events)
            if event.at < self.world.time - _EPS:
                raise EnvironmentalFleetGuardedJointError(
                    "membership boundary was skipped by the fleet-guarded scheduler"
                )
            if event.kind is EnvironmentalFleetJointMembershipEventKind.VESSEL_ARRIVAL:
                assert event.vessel is not None
                self.world.add_vessel(
                    event.vessel,
                    limits=event.limits,
                    environmental_drift_velocity=event.drift,
                )
            elif event.kind is EnvironmentalFleetJointMembershipEventKind.VESSEL_REMOVAL:
                assert event.vessel_id is not None
                self.world.remove_vessel(event.vessel_id)
                if event.vessel_id in self.vessel_ids:
                    removed_controlled.append(event.vessel_id)
            else:  # pragma: no cover
                raise AssertionError(event.kind)
        return tuple(removed_controlled)

    def _observe_current(self) -> EnvironmentalFleetOwnershipObservation:
        if self._fleet_observations and abs(self._fleet_observations[-1].time_min - self.world.time) <= _EPS:
            return self._fleet_observations[-1]
        observation = observe_environmental_fleet_ownership(
            self.registry,
            self.world,
            uncertain_pairs=self.joint_cycle.uncertain_pairs,
            responsibility_context=self.joint_cycle.responsibility_context,
            classification_config=self.joint_cycle.classification_config,
            direction_basis=self.joint_cycle.direction_basis,
        )
        self._fleet_observations.append(observation)
        return observation

    def _assess_current(
        self,
        *,
        requested_policy_action=None,
        removed_controlled: tuple[str, ...] = (),
    ) -> EnvironmentalJointOwnershipDecision | None:
        observation = self._observe_current()
        if removed_controlled:
            self._recapture = EnvironmentalFleetJointRecapture(
                self.world.time,
                EnvironmentalFleetJointRecaptureReason.CONTROLLED_VESSEL_REMOVED,
                observation,
                None,
                removed_controlled,
            )
            return None

        decision = assess_environmental_joint_cycle_ownership(
            self.component_id,
            self.vessel_ids,
            self.pair_keys,
            observation.dispatch,
            requested_policy_action=requested_policy_action,
        )
        self._ownership_decisions.append(decision)
        if decision.action is EnvironmentalJointOwnershipAction.RECAPTURE_REQUIRED:
            self._recapture = EnvironmentalFleetJointRecapture(
                self.world.time,
                EnvironmentalFleetJointRecaptureReason.OWNERSHIP_CHANGED,
                observation,
                decision,
            )
        return decision

    def _before_policy_action(self, policy: EnvironmentalJointExecutionPolicyDecision) -> None:
        removed = self._apply_due_membership()
        self._assess_current(
            requested_policy_action=policy.action,
            removed_controlled=removed,
        )
        if self._recapture is not None:
            raise _JointRecaptureSignal(self._recapture)

    def _advance_review_after_current(self) -> None:
        if self._next_review_time is None:
            return
        while self._next_review_time <= self.world.time + _EPS:
            self._next_review_time += self.config.review_interval_min

    def _next_membership_time(self) -> float | None:
        return self._membership_events[0].at if self._membership_events else None

    def start(self) -> EnvironmentalJointOwnershipDecision | None:
        if self._started:
            raise EnvironmentalFleetGuardedJointError("a113 session is already started")
        self._started = True
        removed = self._apply_due_membership()
        decision = self._assess_current(removed_controlled=removed)
        self._next_review_time = self.world.time + self.config.review_interval_min
        return decision

    def _joint_result_at_boundary(self) -> EnvironmentalJointControlCycleResult:
        return self.joint_cycle.result(EnvironmentalJointControlStopReason.END_TIME)

    def _result(
        self,
        reason: EnvironmentalFleetGuardedJointStopReason,
        joint_result: EnvironmentalJointControlCycleResult,
    ) -> EnvironmentalFleetGuardedJointResult:
        return EnvironmentalFleetGuardedJointResult(
            self.world.snapshot(),
            reason,
            joint_result,
            self._recapture,
            self.fleet_observations,
            self.ownership_decisions,
        )

    def run_until(self, end_time: float) -> EnvironmentalFleetGuardedJointResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        if not self._started:
            self.start()
        if self._recapture is not None:
            return self._result(
                EnvironmentalFleetGuardedJointStopReason.RECAPTURE_REQUIRED,
                self._joint_result_at_boundary(),
            )

        for _ in range(self.config.max_boundaries_per_run):
            if self.world.time >= end_time - _EPS:
                return self._result(
                    EnvironmentalFleetGuardedJointStopReason.END_TIME,
                    self._joint_result_at_boundary(),
                )

            candidates = [end_time]
            if self._next_review_time is not None:
                candidates.append(self._next_review_time)
            membership_time = self._next_membership_time()
            if membership_time is not None:
                candidates.append(membership_time)
            horizon = min(candidates)

            try:
                joint_result = self.joint_cycle.run_until(horizon)
            except _JointRecaptureSignal as signal:
                self._recapture = signal.recapture
                return self._result(
                    EnvironmentalFleetGuardedJointStopReason.RECAPTURE_REQUIRED,
                    self._joint_result_at_boundary(),
                )

            if joint_result.stop_reason is EnvironmentalJointControlStopReason.HANDOFF_ELIGIBLE:
                return self._result(
                    EnvironmentalFleetGuardedJointStopReason.HANDOFF_ELIGIBLE,
                    joint_result,
                )

            removed = self._apply_due_membership()
            fleet_due = (
                (self._next_review_time is not None and self._next_review_time <= self.world.time + _EPS)
                or bool(removed)
                or (membership_time is not None and membership_time <= self.world.time + _EPS)
            )
            if fleet_due:
                self._assess_current(removed_controlled=removed)
                self._advance_review_after_current()
                if self._recapture is not None:
                    return self._result(
                        EnvironmentalFleetGuardedJointStopReason.RECAPTURE_REQUIRED,
                        joint_result,
                    )

            if self.world.time >= end_time - _EPS:
                return self._result(EnvironmentalFleetGuardedJointStopReason.END_TIME, joint_result)

        raise EnvironmentalFleetGuardedJointError(
            f"a113 exceeded max_boundaries_per_run={self.config.max_boundaries_per_run}"
        )
