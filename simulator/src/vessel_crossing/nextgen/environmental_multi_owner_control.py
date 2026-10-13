"""Shared-clock execution coordinator for several environmental PAIR/JOINT owners.

A117 is the first layer that actually *runs* the several disjoint owners created by
A116.  The owners still keep their existing a100/a102 and a112/a113 local state
machines, but physical time and fleet observation are owned here:

- one ``EnvironmentalWorld2D`` clock is advanced once to the next real global boundary;
- PAIR/JOINT execution observers are advanced coherently at that common timestamp;
- local lifecycle evidence is assimilated before topology is reconsidered;
- caller-owned membership events are applied once;
- exactly one canonical fleet-registry observation is produced for that physical time;
- every current owner is checked against that same observation before any due local
  policy may write a replacement command;
- if any owner has changed, the coordinator stops at a clean recapture boundary and no
  due owner policy is consumed.

A117 intentionally stops when a PAIR leaves AVOIDANCE for route handoff, or when a
JOINT owner reaches local handoff eligibility.  Shared post-avoidance route execution
and fleet-safe nominal handoff remain a later layer.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from math import isfinite
from typing import Callable, Iterable

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .encounter_classification import EncounterClassificationConfig
from .encounter_registry import EncounterRegistry, PairKey, canonical_pair_key
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_fleet_guarded_joint_control import EnvironmentalFleetGuardedJointControlSession
from .environmental_fleet_guarded_pair_control import EnvironmentalFleetGuardedPairControlSession
from .environmental_fleet_joint_ownership import EnvironmentalJointOwnershipAction
from .environmental_fleet_multi_owner_recapture import (
    EnvironmentalFleetMultiOwnerRecaptureAction,
    EnvironmentalFleetMultiOwnerRecaptureResult,
    EnvironmentalStartedJointOwner,
    EnvironmentalStartedOwner,
    EnvironmentalStartedPairOwner,
)
from .environmental_fleet_ownership import (
    EnvironmentalFleetOwnershipObservation,
    EnvironmentalPairOwnershipAction,
    observe_environmental_fleet_ownership,
)
from .environmental_joint_control_cycle import EnvironmentalJointControlStage
from .environmental_joint_execution_policy import decide_environmental_joint_execution_policy
from .environmental_joint_execution_session import (
    EnvironmentalJointExecutionObservationReason,
    EnvironmentalJointExecutionSessionInvariantViolation,
    EnvironmentalJointExecutionStopReason,
)
from .environmental_pair_control_cycle import EnvironmentalPairControlStage
from .environmental_pair_execution_policy import (
    EnvironmentalPairExecutionPolicyAction,
    decide_environmental_pair_execution_policy,
)
from .environmental_pair_execution_session import (
    EnvironmentalPairExecutionObservationReason,
    EnvironmentalPairExecutionSessionInvariantViolation,
    EnvironmentalPairExecutionStopReason,
)
from .environmental_pair_lifecycle_execution import EnvironmentalPairLifecycleExecutionRecord
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .responsibility import ResponsibilityContext

_EPS = 1e-10


class EnvironmentalMultiOwnerControlError(RuntimeError):
    """Base error for invalid A117 shared-clock orchestration."""


class EnvironmentalMultiOwnerControlStopReason(str, Enum):
    END_TIME = "end_time"
    RECAPTURE_REQUIRED = "recapture_required"
    OWNER_STAGE_TRANSFER_REQUIRED = "owner_stage_transfer_required"


class EnvironmentalMultiOwnerMembershipEventKind(str, Enum):
    VESSEL_ARRIVAL = "vessel_arrival"
    VESSEL_REMOVAL = "vessel_removal"


class EnvironmentalMultiOwnerMembershipApplicationOutcome(str, Enum):
    APPLIED = "applied"
    STALE_INCARNATION_SKIPPED = "stale_incarnation_skipped"


@dataclass(frozen=True, slots=True)
class EnvironmentalMultiOwnerMembershipApplication:
    """Audit evidence for one due membership event on the shared calendar."""

    time_min: float
    kind: EnvironmentalMultiOwnerMembershipEventKind
    vessel_id: str
    incarnation: int | None
    outcome: EnvironmentalMultiOwnerMembershipApplicationOutcome

    @property
    def applied(self) -> bool:
        return self.outcome is EnvironmentalMultiOwnerMembershipApplicationOutcome.APPLIED


@dataclass(frozen=True, slots=True)
class EnvironmentalMultiOwnerControlConfig:
    max_boundaries_per_run: int = 8192

    def __post_init__(self) -> None:
        if self.max_boundaries_per_run < 1:
            raise ValueError("max_boundaries_per_run must be at least 1")


@dataclass(order=True, frozen=True, slots=True)
class _MembershipEvent:
    at: float
    sequence: int
    kind: EnvironmentalMultiOwnerMembershipEventKind = field(compare=False)
    vessel: WaterRelativeVesselState | None = field(compare=False, default=None)
    vessel_id: str | None = field(compare=False, default=None)
    limits: ManeuverLimits | None = field(compare=False, default=None)
    drift: Vector2 | None = field(compare=False, default=None)
    incarnation: int | None = field(compare=False, default=None)


@dataclass(frozen=True, slots=True)
class EnvironmentalMultiOwnerBoundary:
    time_min: float
    execution_due_owner_ids: tuple[str, ...]
    fleet_review_due_owner_ids: tuple[str, ...]
    membership_event_count: int
    removed_vessel_ids: tuple[str, ...]
    fleet_observation: EnvironmentalFleetOwnershipObservation | None
    recapture_owner_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentalMultiOwnerControlResult:
    source: EnvironmentalFleetMultiOwnerRecaptureResult | None
    snapshot: EnvironmentalWorldSnapshot
    stop_reason: EnvironmentalMultiOwnerControlStopReason
    boundaries: tuple[EnvironmentalMultiOwnerBoundary, ...]
    fleet_observations: tuple[EnvironmentalFleetOwnershipObservation, ...]
    recapture_observation: EnvironmentalFleetOwnershipObservation | None
    recapture_owner_ids: tuple[str, ...]
    stage_transfer_owner_ids: tuple[str, ...]

    @property
    def recapture_required(self) -> bool:
        return self.stop_reason is EnvironmentalMultiOwnerControlStopReason.RECAPTURE_REQUIRED


class EnvironmentalMultiOwnerControlSession:
    """Run all A116 replacement owners on one environmental physical clock."""

    def __init__(
        self,
        registry: EncounterRegistry,
        world: EnvironmentalWorld2D,
        source: EnvironmentalFleetMultiOwnerRecaptureResult,
        *,
        config: EnvironmentalMultiOwnerControlConfig = EnvironmentalMultiOwnerControlConfig(),
    ) -> None:
        if not isinstance(registry, EncounterRegistry):
            raise TypeError("registry must be EncounterRegistry")
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if not isinstance(source, EnvironmentalFleetMultiOwnerRecaptureResult):
            raise TypeError("source must be EnvironmentalFleetMultiOwnerRecaptureResult")
        if source.action is not EnvironmentalFleetMultiOwnerRecaptureAction.MULTI_OWNER_CYCLES_STARTED:
            raise EnvironmentalMultiOwnerControlError(
                "a117 requires an a116 MULTI_OWNER_CYCLES_STARTED result"
            )
        if not isinstance(config, EnvironmentalMultiOwnerControlConfig):
            raise TypeError("config must be EnvironmentalMultiOwnerControlConfig")
        if world.snapshot() != source.snapshot_after or abs(world.time - source.snapshot_after.time) > _EPS:
            raise EnvironmentalMultiOwnerControlError(
                "world no longer matches the a116 multi-owner bootstrap boundary"
            )
        if len(source.started_owners) < 2:
            raise EnvironmentalMultiOwnerControlError("a117 requires at least two started owners")

        claimed: set[str] = set()
        for started in source.started_owners:
            overlap = claimed.intersection(started.owner.vessel_ids)
            if overlap:
                raise EnvironmentalMultiOwnerControlError(
                    f"a116 started owners overlap command channels: {sorted(overlap)!r}"
                )
            claimed.update(started.owner.vessel_ids)
            guard = started.guarded_session
            if guard.world is not world:
                raise EnvironmentalMultiOwnerControlError("all owners must share one EnvironmentalWorld2D")
            if getattr(guard, "_membership_events", None):
                raise EnvironmentalMultiOwnerControlError(
                    "owner-local membership queues must be empty before A117 takes calendar ownership"
                )
            if not getattr(guard, "_started", False):
                raise EnvironmentalMultiOwnerControlError("A116 owner guard must already be started")

        self.registry = registry
        self.world = world
        self.source = source
        self.config = config
        self.started_owners: tuple[EnvironmentalStartedOwner, ...] = source.started_owners
        self._membership_events: list[_MembershipEvent] = []
        self._counter = count()
        self._membership_applications: list[EnvironmentalMultiOwnerMembershipApplication] = []
        self._membership_application_hooks: list[Callable[[EnvironmentalMultiOwnerMembershipApplication], None]] = []
        self._active_incarnations: dict[str, int | None] = {
            vessel.vessel_id: None for vessel in world.snapshot().vessels
        }
        self._boundaries: list[EnvironmentalMultiOwnerBoundary] = []

        initial = source.source.source.recapture.fleet_observation  # a114 -> a113 recapture
        if abs(initial.time_min - world.time) > _EPS:
            raise EnvironmentalMultiOwnerControlError(
                "a116 committed fleet observation does not match coordinator start time"
            )
        self._fleet_observations: list[EnvironmentalFleetOwnershipObservation] = [initial]
        for started in self.started_owners:
            guard_obs = started.guarded_session.fleet_observations
            if not guard_obs or guard_obs[-1] != initial:
                raise EnvironmentalMultiOwnerControlError(
                    "all A116 owner guards must begin from the same committed fleet observation"
                )

        self._uncertain_pairs, self._responsibility_context, self._classification_config, self._direction_basis = (
            self._observation_settings()
        )

    @classmethod
    def from_idle_observation(
        cls,
        registry: EncounterRegistry,
        world: EnvironmentalWorld2D,
        observation: EnvironmentalFleetOwnershipObservation,
        *,
        uncertain_pairs: Iterable[PairKey] = (),
        responsibility_context: ResponsibilityContext | None = None,
        classification_config: EncounterClassificationConfig | None = None,
        direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
        config: EnvironmentalMultiOwnerControlConfig = EnvironmentalMultiOwnerControlConfig(),
    ) -> "EnvironmentalMultiOwnerControlSession":
        """Construct an ownerless shared coordinator from one already-committed observation.

        A124 uses this only for true cold start.  It performs no registry update and owns no
        command channel yet; A123 may later atomically attach fresh owners to this coordinator.
        """

        if not isinstance(registry, EncounterRegistry):
            raise TypeError("registry must be EncounterRegistry")
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if not isinstance(observation, EnvironmentalFleetOwnershipObservation):
            raise TypeError("observation must be EnvironmentalFleetOwnershipObservation")
        if not isinstance(config, EnvironmentalMultiOwnerControlConfig):
            raise TypeError("config must be EnvironmentalMultiOwnerControlConfig")
        if abs(observation.time_min - world.time) > _EPS:
            raise EnvironmentalMultiOwnerControlError(
                "idle coordinator observation must match current environmental world time"
            )
        if getattr(registry, "_last_update_time", None) is None or abs(
            registry._last_update_time - observation.time_min  # noqa: SLF001
        ) > _EPS:
            raise EnvironmentalMultiOwnerControlError(
                "idle coordinator requires an observation already committed by the registry"
            )
        if not isinstance(direction_basis, EnvironmentalEncounterDirectionBasis):
            raise TypeError("direction_basis must be EnvironmentalEncounterDirectionBasis")

        self = cls.__new__(cls)
        self.registry = registry
        self.world = world
        self.source = None
        self.config = config
        self.started_owners = ()
        self._membership_events = []
        self._counter = count()
        self._membership_applications = []
        self._membership_application_hooks = []
        self._active_incarnations = {vessel.vessel_id: None for vessel in world.snapshot().vessels}
        self._boundaries = []
        self._fleet_observations = [observation]
        self._uncertain_pairs = tuple(sorted(canonical_pair_key(*key) for key in uncertain_pairs))
        self._responsibility_context = responsibility_context
        self._classification_config = classification_config
        self._direction_basis = direction_basis
        return self

    @property
    def fleet_observations(self) -> tuple[EnvironmentalFleetOwnershipObservation, ...]:
        return tuple(self._fleet_observations)

    @property
    def boundaries(self) -> tuple[EnvironmentalMultiOwnerBoundary, ...]:
        return tuple(self._boundaries)

    @property
    def membership_applications(self) -> tuple[EnvironmentalMultiOwnerMembershipApplication, ...]:
        return tuple(self._membership_applications)

    def add_membership_application_hook(
        self,
        hook: Callable[[EnvironmentalMultiOwnerMembershipApplication], None],
    ) -> None:
        if not callable(hook):
            raise TypeError("membership application hook must be callable")
        if hook not in self._membership_application_hooks:
            self._membership_application_hooks.append(hook)

    def _record_membership_application(
        self,
        application: EnvironmentalMultiOwnerMembershipApplication,
    ) -> None:
        self._membership_applications.append(application)
        for hook in tuple(self._membership_application_hooks):
            hook(application)

    def active_incarnation_for(self, vessel_id: str) -> int | None:
        if vessel_id not in {item.vessel_id for item in self.world.snapshot().vessels}:
            raise KeyError(vessel_id)
        return self._active_incarnations.get(vessel_id)

    def note_external_vessel_removal(self, vessel_id: str) -> int | None:
        """Retire incarnation identity after another runtime-owned physical writer removed a vessel."""

        if vessel_id in {item.vessel_id for item in self.world.snapshot().vessels}:
            raise EnvironmentalMultiOwnerControlError(
                f"cannot retire incarnation identity while vessel {vessel_id!r} remains active"
            )
        return self._active_incarnations.pop(vessel_id, None)

    def _adopt_replacement_owner_set(
        self,
        started_owners: tuple[EnvironmentalStartedOwner, ...],
        *,
        committed_observation: EnvironmentalFleetOwnershipObservation,
    ) -> None:
        """Atomically adopt an already-published replacement owner set.

        A118 owns planning and the combined same-timestamp command transaction.  This
        method deliberately performs no registry read and no command write; it only
        switches the runtime sessions after the transaction has succeeded.
        """

        if not started_owners:
            # A120 may retire the final avoidance owner while route owners continue on
            # the same shared clock.  Keep the existing observation settings so this
            # coordinator can still produce canonical fleet evidence for route guards.
            self.started_owners = ()
            return
        if abs(committed_observation.time_min - self.world.time) > _EPS:
            raise EnvironmentalMultiOwnerControlError(
                "replacement owner observation must match the current shared-clock timestamp"
            )
        if not self._fleet_observations or self._fleet_observations[-1] != committed_observation:
            raise EnvironmentalMultiOwnerControlError(
                "replacement owner observation is not the coordinator's committed canonical observation"
            )

        claimed: set[str] = set()
        for started in started_owners:
            overlap = claimed.intersection(started.owner.vessel_ids)
            if overlap:
                raise EnvironmentalMultiOwnerControlError(
                    f"replacement owners overlap command channels: {sorted(overlap)!r}"
                )
            claimed.update(started.owner.vessel_ids)
            guard = started.guarded_session
            if guard.world is not self.world:
                raise EnvironmentalMultiOwnerControlError(
                    "replacement owners must share the coordinator EnvironmentalWorld2D"
                )
            if getattr(guard, "_membership_events", None):
                raise EnvironmentalMultiOwnerControlError(
                    "replacement owner-local membership queues must be empty"
                )
            if not getattr(guard, "_started", False):
                raise EnvironmentalMultiOwnerControlError(
                    "replacement owner guard must already be started"
                )
            observations = guard.fleet_observations
            if not observations or observations[-1] != committed_observation:
                raise EnvironmentalMultiOwnerControlError(
                    "replacement owner guard is not bound to the committed canonical observation"
                )

        self.started_owners = started_owners
        (
            self._uncertain_pairs,
            self._responsibility_context,
            self._classification_config,
            self._direction_basis,
        ) = self._observation_settings()

    def _observation_settings(self):
        uncertain: set[PairKey] = set()
        contexts: list[ResponsibilityContext | None] = []
        classifications: list[EncounterClassificationConfig | None] = []
        directions: list[EnvironmentalEncounterDirectionBasis] = []
        for started in self.started_owners:
            if isinstance(started, EnvironmentalStartedPairOwner):
                cycle = started.pair_cycle
                if cycle.uncertain:
                    uncertain.add(canonical_pair_key(*started.owner.vessel_ids))
            else:
                cycle = started.joint_cycle
                uncertain.update(canonical_pair_key(*key) for key in cycle.uncertain_pairs)
            contexts.append(cycle.responsibility_context)
            classifications.append(cycle.classification_config)
            directions.append(cycle.direction_basis)
        if any(item != contexts[0] for item in contexts[1:]):
            raise EnvironmentalMultiOwnerControlError(
                "all coordinated owners must share one responsibility_context"
            )
        if any(item != classifications[0] for item in classifications[1:]):
            raise EnvironmentalMultiOwnerControlError(
                "all coordinated owners must share one encounter classification config"
            )
        if any(item is not directions[0] for item in directions[1:]):
            raise EnvironmentalMultiOwnerControlError(
                "all coordinated owners must share one environmental direction basis"
            )
        return tuple(sorted(uncertain)), contexts[0], classifications[0], directions[0]

    def _validate_future_membership_time(self, at: float) -> None:
        if not isfinite(at):
            raise ValueError("membership event time must be finite")
        if at <= self.world.time + _EPS:
            raise ValueError("membership event time must be strictly after current world time")

    @staticmethod
    def _validate_incarnation(incarnation: int | None) -> None:
        if incarnation is not None and (not isinstance(incarnation, int) or isinstance(incarnation, bool) or incarnation < 1):
            raise ValueError("incarnation must be a positive integer or None")

    def schedule_arrival(
        self,
        *,
        at: float,
        vessel: WaterRelativeVesselState,
        limits: ManeuverLimits | None = None,
        environmental_drift_velocity: Vector2 | None = None,
        incarnation: int | None = None,
    ) -> None:
        self._validate_future_membership_time(at)
        self._validate_incarnation(incarnation)
        if not isinstance(vessel, WaterRelativeVesselState):
            raise TypeError("vessel must be WaterRelativeVesselState")
        heapq.heappush(
            self._membership_events,
            _MembershipEvent(
                float(at),
                next(self._counter),
                EnvironmentalMultiOwnerMembershipEventKind.VESSEL_ARRIVAL,
                vessel=vessel,
                limits=limits,
                drift=environmental_drift_velocity,
                incarnation=incarnation,
            ),
        )

    def schedule_removal(
        self,
        *,
        at: float,
        vessel_id: str,
        incarnation: int | None = None,
    ) -> None:
        self._validate_future_membership_time(at)
        self._validate_incarnation(incarnation)
        if not vessel_id:
            raise ValueError("vessel_id must be non-empty")
        heapq.heappush(
            self._membership_events,
            _MembershipEvent(
                float(at),
                next(self._counter),
                EnvironmentalMultiOwnerMembershipEventKind.VESSEL_REMOVAL,
                vessel_id=vessel_id,
                incarnation=incarnation,
            ),
        )

    def schedule_incarnation_episode(
        self,
        *,
        at: float,
        vessel: WaterRelativeVesselState,
        incarnation: int,
        limits: ManeuverLimits | None = None,
        environmental_drift_velocity: Vector2 | None = None,
        remove_at: float | None = None,
    ) -> None:
        """Atomically queue one generation-tagged arrival and its optional departure."""

        self._validate_future_membership_time(at)
        self._validate_incarnation(incarnation)
        if not isinstance(vessel, WaterRelativeVesselState):
            raise TypeError("vessel must be WaterRelativeVesselState")
        if remove_at is not None:
            self._validate_future_membership_time(remove_at)
            if remove_at <= at + _EPS:
                raise ValueError("incarnation removal time must be strictly after arrival time")

        events = [
            _MembershipEvent(
                float(at),
                next(self._counter),
                EnvironmentalMultiOwnerMembershipEventKind.VESSEL_ARRIVAL,
                vessel=vessel,
                limits=limits,
                drift=environmental_drift_velocity,
                incarnation=incarnation,
            )
        ]
        if remove_at is not None:
            events.append(
                _MembershipEvent(
                    float(remove_at),
                    next(self._counter),
                    EnvironmentalMultiOwnerMembershipEventKind.VESSEL_REMOVAL,
                    vessel_id=vessel.vessel_id,
                    incarnation=incarnation,
                )
            )
        combined = [*self._membership_events, *events]
        heapq.heapify(combined)
        self._membership_events = combined

    @staticmethod
    def _membership_event_vessel_id(event: _MembershipEvent) -> str:
        if event.vessel is not None:
            return event.vessel.vessel_id
        assert event.vessel_id is not None
        return event.vessel_id

    def cancel_incarnation_events(self, *, vessel_id: str, incarnation: int) -> int:
        """Atomically remove all not-yet-applied calendar events for one incarnation."""

        if not vessel_id:
            raise ValueError("vessel_id must be non-empty")
        self._validate_incarnation(incarnation)
        if self._active_incarnations.get(vessel_id) == incarnation:
            raise EnvironmentalMultiOwnerControlError(
                f"cannot cancel membership events for active incarnation {incarnation} of {vessel_id!r}"
            )
        matching = tuple(
            event
            for event in self._membership_events
            if event.incarnation == incarnation and self._membership_event_vessel_id(event) == vessel_id
        )
        if not matching:
            raise EnvironmentalMultiOwnerControlError(
                f"no pending membership events for incarnation {incarnation} of {vessel_id!r}"
            )
        doomed = set(matching)
        replacement = [event for event in self._membership_events if event not in doomed]
        heapq.heapify(replacement)
        self._membership_events = replacement
        return len(matching)

    def amend_incarnation_episode_events(
        self,
        *,
        vessel: WaterRelativeVesselState,
        incarnation: int,
        at: float,
        limits: ManeuverLimits | None = None,
        environmental_drift_velocity: Vector2 | None = None,
        remove_at: float | None = None,
    ) -> None:
        """Replace a future generation's queued episode while preserving event order keys."""

        if not isinstance(vessel, WaterRelativeVesselState):
            raise TypeError("vessel must be WaterRelativeVesselState")
        self._validate_incarnation(incarnation)
        self._validate_future_membership_time(at)
        if remove_at is not None:
            self._validate_future_membership_time(remove_at)
            if remove_at <= at + _EPS:
                raise ValueError("incarnation removal time must be strictly after arrival time")
        vessel_id = vessel.vessel_id
        if self._active_incarnations.get(vessel_id) == incarnation:
            raise EnvironmentalMultiOwnerControlError(
                f"cannot amend active incarnation {incarnation} of {vessel_id!r}"
            )

        matching = tuple(
            event
            for event in self._membership_events
            if event.incarnation == incarnation and self._membership_event_vessel_id(event) == vessel_id
        )
        arrivals = tuple(
            event for event in matching
            if event.kind is EnvironmentalMultiOwnerMembershipEventKind.VESSEL_ARRIVAL
        )
        removals = tuple(
            event for event in matching
            if event.kind is EnvironmentalMultiOwnerMembershipEventKind.VESSEL_REMOVAL
        )
        if len(arrivals) != 1 or len(removals) > 1 or len(matching) != len(arrivals) + len(removals):
            raise EnvironmentalMultiOwnerControlError(
                f"pending incarnation {incarnation} of {vessel_id!r} does not have one canonical episode"
            )

        replacement_events = [
            _MembershipEvent(
                float(at),
                arrivals[0].sequence,
                EnvironmentalMultiOwnerMembershipEventKind.VESSEL_ARRIVAL,
                vessel=vessel,
                limits=limits,
                drift=environmental_drift_velocity,
                incarnation=incarnation,
            )
        ]
        if remove_at is not None:
            removal_sequence = removals[0].sequence if removals else next(self._counter)
            replacement_events.append(
                _MembershipEvent(
                    float(remove_at),
                    removal_sequence,
                    EnvironmentalMultiOwnerMembershipEventKind.VESSEL_REMOVAL,
                    vessel_id=vessel_id,
                    incarnation=incarnation,
                )
            )

        doomed = set(matching)
        replacement = [event for event in self._membership_events if event not in doomed]
        replacement.extend(replacement_events)
        heapq.heapify(replacement)
        self._membership_events = replacement

    def _next_membership_time(self) -> float | None:
        return self._membership_events[0].at if self._membership_events else None

    def _apply_due_membership(self) -> tuple[int, tuple[str, ...]]:
        count_events = 0
        removed: list[str] = []
        while self._membership_events and self._membership_events[0].at <= self.world.time + _EPS:
            event = heapq.heappop(self._membership_events)
            if event.at < self.world.time - _EPS:
                raise EnvironmentalMultiOwnerControlError(
                    "membership boundary was skipped by the shared coordinator"
                )
            count_events += 1
            if event.kind is EnvironmentalMultiOwnerMembershipEventKind.VESSEL_ARRIVAL:
                assert event.vessel is not None
                vessel_id = event.vessel.vessel_id
                if event.incarnation is not None and vessel_id in self._active_incarnations:
                    self._record_membership_application(
                        EnvironmentalMultiOwnerMembershipApplication(
                            self.world.time,
                            event.kind,
                            vessel_id,
                            event.incarnation,
                            EnvironmentalMultiOwnerMembershipApplicationOutcome.STALE_INCARNATION_SKIPPED,
                        )
                    )
                    continue
                self.world.add_vessel(
                    event.vessel,
                    limits=event.limits,
                    environmental_drift_velocity=event.drift,
                )
                self._active_incarnations[vessel_id] = event.incarnation
                self._record_membership_application(
                    EnvironmentalMultiOwnerMembershipApplication(
                        self.world.time,
                        event.kind,
                        vessel_id,
                        event.incarnation,
                        EnvironmentalMultiOwnerMembershipApplicationOutcome.APPLIED,
                    )
                )
            else:
                assert event.vessel_id is not None
                vessel_id = event.vessel_id
                if event.incarnation is not None and self._active_incarnations.get(vessel_id) != event.incarnation:
                    self._record_membership_application(
                        EnvironmentalMultiOwnerMembershipApplication(
                            self.world.time,
                            event.kind,
                            vessel_id,
                            event.incarnation,
                            EnvironmentalMultiOwnerMembershipApplicationOutcome.STALE_INCARNATION_SKIPPED,
                        )
                    )
                    continue
                effective_incarnation = self._active_incarnations.get(vessel_id)
                self.world.remove_vessel(vessel_id)
                self._active_incarnations.pop(vessel_id, None)
                removed.append(vessel_id)
                self._record_membership_application(
                    EnvironmentalMultiOwnerMembershipApplication(
                        self.world.time,
                        event.kind,
                        vessel_id,
                        event.incarnation if event.incarnation is not None else effective_incarnation,
                        EnvironmentalMultiOwnerMembershipApplicationOutcome.APPLIED,
                    )
                )
        return count_events, tuple(removed)

    def _execution_boundary_time(self, started: EnvironmentalStartedOwner) -> float | None:
        if isinstance(started, EnvironmentalStartedPairOwner):
            cycle = started.pair_cycle
            if cycle.stage is not EnvironmentalPairControlStage.AVOIDANCE:
                return None
            if cycle._pending_policy is not None:  # noqa: SLF001 - package orchestration primitive
                return self.world.time
            return cycle._execution_session.next_boundary_time()  # noqa: SLF001
        cycle = started.joint_cycle
        if cycle.stage is not EnvironmentalJointControlStage.AVOIDANCE:
            return None
        if cycle._pending_policy is not None:  # noqa: SLF001
            return self.world.time
        return cycle._execution_session.next_boundary_time()  # noqa: SLF001

    def _stage_transfer_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for started in self.started_owners:
            if isinstance(started, EnvironmentalStartedPairOwner):
                if started.pair_cycle.stage is not EnvironmentalPairControlStage.AVOIDANCE:
                    ids.append(started.owner.component_id)
            elif started.joint_cycle.stage is not EnvironmentalJointControlStage.AVOIDANCE:
                ids.append(started.owner.component_id)
        return tuple(ids)

    def _shared_observation(self) -> EnvironmentalFleetOwnershipObservation:
        if self._fleet_observations and abs(self._fleet_observations[-1].time_min - self.world.time) <= _EPS:
            return self._fleet_observations[-1]
        observation = observe_environmental_fleet_ownership(
            self.registry,
            self.world,
            uncertain_pairs=self._uncertain_pairs,
            responsibility_context=self._responsibility_context,
            classification_config=self._classification_config,
            direction_basis=self._direction_basis,
        )
        self._fleet_observations.append(observation)
        for started in self.started_owners:
            guard = started.guarded_session
            if not guard._fleet_observations or abs(guard._fleet_observations[-1].time_min - self.world.time) > _EPS:  # noqa: SLF001
                guard._fleet_observations.append(observation)  # noqa: SLF001
        return observation

    @staticmethod
    def _controlled_removed(started: EnvironmentalStartedOwner, removed: tuple[str, ...]) -> tuple[str, ...]:
        controlled = set(started.owner.vessel_ids)
        return tuple(vessel_id for vessel_id in removed if vessel_id in controlled)

    def _assess_all_owners(
        self,
        removed: tuple[str, ...],
    ) -> tuple[EnvironmentalFleetOwnershipObservation, tuple[str, ...]]:
        observation = self._shared_observation()
        recapture: list[str] = []
        for started in self.started_owners:
            controlled_removed = self._controlled_removed(started, removed)
            if isinstance(started, EnvironmentalStartedPairOwner):
                guard: EnvironmentalFleetGuardedPairControlSession = started.guarded_session
                pending = started.pair_cycle._pending_policy  # noqa: SLF001
                requested = None if pending is None else pending.action
                decision = guard._assess_current(  # noqa: SLF001
                    started.pair_cycle.stage,
                    requested_policy_action=requested,
                    removed_controlled=controlled_removed,
                )
                if guard.recapture is not None or (
                    decision is not None and decision.action is EnvironmentalPairOwnershipAction.RECAPTURE_REQUIRED
                ):
                    recapture.append(started.owner.component_id)
            else:
                guard_j: EnvironmentalFleetGuardedJointControlSession = started.guarded_session
                pending = started.joint_cycle._pending_policy  # noqa: SLF001
                requested = None if pending is None else pending.action
                decision_j = guard_j._assess_current(  # noqa: SLF001
                    requested_policy_action=requested,
                    removed_controlled=controlled_removed,
                )
                if guard_j.recapture is not None or (
                    decision_j is not None and decision_j.action is EnvironmentalJointOwnershipAction.RECAPTURE_REQUIRED
                ):
                    recapture.append(started.owner.component_id)
        return observation, tuple(recapture)

    @staticmethod
    def _pair_adopt_shared_advance(
        started: EnvironmentalStartedPairOwner,
        *,
        due: bool,
        command_event_due: bool,
    ) -> None:
        cycle = started.pair_cycle
        execution = cycle._execution_session  # noqa: SLF001
        world = cycle.world
        if not due:
            try:
                commands = (world.command_for(execution.first_id), world.command_for(execution.second_id))
            except KeyError as exc:
                raise EnvironmentalMultiOwnerControlError(
                    "PAIR vessel disappeared during shared physical advance"
                ) from exc
            if commands != execution._last_commands:  # noqa: SLF001
                raise EnvironmentalMultiOwnerControlError(
                    "non-due PAIR command changed during shared physical advance"
                )
            execution._last_snapshot = world.snapshot()  # noqa: SLF001
            execution._last_commands = commands  # noqa: SLF001
            return

        boundary = world.time
        reasons: list[EnvironmentalPairExecutionObservationReason] = []
        review_due = execution._next_review_time is not None and abs(execution._next_review_time - boundary) <= _EPS  # noqa: SLF001
        if review_due:
            reasons.append(EnvironmentalPairExecutionObservationReason.PERIODIC_REVIEW)
            execution._advance_next_review_after(boundary)  # noqa: SLF001
        if command_event_due:
            reasons.append(EnvironmentalPairExecutionObservationReason.COMMAND_EVENT)
        first_due = abs(execution.first_completion_time - boundary) <= _EPS
        second_due = abs(execution.second_completion_time - boundary) <= _EPS
        first_reached, second_reached = execution._target_reached_flags()  # noqa: SLF001
        if first_due:
            if not first_reached:
                raise EnvironmentalPairExecutionSessionInvariantViolation(
                    "first selected command target was not reached at shared transition completion"
                )
            reasons.append(EnvironmentalPairExecutionObservationReason.FIRST_TARGET_REACHED)
        if second_due:
            if not second_reached:
                raise EnvironmentalPairExecutionSessionInvariantViolation(
                    "second selected command target was not reached at shared transition completion"
                )
            reasons.append(EnvironmentalPairExecutionObservationReason.SECOND_TARGET_REACHED)

        commands = (world.command_for(execution.first_id), world.command_for(execution.second_id))
        selected = (execution.applied.plan.first_command, execution.applied.plan.second_command)
        overridden = commands != selected
        complete = first_reached and second_reached and (
            not isfinite(execution.completion_time) or boundary >= execution.completion_time - _EPS
        )
        if overridden and not complete:
            reasons.append(EnvironmentalPairExecutionObservationReason.COMMAND_OVERRIDDEN)
        record = execution._observation(tuple(reasons))  # noqa: SLF001
        execution._history.append(record)  # noqa: SLF001
        execution._last_snapshot = world.snapshot()  # noqa: SLF001
        execution._last_commands = commands  # noqa: SLF001
        if complete:
            execution._terminal_reason = EnvironmentalPairExecutionStopReason.COMPLETE  # noqa: SLF001
        elif overridden:
            execution._terminal_reason = EnvironmentalPairExecutionStopReason.COMMAND_OVERRIDDEN  # noqa: SLF001

        wrapped = cycle._lifecycle_execution  # noqa: SLF001
        decision = wrapped._assimilate(record)  # noqa: SLF001
        combined = EnvironmentalPairLifecycleExecutionRecord(record, decision)
        wrapped._history.append(combined)  # noqa: SLF001
        wrapped._latest_decision = decision  # noqa: SLF001
        cycle._latest_decision = decision  # noqa: SLF001
        policy = decide_environmental_pair_execution_policy(combined)
        cycle._policy_history.append(policy)  # noqa: SLF001
        cycle._pending_policy = policy  # noqa: SLF001

    @staticmethod
    def _joint_adopt_shared_advance(
        started: EnvironmentalStartedJointOwner,
        *,
        due: bool,
        command_event_due: bool,
    ) -> None:
        cycle = started.joint_cycle
        execution = cycle._execution_session  # noqa: SLF001
        world = cycle.world
        if not due:
            try:
                commands = tuple((v, world.command_for(v)) for v in execution.vessel_ids)
            except KeyError as exc:
                raise EnvironmentalMultiOwnerControlError(
                    "JOINT vessel disappeared during shared physical advance"
                ) from exc
            if commands != execution._last_commands:  # noqa: SLF001
                raise EnvironmentalMultiOwnerControlError(
                    "non-due JOINT command changed during shared physical advance"
                )
            execution._last_snapshot = world.snapshot()  # noqa: SLF001
            execution._last_commands = commands  # noqa: SLF001
            return

        boundary = world.time
        reasons: list[EnvironmentalJointExecutionObservationReason] = []
        review_due = execution._next_review_time is not None and abs(execution._next_review_time - boundary) <= _EPS  # noqa: SLF001
        if review_due:
            reasons.append(EnvironmentalJointExecutionObservationReason.PERIODIC_REVIEW)
            execution._advance_next_review_after(boundary)  # noqa: SLF001
        if command_event_due:
            reasons.append(EnvironmentalJointExecutionObservationReason.COMMAND_EVENT)
        due_ids = tuple(
            vessel_id
            for vessel_id, completion_time in execution.target_completion_times
            if abs(completion_time - boundary) <= _EPS
        )
        reached = execution._target_reached_ids()  # noqa: SLF001
        if due_ids:
            missing = tuple(vessel_id for vessel_id in due_ids if vessel_id not in reached)
            if missing:
                details = []
                snap = world.snapshot()
                for vessel_id in missing:
                    state = snap.by_id(vessel_id)
                    command = execution.applied.plan.command_for(vessel_id)
                    details.append((
                        vessel_id, boundary, execution.completion_time_for(vessel_id),
                        state.through_water_heading_deg, command.target_through_water_heading_deg,
                        state.through_water_speed, command.target_through_water_speed,
                    ))
                raise EnvironmentalJointExecutionSessionInvariantViolation(
                    f"selected JOINT targets not reached at shared completion: {missing}; details={details}; all_completion_times={execution.target_completion_times}"
                )
            reasons.append(EnvironmentalJointExecutionObservationReason.TARGET_REACHED)
        commands = tuple((v, world.command_for(v)) for v in execution.vessel_ids)
        selected = execution.applied.plan.selected_commands
        complete = len(reached) == len(execution.vessel_ids) and (
            not isfinite(execution.completion_time) or boundary >= execution.completion_time - _EPS
        )
        overridden = commands != selected
        if overridden and not complete:
            reasons.append(EnvironmentalJointExecutionObservationReason.COMMAND_OVERRIDDEN)
        record = execution._observation(tuple(reasons))  # noqa: SLF001
        execution._history.append(record)  # noqa: SLF001
        execution._last_snapshot = world.snapshot()  # noqa: SLF001
        execution._last_commands = commands  # noqa: SLF001
        if complete:
            execution._terminal_reason = EnvironmentalJointExecutionStopReason.COMPLETE  # noqa: SLF001
        elif overridden:
            execution._terminal_reason = EnvironmentalJointExecutionStopReason.COMMAND_OVERRIDDEN  # noqa: SLF001

        wrapped = cycle._lifecycle_execution  # noqa: SLF001
        combined = wrapped._assimilate_record(record)  # noqa: SLF001
        cycle._latest_pair_decisions = combined.pair_decisions  # noqa: SLF001
        policy = decide_environmental_joint_execution_policy(combined)
        cycle._policy_history.append(policy)  # noqa: SLF001
        cycle._pending_policy = policy  # noqa: SLF001

    def _sync_all_after_shared_advance(
        self,
        execution_times: dict[str, float | None],
        *,
        command_event_due: bool,
    ) -> tuple[str, ...]:
        due_ids: list[str] = []
        for started in self.started_owners:
            due = (
                execution_times[started.owner.component_id] is not None
                and abs(execution_times[started.owner.component_id] - self.world.time) <= _EPS
            )
            if due:
                due_ids.append(started.owner.component_id)
            if isinstance(started, EnvironmentalStartedPairOwner):
                self._pair_adopt_shared_advance(started, due=due, command_event_due=command_event_due)
            else:
                self._joint_adopt_shared_advance(started, due=due, command_event_due=command_event_due)
        return tuple(due_ids)

    def _drain_current_policies(self) -> tuple[str, ...]:
        """Consume zero-time local policies only after shared ownership has been admitted."""
        for _ in range(128):
            changed = False
            for started in self.started_owners:
                if isinstance(started, EnvironmentalStartedPairOwner):
                    cycle = started.pair_cycle
                    if cycle.stage is EnvironmentalPairControlStage.AVOIDANCE and cycle._pending_policy is not None:  # noqa: SLF001
                        cycle._apply_pending_policy()  # noqa: SLF001
                        changed = True
                else:
                    cycle = started.joint_cycle
                    if cycle.stage is EnvironmentalJointControlStage.AVOIDANCE and cycle._pending_policy is not None:  # noqa: SLF001
                        cycle._apply_pending_policy()  # noqa: SLF001
                        changed = True
            transfer = self._stage_transfer_ids()
            if transfer:
                return transfer
            if not changed:
                return ()
        raise EnvironmentalMultiOwnerControlError("too many zero-time owner policy transitions")

    def _advance_guard_reviews(self) -> tuple[str, ...]:
        due: list[str] = []
        for started in self.started_owners:
            guard = started.guarded_session
            if guard.next_review_time is not None and guard.next_review_time <= self.world.time + _EPS:
                due.append(started.owner.component_id)
                guard._advance_review_after_current()  # noqa: SLF001
        return tuple(due)

    def _result(
        self,
        reason: EnvironmentalMultiOwnerControlStopReason,
        *,
        recapture_observation: EnvironmentalFleetOwnershipObservation | None = None,
        recapture_owner_ids: tuple[str, ...] = (),
        stage_transfer_owner_ids: tuple[str, ...] = (),
    ) -> EnvironmentalMultiOwnerControlResult:
        return EnvironmentalMultiOwnerControlResult(
            self.source,
            self.world.snapshot(),
            reason,
            self.boundaries,
            self.fleet_observations,
            recapture_observation,
            recapture_owner_ids,
            stage_transfer_owner_ids,
        )

    def run_until(self, end_time: float) -> EnvironmentalMultiOwnerControlResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")

        # A116 already admitted every owner at the current timestamp using the same
        # canonical observation.  Drain their initial APPLIED policies without creating
        # a duplicate registry update.
        transfer = self._drain_current_policies()
        if transfer:
            return self._result(
                EnvironmentalMultiOwnerControlStopReason.OWNER_STAGE_TRANSFER_REQUIRED,
                stage_transfer_owner_ids=transfer,
            )

        for _ in range(self.config.max_boundaries_per_run):
            if self.world.time >= end_time - _EPS:
                return self._result(EnvironmentalMultiOwnerControlStopReason.END_TIME)

            execution_times = {
                started.owner.component_id: self._execution_boundary_time(started)
                for started in self.started_owners
            }
            # Pending policies must have been drained above; a current-time execution
            # boundary here would otherwise produce a zero-time scheduler loop.
            if any(
                time is not None and time <= self.world.time + _EPS
                for time in execution_times.values()
            ):
                raise EnvironmentalMultiOwnerControlError(
                    "owner reports a non-future execution boundary after policy drain"
                )

            candidates: list[float] = [end_time]
            candidates.extend(time for time in execution_times.values() if time is not None)
            candidates.extend(
                guard.next_review_time
                for guard in (started.guarded_session for started in self.started_owners)
                if guard.next_review_time is not None
            )
            membership_time = self._next_membership_time()
            if membership_time is not None:
                candidates.append(membership_time)
            boundary = min(candidates)
            real_boundary = boundary < end_time - _EPS or any(
                time is not None and abs(time - boundary) <= _EPS
                for time in execution_times.values()
            ) or any(
                started.guarded_session.next_review_time is not None
                and abs(started.guarded_session.next_review_time - boundary) <= _EPS
                for started in self.started_owners
            ) or (membership_time is not None and abs(membership_time - boundary) <= _EPS)

            command_time = self.world.next_command_time()
            command_event_due = command_time is not None and abs(command_time - boundary) <= _EPS
            self.world.run_until(boundary)
            execution_due = self._sync_all_after_shared_advance(
                execution_times,
                command_event_due=command_event_due,
            )

            if not real_boundary:
                return self._result(EnvironmentalMultiOwnerControlStopReason.END_TIME)

            membership_count, removed = self._apply_due_membership()

            # Preserve the original a102/a113 observation contracts while sharing one
            # registry call. PAIR CONTINUE boundaries do not need a fleet read merely
            # because a93 reviewed execution; PAIR replans do. JOINT a113, by contrast,
            # gates every real a110 policy boundary, including CONTINUE. Fleet review
            # and membership boundaries always require the canonical observation.
            review_due = tuple(
                started.owner.component_id
                for started in self.started_owners
                if started.guarded_session.next_review_time is not None
                and started.guarded_session.next_review_time <= self.world.time + _EPS
            )
            policy_observation_required = False
            for started in self.started_owners:
                if isinstance(started, EnvironmentalStartedJointOwner):
                    if started.joint_cycle._pending_policy is not None:  # noqa: SLF001
                        policy_observation_required = True
                        break
                else:
                    pending = started.pair_cycle._pending_policy  # noqa: SLF001
                    if (
                        pending is not None
                        and pending.action is EnvironmentalPairExecutionPolicyAction.REPLAN_REQUIRED
                    ):
                        policy_observation_required = True
                        break

            observation = None
            recapture_ids: tuple[str, ...] = ()
            if membership_count or review_due or policy_observation_required:
                observation, recapture_ids = self._assess_all_owners(removed)
            if review_due:
                advanced = self._advance_guard_reviews()
                if advanced != review_due:
                    raise EnvironmentalMultiOwnerControlError(
                        "fleet-review due set changed while advancing shared review calendars"
                    )

            record = EnvironmentalMultiOwnerBoundary(
                self.world.time,
                execution_due,
                review_due,
                membership_count,
                removed,
                observation,
                recapture_ids,
            )
            self._boundaries.append(record)

            if recapture_ids:
                assert observation is not None
                return self._result(
                    EnvironmentalMultiOwnerControlStopReason.RECAPTURE_REQUIRED,
                    recapture_observation=observation,
                    recapture_owner_ids=recapture_ids,
                )

            transfer = self._drain_current_policies()
            if transfer:
                return self._result(
                    EnvironmentalMultiOwnerControlStopReason.OWNER_STAGE_TRANSFER_REQUIRED,
                    stage_transfer_owner_ids=transfer,
                )

            if self.world.time >= end_time - _EPS:
                return self._result(EnvironmentalMultiOwnerControlStopReason.END_TIME)

        raise EnvironmentalMultiOwnerControlError(
            f"a117 exceeded max_boundaries_per_run={self.config.max_boundaries_per_run}"
        )
