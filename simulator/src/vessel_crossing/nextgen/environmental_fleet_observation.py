"""Persistent read-only fleet observation calendar for ``EnvironmentalWorld2D``.

A81 vertically exercises the a80 environmental registry across physical time without
introducing maneuver selection or command ownership.  The session merges three kinds of
observation boundary on one monotonic calendar:

- periodic review times;
- water-relative command event times already queued in ``EnvironmentalWorld2D``;
- explicit vessel arrival/removal times owned by this observation session.

All boundary causes at one timestamp are applied first and produce exactly one registry
observation, admission batch and encounter-component batch.  The session never creates
or changes a vessel maneuver command itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
from itertools import count
from math import isfinite
from typing import Iterable

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .encounter_classification import EncounterClassificationConfig
from .encounter_components import EncounterComponentBatch, build_active_encounter_components
from .encounter_registry import EncounterRegistry, EncounterRegistryUpdate, PairKey
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_registry import update_environmental_registry
from .environmental_world import EnvironmentalWorld2D
from .responsibility import ResponsibilityContext
from .supervision_admission import SupervisionAdmissionBatch, admit_supervision

_EPS = 1e-12


class EnvironmentalFleetObservationReason(str, Enum):
    INITIAL = "initial"
    PERIODIC_REVIEW = "periodic_review"
    COMMAND_EVENT = "command_event"
    VESSEL_ARRIVAL = "vessel_arrival"
    VESSEL_REMOVAL = "vessel_removal"


_REASON_ORDER = (
    EnvironmentalFleetObservationReason.INITIAL,
    EnvironmentalFleetObservationReason.PERIODIC_REVIEW,
    EnvironmentalFleetObservationReason.COMMAND_EVENT,
    EnvironmentalFleetObservationReason.VESSEL_ARRIVAL,
    EnvironmentalFleetObservationReason.VESSEL_REMOVAL,
)


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetObservationRecord:
    time_min: float
    reasons: tuple[EnvironmentalFleetObservationReason, ...]
    registry_update: EncounterRegistryUpdate
    admission: SupervisionAdmissionBatch
    components: EncounterComponentBatch


@dataclass(order=True, frozen=True, slots=True)
class _MembershipEvent:
    at: float
    sequence: int
    kind: EnvironmentalFleetObservationReason = field(compare=False)
    vessel: WaterRelativeVesselState | None = field(compare=False, default=None)
    vessel_id: str | None = field(compare=False, default=None)
    limits: ManeuverLimits | None = field(compare=False, default=None)
    environmental_drift_velocity: Vector2 | None = field(compare=False, default=None)


class EnvironmentalFleetObservationSession:
    """Read-only persistent registry/admission/component observation session.

    ``review_interval_min`` is a model/scheduler cadence, not a controller time step.
    Physical motion remains event-driven in ``EnvironmentalWorld2D``.  The session only
    decides when to sample fleet encounter state and never schedules maneuver commands.
    """

    def __init__(
        self,
        world: EnvironmentalWorld2D,
        registry: EncounterRegistry,
        *,
        review_interval_min: float,
        uncertain_pairs: Iterable[PairKey] = (),
        responsibility_context: ResponsibilityContext | None = None,
        classification_config: EncounterClassificationConfig | None = None,
        direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    ) -> None:
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if not isinstance(registry, EncounterRegistry):
            raise TypeError("registry must be EncounterRegistry")
        if not isfinite(review_interval_min) or review_interval_min <= 0.0:
            raise ValueError("review_interval_min must be finite and positive")
        self.world = world
        self.registry = registry
        self.review_interval_min = float(review_interval_min)
        self.uncertain_pairs = tuple(uncertain_pairs)
        self.responsibility_context = responsibility_context
        self.classification_config = classification_config
        self.direction_basis = direction_basis
        self._started = False
        self._next_review_time: float | None = None
        self._history: list[EnvironmentalFleetObservationRecord] = []
        self._membership_events: list[_MembershipEvent] = []
        self._counter = count()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def history(self) -> tuple[EnvironmentalFleetObservationRecord, ...]:
        return tuple(self._history)

    @property
    def next_review_time(self) -> float | None:
        return self._next_review_time

    def schedule_arrival(
        self,
        *,
        at: float,
        vessel: WaterRelativeVesselState,
        limits: ManeuverLimits | None = None,
        environmental_drift_velocity: Vector2 | None = None,
    ) -> None:
        self._validate_future_membership_time(at)
        heapq.heappush(
            self._membership_events,
            _MembershipEvent(
                float(at),
                next(self._counter),
                EnvironmentalFleetObservationReason.VESSEL_ARRIVAL,
                vessel=vessel,
                limits=limits,
                environmental_drift_velocity=environmental_drift_velocity,
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
                EnvironmentalFleetObservationReason.VESSEL_REMOVAL,
                vessel_id=vessel_id,
            ),
        )

    def _validate_future_membership_time(self, at: float) -> None:
        if not isfinite(at):
            raise ValueError("membership event time must be finite")
        if at <= self.world.time + _EPS:
            raise ValueError("membership event time must be strictly after current world time")

    def _observe(
        self, reasons: Iterable[EnvironmentalFleetObservationReason]
    ) -> EnvironmentalFleetObservationRecord:
        unique = set(reasons)
        ordered_reasons = tuple(reason for reason in _REASON_ORDER if reason in unique)
        if not ordered_reasons:
            raise ValueError("at least one observation reason is required")
        update = update_environmental_registry(
            self.registry,
            self.world,
            uncertain_pairs=self.uncertain_pairs,
            responsibility_context=self.responsibility_context,
            classification_config=self.classification_config,
            direction_basis=self.direction_basis,
        )
        admission = admit_supervision(update)
        components = build_active_encounter_components(admission)
        record = EnvironmentalFleetObservationRecord(
            self.world.time,
            ordered_reasons,
            update,
            admission,
            components,
        )
        self._history.append(record)
        return record

    def start(self) -> EnvironmentalFleetObservationRecord:
        if self._started:
            raise ValueError("environmental fleet observation session is already started")
        reasons = {EnvironmentalFleetObservationReason.INITIAL}
        if self.world.next_command_time() is not None and abs(self.world.next_command_time() - self.world.time) <= _EPS:
            self.world.run_until(self.world.time)
            reasons.add(EnvironmentalFleetObservationReason.COMMAND_EVENT)
        record = self._observe(reasons)
        self._started = True
        self._next_review_time = self.world.time + self.review_interval_min
        return record

    def next_boundary_time(self) -> float | None:
        if not self._started:
            raise ValueError("environmental fleet observation session must be started")
        candidates: list[float] = []
        if self._next_review_time is not None:
            candidates.append(self._next_review_time)
        command_time = self.world.next_command_time()
        if command_time is not None:
            candidates.append(command_time)
        if self._membership_events:
            candidates.append(self._membership_events[0].at)
        return min(candidates) if candidates else None

    def _apply_membership_events_at_current_time(
        self,
    ) -> set[EnvironmentalFleetObservationReason]:
        reasons: set[EnvironmentalFleetObservationReason] = set()
        while self._membership_events and abs(self._membership_events[0].at - self.world.time) <= _EPS:
            event = heapq.heappop(self._membership_events)
            if event.kind is EnvironmentalFleetObservationReason.VESSEL_ARRIVAL:
                assert event.vessel is not None
                self.world.add_vessel(
                    event.vessel,
                    limits=event.limits,
                    environmental_drift_velocity=event.environmental_drift_velocity,
                )
                reasons.add(EnvironmentalFleetObservationReason.VESSEL_ARRIVAL)
            elif event.kind is EnvironmentalFleetObservationReason.VESSEL_REMOVAL:
                assert event.vessel_id is not None
                self.world.remove_vessel(event.vessel_id)
                reasons.add(EnvironmentalFleetObservationReason.VESSEL_REMOVAL)
            else:  # pragma: no cover - internal enum exhaustiveness
                raise AssertionError(event.kind)
        return reasons

    def run_until(self, end_time: float) -> tuple[EnvironmentalFleetObservationRecord, ...]:
        if not self._started:
            raise ValueError("environmental fleet observation session must be started")
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")

        produced: list[EnvironmentalFleetObservationRecord] = []
        while True:
            boundary = self.next_boundary_time()
            if boundary is None or boundary > end_time + _EPS:
                break

            command_time = self.world.next_command_time()
            command_due = command_time is not None and abs(command_time - boundary) <= _EPS
            periodic_due = (
                self._next_review_time is not None
                and abs(self._next_review_time - boundary) <= _EPS
            )

            self.world.run_until(boundary)
            reasons: set[EnvironmentalFleetObservationReason] = set()
            if command_due:
                reasons.add(EnvironmentalFleetObservationReason.COMMAND_EVENT)
            reasons.update(self._apply_membership_events_at_current_time())
            if periodic_due:
                reasons.add(EnvironmentalFleetObservationReason.PERIODIC_REVIEW)
                assert self._next_review_time is not None
                self._next_review_time += self.review_interval_min

            produced.append(self._observe(reasons))

        self.world.run_until(end_time)
        return tuple(produced)
