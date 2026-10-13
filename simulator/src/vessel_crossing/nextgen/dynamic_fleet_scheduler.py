"""Dynamic-arrival nominal fleet scheduling on the shared World2D clock."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .dynamics import ManeuverLimits, Vessel2DState
from .encounter_registry import EncounterRegistry, EncounterRegistryUpdate
from .follower import RouteFollowerState, RouteGuidance
from .nominal_control_session import DynamicNominalControlSession
from .nominal_fleet_scheduler import NominalFleetScheduleUnavailable, NominalFleetTarget
from .route_exit import RouteEndDisposition
from .world import World2D, WorldSnapshot
from .vessel_classes import VesselClassProfile

_EPS = 1e-10


class DynamicFleetStopReason(str, Enum):
    ALL_COMPLETE = "all_complete"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class NominalVesselArrival:
    at: float
    vessel: Vessel2DState
    target: NominalFleetTarget
    limits: ManeuverLimits | None = None
    vessel_class: VesselClassProfile | None = None

    def __post_init__(self) -> None:
        if self.at < 0:
            raise ValueError("arrival time must be non-negative")
        if self.vessel.vessel_id != self.target.vessel_id:
            raise ValueError("arrival vessel_id must match target vessel_id")
        if self.vessel_class is not None:
            class_limits = self.vessel_class.maneuver_limits
            if self.limits is None:
                object.__setattr__(self, "limits", class_limits)
            elif self.limits != class_limits:
                raise ValueError("arrival maneuver limits conflict with vessel_class profile")
            envelope = self.vessel_class.speed_envelope
            if envelope is not None:
                if not envelope.permits(self.vessel.speed):
                    raise ValueError("arrival vessel speed exceeds vessel_class speed envelope")
                if not envelope.permits(self.target.navigator.follower.cruise_speed):
                    raise ValueError("nominal cruise speed exceeds vessel_class speed envelope")

    @property
    def vessel_class_id(self) -> str | None:
        return None if self.vessel_class is None else self.vessel_class.class_id


@dataclass(frozen=True, slots=True)
class DynamicFleetArrivalRecord:
    time: float
    vessel_id: str
    initial_state: Vessel2DState


@dataclass(frozen=True, slots=True)
class DynamicFleetGuidanceStep:
    time: float
    vessel_id: str
    guidance: RouteGuidance
    next_review_at: float | None


@dataclass(frozen=True, slots=True)
class DynamicFleetVesselResult:
    vessel_id: str
    arrived_at: float
    follower_state: RouteFollowerState
    guidance: RouteGuidance
    next_review_at: float | None
    disposition: RouteEndDisposition | None

    @property
    def completed(self) -> bool:
        return self.follower_state.completed


@dataclass(frozen=True, slots=True)
class DynamicFleetRunResult:
    snapshot: WorldSnapshot
    vessels: tuple[DynamicFleetVesselResult, ...]
    arrivals: tuple[DynamicFleetArrivalRecord, ...]
    guidance_steps: tuple[DynamicFleetGuidanceStep, ...]
    encounter_updates: tuple[EncounterRegistryUpdate, ...]
    pending_arrivals: tuple[NominalVesselArrival, ...]
    stop_reason: DynamicFleetStopReason

    @property
    def all_complete(self) -> bool:
        return self.stop_reason is DynamicFleetStopReason.ALL_COMPLETE

    def by_id(self, vessel_id: str) -> DynamicFleetVesselResult:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)


class DynamicNominalFleetScheduler:
    """Run the preemptible nominal event source to a horizon with no external owners.

    A33 factors the event-boundary mechanics into ``DynamicNominalControlSession``.
    This compatibility wrapper retains the a20/a23 run-to-horizon API for scenarios in
    which nominal control is never pre-empted by encounter supervision.
    """

    def __init__(
        self,
        world: World2D,
        arrivals: Iterable[NominalVesselArrival],
        *,
        encounter_registry: EncounterRegistry | None = None,
    ) -> None:
        self.world = world
        self.encounter_registry = encounter_registry
        self.arrivals = tuple(arrivals)
        # Validate immediately, before any world mutation, exactly as the legacy wrapper
        # contract expects.  A fresh session is created for the actual run below.
        DynamicNominalControlSession(
            world,
            self.arrivals,
            profile_catalog=(None if encounter_registry is None else encounter_registry.profile_catalog),
        )

    @staticmethod
    def _vessel_results(session: DynamicNominalControlSession) -> tuple[DynamicFleetVesselResult, ...]:
        result: list[DynamicFleetVesselResult] = []
        for item in session.vessel_states:
            if item.guidance is None:
                raise NominalFleetScheduleUnavailable("arrived vessel has no nominal guidance")
            if item.suspended:
                raise NominalFleetScheduleUnavailable(
                    "compatibility dynamic scheduler cannot return suspended nominal control"
                )
            result.append(
                DynamicFleetVesselResult(
                    item.vessel_id,
                    item.arrived_at,
                    item.follower_state,
                    item.guidance,
                    item.next_review_at,
                    item.disposition,
                )
            )
        return tuple(result)

    @staticmethod
    def _arrival_records(session: DynamicNominalControlSession) -> tuple[DynamicFleetArrivalRecord, ...]:
        return tuple(
            DynamicFleetArrivalRecord(item.time, item.vessel_id, item.initial_state)
            for item in session.arrival_records
        )

    @staticmethod
    def _guidance_steps(session: DynamicNominalControlSession) -> tuple[DynamicFleetGuidanceStep, ...]:
        return tuple(
            DynamicFleetGuidanceStep(item.time, item.vessel_id, item.guidance, item.next_review_at)
            for item in session.guidance_steps
        )

    def run_until(self, *, end_time: float, max_events: int = 10000) -> DynamicFleetRunResult:
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        if max_events < 1:
            raise ValueError("max_events must be at least 1")

        session = DynamicNominalControlSession(
            self.world,
            self.arrivals,
            profile_catalog=(
                None if self.encounter_registry is None else self.encounter_registry.profile_catalog
            ),
        )
        encounter_updates: list[EncounterRegistryUpdate] = []

        for _ in range(max_events):
            if session.all_routes_complete:
                return DynamicFleetRunResult(
                    self.world.snapshot(),
                    self._vessel_results(session),
                    self._arrival_records(session),
                    self._guidance_steps(session),
                    tuple(encounter_updates),
                    (),
                    DynamicFleetStopReason.ALL_COMPLETE,
                )
            if self.world.time >= end_time - _EPS:
                return DynamicFleetRunResult(
                    self.world.snapshot(),
                    self._vessel_results(session),
                    self._arrival_records(session),
                    self._guidance_steps(session),
                    tuple(encounter_updates),
                    tuple(session.pending_arrivals),
                    DynamicFleetStopReason.END_TIME,
                )

            next_event = session.next_event_time()
            if next_event is None:
                raise NominalFleetScheduleUnavailable("no future event exists for unfinished dynamic fleet")
            next_time = min(end_time, next_event)
            self.world.run_until(next_time)
            if next_event <= self.world.time + _EPS:
                batch = session.process_current_time()
                if self.encounter_registry is not None and batch.had_event:
                    encounter_updates.append(self.encounter_registry.update(self.world))

        raise NominalFleetScheduleUnavailable("dynamic nominal fleet scheduler exceeded max_events")
