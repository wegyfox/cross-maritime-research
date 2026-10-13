"""Preemptible nominal arrival/route-review event source for mission orchestration.

Unlike the earlier run-to-horizon nominal schedulers, this session does not advance
``World2D`` on its own.  It exposes the next nominal event time and processes only the
world's *current* timestamp.  That separation lets a mission-level calendar merge
nominal arrivals/reviews with avoidance-owner reviews on one shared clock.

Nominal control can be suspended immediately for vessels whose command channel has been
claimed by encounter supervision.  Suspension clears the next nominal review so route
guidance cannot later overwrite an avoidance command.  Resumption is intentionally not
provided here: after an avoidance maneuver, route recovery/handoff must explicitly
reconcile physical state with route progress before nominal control becomes valid again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .dynamics import ManeuverLimits, Vessel2DState
from .follower import RouteFollowerState, RouteGuidance
from .nominal_fleet_scheduler import NominalFleetScheduleUnavailable, NominalFleetTarget
from .route_exit import RouteEndDisposition, apply_route_end_action
from .route_handoff import RouteHandoffTarget, RouteHandoffVesselResult
from .world import World2D
from .vessel_classes import VesselClassProfile, VesselProfileCatalog

_EPS = 1e-10


class NominalArrivalSpec(Protocol):
    at: float
    vessel: Vessel2DState
    target: NominalFleetTarget
    limits: ManeuverLimits | None


@dataclass(frozen=True, slots=True)
class NominalControlArrivalRecord:
    time: float
    vessel_id: str
    initial_state: Vessel2DState


@dataclass(frozen=True, slots=True)
class NominalControlGuidanceStep:
    time: float
    vessel_id: str
    guidance: RouteGuidance
    next_review_at: float | None


@dataclass(frozen=True, slots=True)
class NominalControlSuspension:
    time: float
    vessel_id: str
    follower_state: RouteFollowerState
    previous_next_review_at: float | None


@dataclass(frozen=True, slots=True)
class NominalControlResumption:
    time: float
    vessel_id: str
    follower_state: RouteFollowerState
    next_review_at: float | None


@dataclass(frozen=True, slots=True)
class NominalControlRecoveryCheckpoint:
    time: float
    vessel_id: str
    follower_state: RouteFollowerState


@dataclass(frozen=True, slots=True)
class NominalControlEventBatch:
    time: float
    arrivals: tuple[NominalControlArrivalRecord, ...]
    guidance_steps: tuple[NominalControlGuidanceStep, ...]
    suspensions: tuple[NominalControlSuspension, ...]

    @property
    def had_event(self) -> bool:
        return bool(self.arrivals or self.guidance_steps or self.suspensions)


@dataclass(frozen=True, slots=True)
class NominalControlVesselState:
    vessel_id: str
    arrived_at: float
    follower_state: RouteFollowerState
    guidance: RouteGuidance | None
    next_review_at: float | None
    disposition: RouteEndDisposition | None
    suspended: bool

    @property
    def completed(self) -> bool:
        return self.follower_state.completed


@dataclass(slots=True)
class _Cursor:
    arrival: NominalArrivalSpec
    target: NominalFleetTarget
    state: RouteFollowerState
    guidance: RouteGuidance | None = None
    next_review_at: float | None = None
    disposition: RouteEndDisposition | None = None
    suspended: bool = False

    @property
    def vessel_id(self) -> str:
        return self.arrival.vessel.vessel_id

    def public(self) -> NominalControlVesselState:
        return NominalControlVesselState(
            self.vessel_id,
            self.arrival.at,
            self.state,
            self.guidance,
            self.next_review_at,
            self.disposition,
            self.suspended,
        )


class DynamicNominalControlSession:
    """Stateful nominal event source that never advances physical time itself."""

    def __init__(
        self,
        world: World2D,
        arrivals: Iterable[NominalArrivalSpec],
        *,
        profile_catalog: VesselProfileCatalog | None = None,
    ) -> None:
        self.world = world
        self.profile_catalog = profile_catalog
        indexed = list(enumerate(arrivals))
        if not indexed:
            raise ValueError("at least one arrival is required")
        indexed.sort(key=lambda item: (item[1].at, item[0]))
        self.arrivals = tuple(item[1] for item in indexed)
        ids = [item.vessel.vessel_id for item in self.arrivals]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate arrival vessel_id")
        existing = {v.vessel_id for v in world.snapshot().vessels}
        overlap = existing.intersection(ids)
        if overlap:
            raise ValueError(f"arrival vessel ids already exist in world: {sorted(overlap)}")
        if self.arrivals[0].at < world.time - _EPS:
            raise ValueError("arrival cannot be scheduled in the past")

        self._pending_index = 0
        self._cursors: dict[str, _Cursor] = {}
        self._arrival_records: list[NominalControlArrivalRecord] = []
        self._guidance_steps: list[NominalControlGuidanceStep] = []
        self._suspensions: list[NominalControlSuspension] = []
        self._resumptions: list[NominalControlResumption] = []
        self._recovery_checkpoints: list[NominalControlRecoveryCheckpoint] = []

    @property
    def pending_arrivals(self) -> tuple[NominalArrivalSpec, ...]:
        return self.arrivals[self._pending_index :]

    @property
    def vessel_states(self) -> tuple[NominalControlVesselState, ...]:
        return tuple(self._cursors[vessel_id].public() for vessel_id in sorted(self._cursors))

    @property
    def arrival_records(self) -> tuple[NominalControlArrivalRecord, ...]:
        return tuple(self._arrival_records)

    @property
    def guidance_steps(self) -> tuple[NominalControlGuidanceStep, ...]:
        return tuple(self._guidance_steps)

    @property
    def suspension_records(self) -> tuple[NominalControlSuspension, ...]:
        return tuple(self._suspensions)

    @property
    def resumption_records(self) -> tuple[NominalControlResumption, ...]:
        return tuple(self._resumptions)

    @property
    def recovery_checkpoint_records(self) -> tuple[NominalControlRecoveryCheckpoint, ...]:
        return tuple(self._recovery_checkpoints)

    @property
    def all_routes_complete(self) -> bool:
        return self._pending_index >= len(self.arrivals) and all(
            cursor.state.completed for cursor in self._cursors.values()
        )

    def by_id(self, vessel_id: str) -> NominalControlVesselState:
        return self._cursors[vessel_id].public()

    def current_target(self, vessel_id: str) -> NominalFleetTarget:
        """Return the route target currently bound to one arrived nominal cursor."""
        return self._cursors[vessel_id].target

    def rebind_suspended_targets(
        self,
        targets: Iterable[NominalFleetTarget],
    ) -> tuple[NominalFleetTarget, ...]:
        """Atomically replace route targets for suspended, unfinished vessels.

        Arrival records are immutable historical spawn evidence.  A reroute therefore
        changes only the cursor's current target/route-coordinate state.  Physical vessel
        state and the currently executing world command are deliberately untouched; an
        exact route-handoff/recovery step must still establish valid nominal ownership.
        """
        items = tuple(sorted(tuple(targets), key=lambda item: item.vessel_id))
        if not items:
            raise NominalFleetScheduleUnavailable("replacement target set must be non-empty")
        ids = [item.vessel_id for item in items]
        if len(set(ids)) != len(ids):
            raise NominalFleetScheduleUnavailable("duplicate replacement target vessel_id")

        snapshot = self.world.snapshot()
        validated: list[tuple[_Cursor, NominalFleetTarget]] = []
        for target in items:
            cursor = self._cursors.get(target.vessel_id)
            if cursor is None:
                raise NominalFleetScheduleUnavailable(
                    f"cannot rebind unknown vessel {target.vessel_id}"
                )
            if not cursor.suspended:
                raise NominalFleetScheduleUnavailable(
                    f"replacement target {target.vessel_id} is not suspended"
                )
            if cursor.state.completed:
                raise NominalFleetScheduleUnavailable(
                    f"completed vessel {target.vessel_id} cannot be rerouted"
                )
            if target.follower_state.completed:
                raise NominalFleetScheduleUnavailable(
                    f"replacement target {target.vessel_id} must not start completed"
                )
            try:
                snapshot.by_id(target.vessel_id)
            except KeyError as exc:
                raise NominalFleetScheduleUnavailable(
                    f"replacement target vessel {target.vessel_id} is not in the world"
                ) from exc

            if self.profile_catalog is not None:
                profile = self.profile_catalog.profile_for(target.vessel_id)
                if profile is not None and profile.speed_envelope is not None:
                    cruise_speed = target.navigator.follower.cruise_speed
                    if not profile.speed_envelope.permits(cruise_speed):
                        raise NominalFleetScheduleUnavailable(
                            f"replacement nominal cruise speed exceeds vessel profile speed envelope for {target.vessel_id}"
                        )
            validated.append((cursor, target))

        # Mutate only after the complete batch is known valid.
        for cursor, target in validated:
            cursor.target = target
            cursor.state = target.follower_state
            cursor.guidance = None
            cursor.next_review_at = None
            cursor.disposition = None
        return items

    def next_event_time(self) -> float | None:
        candidates: list[float] = []
        if self._pending_index < len(self.arrivals):
            candidates.append(self.arrivals[self._pending_index].at)
        candidates.extend(
            cursor.next_review_at
            for cursor in self._cursors.values()
            if not cursor.suspended
            and not cursor.state.completed
            and cursor.next_review_at is not None
        )
        if not candidates:
            return None
        value = min(candidates)
        if value < self.world.time - _EPS:
            raise NominalFleetScheduleUnavailable("nominal session event time moved backwards")
        return value

    def _compute(self, cursor: _Cursor) -> RouteGuidance:
        vessel = self.world.snapshot().by_id(cursor.vessel_id)
        guidance = cursor.target.navigator.follower.guidance(
            vessel, cursor.state, self.world.limits_for(cursor.vessel_id)
        )
        if guidance.follower_state.along_track + _EPS < cursor.state.along_track:
            raise NominalFleetScheduleUnavailable("route progress regressed")
        return guidance

    def _apply(
        self,
        cursor: _Cursor,
        guidance: RouteGuidance,
        batch_steps: list[NominalControlGuidanceStep],
    ) -> None:
        cursor.state = guidance.follower_state
        cursor.guidance = guidance
        self.world.schedule_command(at=self.world.time, vessel_id=cursor.vessel_id, command=guidance.command)
        if cursor.state.completed:
            cursor.next_review_at = None
            self.world.run_until(self.world.time)
            cursor.disposition = apply_route_end_action(
                self.world,
                vessel_id=cursor.vessel_id,
                guidance=guidance,
                action=cursor.target.end_action,
            )
            if (
                cursor.disposition.removed
                and self.profile_catalog is not None
            ):
                self.profile_catalog.remove(cursor.vessel_id)
            step = NominalControlGuidanceStep(self.world.time, cursor.vessel_id, guidance, None)
            self._guidance_steps.append(step)
            batch_steps.append(step)
            return
        if guidance.next_review_after is None:
            raise NominalFleetScheduleUnavailable(
                f"exact nominal review unavailable for {cursor.vessel_id}"
            )
        cursor.next_review_at = self.world.time + max(0.0, guidance.next_review_after)
        step = NominalControlGuidanceStep(
            self.world.time, cursor.vessel_id, guidance, cursor.next_review_at
        )
        self._guidance_steps.append(step)
        batch_steps.append(step)

    def _refresh(self, cursor: _Cursor, batch_steps: list[NominalControlGuidanceStep]) -> None:
        if cursor.suspended or cursor.state.completed:
            return
        for _ in range(24):
            guidance = self._compute(cursor)
            self._apply(cursor, guidance, batch_steps)
            if cursor.state.completed:
                return
            assert guidance.next_review_after is not None
            if guidance.next_review_after > _EPS:
                return
            self.world.run_until(self.world.time)
        raise NominalFleetScheduleUnavailable("too many zero-time nominal session transitions")

    def suspend(self, vessel_ids: Iterable[str]) -> tuple[NominalControlSuspension, ...]:
        """Immediately remove nominal command-review ownership from arrived vessels."""
        result: list[NominalControlSuspension] = []
        for vessel_id in sorted(set(vessel_ids)):
            cursor = self._cursors.get(vessel_id)
            if cursor is None or cursor.suspended or cursor.state.completed:
                continue
            item = NominalControlSuspension(
                self.world.time,
                vessel_id,
                cursor.state,
                cursor.next_review_at,
            )
            cursor.suspended = True
            cursor.next_review_at = None
            self._suspensions.append(item)
            result.append(item)
        return tuple(result)


    def route_handoff_targets(self, vessel_ids: Iterable[str]) -> tuple[RouteHandoffTarget, ...]:
        """Build exact recovery targets from suspended nominal cursors."""
        targets: list[RouteHandoffTarget] = []
        for vessel_id in sorted(set(vessel_ids)):
            cursor = self._cursors.get(vessel_id)
            if cursor is None:
                raise NominalFleetScheduleUnavailable(
                    f"cannot build route handoff target for unknown vessel {vessel_id}"
                )
            if not cursor.suspended:
                raise NominalFleetScheduleUnavailable(
                    f"route handoff target {vessel_id} is not suspended"
                )
            if cursor.state.completed:
                raise NominalFleetScheduleUnavailable(
                    f"completed vessel {vessel_id} does not require route handoff"
                )
            targets.append(
                RouteHandoffTarget(
                    vessel_id,
                    cursor.target.navigator,
                    cursor.state,
                )
            )
        if not targets:
            raise NominalFleetScheduleUnavailable("route handoff target set must be non-empty")
        return tuple(targets)

    def checkpoint_recovery_progress(
        self,
        results: Iterable[RouteHandoffVesselResult],
    ) -> tuple[NominalControlRecoveryCheckpoint, ...]:
        """Persist non-regressing route progress while keeping cursors suspended.

        This is used when an in-progress recovery group is pre-empted or repartitioned.
        Physical recovery state is deliberately not copied into nominal control; a new
        route-handoff session will rederive recovery geometry from the current world
        state while respecting the checkpointed minimum along-track progress.
        """
        items = tuple(sorted(tuple(results), key=lambda item: item.vessel_id))
        if not items:
            raise NominalFleetScheduleUnavailable("recovery checkpoint results must be non-empty")
        ids = [item.vessel_id for item in items]
        if len(set(ids)) != len(ids):
            raise NominalFleetScheduleUnavailable("duplicate recovery checkpoint vessel_id")
        validated: list[tuple[_Cursor, RouteHandoffVesselResult]] = []
        for item in items:
            cursor = self._cursors.get(item.vessel_id)
            if cursor is None:
                raise NominalFleetScheduleUnavailable(
                    f"cannot checkpoint unknown vessel {item.vessel_id}"
                )
            if not cursor.suspended:
                raise NominalFleetScheduleUnavailable(
                    f"vessel {item.vessel_id} is not suspended"
                )
            if item.follower_state.along_track + _EPS < cursor.state.along_track:
                raise NominalFleetScheduleUnavailable(
                    f"recovery checkpoint regressed progress for {item.vessel_id}"
                )
            validated.append((cursor, item))

        records: list[NominalControlRecoveryCheckpoint] = []
        for cursor, item in validated:
            cursor.state = item.follower_state
            cursor.guidance = None
            cursor.next_review_at = None
            record = NominalControlRecoveryCheckpoint(
                self.world.time, cursor.vessel_id, cursor.state
            )
            self._recovery_checkpoints.append(record)
            records.append(record)
        return tuple(records)

    def resume_from_handoff(
        self,
        results: Iterable[RouteHandoffVesselResult],
    ) -> tuple[NominalControlResumption, ...]:
        """Return nominal-ready recovery results to the route-review event source.

        The handoff layer has already issued the current nominal guidance command at the
        same world timestamp.  This method only transfers route state and future review
        ownership back to the nominal session.
        """
        items = tuple(sorted(tuple(results), key=lambda item: item.vessel_id))
        if not items:
            raise NominalFleetScheduleUnavailable("route handoff results must be non-empty")
        ids = [item.vessel_id for item in items]
        if len(set(ids)) != len(ids):
            raise NominalFleetScheduleUnavailable("duplicate route handoff result vessel_id")

        # Validate the entire batch first so resumption is atomic.
        validated: list[tuple[_Cursor, RouteHandoffVesselResult]] = []
        for item in items:
            cursor = self._cursors.get(item.vessel_id)
            if cursor is None:
                raise NominalFleetScheduleUnavailable(
                    f"cannot resume unknown vessel {item.vessel_id}"
                )
            if not cursor.suspended:
                raise NominalFleetScheduleUnavailable(
                    f"vessel {item.vessel_id} is not suspended"
                )
            if not item.nominal_ready or item.nominal_guidance is None:
                raise NominalFleetScheduleUnavailable(
                    f"route handoff for {item.vessel_id} is not nominal-ready"
                )
            if item.follower_state.along_track + _EPS < cursor.state.along_track:
                raise NominalFleetScheduleUnavailable(
                    f"route handoff regressed progress for {item.vessel_id}"
                )
            if item.nominal_guidance.follower_state != item.follower_state:
                raise NominalFleetScheduleUnavailable(
                    f"route handoff guidance/state mismatch for {item.vessel_id}"
                )
            if (
                item.next_review_at is not None
                and item.next_review_at < self.world.time - _EPS
            ):
                raise NominalFleetScheduleUnavailable(
                    f"route handoff review for {item.vessel_id} is stale"
                )
            validated.append((cursor, item))

        records: list[NominalControlResumption] = []
        for cursor, item in validated:
            cursor.state = item.follower_state
            cursor.guidance = item.nominal_guidance
            cursor.next_review_at = item.next_review_at
            cursor.suspended = False
            if cursor.state.completed:
                cursor.next_review_at = None
                self.world.run_until(self.world.time)
                cursor.disposition = apply_route_end_action(
                    self.world,
                    vessel_id=cursor.vessel_id,
                    guidance=item.nominal_guidance,
                    action=cursor.target.end_action,
                )
            record = NominalControlResumption(
                self.world.time,
                cursor.vessel_id,
                cursor.state,
                cursor.next_review_at,
            )
            self._resumptions.append(record)
            records.append(record)
        return tuple(records)

    def process_current_time(
        self,
        *,
        blocked_vessel_ids: Iterable[str] = (),
    ) -> NominalControlEventBatch:
        """Process arrivals and unblocked route reviews due at ``world.time``.

        Phase order is preserved from a20: arrivals are inserted and initially guided
        before pre-existing route reviews.  ``blocked_vessel_ids`` is a command-ownership
        safety gate; arrived blocked vessels are suspended instead of issuing nominal
        guidance, and already-active blocked vessels are suspended before any due review.
        """
        blocked = set(blocked_vessel_ids)
        batch_suspensions = list(self.suspend(blocked))
        batch_arrivals: list[NominalControlArrivalRecord] = []
        batch_steps: list[NominalControlGuidanceStep] = []
        new_ids: set[str] = set()

        while (
            self._pending_index < len(self.arrivals)
            and self.arrivals[self._pending_index].at <= self.world.time + _EPS
        ):
            arrival = self.arrivals[self._pending_index]
            if arrival.at < self.world.time - _EPS:
                raise NominalFleetScheduleUnavailable("missed nominal arrival event")
            vessel_class = getattr(arrival, "vessel_class", None)
            if vessel_class is not None and not isinstance(vessel_class, VesselClassProfile):
                raise ValueError("arrival vessel_class must be a VesselClassProfile or None")
            if self.profile_catalog is not None and vessel_class is not None:
                existing = self.profile_catalog.profile_for(arrival.vessel.vessel_id)
                if existing is not None and existing != vessel_class:
                    raise ValueError(f"vessel profile conflict for {arrival.vessel.vessel_id}")
            self.world.add_vessel(arrival.vessel, limits=arrival.limits)
            if self.profile_catalog is not None and vessel_class is not None:
                self.profile_catalog.assign(arrival.vessel.vessel_id, vessel_class)
            cursor = _Cursor(arrival, arrival.target, arrival.target.follower_state)
            self._cursors[cursor.vessel_id] = cursor
            new_ids.add(cursor.vessel_id)
            record = NominalControlArrivalRecord(self.world.time, cursor.vessel_id, arrival.vessel)
            self._arrival_records.append(record)
            batch_arrivals.append(record)
            self._pending_index += 1
            if cursor.vessel_id in blocked:
                item = NominalControlSuspension(
                    self.world.time, cursor.vessel_id, cursor.state, None
                )
                cursor.suspended = True
                self._suspensions.append(item)
                batch_suspensions.append(item)
            else:
                self._refresh(cursor, batch_steps)

        due = [
            cursor
            for cursor in self._cursors.values()
            if cursor.vessel_id not in new_ids
            and not cursor.suspended
            and not cursor.state.completed
            and cursor.next_review_at is not None
            and cursor.next_review_at <= self.world.time + _EPS
        ]
        for cursor in sorted(due, key=lambda item: item.vessel_id):
            self._refresh(cursor, batch_steps)
        self.world.run_until(self.world.time)

        return NominalControlEventBatch(
            self.world.time,
            tuple(batch_arrivals),
            tuple(batch_steps),
            tuple(batch_suspensions),
        )
