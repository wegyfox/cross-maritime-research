"""Event-driven vessel lifecycle and conservative same-stream control.

v0.5 adds dynamic speed commands, explicit stale-event invalidation and a
simple longitudinal following model.  Crossing-control policies are still
intentionally excluded: this checkpoint proves that manoeuvres can safely
change previously scheduled analytical events before COLREG/FCFS logic is
layered on top.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from math import inf, isinf

from .control import PolicyController, PolicyEvaluation
from .events import EventKind, EventQueue, SimulationEvent
from .kinematics import SpeedLimits, time_to_cover_distance, time_to_reach_speed_target
from .vessel import Vessel, VesselStatus, advance_vessel
from .zones import AxisAlignedConflictZone, ConflictZonePassage


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    speed_limits: SpeedLimits
    min_entry_spacing: float = 0.0
    policy_controller: PolicyController | None = None
    conflict_zone: AxisAlignedConflictZone | None = None

    def __post_init__(self) -> None:
        if self.min_entry_spacing < 0:
            raise ValueError("min_entry_spacing must be non-negative")

    @property
    def min_same_stream_spacing(self) -> float:
        """Current baseline uses the entry spacing as the route spacing too."""
        return self.min_entry_spacing


@dataclass(frozen=True, slots=True)
class SimulationCounts:
    scheduled: int
    waiting: int
    active: int
    completed: int

    @property
    def total(self) -> int:
        return self.scheduled + self.waiting + self.active + self.completed


@dataclass(frozen=True, slots=True)
class EffectiveTargetChange:
    """One discrete commanded maneuver in the longitudinal control state.

    The legacy program counted a maneuver whenever the speed regime changed.
    The modern event-driven engine has continuous acceleration, so the closest
    auditable analogue is a change in the *effective target speed*: either an
    own-policy command or a same-stream constraint propagated from a leader.
    """

    time: float
    vessel_id: str
    route_name: str
    previous_target: float
    new_target: float

    @property
    def is_reduction(self) -> bool:
        return self.new_target < self.previous_target

    @property
    def is_restore(self) -> bool:
        return self.new_target > self.previous_target


class Simulation:
    """Event-driven lifecycle for fixed-route vessels.

    Same-route model in v0.5
    ------------------------
    Vessels are admitted FIFO with ``min_entry_spacing``.  Once active, each
    vessel keeps its own ``target_speed`` command, but its *effective* target
    cannot exceed the effective target of the vessel ahead on the same route.
    Therefore a leader slowdown/stop propagates through the convoy immediately.

    This is deliberately conservative rather than a calibrated ship-following
    model.  It provides a deterministic no-overtaking baseline on which more
    realistic longitudinal behavior can later be substituted.
    """

    _EPSILON = 1e-10
    _CONTROL_EVENT_ID = "__control__"

    def __init__(self, vessels: list[Vessel], config: SimulationConfig) -> None:
        ids = [v.vessel_id for v in vessels]
        if len(ids) != len(set(ids)):
            raise ValueError("vessel IDs must be unique")
        if any(v.status is not VesselStatus.SCHEDULED for v in vessels):
            raise ValueError("all input vessels must initially be scheduled")

        self.config = config
        self.time = 0.0
        self._total_generated = len(vessels)
        self._vessels = {v.vessel_id: v for v in vessels}
        self._events = EventQueue()
        self._event_generations: dict[tuple[EventKind, str], int] = defaultdict(int)
        self._waiting_by_route: dict[str, deque[str]] = defaultdict(deque)
        self._active_ids: set[str] = set()
        self._completed_ids: set[str] = set()
        self._last_policy_evaluation: PolicyEvaluation | None = None
        self._policy_evaluation_count = 0
        self._zone_passages: dict[str, ConflictZonePassage] = {}
        self._effective_target_state: dict[str, float] = {}
        self._effective_target_changes: list[EffectiveTargetChange] = []

        if config.conflict_zone is not None:
            for vessel in vessels:
                if config.conflict_zone.route_interval(vessel.route) is not None:
                    self._zone_passages[vessel.vessel_id] = ConflictZonePassage(
                        vessel.vessel_id
                    )

        for vessel in vessels:
            self._schedule_unique_event(
                vessel.arrival_time,
                EventKind.ARRIVAL,
                vessel.vessel_id,
            )

    @property
    def vessels(self) -> tuple[Vessel, ...]:
        return tuple(self._vessels.values())

    @property
    def counts(self) -> SimulationCounts:
        counts = {status: 0 for status in VesselStatus}
        for vessel in self._vessels.values():
            counts[vessel.status] += 1
        return SimulationCounts(
            scheduled=counts[VesselStatus.SCHEDULED],
            waiting=counts[VesselStatus.WAITING],
            active=counts[VesselStatus.ACTIVE],
            completed=counts[VesselStatus.COMPLETED],
        )

    def vessel(self, vessel_id: str) -> Vessel:
        return self._vessels[vessel_id]

    @property
    def zone_passages(self) -> tuple[ConflictZonePassage, ...]:
        """Immutable measurement records for vessels whose routes cross the zone."""
        return tuple(
            self._zone_passages[vessel_id]
            for vessel_id in sorted(self._zone_passages)
        )

    def zone_passage(self, vessel_id: str) -> ConflictZonePassage:
        return self._zone_passages[vessel_id]

    @property
    def last_policy_evaluation(self) -> PolicyEvaluation | None:
        return self._last_policy_evaluation

    @property
    def policy_evaluation_count(self) -> int:
        return self._policy_evaluation_count

    @property
    def effective_target_changes(self) -> tuple[EffectiveTargetChange, ...]:
        """Immutable chronological maneuver-command log."""
        return tuple(self._effective_target_changes)

    def effective_target_speed(self, vessel_id: str) -> float:
        """Return the speed command after same-stream leader constraints."""
        vessel = self._vessels[vessel_id]
        own_target = self._own_target(vessel)
        if vessel.status is not VesselStatus.ACTIVE:
            return own_target
        return self._effective_targets_for_route(vessel.route.name)[vessel_id]

    def command_speed(self, vessel_id: str, target_speed: float) -> None:
        """Change one active vessel's own speed command at current sim time.

        Any completion/release predictions affected on the same route are
        immediately re-versioned and rescheduled.  Old heap entries are left in
        place and later discarded by generation-token validation.
        """
        vessel = self._vessels[vessel_id]
        if vessel.status is not VesselStatus.ACTIVE:
            raise ValueError("speed commands can only be applied to active vessels")

        updated = vessel.with_target_speed(target_speed)
        if abs(self._own_target(vessel) - self._own_target(updated)) <= self._EPSILON:
            return

        self._apply_target_updates({vessel_id: target_speed})
        self._assert_invariants()

    def run(self, *, until: float | None = None) -> None:
        if until is not None and until < self.time:
            raise ValueError("until cannot be earlier than current simulation time")

        while True:
            self._discard_stale_front()
            if not self._events:
                break

            next_event = self._events.peek()
            if until is not None and next_event.time > until:
                self._advance_active_to(until)
                self.time = until
                self._assert_invariants()
                return
            self.step()

        if until is not None and until > self.time:
            self._advance_active_to(until)
            self.time = until
        self._assert_invariants()

    def step(self) -> SimulationEvent:
        self._discard_stale_front()
        if not self._events:
            raise IndexError("step called with no current events")

        event = self._events.pop()
        if not self._event_is_current(event):  # defensive; front was checked.
            raise RuntimeError("stale event escaped front pruning")
        if event.time + self._EPSILON < self.time:
            raise RuntimeError("event queue attempted to move simulation time backwards")

        # Mark this exact event token as consumed before its handler potentially
        # schedules a replacement of the same semantic event.
        self._consume_event(event)

        self._advance_active_to(event.time)
        self.time = event.time

        if event.kind is EventKind.ARRIVAL:
            self._handle_arrival(event.vessel_id)
        elif event.kind is EventKind.RELEASE:
            self._handle_release(event.vessel_id)
        elif event.kind is EventKind.CONFLICT_ZONE_ENTRY:
            self._handle_conflict_zone_entry(event.vessel_id)
        elif event.kind is EventKind.CONFLICT_ZONE_EXIT:
            self._handle_conflict_zone_exit(event.vessel_id)
        elif event.kind is EventKind.COMPLETION:
            self._handle_completion(event.vessel_id)
        elif event.kind is EventKind.CONTROL_REVIEW:
            self._handle_control_review()
        else:  # pragma: no cover - Enum makes this defensive.
            raise RuntimeError(f"unsupported event kind {event.kind}")

        self._evaluate_policy()
        self._assert_invariants()
        return event

    def _evaluate_policy(self) -> None:
        controller = self.config.policy_controller
        if controller is None:
            return

        evaluation = controller.evaluate(time=self.time, vessels=self.vessels)
        self._last_policy_evaluation = evaluation
        self._policy_evaluation_count += 1

        updates = {
            command.vessel_id: command.target_speed
            for command in evaluation.commands
            if abs(
                self._own_target(self._vessels[command.vessel_id])
                - command.target_speed
            )
            > self._EPSILON
        }
        if updates:
            self._apply_target_updates(updates)

        # Current positions/velocities do not change instantaneously when a
        # target command is installed, so the same snapshot remains valid for
        # horizon-entry timing. Speed-transition timing is computed separately
        # from the updated effective targets. Do not call policy.decide() twice
        # at the same simulation instant.
        self._reschedule_control_review(evaluation)

    def _apply_target_updates(self, updates: dict[str, float]) -> None:
        """Apply a batch atomically, then reschedule each affected route once."""
        affected_routes: set[str] = set()
        for vessel_id, target_speed in updates.items():
            vessel = self._vessels[vessel_id]
            if vessel.status is not VesselStatus.ACTIVE:
                raise ValueError("speed commands can only be applied to active vessels")
            updated = vessel.with_target_speed(target_speed)
            if abs(self._own_target(vessel) - self._own_target(updated)) <= self._EPSILON:
                continue
            self._vessels[vessel_id] = updated
            affected_routes.add(vessel.route.name)

        for route_name in affected_routes:
            self._reschedule_route_predictions(route_name)

    def _reschedule_control_review(self, evaluation: PolicyEvaluation) -> None:
        controller = self.config.policy_controller
        if controller is None:
            return

        horizon_delay = controller.next_horizon_review_delay(evaluation)
        danger_delay = controller.next_danger_review_delay(evaluation)
        recovery_delay = controller.next_recovery_review_delay(evaluation)
        transition_delay = self._next_speed_transition_review_delay()
        delay = min(horizon_delay, danger_delay, recovery_delay, transition_delay)

        if isinf(delay):
            self._invalidate_event(EventKind.CONTROL_REVIEW, self._CONTROL_EVENT_ID)
            return

        # Exact zero would immediately reschedule itself forever.  A zero-time
        # condition is already visible in the snapshot that was just evaluated.
        if delay <= self._EPSILON:
            self._invalidate_event(EventKind.CONTROL_REVIEW, self._CONTROL_EVENT_ID)
            return

        self._schedule_unique_event(
            self.time + delay,
            EventKind.CONTROL_REVIEW,
            self._CONTROL_EVENT_ID,
        )

    def _next_speed_transition_review_delay(self) -> float:
        best = inf
        for route_name in self._active_route_names():
            targets = self._effective_targets_for_route(route_name)
            for vessel_id, target in targets.items():
                vessel = self._vessels[vessel_id]
                delay = time_to_reach_speed_target(
                    vessel.current_speed,
                    target,
                    self.config.speed_limits,
                )
                if delay > self._EPSILON:
                    best = min(best, delay)
        return best

    def _advance_active_to(self, new_time: float) -> None:
        delta = new_time - self.time
        if delta < -self._EPSILON:
            raise RuntimeError("simulation time cannot decrease")
        if delta <= self._EPSILON:
            return

        effective_targets: dict[str, float] = {}
        for route_name in self._active_route_names():
            effective_targets.update(self._effective_targets_for_route(route_name))

        for vessel_id in list(self._active_ids):
            vessel = self._vessels[vessel_id]
            updated = advance_vessel(
                vessel,
                target_speed=effective_targets[vessel_id],
                duration=delta,
                limits=self.config.speed_limits,
            )
            if updated.status is VesselStatus.COMPLETED:
                updated = replace(updated, completed_at=new_time)
            self._vessels[vessel_id] = updated

            # Normally completion is processed by its current analytical event
            # exactly at this time.  This numerical guard makes tiny coincidence
            # harmless, while the handler will perform route rescheduling.
            if updated.status is VesselStatus.COMPLETED:
                self._active_ids.discard(vessel_id)
                self._completed_ids.add(vessel_id)

    def _handle_arrival(self, vessel_id: str) -> None:
        vessel = self._vessels[vessel_id]
        if vessel.status is not VesselStatus.SCHEDULED:
            return

        route_queue = self._waiting_by_route[vessel.route.name]
        if route_queue or not self._source_is_clear(vessel.route.name):
            self._vessels[vessel_id] = replace(vessel, status=VesselStatus.WAITING)
            route_queue.append(vessel_id)
            if len(route_queue) == 1:
                self._schedule_front_release(vessel.route.name)
            return

        self._activate(vessel_id)

    def _handle_release(self, vessel_id: str) -> None:
        vessel = self._vessels[vessel_id]
        if vessel.status is not VesselStatus.WAITING:
            return

        route_queue = self._waiting_by_route[vessel.route.name]
        if not route_queue or route_queue[0] != vessel_id:
            return

        if not self._source_is_clear(vessel.route.name):
            self._schedule_front_release(vessel.route.name)
            return

        route_queue.popleft()
        self._activate(vessel_id)

    def _handle_completion(self, vessel_id: str) -> None:
        vessel = self._vessels[vessel_id]
        route_name = vessel.route.name

        if vessel.status is VesselStatus.COMPLETED:
            self._active_ids.discard(vessel_id)
            self._completed_ids.add(vessel_id)
            self._effective_target_state.pop(vessel_id, None)
            self._reschedule_route_predictions(route_name)
            return

        if vessel.status is not VesselStatus.ACTIVE:
            return

        if vessel.route.length - vessel.progress <= self._EPSILON:
            vessel = replace(
                vessel,
                progress=vessel.route.length,
                current_speed=0.0,
                status=VesselStatus.COMPLETED,
                completed_at=self.time,
            )
            self._vessels[vessel_id] = vessel
            self._active_ids.discard(vessel_id)
            self._completed_ids.add(vessel_id)
            self._effective_target_state.pop(vessel_id, None)
            self._reschedule_route_predictions(route_name)
            return

        # A current generation completion event should coincide with route end.
        # If it does not, a prediction was inconsistent and should fail loudly.
        raise RuntimeError("current completion event fired before route completion")

    def _handle_conflict_zone_entry(self, vessel_id: str) -> None:
        passage = self._zone_passages.get(vessel_id)
        zone = self.config.conflict_zone
        if passage is None or zone is None or passage.entered_at is not None:
            return
        vessel = self._vessels[vessel_id]
        interval = zone.route_interval(vessel.route)
        if interval is None:
            return
        if vessel.progress + self._EPSILON < interval.entry_progress:
            raise RuntimeError("current conflict-zone entry event fired before boundary")
        self._zone_passages[vessel_id] = replace(passage, entered_at=self.time)
        self._reschedule_route_predictions(vessel.route.name)

    def _handle_conflict_zone_exit(self, vessel_id: str) -> None:
        passage = self._zone_passages.get(vessel_id)
        zone = self.config.conflict_zone
        if passage is None or zone is None or passage.exited_at is not None:
            return
        if passage.entered_at is None:
            raise RuntimeError("conflict-zone exit observed before entry")
        vessel = self._vessels[vessel_id]
        interval = zone.route_interval(vessel.route)
        if interval is None:
            return
        if vessel.progress + self._EPSILON < interval.exit_progress:
            raise RuntimeError("current conflict-zone exit event fired before boundary")
        self._zone_passages[vessel_id] = replace(passage, exited_at=self.time)
        self._reschedule_route_predictions(vessel.route.name)

    def _handle_control_review(self) -> None:
        """Control review has no direct state mutation; propagation already ran."""
        return

    def _activate(self, vessel_id: str) -> None:
        vessel = self._vessels[vessel_id]
        # A waiting vessel is still outside the modeled route.  Its approach
        # dynamics before the boundary are intentionally out of scope, so the
        # admission boundary may constrain its entry speed to the nearest
        # same-route leader's current speed.  Without this boundary condition a
        # 6 kn follower could be inserted behind a leader already slowed to
        # 4 kn, violating the conservative no-overtaking baseline immediately
        # before bounded deceleration had any time to act.
        leader = self._closest_active_to_source(vessel.route.name)
        entry_speed = vessel.current_speed
        if leader is not None:
            entry_speed = min(entry_speed, leader.current_speed)

        activated = replace(
            vessel.activate(at=self.time),
            current_speed=entry_speed,
        )
        self._vessels[vessel_id] = activated
        self._active_ids.add(vessel_id)
        self._reschedule_route_predictions(activated.route.name)

    def _reschedule_route_predictions(self, route_name: str) -> None:
        """Refresh completion and source-release predictions for one route."""
        effective_targets = self._effective_targets_for_route(route_name)
        self._record_effective_target_changes(route_name, effective_targets)

        for vessel_id in self._active_ids_on_route(route_name):
            vessel = self._vessels[vessel_id]
            remaining = vessel.route.length - vessel.progress
            completion_delta = time_to_cover_distance(
                vessel.current_speed,
                effective_targets[vessel_id],
                remaining,
                self.config.speed_limits,
            )
            if isinf(completion_delta):
                self._invalidate_event(EventKind.COMPLETION, vessel_id)
            else:
                self._schedule_unique_event(
                    self.time + completion_delta,
                    EventKind.COMPLETION,
                    vessel_id,
                )

            self._reschedule_zone_prediction(
                vessel,
                effective_target=effective_targets[vessel_id],
            )

        self._schedule_front_release(route_name, effective_targets=effective_targets)

    def _record_effective_target_changes(
        self,
        route_name: str,
        effective_targets: dict[str, float],
    ) -> None:
        """Record discrete effective-target transitions exactly once."""

        active_ids = set(effective_targets)
        for vessel_id in sorted(active_ids):
            vessel = self._vessels[vessel_id]
            new_target = effective_targets[vessel_id]
            previous_target = self._effective_target_state.get(vessel_id, vessel.current_speed)
            if abs(previous_target - new_target) > self._EPSILON:
                self._effective_target_changes.append(
                    EffectiveTargetChange(
                        time=self.time,
                        vessel_id=vessel_id,
                        route_name=route_name,
                        previous_target=previous_target,
                        new_target=new_target,
                    )
                )
            self._effective_target_state[vessel_id] = new_target

        for vessel_id in tuple(self._effective_target_state):
            vessel = self._vessels.get(vessel_id)
            if vessel is not None and vessel.route.name == route_name and vessel_id not in active_ids:
                self._effective_target_state.pop(vessel_id, None)

    def _reschedule_zone_prediction(
        self,
        vessel: Vessel,
        *,
        effective_target: float,
    ) -> None:
        """Refresh generation-safe analytical entry/exit predictions."""
        zone = self.config.conflict_zone
        if zone is None or vessel.vessel_id not in self._zone_passages:
            return

        interval = zone.route_interval(vessel.route)
        if interval is None:
            self._invalidate_zone_events(vessel.vessel_id)
            return

        passage = self._zone_passages[vessel.vessel_id]
        if passage.entered_at is None:
            self._invalidate_event(EventKind.CONFLICT_ZONE_EXIT, vessel.vessel_id)
            distance = max(0.0, interval.entry_progress - vessel.progress)
            delay = time_to_cover_distance(
                vessel.current_speed,
                effective_target,
                distance,
                self.config.speed_limits,
            )
            if isinf(delay):
                self._invalidate_event(EventKind.CONFLICT_ZONE_ENTRY, vessel.vessel_id)
            else:
                self._schedule_unique_event(
                    self.time + delay,
                    EventKind.CONFLICT_ZONE_ENTRY,
                    vessel.vessel_id,
                )
            return

        self._invalidate_event(EventKind.CONFLICT_ZONE_ENTRY, vessel.vessel_id)
        if passage.exited_at is not None:
            self._invalidate_event(EventKind.CONFLICT_ZONE_EXIT, vessel.vessel_id)
            return

        distance = max(0.0, interval.exit_progress - vessel.progress)
        delay = time_to_cover_distance(
            vessel.current_speed,
            effective_target,
            distance,
            self.config.speed_limits,
        )
        if isinf(delay):
            self._invalidate_event(EventKind.CONFLICT_ZONE_EXIT, vessel.vessel_id)
        else:
            self._schedule_unique_event(
                self.time + delay,
                EventKind.CONFLICT_ZONE_EXIT,
                vessel.vessel_id,
            )

    def _invalidate_zone_events(self, vessel_id: str) -> None:
        self._invalidate_event(EventKind.CONFLICT_ZONE_ENTRY, vessel_id)
        self._invalidate_event(EventKind.CONFLICT_ZONE_EXIT, vessel_id)

    def _source_is_clear(self, route_name: str) -> bool:
        if self.config.min_entry_spacing == 0:
            return True
        closest = self._closest_active_to_source(route_name)
        return closest is None or closest.progress + self._EPSILON >= self.config.min_entry_spacing

    def _closest_active_to_source(self, route_name: str) -> Vessel | None:
        candidates = [
            self._vessels[vessel_id]
            for vessel_id in self._active_ids_on_route(route_name)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda v: v.progress)

    def _schedule_front_release(
        self,
        route_name: str,
        *,
        effective_targets: dict[str, float] | None = None,
    ) -> None:
        route_queue = self._waiting_by_route[route_name]
        if not route_queue:
            return

        vessel_id = route_queue[0]
        if self._source_is_clear(route_name):
            self._schedule_unique_event(self.time, EventKind.RELEASE, vessel_id)
            return

        leader = self._closest_active_to_source(route_name)
        if leader is None:
            self._schedule_unique_event(self.time, EventKind.RELEASE, vessel_id)
            return

        if effective_targets is None:
            effective_targets = self._effective_targets_for_route(route_name)
        leader_target = effective_targets[leader.vessel_id]

        distance_to_clear = max(0.0, self.config.min_entry_spacing - leader.progress)
        time_to_spacing = time_to_cover_distance(
            leader.current_speed,
            leader_target,
            distance_to_clear,
            self.config.speed_limits,
        )
        remaining_route = leader.route.length - leader.progress
        time_to_completion = time_to_cover_distance(
            leader.current_speed,
            leader_target,
            remaining_route,
            self.config.speed_limits,
        )
        delay = min(time_to_spacing, time_to_completion)
        if isinf(delay):
            self._invalidate_event(EventKind.RELEASE, vessel_id)
        else:
            self._schedule_unique_event(self.time + delay, EventKind.RELEASE, vessel_id)

    def _effective_targets_for_route(self, route_name: str) -> dict[str, float]:
        """Return front-to-back effective speed targets for active vessels."""
        ordered = sorted(
            (self._vessels[vessel_id] for vessel_id in self._active_ids_on_route(route_name)),
            key=lambda v: (-v.progress, v.vessel_id),
        )
        result: dict[str, float] = {}
        leader_target: float | None = None
        for vessel in ordered:
            own = self._own_target(vessel)
            effective = own if leader_target is None else min(own, leader_target)
            result[vessel.vessel_id] = effective
            leader_target = effective
        return result

    def _active_ids_on_route(self, route_name: str) -> list[str]:
        return [
            vessel_id
            for vessel_id in self._active_ids
            if self._vessels[vessel_id].route.name == route_name
        ]

    def _active_route_names(self) -> set[str]:
        return {self._vessels[vessel_id].route.name for vessel_id in self._active_ids}

    @staticmethod
    def _own_target(vessel: Vessel) -> float:
        # Vessel.__post_init__ normalizes None to a float.
        if vessel.target_speed is None:  # pragma: no cover - defensive typing guard.
            return vessel.desired_speed
        return vessel.target_speed

    def _event_key(self, kind: EventKind, vessel_id: str) -> tuple[EventKind, str]:
        return (kind, vessel_id)

    def _schedule_unique_event(
        self,
        time: float,
        kind: EventKind,
        vessel_id: str,
    ) -> SimulationEvent:
        key = self._event_key(kind, vessel_id)
        generation = self._event_generations[key] + 1
        self._event_generations[key] = generation
        return self._events.push(
            time,
            kind,
            vessel_id,
            generation=generation,
        )

    def _invalidate_event(self, kind: EventKind, vessel_id: str) -> None:
        key = self._event_key(kind, vessel_id)
        self._event_generations[key] += 1

    def _event_is_current(self, event: SimulationEvent) -> bool:
        return event.generation == self._event_generations[
            self._event_key(event.kind, event.vessel_id)
        ]

    def _consume_event(self, event: SimulationEvent) -> None:
        self._invalidate_event(event.kind, event.vessel_id)

    def _discard_stale_front(self) -> None:
        while self._events and not self._event_is_current(self._events.peek()):
            self._events.pop()

    def _assert_invariants(self) -> None:
        counts = self.counts
        if counts.total != self._total_generated:
            raise RuntimeError("vessel conservation invariant violated")

        if len(self._vessels) != self._total_generated:
            raise RuntimeError("vessel ID uniqueness/conservation violated")

        for vessel in self._vessels.values():
            if vessel.progress < -self._EPSILON:
                raise RuntimeError("negative vessel progress")
            if vessel.progress - vessel.route.length > self._EPSILON:
                raise RuntimeError("vessel advanced beyond route end")
            own_target = self._own_target(vessel)
            if own_target < -self._EPSILON:
                raise RuntimeError("negative target speed")
            if own_target - vessel.desired_speed > self._EPSILON:
                raise RuntimeError("target speed exceeds desired speed")
            if vessel.status in (VesselStatus.SCHEDULED, VesselStatus.WAITING):
                if vessel.activated_at is not None or vessel.completed_at is not None:
                    raise RuntimeError("inactive vessel carries lifecycle timestamps")
            elif vessel.status is VesselStatus.ACTIVE:
                if vessel.activated_at is None:
                    raise RuntimeError("active vessel missing activated_at")
                if vessel.completed_at is not None:
                    raise RuntimeError("active vessel has completed_at")
            elif vessel.status is VesselStatus.COMPLETED:
                if vessel.activated_at is None or vessel.completed_at is None:
                    raise RuntimeError("completed vessel missing lifecycle timestamps")

        if self.config.conflict_zone is not None:
            for vessel_id, passage in self._zone_passages.items():
                if passage.exited_at is not None and passage.entered_at is None:
                    raise RuntimeError("conflict-zone exit exists without entry")
                vessel = self._vessels[vessel_id]
                interval = self.config.conflict_zone.route_interval(vessel.route)
                if interval is None:
                    raise RuntimeError("zone passage exists for non-intersecting route")
                if (
                    passage.entered_at is not None
                    and vessel.progress + self._EPSILON < interval.entry_progress
                ):
                    raise RuntimeError("zone entry timestamp precedes geometric entry")
                if (
                    passage.exited_at is not None
                    and vessel.progress + self._EPSILON < interval.exit_progress
                ):
                    raise RuntimeError("zone exit timestamp precedes geometric exit")

        spacing = self.config.min_same_stream_spacing
        if spacing > 0:
            for route_name in self._active_route_names():
                ordered = sorted(
                    (self._vessels[vessel_id] for vessel_id in self._active_ids_on_route(route_name)),
                    key=lambda v: (-v.progress, v.vessel_id),
                )
                for leader, follower in zip(ordered, ordered[1:]):
                    gap = leader.progress - follower.progress
                    if gap + self._EPSILON < spacing:
                        raise RuntimeError("same-stream minimum spacing invariant violated")
                    if follower.current_speed - leader.current_speed > self._EPSILON:
                        raise RuntimeError("same-stream follower became faster than leader")
