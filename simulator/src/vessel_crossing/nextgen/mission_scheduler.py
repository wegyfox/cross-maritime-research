"""Unified nominal / avoidance / route-recovery mission event calendar.

A38 makes the unified mission state machine persistent across caller horizons.  Nominal,
avoidance and route-recovery owners remain on one ``World2D`` event clock, but their
review state now lives in ``UnifiedMissionSession`` instead of local variables inside a
one-shot run loop.  A pure horizon stop therefore cannot reset review cadence, recovery
ownership or initialization state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Iterable

from .encounter_registry import EncounterRegistry, EncounterRegistryUpdate
from .fleet_review_session import (
    FleetAppliedOwnerPlan,
    FleetReviewControlSession,
    FleetReviewController,
    FleetReviewSessionStatus,
    FleetReviewSessionUpdate,
)
from .fleet_supervision import FleetCommandOwner, FleetSupervisionDispatch, build_fleet_supervision_dispatch
from .mission_safety import (
    MissionSafetyExperimentSummary,
    MissionSafetyRecorder,
    PairSafetyEpisode,
    TimedPairSafetyIntervalMetric,
)
from .safety_metrics import FleetSafetyMetricSummary, PairSafetyMetricConfig
from .nominal_control_session import (
    DynamicNominalControlSession,
    NominalArrivalSpec,
    NominalControlEventBatch,
    NominalControlVesselState,
)
from .nominal_fleet_scheduler import NominalFleetTarget
from .route_handoff import FleetRouteHandoffUpdate
from .route_recovery_manager import (
    FleetRouteRecoveryManager,
    RouteRecoveryGroupUnavailable,
    RouteRecoveryGroupUpdate,
)
from .world import World2D, WorldSnapshot

_EPS = 1e-10


class MissionSchedulerError(RuntimeError):
    """Base fail-closed mission-calendar error."""


class MissionStopReason(str, Enum):
    END_TIME = "end_time"
    ALL_NOMINAL_COMPLETE = "all_nominal_complete"
    ROUTE_HANDOFF_REQUIRED = "route_handoff_required"
    ROUTE_RECOVERY_UNAVAILABLE = "route_recovery_unavailable"
    # Kept for compatibility with a36 result handling. A37 normally resolves these
    # transitions on the same timestamp instead of returning them.
    RECOVERY_PREEMPTED = "recovery_preempted"
    RECOVERY_CONFLICT = "recovery_conflict"
    NO_EVENTS = "no_events"


MissionControllerFactory = Callable[[FleetCommandOwner], FleetReviewController]
MissionObserver = Callable[[World2D], EncounterRegistryUpdate]


@dataclass(frozen=True, slots=True)
class MissionCalendarEvent:
    time_min: float
    nominal_batch: NominalControlEventBatch | None
    registry_update: EncounterRegistryUpdate
    dispatch: FleetSupervisionDispatch
    fleet_update: FleetReviewSessionUpdate | None
    fleet_session_rebuilt: bool
    initial_owner_plans: tuple[FleetAppliedOwnerPlan, ...]
    owned_vessel_ids: tuple[str, ...]
    route_handoff_update: FleetRouteHandoffUpdate | None = None
    resumed_vessel_ids: tuple[str, ...] = ()
    recovery_group_updates: tuple[RouteRecoveryGroupUpdate, ...] = ()
    preempted_recovery_vessel_ids: tuple[str, ...] = ()
    recovery_unavailable_groups: tuple[RouteRecoveryGroupUnavailable, ...] = ()


@dataclass(frozen=True, slots=True)
class MissionRunResult:
    snapshot: WorldSnapshot
    stop_reason: MissionStopReason
    events: tuple[MissionCalendarEvent, ...]
    nominal_states: tuple[NominalControlVesselState, ...]
    active_component_ids: tuple[str, ...]
    next_fleet_review_times: tuple[tuple[str, float], ...]
    route_handoff_vessel_ids: tuple[str, ...]
    active_recovery_vessel_ids: tuple[str, ...] = ()
    active_recovery_group_ids: tuple[str, ...] = ()
    recovery_unavailable_groups: tuple[RouteRecoveryGroupUnavailable, ...] = ()
    safety_intervals: tuple[TimedPairSafetyIntervalMetric, ...] = ()
    safety_episodes: tuple[PairSafetyEpisode, ...] = ()
    safety_summary: FleetSafetyMetricSummary | None = None
    safety_experiment_summary: MissionSafetyExperimentSummary | None = None

    @property
    def route_handoff_required(self) -> bool:
        return bool(self.route_handoff_vessel_ids)



class _UnifiedMissionCore:
    """Shared construction and fail-closed helpers for mission sessions/wrappers."""

    def __init__(
        self,
        world: World2D,
        arrivals: Iterable[NominalArrivalSpec],
        registry: EncounterRegistry,
        *,
        controller_factory: MissionControllerFactory,
        observer: MissionObserver | None = None,
        auto_route_handoff: bool = True,
    ) -> None:
        if not callable(controller_factory):
            raise ValueError("controller_factory must be callable")
        self.world = world
        self.nominal = DynamicNominalControlSession(
            world, arrivals, profile_catalog=registry.profile_catalog
        )
        self.registry = registry
        self.controller_factory = controller_factory
        self.observer: MissionObserver = observer if observer is not None else registry.update
        if not callable(self.observer):
            raise ValueError("observer must be callable")
        self.auto_route_handoff = bool(auto_route_handoff)

    def _controllers_for(self, dispatch: FleetSupervisionDispatch) -> tuple[FleetReviewController, ...]:
        controllers: list[FleetReviewController] = []
        for owner in dispatch.owners:
            controller = self.controller_factory(owner)
            if controller.component_id != owner.component_id:
                raise MissionSchedulerError("controller_factory returned wrong component_id")
            if controller.backend is not owner.backend:
                raise MissionSchedulerError("controller_factory returned wrong backend")
            controllers.append(controller)
        return tuple(controllers)

    @staticmethod
    def _nominal_state_map(session: DynamicNominalControlSession) -> dict[str, NominalControlVesselState]:
        return {state.vessel_id: state for state in session.vessel_states}

    def _handoff_ids(
        self,
        previous_owned: set[str],
        current_owned: set[str],
    ) -> tuple[str, ...]:
        released = previous_owned - current_owned
        states = self._nominal_state_map(self.nominal)
        result = []
        for vessel_id in sorted(released):
            state = states.get(vessel_id)
            if state is not None and state.suspended and not state.completed:
                result.append(vessel_id)
        return tuple(result)

    def _controllers_session(
        self,
        update: EncounterRegistryUpdate,
        dispatch: FleetSupervisionDispatch,
    ) -> tuple[FleetReviewControlSession | None, tuple[FleetAppliedOwnerPlan, ...]]:
        if not dispatch.owners:
            return None, ()
        self.nominal.suspend(dispatch.owned_vessel_ids)
        session = FleetReviewControlSession(
            self.world,
            update,
            dispatch,
            controllers=self._controllers_for(dispatch),
            profile_catalog=self.registry.profile_catalog,
        )
        plans = session.start()
        return session, plans


class UnifiedMissionSession(_UnifiedMissionCore):
    """Persistent unified mission state machine advanced by caller-supplied horizons.

    ``advance_until`` returns only the calendar events processed by that call.  Persistent
    fleet-review and route-recovery state remains owned by the session and is exposed via
    properties/result fields.  Formal events at exactly ``end_time`` are processed fully;
    a caller horizon never changes what a real event does.
    """

    def __init__(
        self,
        world: World2D,
        arrivals: Iterable[NominalArrivalSpec],
        registry: EncounterRegistry,
        *,
        controller_factory: MissionControllerFactory,
        observer: MissionObserver | None = None,
        auto_route_handoff: bool = True,
        safety_metric_config: PairSafetyMetricConfig | None = None,
    ) -> None:
        super().__init__(
            world,
            arrivals,
            registry,
            controller_factory=controller_factory,
            observer=observer,
            auto_route_handoff=auto_route_handoff,
        )
        self.fleet_session: FleetReviewControlSession | None = None
        self.recovery_manager = FleetRouteRecoveryManager(world)
        self._advance_calls = 0
        self._has_registry_boundary = False
        self._manual_handoff_ids: tuple[str, ...] = ()
        self._recovery_unavailable_groups: tuple[RouteRecoveryGroupUnavailable, ...] = ()
        self._event_history: list[MissionCalendarEvent] = []
        self.safety_recorder = (
            None
            if safety_metric_config is None
            else MissionSafetyRecorder(
                profile_catalog=self.registry.profile_catalog,
                config=safety_metric_config,
            )
        )
        self._safety_call_start_index = 0
        self._expected_snapshot = world.snapshot()

    @property
    def event_history(self) -> tuple[MissionCalendarEvent, ...]:
        return tuple(self._event_history)

    @property
    def active_component_ids(self) -> tuple[str, ...]:
        return () if self.fleet_session is None else self.fleet_session.active_component_ids

    @property
    def next_fleet_review_times(self) -> tuple[tuple[str, float], ...]:
        return () if self.fleet_session is None else self.fleet_session.next_review_times

    @property
    def active_recovery_group_ids(self) -> tuple[str, ...]:
        return self.recovery_manager.group_ids

    @property
    def active_recovery_vessel_ids(self) -> tuple[str, ...]:
        return self.recovery_manager.owned_vessel_ids

    @property
    def recovery_unavailable_groups(self) -> tuple[RouteRecoveryGroupUnavailable, ...]:
        return self._recovery_unavailable_groups

    @property
    def safety_interval_history(self) -> tuple[TimedPairSafetyIntervalMetric, ...]:
        return () if self.safety_recorder is None else self.safety_recorder.history

    @property
    def safety_episodes(self) -> tuple[PairSafetyEpisode, ...]:
        return () if self.safety_recorder is None else self.safety_recorder.episodes

    @property
    def safety_summary(self) -> FleetSafetyMetricSummary | None:
        return None if self.safety_recorder is None else self.safety_recorder.summary

    @property
    def safety_experiment_summary(self) -> MissionSafetyExperimentSummary | None:
        return None if self.safety_recorder is None else self.safety_recorder.experiment_summary

    def resolve_recovery_unavailable(
        self,
        replacement_targets: Iterable[NominalFleetTarget],
    ) -> tuple[RouteRecoveryGroupUpdate | RouteRecoveryGroupUnavailable, ...]:
        """Resolve the persistent a61 boundary using caller-supplied route targets.

        Mission control never invents a route.  The caller must replace exactly the
        vessels that failed route recovery.  Target rebinding is atomic inside the
        nominal session; each affected recovery group is then rebuilt on the unchanged
        mission timestamp and may either become nominal-ready, remain in exact recovery,
        or surface another structured recovery-unavailable boundary.
        """
        self._assert_external_world_unchanged()
        groups = self._recovery_unavailable_groups
        if not groups:
            raise MissionSchedulerError("no route-recovery-unavailable boundary is active")

        targets = tuple(replacement_targets)
        expected_ids = tuple(sorted(group.failed_vessel_id for group in groups))
        provided_ids = tuple(sorted(target.vessel_id for target in targets))
        if len(set(provided_ids)) != len(provided_ids):
            raise ValueError("replacement target vessel ids must be unique")
        if provided_ids != expected_ids:
            raise ValueError(
                "replacement targets must match exactly the failed recovery vessel ids: "
                f"expected {expected_ids}, got {provided_ids}"
            )

        avoidance_owned = (
            set() if self.fleet_session is None else set(self.fleet_session.owned_vessel_ids)
        )
        overlap = avoidance_owned.intersection(provided_ids)
        if overlap:
            raise MissionSchedulerError(
                f"cannot reroute vessels currently owned by avoidance: {sorted(overlap)}"
            )

        # Rebinding validates the full batch before changing any nominal cursor.
        self.nominal.rebind_suspended_targets(targets)
        self._recovery_unavailable_groups = ()

        rebuilt: list[RouteRecoveryGroupUpdate | RouteRecoveryGroupUnavailable] = []
        unavailable: list[RouteRecoveryGroupUnavailable] = []
        for old_group in groups:
            item, _resumed = self._add_recovery_group(old_group.vessel_ids)
            rebuilt.append(item)
            if isinstance(item, RouteRecoveryGroupUnavailable):
                # Preserve the latest non-regressing progress even when the replacement
                # target itself cannot be recovered.
                self.nominal.checkpoint_recovery_progress(item.vessel_results)
                unavailable.append(item)

        self._recovery_unavailable_groups = tuple(unavailable)
        self._expected_snapshot = self.world.snapshot()
        return tuple(rebuilt)

    def _assert_external_world_unchanged(self) -> None:
        if self.world.snapshot() != self._expected_snapshot:
            raise MissionSchedulerError(
                "world changed outside the persistent mission session between advance calls"
            )

    def _advance_world_to(self, end_time: float) -> None:
        """Advance physics while recording command-homogeneous safety intervals.

        Metric-only subdivisions never invoke the observer or any control policy.  They
        exist solely so one a64 interval cannot straddle a queued command replacement.
        """
        if self.safety_recorder is None:
            self.world.run_until(end_time)
            return

        while self.world.time < end_time - _EPS:
            next_command = self.world.next_command_time()
            if next_command is not None and next_command <= self.world.time + _EPS:
                self.world.run_until(self.world.time)
                continue
            boundary = end_time
            if next_command is not None and next_command < boundary - _EPS:
                boundary = next_command
            duration = boundary - self.world.time
            if duration > _EPS:
                self.safety_recorder.observe_current_interval(self.world, duration_min=duration)
            self.world.run_until(boundary)

    def _make_result(
        self,
        *,
        stop_reason: MissionStopReason,
        events: list[MissionCalendarEvent],
        route_handoff_ids: tuple[str, ...] = (),
    ) -> MissionRunResult:
        history = self.safety_interval_history
        result = MissionRunResult(
            snapshot=self.world.snapshot(),
            stop_reason=stop_reason,
            events=tuple(events),
            nominal_states=self.nominal.vessel_states,
            active_component_ids=self.active_component_ids,
            next_fleet_review_times=self.next_fleet_review_times,
            route_handoff_vessel_ids=route_handoff_ids,
            active_recovery_vessel_ids=self.recovery_manager.owned_vessel_ids,
            active_recovery_group_ids=self.recovery_manager.group_ids,
            recovery_unavailable_groups=self._recovery_unavailable_groups,
            safety_intervals=history[self._safety_call_start_index :],
            safety_episodes=self.safety_episodes,
            safety_summary=self.safety_summary,
            safety_experiment_summary=self.safety_experiment_summary,
        )
        self._expected_snapshot = result.snapshot
        return result

    def _record_event(self, events: list[MissionCalendarEvent], event: MissionCalendarEvent) -> None:
        events.append(event)
        self._event_history.append(event)

    def _start_fleet_session(
        self,
        update: EncounterRegistryUpdate,
        dispatch: FleetSupervisionDispatch,
    ) -> tuple[FleetAppliedOwnerPlan, ...]:
        self.fleet_session, plans = self._controllers_session(update, dispatch)
        return plans

    def _add_recovery_group(
        self,
        vessel_ids: tuple[str, ...],
    ) -> tuple[RouteRecoveryGroupUpdate | RouteRecoveryGroupUnavailable, tuple[str, ...]]:
        item = self.recovery_manager.add_group(
            self.nominal.route_handoff_targets(vessel_ids),
            source_snapshot=self.world.snapshot(),
        )
        if isinstance(item, RouteRecoveryGroupUnavailable):
            return item, ()
        if item.completed:
            records = self.nominal.resume_from_handoff(item.update.vessel_results)
            return item, tuple(record.vessel_id for record in records)
        return item, ()

    def _validate_initial_boundary(self, initial_update: EncounterRegistryUpdate | None) -> None:
        if self._advance_calls > 0 and initial_update is not None:
            raise ValueError("initial_update may be supplied only on the first advance_until call")
        first_nominal = self.nominal.next_event_time()
        if initial_update is not None and first_nominal is not None and first_nominal <= self.world.time + _EPS:
            raise ValueError(
                "initial_update cannot precede a nominal event at the same current timestamp; "
                "let the mission calendar process nominal phases before the single observation"
            )
        if (
            not self._has_registry_boundary
            and initial_update is None
            and self.world.snapshot().vessels
            and (first_nominal is None or first_nominal > self.world.time + _EPS)
        ):
            raise MissionSchedulerError(
                "initial_update is required when the world already contains vessels before "
                "the next nominal event"
            )

    def advance_until(
        self,
        *,
        end_time: float,
        initial_update: EncounterRegistryUpdate | None = None,
        max_events: int = 10000,
    ) -> MissionRunResult:
        if not isfinite(end_time) or end_time < self.world.time - _EPS:
            raise ValueError("end_time must be finite and not precede world time")
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._assert_external_world_unchanged()
        self._validate_initial_boundary(initial_update)
        self._advance_calls += 1
        self._safety_call_start_index = len(self.safety_interval_history)
        events: list[MissionCalendarEvent] = []

        if self._manual_handoff_ids:
            return self._make_result(
                stop_reason=MissionStopReason.ROUTE_HANDOFF_REQUIRED,
                events=events,
                route_handoff_ids=self._manual_handoff_ids,
            )

        if self._recovery_unavailable_groups:
            return self._make_result(
                stop_reason=MissionStopReason.ROUTE_RECOVERY_UNAVAILABLE,
                events=events,
            )

        if initial_update is not None:
            if abs(initial_update.time_min - self.world.time) > _EPS:
                raise ValueError("initial_update time must match current world time")
            dispatch = build_fleet_supervision_dispatch(initial_update)
            self.nominal.suspend(dispatch.owned_vessel_ids)
            initial_plans = self._start_fleet_session(initial_update, dispatch)
            self._has_registry_boundary = True
            self._record_event(
                events,
                MissionCalendarEvent(
                    self.world.time,
                    None,
                    initial_update,
                    dispatch,
                    None,
                    self.fleet_session is not None,
                    initial_plans,
                    dispatch.owned_vessel_ids,
                ),
            )

        for _ in range(max_events):
            if self.world.time >= end_time - _EPS:
                return self._make_result(stop_reason=MissionStopReason.END_TIME, events=events)

            nominal_time = self.nominal.next_event_time()
            fleet_time = None if self.fleet_session is None else self.fleet_session.next_event_time()
            recovery_time = self.recovery_manager.next_event_time()
            available = [value for value in (nominal_time, fleet_time, recovery_time) if value is not None]

            if not available:
                if (
                    self.nominal.all_routes_complete
                    and self.fleet_session is None
                    and not self.recovery_manager.group_ids
                ):
                    return self._make_result(
                        stop_reason=MissionStopReason.ALL_NOMINAL_COMPLETE,
                        events=events,
                    )
                if self.world.time < end_time - _EPS:
                    self._advance_world_to(end_time)
                    return self._make_result(stop_reason=MissionStopReason.END_TIME, events=events)
                return self._make_result(stop_reason=MissionStopReason.NO_EVENTS, events=events)

            event_time = min(available)
            if event_time > end_time + _EPS:
                self._advance_world_to(end_time)
                return self._make_result(stop_reason=MissionStopReason.END_TIME, events=events)
            self._advance_world_to(event_time)

            nominal_due = nominal_time is not None and nominal_time <= self.world.time + _EPS
            fleet_due = fleet_time is not None and fleet_time <= self.world.time + _EPS
            recovery_due = recovery_time is not None and recovery_time <= self.world.time + _EPS
            if not nominal_due and not fleet_due and not recovery_due:
                raise MissionSchedulerError("calendar advanced to a timestamp with no due event")
            due_recovery_groups = self.recovery_manager.due_group_ids() if recovery_due else ()

            previous_owned = set() if self.fleet_session is None else set(self.fleet_session.owned_vessel_ids)
            recovering_before = set(self.recovery_manager.owned_vessel_ids)
            nominal_batch = None
            if nominal_due:
                nominal_batch = self.nominal.process_current_time(
                    blocked_vessel_ids=previous_owned | recovering_before
                )

            update = self.observer(self.world)
            self._has_registry_boundary = True
            if abs(update.time_min - self.world.time) > _EPS:
                raise MissionSchedulerError("observer returned an update for the wrong timestamp")
            dispatch = build_fleet_supervision_dispatch(update)
            current_owned = set(dispatch.owned_vessel_ids)
            self.nominal.suspend(current_owned)
            handoff_ids = self._handoff_ids(previous_owned, current_owned)

            recovery_updates: list[RouteRecoveryGroupUpdate] = []
            recovery_unavailable: list[RouteRecoveryGroupUnavailable] = []
            resumed_ids: list[str] = []
            preempted_ids: set[str] = set()

            preemptions = self.recovery_manager.pop_overlapping(current_owned)
            for preemption in preemptions:
                self.nominal.checkpoint_recovery_progress(preemption.vessel_results)
                preempted_ids.update(set(preemption.vessel_ids).intersection(current_owned))
                survivors = tuple(sorted(set(preemption.vessel_ids) - current_owned))
                if survivors:
                    recovery_item, resumed = self._add_recovery_group(survivors)
                    if isinstance(recovery_item, RouteRecoveryGroupUnavailable):
                        recovery_unavailable.append(recovery_item)
                    else:
                        recovery_updates.append(recovery_item)
                    resumed_ids.extend(resumed)

            fleet_update: FleetReviewSessionUpdate | None = None
            rebuilt = False
            initial_plans: tuple[FleetAppliedOwnerPlan, ...] = ()

            if self.fleet_session is None:
                if dispatch.owners:
                    initial_plans = self._start_fleet_session(update, dispatch)
                    rebuilt = self.fleet_session is not None
            else:
                fleet_update = self.fleet_session.ingest_registry_update(
                    update,
                    apply_due_plans=True,
                )
                if fleet_update.status is FleetReviewSessionStatus.TOPOLOGY_CHANGED:
                    initial_plans = self._start_fleet_session(update, dispatch)
                    rebuilt = self.fleet_session is not None
                elif fleet_update.status is FleetReviewSessionStatus.ALL_RELEASED:
                    self.fleet_session = None

            if handoff_ids:
                if not self.auto_route_handoff:
                    self._manual_handoff_ids = handoff_ids
                    self._record_event(
                        events,
                        MissionCalendarEvent(
                            self.world.time,
                            nominal_batch,
                            update,
                            dispatch,
                            fleet_update,
                            rebuilt,
                            initial_plans,
                            dispatch.owned_vessel_ids,
                            preempted_recovery_vessel_ids=tuple(sorted(preempted_ids)),
                        ),
                    )
                    return self._make_result(
                        stop_reason=MissionStopReason.ROUTE_HANDOFF_REQUIRED,
                        events=events,
                        route_handoff_ids=handoff_ids,
                    )
                recovery_item, resumed = self._add_recovery_group(handoff_ids)
                if isinstance(recovery_item, RouteRecoveryGroupUnavailable):
                    recovery_unavailable.append(recovery_item)
                else:
                    recovery_updates.append(recovery_item)
                resumed_ids.extend(resumed)

            surviving_due = tuple(
                group_id
                for group_id in due_recovery_groups
                if group_id in self.recovery_manager.group_ids
            )
            for recovery_item in self.recovery_manager.process_groups_current_time(surviving_due):
                if isinstance(recovery_item, RouteRecoveryGroupUnavailable):
                    recovery_unavailable.append(recovery_item)
                    continue
                recovery_updates.append(recovery_item)
                if recovery_item.completed:
                    records = self.nominal.resume_from_handoff(recovery_item.update.vessel_results)
                    resumed_ids.extend(record.vessel_id for record in records)

            primary_handoff = recovery_updates[-1].update if recovery_updates else None
            self._record_event(
                events,
                MissionCalendarEvent(
                    self.world.time,
                    nominal_batch,
                    update,
                    dispatch,
                    fleet_update,
                    rebuilt,
                    initial_plans,
                    dispatch.owned_vessel_ids,
                    primary_handoff,
                    tuple(sorted(set(resumed_ids))),
                    tuple(recovery_updates),
                    tuple(sorted(preempted_ids)),
                    tuple(recovery_unavailable),
                ),
            )

            if recovery_unavailable:
                for item in recovery_unavailable:
                    self.nominal.checkpoint_recovery_progress(item.vessel_results)
                self._recovery_unavailable_groups = tuple(recovery_unavailable)
                return self._make_result(
                    stop_reason=MissionStopReason.ROUTE_RECOVERY_UNAVAILABLE,
                    events=events,
                )

            if self.world.time >= end_time - _EPS:
                return self._make_result(stop_reason=MissionStopReason.END_TIME, events=events)

            if (
                self.nominal.all_routes_complete
                and self.fleet_session is None
                and not self.recovery_manager.group_ids
            ):
                return self._make_result(
                    stop_reason=MissionStopReason.ALL_NOMINAL_COMPLETE,
                    events=events,
                )

        raise MissionSchedulerError("mission event limit exceeded")


class UnifiedMissionScheduler:
    """Compatibility facade backed by one persistent ``UnifiedMissionSession``.

    Existing one-shot ``run_until`` calls retain their API.  Repeated calls on the same
    scheduler now resume the exact mission state rather than rebuilding fleet/recovery
    sessions from scratch.
    """

    def __init__(
        self,
        world: World2D,
        arrivals: Iterable[NominalArrivalSpec],
        registry: EncounterRegistry,
        *,
        controller_factory: MissionControllerFactory,
        observer: MissionObserver | None = None,
        auto_route_handoff: bool = True,
        safety_metric_config: PairSafetyMetricConfig | None = None,
    ) -> None:
        self.session = UnifiedMissionSession(
            world,
            arrivals,
            registry,
            controller_factory=controller_factory,
            observer=observer,
            auto_route_handoff=auto_route_handoff,
            safety_metric_config=safety_metric_config,
        )

    @property
    def world(self) -> World2D:
        return self.session.world

    @property
    def nominal(self) -> DynamicNominalControlSession:
        return self.session.nominal

    @property
    def registry(self) -> EncounterRegistry:
        return self.session.registry

    @property
    def event_history(self) -> tuple[MissionCalendarEvent, ...]:
        return self.session.event_history

    def resolve_recovery_unavailable(
        self,
        replacement_targets: Iterable[NominalFleetTarget],
    ) -> tuple[RouteRecoveryGroupUpdate | RouteRecoveryGroupUnavailable, ...]:
        return self.session.resolve_recovery_unavailable(replacement_targets)

    def run_until(
        self,
        *,
        end_time: float,
        initial_update: EncounterRegistryUpdate | None = None,
        max_events: int = 10000,
    ) -> MissionRunResult:
        return self.session.advance_until(
            end_time=end_time,
            initial_update=initial_update,
            max_events=max_events,
        )
