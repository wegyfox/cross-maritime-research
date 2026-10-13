"""Mixed shared-clock environmental avoidance and route ownership.

A119 extends the A117 single-clock discipline across the post-avoidance boundary.  A
PAIR owner that clears may continue through the existing A98/A99 route stages while
other PAIR/JOINT avoidance owners remain active.  A JOINT owner that becomes locally
handoff-eligible, and surviving channels explicitly released by A118, may be admitted
as fleet-released route owners from one canonical ownership observation.

The coordinator owns physical advancement and fleet observations.  Before any route
handoff/nominal command refresh, or any avoidance replan, one current canonical fleet
observation is shared across all active owners.  A new encounter touching a route-owned
vessel therefore recaptures before that route writer can mutate its command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping

from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_fleet_guarded_joint_control import EnvironmentalFleetGuardedJointControlSession
from .environmental_fleet_guarded_pair_control import EnvironmentalFleetGuardedPairControlSession
from .environmental_fleet_joint_ownership import EnvironmentalJointOwnershipAction
from .environmental_fleet_multi_owner_recapture import (
    EnvironmentalStartedJointOwner,
    EnvironmentalStartedOwner,
    EnvironmentalStartedPairOwner,
)
from .environmental_fleet_ownership import (
    EnvironmentalFleetOwnershipObservation,
    EnvironmentalPairOwnershipAction,
)
from .environmental_joint_control_cycle import EnvironmentalJointControlStage
from .environmental_joint_execution_policy import EnvironmentalJointExecutionPolicyAction
from .environmental_multi_owner_control import (
    EnvironmentalMultiOwnerControlResult,
    EnvironmentalMultiOwnerControlSession,
    EnvironmentalMultiOwnerControlStopReason,
)
from .environmental_multi_owner_rebootstrap import (
    EnvironmentalMultiOwnerRebootstrapResult,
)
from .environmental_nominal_route_session import (
    EnvironmentalNominalRouteMode,
    EnvironmentalNominalRouteSession,
    EnvironmentalNominalRouteSessionConfig,
    EnvironmentalNominalRouteStopReason,
)
from .environmental_pair_control_cycle import EnvironmentalPairControlStage
from .environmental_pair_execution_policy import EnvironmentalPairExecutionPolicyAction
from .environmental_released_route_handoff import (
    build_environmental_released_route_handoff_binding,
)
from .environmental_route_handoff import (
    EnvironmentalRouteHandoffBinding,
    EnvironmentalRouteHandoffTarget,
)
from .environmental_route_handoff_session import (
    EnvironmentalRouteHandoffSession,
    EnvironmentalRouteHandoffSessionConfig,
    EnvironmentalRouteHandoffSessionMode,
    EnvironmentalRouteHandoffSessionStopReason,
)
from .environmental_world import EnvironmentalWorldSnapshot
from .route_exit import RouteEndAction
from .same_route_following import SameRouteFollowingSupervisor
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalMixedOwnerControlError(RuntimeError):
    """Base error for invalid A119 mixed-owner orchestration."""


class EnvironmentalMixedOwnerStopReason(str, Enum):
    END_TIME = "end_time"
    RECAPTURE_REQUIRED = "recapture_required"
    ROUTE_DISPOSITION_REQUIRED = "route_disposition_required"
    COMMAND_OVERRIDDEN = "command_overridden"
    ALL_CONTROL_COMPLETE = "all_control_complete"


class EnvironmentalMixedRouteStage(str, Enum):
    ROUTE_HANDOFF = "route_handoff"
    NOMINAL_ROUTE = "nominal_route"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class EnvironmentalMixedOwnerControlConfig:
    route_handoff: EnvironmentalRouteHandoffSessionConfig = EnvironmentalRouteHandoffSessionConfig()
    nominal_route: EnvironmentalNominalRouteSessionConfig = EnvironmentalNominalRouteSessionConfig()
    route_fleet_review_interval_min: float = 1.0
    max_boundaries_per_run: int = 8192

    def __post_init__(self) -> None:
        if not isinstance(self.route_handoff, EnvironmentalRouteHandoffSessionConfig):
            raise TypeError("route_handoff must be EnvironmentalRouteHandoffSessionConfig")
        if not isinstance(self.nominal_route, EnvironmentalNominalRouteSessionConfig):
            raise TypeError("nominal_route must be EnvironmentalNominalRouteSessionConfig")
        if not isfinite(self.route_fleet_review_interval_min) or self.route_fleet_review_interval_min <= 0.0:
            raise ValueError("route_fleet_review_interval_min must be finite and positive")
        if self.max_boundaries_per_run < 1:
            raise ValueError("max_boundaries_per_run must be at least 1")


@dataclass(slots=True)
class _RouteOwner:
    owner_id: str
    vessel_ids: tuple[str, ...]
    binding: EnvironmentalRouteHandoffBinding
    handoff_session: EnvironmentalRouteHandoffSession
    stage: EnvironmentalMixedRouteStage = EnvironmentalMixedRouteStage.ROUTE_HANDOFF
    nominal_session: EnvironmentalNominalRouteSession | None = None
    pair_source: EnvironmentalStartedPairOwner | None = None
    next_fleet_review_time: float | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentalMixedRouteCompletion:
    owner_id: str
    vessel_ids: tuple[str, ...]
    expected_commands: tuple[tuple[str, WaterRelativeManeuverCommand], ...]


@dataclass(frozen=True, slots=True)
class EnvironmentalMixedSameRouteSupervision:
    time_min: float
    owner_id: str
    follower_id: str
    leader_id: str
    previous_command: WaterRelativeManeuverCommand
    issued_command: WaterRelativeManeuverCommand
    max_follower_speed_nm_per_min: float


@dataclass(frozen=True, slots=True)
class EnvironmentalMixedOwnerBoundary:
    time_min: float
    avoidance_execution_due_owner_ids: tuple[str, ...]
    route_due_owner_ids: tuple[str, ...]
    fleet_review_due_owner_ids: tuple[str, ...]
    membership_event_count: int
    removed_vessel_ids: tuple[str, ...]
    fleet_observation: EnvironmentalFleetOwnershipObservation | None
    recapture_owner_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentalMixedOwnerControlResult:
    snapshot: EnvironmentalWorldSnapshot
    stop_reason: EnvironmentalMixedOwnerStopReason
    boundaries: tuple[EnvironmentalMixedOwnerBoundary, ...]
    fleet_observations: tuple[EnvironmentalFleetOwnershipObservation, ...]
    avoidance_owner_ids: tuple[str, ...]
    route_owner_ids: tuple[str, ...]
    completed_route_owner_ids: tuple[str, ...]
    recapture_observation: EnvironmentalFleetOwnershipObservation | None = None
    recapture_owner_ids: tuple[str, ...] = ()
    overridden_vessel_ids: tuple[str, ...] = ()
    route_disposition_observation: EnvironmentalFleetOwnershipObservation | None = None
    route_completions: tuple[EnvironmentalMixedRouteCompletion, ...] = ()
    same_route_supervisions: tuple[EnvironmentalMixedSameRouteSupervision, ...] = ()


class EnvironmentalMixedOwnerControlSession:
    """Run avoidance and post-avoidance route owners on one EnvironmentalWorld2D clock."""

    def __init__(
        self,
        coordinator: EnvironmentalMultiOwnerControlSession,
        source: EnvironmentalMultiOwnerControlResult | EnvironmentalMultiOwnerRebootstrapResult,
        route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
        *,
        config: EnvironmentalMixedOwnerControlConfig = EnvironmentalMixedOwnerControlConfig(),
        profile_catalog: VesselProfileCatalog | None = None,
        same_route_following: SameRouteFollowingSupervisor | None = None,
        persistent_same_route_following: SameRouteFollowingSupervisor | None = None,
    ) -> None:
        if not isinstance(coordinator, EnvironmentalMultiOwnerControlSession):
            raise TypeError("coordinator must be EnvironmentalMultiOwnerControlSession")
        if not isinstance(source, (EnvironmentalMultiOwnerControlResult, EnvironmentalMultiOwnerRebootstrapResult)):
            raise TypeError("source must be an a117 control result or a118 rebootstrap result")
        if not isinstance(route_targets, Mapping):
            raise TypeError("route_targets must be a mapping")
        if not isinstance(config, EnvironmentalMixedOwnerControlConfig):
            raise TypeError("config must be EnvironmentalMixedOwnerControlConfig")
        if same_route_following is not None and not isinstance(
            same_route_following, SameRouteFollowingSupervisor
        ):
            raise TypeError("same_route_following must be SameRouteFollowingSupervisor or None")
        if persistent_same_route_following is not None and not isinstance(
            persistent_same_route_following, SameRouteFollowingSupervisor
        ):
            raise TypeError(
                "persistent_same_route_following must be SameRouteFollowingSupervisor or None"
            )

        self.coordinator = coordinator
        self.registry = coordinator.registry
        self.world = coordinator.world
        self.source = source
        self.route_targets = route_targets
        self.config = config
        self.profile_catalog = profile_catalog
        self.same_route_following = same_route_following
        self.persistent_same_route_following = persistent_same_route_following
        self._boundaries: list[EnvironmentalMixedOwnerBoundary] = []
        self._route_owners: dict[str, _RouteOwner] = {}
        self._retired_joint_owner_ids: set[str] = set()
        self._completed_route_owner_ids: list[str] = []
        self._same_route_supervisions: list[EnvironmentalMixedSameRouteSupervision] = []
        self._current_write_observation: EnvironmentalFleetOwnershipObservation | None = None
        # Transactional source-control seam. Newly admitted vessels that already touch
        # the canonical avoidance dispatch remain under a zero-write pending owner until
        # same-timestamp recapture commits ordinary route or avoidance ownership.
        self._pending_source_control_ids: dict[str, str] = {}
        # Optional A151 source-boundary extension.  The default remains None so every
        # pre-A151 caller keeps the exact historical A119 calendar/observation path.
        self._source_boundary_extension = None
        # A118 deliberately leaves the old A117 owner tuple untouched when recapture
        # produces no replacement avoidance owner.  Mixed control must therefore bind
        # runtime ownership to the A118 result, not blindly revive those stale sessions.
        if isinstance(source, EnvironmentalMultiOwnerRebootstrapResult):
            self._managed_started_owner_ids = {
                item.owner.component_id for item in source.started_owners
            }
        else:
            self._managed_started_owner_ids = {
                item.owner.component_id for item in coordinator.started_owners
            }

        self._validate_source_boundary()
        self._bootstrap_route_owners()
        self._validate_disjoint_channels()

    @classmethod
    def from_idle_coordinator(
        cls,
        coordinator: EnvironmentalMultiOwnerControlSession,
        route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
        *,
        config: EnvironmentalMixedOwnerControlConfig = EnvironmentalMixedOwnerControlConfig(),
        profile_catalog: VesselProfileCatalog | None = None,
        same_route_following: SameRouteFollowingSupervisor | None = None,
        persistent_same_route_following: SameRouteFollowingSupervisor | None = None,
    ) -> "EnvironmentalMixedOwnerControlSession":
        """Construct an empty mixed stage around an ownerless A117 coordinator."""

        if not isinstance(coordinator, EnvironmentalMultiOwnerControlSession):
            raise TypeError("coordinator must be EnvironmentalMultiOwnerControlSession")
        if not isinstance(route_targets, Mapping):
            raise TypeError("route_targets must be a mapping")
        if not isinstance(config, EnvironmentalMixedOwnerControlConfig):
            raise TypeError("config must be EnvironmentalMixedOwnerControlConfig")
        if same_route_following is not None and not isinstance(
            same_route_following, SameRouteFollowingSupervisor
        ):
            raise TypeError("same_route_following must be SameRouteFollowingSupervisor or None")
        if persistent_same_route_following is not None and not isinstance(
            persistent_same_route_following, SameRouteFollowingSupervisor
        ):
            raise TypeError(
                "persistent_same_route_following must be SameRouteFollowingSupervisor or None"
            )
        if coordinator.started_owners:
            raise EnvironmentalMixedOwnerControlError(
                "idle mixed construction requires an ownerless coordinator"
            )
        if not coordinator.fleet_observations or abs(
            coordinator.fleet_observations[-1].time_min - coordinator.world.time
        ) > _EPS:
            raise EnvironmentalMixedOwnerControlError(
                "idle mixed construction requires current committed fleet evidence"
            )

        self = cls.__new__(cls)
        self.coordinator = coordinator
        self.registry = coordinator.registry
        self.world = coordinator.world
        self.source = None
        self.route_targets = route_targets
        self.config = config
        self.profile_catalog = profile_catalog
        self.same_route_following = same_route_following
        self.persistent_same_route_following = persistent_same_route_following
        self._boundaries = []
        self._route_owners = {}
        self._retired_joint_owner_ids = set()
        self._completed_route_owner_ids = []
        self._same_route_supervisions = []
        self._current_write_observation = None
        self._pending_source_control_ids = {}
        self._source_boundary_extension = None
        self._managed_started_owner_ids = set()
        self._validate_disjoint_channels()
        return self

    def _pending_remove_route_complete_ids(self) -> tuple[str, ...]:
        """Return route-owned REMOVE vessels complete at the current physical timestamp.

        A route session can recognize completion on the same timestamp at which another
        route-owned vessel asks the same-route supervisor for a speed ceiling.  Until the
        outer route-end disposition transaction runs, the completed vessel is still present
        in ``EnvironmentalWorld2D``.  It must not be projected back onto the route endpoint
        and reused as a longitudinal leader while it is pending REMOVE.

        The completion test is delegated to the active route session's own projection
        semantics rather than inferred from the generic same-route geometry.  STOP and
        CONTINUE route-end actions remain visible because they require separate post-route
        physical semantics and must not be silently removed from safety reasoning.
        """

        completed: set[str] = set()
        for owner in self._route_owners.values():
            if owner.stage is EnvironmentalMixedRouteStage.COMPLETE:
                continue
            if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF:
                session = owner.handoff_session
                for vessel_id in owner.vessel_ids:
                    target = owner.binding.for_vessel(vessel_id).target
                    if target.end_action is not RouteEndAction.REMOVE:
                        continue
                    projection, follower_state = session._projection(vessel_id)  # noqa: SLF001
                    route_tolerance = max(
                        target.navigator.follower.position_tolerance,
                        session.config.cross_track_tolerance_nm,
                    )
                    if follower_state.completed and projection.distance <= route_tolerance + _EPS:
                        completed.add(vessel_id)
                continue

            session_n = owner.nominal_session
            if session_n is None:
                raise EnvironmentalMixedOwnerControlError(
                    f"nominal route owner {owner.owner_id!r} lacks a nominal session"
                )
            for vessel_id in owner.vessel_ids:
                target = owner.binding.for_vessel(vessel_id).target
                if target.end_action is not RouteEndAction.REMOVE:
                    continue
                _projection, follower_state = session_n._projection(vessel_id)  # noqa: SLF001
                if follower_state.completed:
                    completed.add(vessel_id)
        return tuple(sorted(completed))

    def _same_route_supervision_targets(self) -> Mapping[str, EnvironmentalRouteHandoffTarget]:
        pending_remove = set(self._pending_remove_route_complete_ids())
        if not pending_remove:
            return self.route_targets
        return {
            vessel_id: target
            for vessel_id, target in self.route_targets.items()
            if vessel_id not in pending_remove
        }

    def _route_command_transform(
        self,
        vessel_id: str,
        command: WaterRelativeManeuverCommand,
    ) -> WaterRelativeManeuverCommand:
        """Apply the global physical same-route speed ceiling to route-owned writers.

        The route controller remains authoritative for heading/ground-track intent.  This
        optional safety layer only lowers the requested through-water speed when the same
        physical following model used by avoidance/source admission requires it.  With no
        configured supervisor the command is returned byte-for-byte unchanged.
        """

        supervisor = self.same_route_following
        if supervisor is None:
            return command
        constraints = supervisor.constraints_for(
            self.world,
            (vessel_id,),
            self._same_route_supervision_targets(),
            self.profile_catalog,
        )
        if not constraints:
            return command
        if len(constraints) != 1:
            raise EnvironmentalMixedOwnerControlError(
                f"same-route route supervision produced multiple constraints for {vessel_id!r}"
            )
        ceiling = constraints[0].max_follower_speed_nm_per_min
        if command.target_through_water_speed <= ceiling + _EPS:
            return command
        return WaterRelativeManeuverCommand(
            target_through_water_heading_deg=command.target_through_water_heading_deg,
            target_through_water_speed=ceiling,
        )

    @property
    def same_route_supervisions(self) -> tuple[EnvironmentalMixedSameRouteSupervision, ...]:
        return tuple(self._same_route_supervisions)

    @staticmethod
    def _route_cursor(owner: _RouteOwner, vessel_id: str):
        if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF:
            return owner.handoff_session._cursors.get(vessel_id)  # noqa: SLF001
        if owner.stage is EnvironmentalMixedRouteStage.NOMINAL_ROUTE and owner.nominal_session is not None:
            return owner.nominal_session._cursors.get(vessel_id)  # noqa: SLF001
        return None

    def _apply_persistent_same_route_route_supervision(
        self,
        owner_ids: tuple[str, ...],
        observation: EnvironmentalFleetOwnershipObservation,
    ) -> tuple[EnvironmentalMixedSameRouteSupervision, ...]:
        """Apply the physical following ceiling even when route navigation is idle.

        A98/A99 route cursors may legitimately have no next navigation review after
        NOMINAL_READY.  Physical same-route following is a different lifecycle: it
        must continue to be reviewed while route authority remains active.  This
        method uses the already-committed fleet observation at the A119 fleet-review
        boundary and only lowers speed; heading/route intent is left untouched.
        """

        supervisor = self.persistent_same_route_following
        if supervisor is None or not owner_ids:
            return ()
        if abs(observation.time_min - self.world.time) > _EPS:
            raise EnvironmentalMixedOwnerControlError(
                "persistent same-route supervision requires current fleet evidence"
            )

        vessel_owner: dict[str, str] = {}
        for owner_id in owner_ids:
            owner = self._route_owners.get(owner_id)
            if owner is None or owner.stage is EnvironmentalMixedRouteStage.COMPLETE:
                continue
            for vessel_id in owner.vessel_ids:
                vessel_owner[vessel_id] = owner_id
        if not vessel_owner:
            return ()

        constraints = supervisor.constraints_for(
            self.world,
            tuple(sorted(vessel_owner)),
            self._same_route_supervision_targets(),
            self.profile_catalog,
        )
        pending: list[tuple[str, WaterRelativeManeuverCommand]] = []
        records: list[EnvironmentalMixedSameRouteSupervision] = []
        for constraint in constraints:
            follower_id = constraint.follower_id
            owner_id = vessel_owner.get(follower_id)
            if owner_id is None:
                continue
            current = self.world.command_for(follower_id)
            ceiling = constraint.max_follower_speed_nm_per_min
            if current.target_through_water_speed <= ceiling + _EPS:
                continue
            command = WaterRelativeManeuverCommand(
                target_through_water_heading_deg=current.target_through_water_heading_deg,
                target_through_water_speed=ceiling,
            )
            pending.append((follower_id, command))
            records.append(
                EnvironmentalMixedSameRouteSupervision(
                    time_min=self.world.time,
                    owner_id=owner_id,
                    follower_id=follower_id,
                    leader_id=constraint.leader_id,
                    previous_command=current,
                    issued_command=command,
                    max_follower_speed_nm_per_min=ceiling,
                )
            )

        if not pending:
            return ()

        previous_observation = self._current_write_observation
        self._current_write_observation = observation
        try:
            for record in records:
                self._route_write_guard(record.owner_id)
            self.world.apply_current_transaction_atomic(commands=tuple(pending))
            for follower_id, command in pending:
                owner = self._route_owners[vessel_owner[follower_id]]
                cursor = self._route_cursor(owner, follower_id)
                if cursor is not None:
                    cursor.issued_command = command
        finally:
            self._current_write_observation = previous_observation

        self._same_route_supervisions.extend(records)
        return tuple(records)

    def _install_source_boundary_extension(self, extension) -> None:
        """Install one opt-in pre-observation source calendar extension.

        A151 deliberately keeps this hook private to the package.  The legacy/A141
        runtime never installs it, so its candidate calendar and observation ordering
        are byte-for-byte governed by the pre-existing branches below.
        """

        if extension is None:
            raise TypeError("source boundary extension must be non-None")
        if self._source_boundary_extension is not None:
            raise EnvironmentalMixedOwnerControlError(
                "a119 source boundary extension is already installed"
            )
        self._source_boundary_extension = extension

    def _source_boundary_time(self, native_frozen_until: float) -> float | None:
        extension = self._source_boundary_extension
        if extension is None:
            return None
        return extension.next_boundary_time(native_frozen_until)

    def _apply_source_pre_observation_phase(
        self,
        *,
        native_boundary_due: bool,
        source_boundary_due: bool,
    ) -> tuple[str, ...]:
        extension = self._source_boundary_extension
        if extension is None:
            return ()
        return extension.apply_pre_observation_phase(
            native_boundary_due=native_boundary_due,
            source_boundary_due=source_boundary_due,
        )

    @property
    def fleet_observations(self) -> tuple[EnvironmentalFleetOwnershipObservation, ...]:
        return self.coordinator.fleet_observations

    @property
    def boundaries(self) -> tuple[EnvironmentalMixedOwnerBoundary, ...]:
        return tuple(self._boundaries)

    def schedule_arrival(self, **kwargs) -> None:
        self.coordinator.schedule_arrival(**kwargs)

    def schedule_removal(self, **kwargs) -> None:
        self.coordinator.schedule_removal(**kwargs)

    def _validate_source_boundary(self) -> None:
        if isinstance(self.source, EnvironmentalMultiOwnerControlResult):
            if self.source.stop_reason is not EnvironmentalMultiOwnerControlStopReason.OWNER_STAGE_TRANSFER_REQUIRED:
                raise EnvironmentalMixedOwnerControlError(
                    "a119 direct a117 source must be OWNER_STAGE_TRANSFER_REQUIRED"
                )
            if self.source.snapshot != self.world.snapshot():
                raise EnvironmentalMixedOwnerControlError(
                    "world no longer matches the a117 owner-stage transfer boundary"
                )
        else:
            if self.source.snapshot_after != self.world.snapshot():
                raise EnvironmentalMixedOwnerControlError(
                    "world no longer matches the a118 rebootstrap boundary"
                )
            if abs(self.source.committed_observation_time - self.world.time) > _EPS:
                raise EnvironmentalMixedOwnerControlError(
                    "a118 committed observation time no longer matches world"
                )

    def _target_tuple(self, vessel_ids: tuple[str, ...]) -> tuple[EnvironmentalRouteHandoffTarget, ...]:
        targets: list[EnvironmentalRouteHandoffTarget] = []
        for vessel_id in vessel_ids:
            try:
                target = self.route_targets[vessel_id]
            except KeyError as exc:
                raise EnvironmentalMixedOwnerControlError(
                    f"missing route target for released vessel {vessel_id!r}"
                ) from exc
            if not isinstance(target, EnvironmentalRouteHandoffTarget):
                raise TypeError(
                    f"route target for {vessel_id!r} must be EnvironmentalRouteHandoffTarget"
                )
            if target.vessel_id != vessel_id:
                raise EnvironmentalMixedOwnerControlError(
                    f"route target key {vessel_id!r} disagrees with target vessel id {target.vessel_id!r}"
                )
            targets.append(target)
        return tuple(targets)

    def _current_observation(self) -> EnvironmentalFleetOwnershipObservation:
        observation = self.coordinator._shared_observation()  # noqa: SLF001 - A119 shares A117 registry ownership
        # A PAIR owner transferred into route handoff is no longer in the coordinator's
        # active avoidance-owner tuple, but its guarded cycle can still be the write guard
        # for the route session.  Publish the one canonical same-timestamp observation to
        # those retained guards as well; otherwise a route command write can attempt a
        # second EncounterRegistry update at the same world time.
        for owner in self._route_owners.values():
            if owner.pair_source is None:
                continue
            guard = owner.pair_source.guarded_session
            if (
                not guard._fleet_observations  # noqa: SLF001
                or abs(guard._fleet_observations[-1].time_min - self.world.time) > _EPS  # noqa: SLF001
            ):
                guard._fleet_observations.append(observation)  # noqa: SLF001
        return observation

    def _route_write_guard(self, owner_id: str) -> None:
        observation = self._current_write_observation
        if observation is None or abs(observation.time_min - self.world.time) > _EPS:
            raise EnvironmentalMixedOwnerControlError(
                f"route owner {owner_id!r} attempted command write without current shared fleet evidence"
            )
        owner = self._route_owners[owner_id]
        touching = tuple(
            item for item in observation.dispatch.owners if set(item.vessel_ids).intersection(owner.vessel_ids)
        )
        if touching:
            raise EnvironmentalMixedOwnerControlError(
                f"route owner {owner_id!r} lost fleet clearance before command write"
            )

    def _make_released_route_owner(
        self,
        owner_id: str,
        vessel_ids: tuple[str, ...],
        observation: EnvironmentalFleetOwnershipObservation,
        source_snapshot: EnvironmentalWorldSnapshot,
        expected_commands: Mapping[str, WaterRelativeManeuverCommand],
    ) -> _RouteOwner:
        targets = self._target_tuple(vessel_ids)
        binding = build_environmental_released_route_handoff_binding(
            self.world,
            observation,
            source_snapshot,
            targets,
            expected_commands,
            profile_catalog=self.profile_catalog,
        )
        handoff = EnvironmentalRouteHandoffSession(
            self.world,
            binding,
            config=self.config.route_handoff,
            profile_catalog=self.profile_catalog,
            before_command_write=lambda owner_id=owner_id: self._route_write_guard(owner_id),
            command_transform=self._route_command_transform if self.same_route_following is not None else None,
        )
        return _RouteOwner(
            owner_id,
            vessel_ids,
            binding,
            handoff,
            next_fleet_review_time=self.world.time + self.config.route_fleet_review_interval_min,
        )

    @staticmethod
    def _joint_expected_commands(
        started: EnvironmentalStartedJointOwner,
    ) -> dict[str, WaterRelativeManeuverCommand]:
        execution = started.joint_cycle._execution_session  # noqa: SLF001
        return dict(execution._last_commands)  # noqa: SLF001

    def _bootstrap_route_owners(self) -> None:
        # A118 released channels already carry the exact command tuple captured before its
        # owner-set transaction.  Their canonical observation is still current.
        if isinstance(self.source, EnvironmentalMultiOwnerRebootstrapResult):
            if self.source.released_vessel_ids:
                observation = self._current_observation()
                expected = dict(self.source.released_commands)
                owner_id = "released:" + "|".join(self.source.released_vessel_ids)
                self._route_owners[owner_id] = self._make_released_route_owner(
                    owner_id,
                    self.source.released_vessel_ids,
                    observation,
                    self.source.snapshot_after,
                    expected,
                )
            return

        transfer_ids = set(self.source.stage_transfer_owner_ids)
        by_id = {started.owner.component_id: started for started in self.coordinator.started_owners}
        missing = transfer_ids.difference(by_id)
        if missing:
            raise EnvironmentalMixedOwnerControlError(
                f"a117 stage-transfer owners are no longer present: {sorted(missing)!r}"
            )

        observation: EnvironmentalFleetOwnershipObservation | None = None
        for owner_id in self.source.stage_transfer_owner_ids:
            started = by_id[owner_id]
            if isinstance(started, EnvironmentalStartedPairOwner):
                cycle = started.pair_cycle
                if cycle.stage is not EnvironmentalPairControlStage.ROUTE_HANDOFF:
                    raise EnvironmentalMixedOwnerControlError(
                        f"PAIR transfer owner {owner_id!r} is not at ROUTE_HANDOFF"
                    )
                if cycle._handoff_binding is None or cycle._handoff_session is None:  # noqa: SLF001
                    raise EnvironmentalMixedOwnerControlError(
                        f"PAIR transfer owner {owner_id!r} lacks its route-handoff state"
                    )
                cycle._handoff_session.command_transform = (  # noqa: SLF001
                    self._route_command_transform if self.same_route_following is not None else None
                )
                self._route_owners[owner_id] = _RouteOwner(
                    owner_id,
                    started.owner.vessel_ids,
                    cycle._handoff_binding,  # noqa: SLF001
                    cycle._handoff_session,  # noqa: SLF001
                    pair_source=started,
                    next_fleet_review_time=started.guarded_session.next_review_time,
                )
                continue

            cycle_j = started.joint_cycle
            if cycle_j.stage is not EnvironmentalJointControlStage.HANDOFF_ELIGIBLE:
                raise EnvironmentalMixedOwnerControlError(
                    f"JOINT transfer owner {owner_id!r} is not HANDOFF_ELIGIBLE"
                )
            if observation is None:
                observation = self._current_observation()
            expected = self._joint_expected_commands(started)
            route_owner = self._make_released_route_owner(
                "released:" + owner_id,
                started.owner.vessel_ids,
                observation,
                self.source.snapshot,
                expected,
            )
            self._route_owners[route_owner.owner_id] = route_owner
            self._retired_joint_owner_ids.add(owner_id)

    def admit_current_clear_route_targets(
        self,
        vessel_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Give current clear vessel channels nominal route authority without re-observation.

        A137 uses this only at an already committed caller/membership boundary.  A vessel
        touched by the canonical avoidance dispatch is deliberately skipped: existing
        A119/A120/A122 ownership transitions will eventually release that channel to route
        control.  Clear channels are bound to their exact current command and snapshot via
        the same released-route admission contract used after avoidance.
        """

        ids = tuple(vessel_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("clear route admission vessel ids must be unique")
        if not ids:
            return ()

        observation = self._current_observation()
        if abs(observation.time_min - self.world.time) > _EPS:
            raise EnvironmentalMixedOwnerControlError(
                "clear route admission requires current committed fleet evidence"
            )
        snapshot = self.world.snapshot()
        active_ids = {item.vessel_id for item in snapshot.vessels}
        missing = set(ids) - active_ids
        if missing:
            raise EnvironmentalMixedOwnerControlError(
                f"clear route admission references inactive vessels: {sorted(missing)!r}"
            )

        claimed = {
            vessel_id
            for started in self._avoidance_started()
            for vessel_id in started.owner.vessel_ids
        }
        claimed.update(
            vessel_id
            for owner in self._route_owners.values()
            if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
            for vessel_id in owner.vessel_ids
        )

        touching_dispatch = {
            vessel_id
            for owner in observation.dispatch.owners
            for vessel_id in owner.vessel_ids
        }
        admitted: list[str] = []
        prepared: list[tuple[str, _RouteOwner]] = []
        sequence = len(self._route_owners) + len(self._completed_route_owner_ids)
        for vessel_id in ids:
            if vessel_id in claimed or vessel_id in touching_dispatch:
                continue
            # Validate target identity before creating any owner object.
            self._target_tuple((vessel_id,))
            sequence += 1
            owner_id = f"nominal:auto:{sequence}:{vessel_id}"
            expected = {vessel_id: self.world.command_for(vessel_id)}
            owner = self._make_released_route_owner(
                owner_id,
                (vessel_id,),
                observation,
                snapshot,
                expected,
            )
            prepared.append((owner_id, owner))
            claimed.add(vessel_id)

        for owner_id, owner in prepared:
            self._route_owners[owner_id] = owner
            admitted.append(owner_id)
        self._validate_disjoint_channels()
        return tuple(admitted)

    def _stage_source_control_ownership(
        self,
        vessel_ids: tuple[str, ...],
        observation: EnvironmentalFleetOwnershipObservation,
    ) -> tuple[str, ...]:
        """Bridge source admission into route or pending-recapture ownership.

        Clear source channels receive ordinary nominal route authority.  A newly
        admitted vessel already touching the canonical avoidance dispatch receives a
        zero-write ``SOURCE_PENDING_RECAPTURE`` owner instead.  This state exists only
        to make the admission→recapture transaction ownership-total; it cannot emit a
        command and is removed as soon as ordinary route/avoidance ownership commits.
        """
        ids = tuple(vessel_ids)
        if not ids:
            return ()
        touching = {
            vessel_id
            for dispatch_owner in observation.dispatch.owners
            for vessel_id in dispatch_owner.vessel_ids
        }
        existing_claims = {
            vessel_id
            for started in self._avoidance_started()
            for vessel_id in started.owner.vessel_ids
        }
        existing_claims.update(
            vessel_id
            for owner in self._route_owners.values()
            if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
            for vessel_id in owner.vessel_ids
        )
        clear_ids = tuple(v for v in ids if v not in touching and v not in existing_claims)
        route_owner_ids = self.admit_current_clear_route_targets(clear_ids)
        for vessel_id in ids:
            if vessel_id in existing_claims or vessel_id in clear_ids:
                self._pending_source_control_ids.pop(vessel_id, None)
                continue
            if vessel_id in touching:
                self._pending_source_control_ids[vessel_id] = f"source-pending:{vessel_id}"
        return route_owner_ids

    def _reconcile_pending_source_control_ownership(self) -> tuple[str, ...]:
        """Drop pending owners once ordinary ownership or physical removal commits."""
        claimed = {
            vessel_id
            for started in self._avoidance_started()
            for vessel_id in started.owner.vessel_ids
        }
        claimed.update(
            vessel_id
            for owner in self._route_owners.values()
            if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
            for vessel_id in owner.vessel_ids
        )
        active = {item.vessel_id for item in self.world.snapshot().vessels}
        cleared = tuple(
            sorted(v for v in self._pending_source_control_ids if v in claimed or v not in active)
        )
        for vessel_id in cleared:
            self._pending_source_control_ids.pop(vessel_id, None)
        return cleared

    def _avoidance_started(self) -> tuple[EnvironmentalStartedOwner, ...]:
        items: list[EnvironmentalStartedOwner] = []
        for started in self.coordinator.started_owners:
            if started.owner.component_id not in self._managed_started_owner_ids:
                continue
            if started.owner.component_id in self._retired_joint_owner_ids:
                continue
            if isinstance(started, EnvironmentalStartedPairOwner):
                if started.pair_cycle.stage is EnvironmentalPairControlStage.AVOIDANCE:
                    items.append(started)
            elif started.joint_cycle.stage is EnvironmentalJointControlStage.AVOIDANCE:
                items.append(started)
        return tuple(items)

    def _validate_disjoint_channels(self) -> None:
        claimed: set[str] = set()
        for started in self._avoidance_started():
            overlap = claimed.intersection(started.owner.vessel_ids)
            if overlap:
                raise EnvironmentalMixedOwnerControlError(
                    f"avoidance owners overlap mixed command channels: {sorted(overlap)!r}"
                )
            claimed.update(started.owner.vessel_ids)
        for owner in self._route_owners.values():
            overlap = claimed.intersection(owner.vessel_ids)
            if overlap:
                raise EnvironmentalMixedOwnerControlError(
                    f"route and avoidance owners overlap command channels: {sorted(overlap)!r}"
                )
            claimed.update(owner.vessel_ids)

    def _assess_current(
        self,
        observation: EnvironmentalFleetOwnershipObservation,
        removed: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        recapture: list[str] = []
        removed_set = set(removed)

        for started in self.coordinator.started_owners:
            owner_id = started.owner.component_id
            if owner_id not in self._managed_started_owner_ids:
                continue
            if owner_id in self._retired_joint_owner_ids:
                continue
            if isinstance(started, EnvironmentalStartedPairOwner):
                stage = started.pair_cycle.stage
                if stage is EnvironmentalPairControlStage.COMPLETE:
                    continue
                guard: EnvironmentalFleetGuardedPairControlSession = started.guarded_session
                if not guard._fleet_observations or guard._fleet_observations[-1] != observation:  # noqa: SLF001
                    guard._fleet_observations.append(observation)  # noqa: SLF001
                controlled_removed = tuple(v for v in removed if v in started.owner.vessel_ids)
                pending = started.pair_cycle._pending_policy  # noqa: SLF001
                requested = None if pending is None else pending.action
                decision = guard._assess_current(  # noqa: SLF001
                    stage,
                    requested_policy_action=requested,
                    removed_controlled=controlled_removed,
                )
                if guard.recapture is not None or (
                    decision is not None and decision.action is EnvironmentalPairOwnershipAction.RECAPTURE_REQUIRED
                ):
                    recapture.append(owner_id)
            else:
                if started.joint_cycle.stage is not EnvironmentalJointControlStage.AVOIDANCE:
                    continue
                guard_j: EnvironmentalFleetGuardedJointControlSession = started.guarded_session
                if not guard_j._fleet_observations or guard_j._fleet_observations[-1] != observation:  # noqa: SLF001
                    guard_j._fleet_observations.append(observation)  # noqa: SLF001
                controlled_removed = tuple(v for v in removed if v in started.owner.vessel_ids)
                pending = started.joint_cycle._pending_policy  # noqa: SLF001
                requested = None if pending is None else pending.action
                decision_j = guard_j._assess_current(  # noqa: SLF001
                    requested_policy_action=requested,
                    removed_controlled=controlled_removed,
                )
                if guard_j.recapture is not None or (
                    decision_j is not None and decision_j.action is EnvironmentalJointOwnershipAction.RECAPTURE_REQUIRED
                ):
                    recapture.append(owner_id)

        for owner_id, owner in self._route_owners.items():
            if owner.stage is EnvironmentalMixedRouteStage.COMPLETE:
                continue
            if removed_set.intersection(owner.vessel_ids):
                recapture.append(owner_id)
                continue
            touching = tuple(
                item for item in observation.dispatch.owners if set(item.vessel_ids).intersection(owner.vessel_ids)
            )
            if touching:
                recapture.append(owner_id)
        return tuple(dict.fromkeys(recapture))

    def _ensure_current_observation(
        self,
        *,
        removed: tuple[str, ...] = (),
    ) -> tuple[EnvironmentalFleetOwnershipObservation, tuple[str, ...]]:
        observation = self._current_observation()
        recapture = self._assess_current(observation, removed)
        return observation, recapture

    def _route_event_time(self, owner: _RouteOwner) -> float | None:
        if owner.stage is EnvironmentalMixedRouteStage.COMPLETE:
            return None
        if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF:
            if not owner.handoff_session._started:  # noqa: SLF001
                return self.world.time
            return owner.handoff_session.next_event_time()
        assert owner.nominal_session is not None
        if not owner.nominal_session._started:  # noqa: SLF001
            return self.world.time
        return owner.nominal_session.next_event_time()

    def _route_override_ids(self, owner: _RouteOwner) -> tuple[str, ...]:
        if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF:
            if not owner.handoff_session._started:  # noqa: SLF001
                return ()
            return owner.handoff_session._detect_override()  # noqa: SLF001
        if owner.stage is EnvironmentalMixedRouteStage.NOMINAL_ROUTE:
            assert owner.nominal_session is not None
            if not owner.nominal_session._started:  # noqa: SLF001
                return ()
            return owner.nominal_session._detect_override()  # noqa: SLF001
        return ()

    def _transition_to_nominal(self, owner: _RouteOwner) -> None:
        handoff = owner.handoff_session
        handoff._terminal_reason = EnvironmentalRouteHandoffSessionStopReason.NOMINAL_READY  # noqa: SLF001
        result = handoff.result(EnvironmentalRouteHandoffSessionStopReason.NOMINAL_READY)
        if owner.pair_source is not None:
            cycle = owner.pair_source.pair_cycle
            cycle._handoff_result = result  # noqa: SLF001
            cycle._nominal_session = EnvironmentalNominalRouteSession(  # noqa: SLF001
                self.world,
                owner.binding,
                result,
                config=cycle.config.nominal_route,
                profile_catalog=cycle.profile_catalog,
                before_command_write=lambda: cycle._guard_command_write(  # noqa: SLF001
                    EnvironmentalPairControlStage.NOMINAL_ROUTE
                ),
                command_transform=self._route_command_transform if self.same_route_following is not None else None,
            )
            cycle._stage = EnvironmentalPairControlStage.NOMINAL_ROUTE  # noqa: SLF001
            owner.nominal_session = cycle._nominal_session  # noqa: SLF001
        else:
            owner.nominal_session = EnvironmentalNominalRouteSession(
                self.world,
                owner.binding,
                result,
                config=self.config.nominal_route,
                profile_catalog=self.profile_catalog,
                before_command_write=lambda owner_id=owner.owner_id: self._route_write_guard(owner_id),
                command_transform=self._route_command_transform if self.same_route_following is not None else None,
            )
        owner.stage = EnvironmentalMixedRouteStage.NOMINAL_ROUTE

    def _mark_route_complete(self, owner: _RouteOwner) -> None:
        owner.stage = EnvironmentalMixedRouteStage.COMPLETE
        if owner.owner_id not in self._completed_route_owner_ids:
            self._completed_route_owner_ids.append(owner.owner_id)
        if owner.pair_source is not None:
            cycle = owner.pair_source.pair_cycle
            cycle._stage = EnvironmentalPairControlStage.COMPLETE  # noqa: SLF001
            from .environmental_pair_control_cycle import EnvironmentalPairControlStopReason
            cycle._terminal_reason = EnvironmentalPairControlStopReason.ROUTE_COMPLETE  # noqa: SLF001

    def _process_route_owner_current(
        self, owner: _RouteOwner
    ) -> tuple[EnvironmentalMixedRouteCompletion, ...]:
        for _ in range(8):
            if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF:
                session = owner.handoff_session
                if not session._started:  # noqa: SLF001
                    candidates = set(owner.vessel_ids)
                    commands_before = {
                        vessel_id: self.world.command_for(vessel_id)
                        for vessel_id in candidates
                    }
                    session.start()
                else:
                    candidates = {
                        vessel_id
                        for vessel_id, cursor in session._cursors.items()  # noqa: SLF001
                        if cursor.next_review_at is not None and cursor.next_review_at <= self.world.time + _EPS
                    }
                    commands_before = {
                        vessel_id: self.world.command_for(vessel_id)
                        for vessel_id in candidates
                    }
                    if candidates:
                        # Read-only preflight of the full due set before the first command write.
                        for vessel_id in candidates:
                            session._desired_for(vessel_id)  # noqa: SLF001
                        session._refresh(candidates)  # noqa: SLF001

                completed_now = tuple(
                    vessel_id
                    for vessel_id in owner.vessel_ids
                    if vessel_id in candidates
                    and session._cursors[vessel_id].mode is EnvironmentalRouteHandoffSessionMode.ROUTE_COMPLETE  # noqa: SLF001
                )
                if completed_now:
                    return (
                        EnvironmentalMixedRouteCompletion(
                            owner_id=owner.owner_id,
                            vessel_ids=completed_now,
                            expected_commands=tuple(
                                (vessel_id, commands_before[vessel_id])
                                for vessel_id in completed_now
                            ),
                        ),
                    )
                if session.nominal_ready:
                    self._transition_to_nominal(owner)
                    continue
                return ()

            if owner.stage is EnvironmentalMixedRouteStage.NOMINAL_ROUTE:
                session_n = owner.nominal_session
                assert session_n is not None
                if not session_n._started:  # noqa: SLF001
                    candidates = set(owner.vessel_ids)
                    commands_before = {
                        vessel_id: self.world.command_for(vessel_id)
                        for vessel_id in candidates
                    }
                    session_n.start()
                else:
                    candidates = {
                        vessel_id
                        for vessel_id, cursor in session_n._cursors.items()  # noqa: SLF001
                        if cursor.next_review_at is not None and cursor.next_review_at <= self.world.time + _EPS
                    }
                    commands_before = {
                        vessel_id: self.world.command_for(vessel_id)
                        for vessel_id in candidates
                    }
                    if candidates:
                        for vessel_id in candidates:
                            session_n._desired_for(vessel_id)  # noqa: SLF001
                        session_n._refresh(candidates)  # noqa: SLF001

                completed_now = tuple(
                    vessel_id
                    for vessel_id in owner.vessel_ids
                    if vessel_id in candidates
                    and session_n._cursors[vessel_id].mode is EnvironmentalNominalRouteMode.ROUTE_COMPLETE  # noqa: SLF001
                )
                if completed_now:
                    return (
                        EnvironmentalMixedRouteCompletion(
                            owner_id=owner.owner_id,
                            vessel_ids=completed_now,
                            expected_commands=tuple(
                                (vessel_id, commands_before[vessel_id])
                                for vessel_id in completed_now
                            ),
                        ),
                    )
                return ()
            return ()
        raise EnvironmentalMixedOwnerControlError("too many zero-time route stage transitions")

    def _admit_new_pair_route_stages(self) -> None:
        for started in self.coordinator.started_owners:
            if started.owner.component_id not in self._managed_started_owner_ids:
                continue
            if not isinstance(started, EnvironmentalStartedPairOwner):
                continue
            owner_id = started.owner.component_id
            if owner_id in self._route_owners:
                continue
            cycle = started.pair_cycle
            if cycle.stage is EnvironmentalPairControlStage.ROUTE_HANDOFF:
                if cycle._handoff_binding is None or cycle._handoff_session is None:  # noqa: SLF001
                    raise EnvironmentalMixedOwnerControlError(
                        f"PAIR owner {owner_id!r} entered route handoff without binding/session"
                    )
                cycle._handoff_session.command_transform = (  # noqa: SLF001
                    self._route_command_transform if self.same_route_following is not None else None
                )
                self._route_owners[owner_id] = _RouteOwner(
                    owner_id,
                    started.owner.vessel_ids,
                    cycle._handoff_binding,  # noqa: SLF001
                    cycle._handoff_session,  # noqa: SLF001
                    pair_source=started,
                    next_fleet_review_time=started.guarded_session.next_review_time,
                )

    def _admit_new_joint_handoffs(self, observation: EnvironmentalFleetOwnershipObservation) -> None:
        for started in self.coordinator.started_owners:
            if started.owner.component_id not in self._managed_started_owner_ids:
                continue
            if not isinstance(started, EnvironmentalStartedJointOwner):
                continue
            owner_id = started.owner.component_id
            if owner_id in self._retired_joint_owner_ids:
                continue
            if started.joint_cycle.stage is not EnvironmentalJointControlStage.HANDOFF_ELIGIBLE:
                continue
            route_owner_id = "released:" + owner_id
            expected = self._joint_expected_commands(started)
            owner = self._make_released_route_owner(
                route_owner_id,
                started.owner.vessel_ids,
                observation,
                self.world.snapshot(),
                expected,
            )
            self._route_owners[route_owner_id] = owner
            self._retired_joint_owner_ids.add(owner_id)

    def _drain_avoidance_policies(self) -> tuple[EnvironmentalFleetOwnershipObservation | None, tuple[str, ...]]:
        latest_observation: EnvironmentalFleetOwnershipObservation | None = None
        for _ in range(64):
            pending_items: list[EnvironmentalStartedOwner] = []
            observation_required = False
            for started in self._avoidance_started():
                if isinstance(started, EnvironmentalStartedPairOwner):
                    pending = started.pair_cycle._pending_policy  # noqa: SLF001
                    if pending is not None:
                        pending_items.append(started)
                        if pending.action is EnvironmentalPairExecutionPolicyAction.REPLAN_REQUIRED:
                            observation_required = True
                else:
                    pending = started.joint_cycle._pending_policy  # noqa: SLF001
                    if pending is not None:
                        pending_items.append(started)
                        observation_required = True
            if not pending_items:
                return latest_observation, ()

            if observation_required:
                latest_observation, recapture = self._ensure_current_observation()
                if recapture:
                    return latest_observation, recapture
                self._current_write_observation = latest_observation

            changed = False
            for started in pending_items:
                if isinstance(started, EnvironmentalStartedPairOwner):
                    started.pair_cycle._apply_pending_policy()  # noqa: SLF001
                else:
                    started.joint_cycle._apply_pending_policy()  # noqa: SLF001
                changed = True
            self._admit_new_pair_route_stages()
            if latest_observation is None and any(
                isinstance(item, EnvironmentalStartedJointOwner)
                and item.joint_cycle.stage is EnvironmentalJointControlStage.HANDOFF_ELIGIBLE
                for item in pending_items
            ):
                latest_observation, recapture = self._ensure_current_observation()
                if recapture:
                    return latest_observation, recapture
            if latest_observation is not None:
                self._admit_new_joint_handoffs(latest_observation)
            if not changed:
                return latest_observation, ()
        raise EnvironmentalMixedOwnerControlError("too many zero-time avoidance policy transitions")

    def _route_due_ids(self) -> tuple[str, ...]:
        due: list[str] = []
        for owner in self._route_owners.values():
            at = self._route_event_time(owner)
            if at is not None and at <= self.world.time + _EPS:
                due.append(owner.owner_id)
        return tuple(due)

    def _route_fleet_review_due_ids(self) -> tuple[str, ...]:
        return tuple(
            owner.owner_id
            for owner in self._route_owners.values()
            if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
            and owner.next_fleet_review_time is not None
            and owner.next_fleet_review_time <= self.world.time + _EPS
        )

    def _advance_route_reviews(self) -> None:
        for owner in self._route_owners.values():
            if owner.next_fleet_review_time is None:
                continue
            while owner.next_fleet_review_time <= self.world.time + _EPS:
                owner.next_fleet_review_time += self.config.route_fleet_review_interval_min

    def _active_owner_ids(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        avoidance = tuple(item.owner.component_id for item in self._avoidance_started())
        route = tuple(
            owner.owner_id
            for owner in self._route_owners.values()
            if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
        )
        return avoidance, route

    def _result(
        self,
        reason: EnvironmentalMixedOwnerStopReason,
        *,
        recapture_observation: EnvironmentalFleetOwnershipObservation | None = None,
        recapture_owner_ids: tuple[str, ...] = (),
        overridden_vessel_ids: tuple[str, ...] = (),
        route_disposition_observation: EnvironmentalFleetOwnershipObservation | None = None,
        route_completions: tuple[EnvironmentalMixedRouteCompletion, ...] = (),
    ) -> EnvironmentalMixedOwnerControlResult:
        avoidance, route = self._active_owner_ids()
        return EnvironmentalMixedOwnerControlResult(
            self.world.snapshot(),
            reason,
            self.boundaries,
            self.fleet_observations,
            avoidance,
            route,
            tuple(self._completed_route_owner_ids),
            recapture_observation,
            recapture_owner_ids,
            overridden_vessel_ids,
            route_disposition_observation,
            route_completions,
            self.same_route_supervisions,
        )

    def _drain_current(self) -> EnvironmentalMixedOwnerControlResult | None:
        observation, recapture = self._drain_avoidance_policies()
        if recapture:
            return self._result(
                EnvironmentalMixedOwnerStopReason.RECAPTURE_REQUIRED,
                recapture_observation=observation,
                recapture_owner_ids=recapture,
            )

        route_due = self._route_due_ids()
        if route_due:
            observation, recapture = self._ensure_current_observation()
            if recapture:
                return self._result(
                    EnvironmentalMixedOwnerStopReason.RECAPTURE_REQUIRED,
                    recapture_observation=observation,
                    recapture_owner_ids=recapture,
                )
            self._current_write_observation = observation
            # Preflight every due owner before the first route mutation.
            for owner_id in route_due:
                owner = self._route_owners[owner_id]
                if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF and owner.handoff_session._started:  # noqa: SLF001
                    due = {
                        v for v, c in owner.handoff_session._cursors.items()  # noqa: SLF001
                        if c.next_review_at is not None and c.next_review_at <= self.world.time + _EPS
                    }
                    for vessel_id in due:
                        owner.handoff_session._desired_for(vessel_id)  # noqa: SLF001
                elif owner.stage is EnvironmentalMixedRouteStage.NOMINAL_ROUTE and owner.nominal_session is not None and owner.nominal_session._started:  # noqa: SLF001
                    due = {
                        v for v, c in owner.nominal_session._cursors.items()  # noqa: SLF001
                        if c.next_review_at is not None and c.next_review_at <= self.world.time + _EPS
                    }
                    for vessel_id in due:
                        owner.nominal_session._desired_for(vessel_id)  # noqa: SLF001
            completions: list[EnvironmentalMixedRouteCompletion] = []
            for owner_id in route_due:
                completions.extend(
                    self._process_route_owner_current(self._route_owners[owner_id])
                )
            self._current_write_observation = None
            if completions:
                return self._result(
                    EnvironmentalMixedOwnerStopReason.ROUTE_DISPOSITION_REQUIRED,
                    route_disposition_observation=observation,
                    route_completions=tuple(completions),
                )

        avoidance, route = self._active_owner_ids()
        if not avoidance and not route:
            return self._result(EnvironmentalMixedOwnerStopReason.ALL_CONTROL_COMPLETE)
        return None

    def run_until(self, end_time: float) -> EnvironmentalMixedOwnerControlResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")

        initial = self._drain_current()
        if initial is not None:
            return initial

        for _ in range(self.config.max_boundaries_per_run):
            if self.world.time >= end_time - _EPS:
                return self._result(EnvironmentalMixedOwnerStopReason.END_TIME)

            avoidance = self._avoidance_started()
            execution_times = {
                started.owner.component_id: self.coordinator._execution_boundary_time(started)  # noqa: SLF001
                for started in avoidance
            }
            if any(at is not None and at <= self.world.time + _EPS for at in execution_times.values()):
                raise EnvironmentalMixedOwnerControlError(
                    "avoidance owner reports a non-future boundary after mixed policy drain"
                )
            route_times = {
                owner.owner_id: self._route_event_time(owner)
                for owner in self._route_owners.values()
                if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
            }
            if any(at is not None and at <= self.world.time + _EPS for at in route_times.values()):
                current = self._drain_current()
                if current is not None:
                    return current
                continue

            candidates: list[float] = [end_time]
            candidates.extend(at for at in execution_times.values() if at is not None)
            candidates.extend(at for at in route_times.values() if at is not None)
            for started in avoidance:
                review = started.guarded_session.next_review_time
                if review is not None:
                    candidates.append(review)
            for owner in self._route_owners.values():
                if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE and owner.next_fleet_review_time is not None:
                    candidates.append(owner.next_fleet_review_time)
            membership_time = self.coordinator._next_membership_time()  # noqa: SLF001
            if membership_time is not None:
                candidates.append(membership_time)
            command_time = self.world.next_command_time()
            if command_time is not None:
                candidates.append(command_time)

            native_boundary = min(candidates)
            source_boundary = self._source_boundary_time(native_boundary)
            if source_boundary is not None:
                if source_boundary < self.world.time - _EPS:
                    raise EnvironmentalMixedOwnerControlError(
                        "source extension proposed a boundary in the past"
                    )
                if source_boundary > native_boundary + _EPS:
                    raise EnvironmentalMixedOwnerControlError(
                        "source extension proposed a boundary beyond its frozen horizon"
                    )
            boundary = (
                source_boundary
                if source_boundary is not None and source_boundary < native_boundary - _EPS
                else native_boundary
            )
            source_boundary_due = (
                source_boundary is not None and abs(source_boundary - boundary) <= _EPS
            )
            native_boundary_reached = abs(native_boundary - boundary) <= _EPS
            native_boundary_due = native_boundary_reached and (
                any(
                    at is not None and abs(at - native_boundary) <= _EPS
                    for at in execution_times.values()
                )
                or any(
                    at is not None and abs(at - native_boundary) <= _EPS
                    for at in route_times.values()
                )
                or any(
                    started.guarded_session.next_review_time is not None
                    and abs(started.guarded_session.next_review_time - native_boundary) <= _EPS
                    for started in avoidance
                )
                or any(
                    owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
                    and owner.next_fleet_review_time is not None
                    and abs(owner.next_fleet_review_time - native_boundary) <= _EPS
                    for owner in self._route_owners.values()
                )
                or (membership_time is not None and abs(membership_time - native_boundary) <= _EPS)
                or (command_time is not None and abs(command_time - native_boundary) <= _EPS)
            )

            command_event_due = command_time is not None and abs(command_time - boundary) <= _EPS
            self.world.run_until(boundary)

            execution_due: list[str] = []
            for started in avoidance:
                owner_id = started.owner.component_id
                due = execution_times[owner_id] is not None and abs(execution_times[owner_id] - self.world.time) <= _EPS
                if due:
                    execution_due.append(owner_id)
                if isinstance(started, EnvironmentalStartedPairOwner):
                    self.coordinator._pair_adopt_shared_advance(  # noqa: SLF001
                        started, due=due, command_event_due=command_event_due
                    )
                else:
                    self.coordinator._joint_adopt_shared_advance(  # noqa: SLF001
                        started, due=due, command_event_due=command_event_due
                    )

            overridden: list[str] = []
            for owner in self._route_owners.values():
                overridden.extend(self._route_override_ids(owner))
            if overridden:
                return self._result(
                    EnvironmentalMixedOwnerStopReason.COMMAND_OVERRIDDEN,
                    overridden_vessel_ids=tuple(dict.fromkeys(overridden)),
                )

            membership_count, removed = self.coordinator._apply_due_membership()  # noqa: SLF001

            # A151 source membership is a pre-observation phase of this same physical
            # boundary.  Native membership/command mutations are applied first; only
            # then may the FIFO source resolver commit an eligible current-time batch.
            # The resulting membership is therefore visible to the one canonical fleet
            # observation below, never to a second same-timestamp registry update.
            source_vessel_ids = self._apply_source_pre_observation_phase(
                native_boundary_due=native_boundary_due,
                source_boundary_due=source_boundary_due,
            )
            membership_count += len(source_vessel_ids)

            route_due = tuple(
                owner_id
                for owner_id, at in route_times.items()
                if at is not None and abs(at - self.world.time) <= _EPS
            )
            avoidance_review_due = tuple(
                started.owner.component_id
                for started in avoidance
                if started.guarded_session.next_review_time is not None
                and started.guarded_session.next_review_time <= self.world.time + _EPS
            )
            route_review_due = self._route_fleet_review_due_ids()

            policy_observation_required = False
            for started in avoidance:
                if isinstance(started, EnvironmentalStartedJointOwner):
                    if started.joint_cycle._pending_policy is not None:  # noqa: SLF001
                        policy_observation_required = True
                        break
                else:
                    pending = started.pair_cycle._pending_policy  # noqa: SLF001
                    if pending is not None and pending.action is EnvironmentalPairExecutionPolicyAction.REPLAN_REQUIRED:
                        policy_observation_required = True
                        break

            # A source-only wake that remains blocked is not a navigation observation.
            # Its only state change is A150 generation invalidation/replanning.
            if (
                source_boundary_due
                and not native_boundary_due
                and not source_vessel_ids
                and not execution_due
                and not route_due
                and not avoidance_review_due
                and not route_review_due
                and membership_count == 0
                and not policy_observation_required
            ):
                continue

            observation = None
            recapture: tuple[str, ...] = ()
            if membership_count or avoidance_review_due or route_review_due or route_due or policy_observation_required:
                observation, recapture = self._ensure_current_observation(removed=removed)

            if source_vessel_ids:
                if observation is None:
                    raise EnvironmentalMixedOwnerControlError(
                        "source admission did not produce canonical fleet evidence"
                    )
                self._stage_source_control_ownership(source_vessel_ids, observation)
                # A newly admitted touching source is already ownership-accounted by a
                # zero-write SOURCE_PENDING_RECAPTURE owner.  That transactional owner
                # is itself the same-timestamp recapture authority: A120 can seed from
                # the admitted vessel and recover the complete canonical component.
                pending_source_owner_ids = tuple(
                    self._pending_source_control_ids[vessel_id]
                    for vessel_id in source_vessel_ids
                    if vessel_id in self._pending_source_control_ids
                )
                if pending_source_owner_ids:
                    recapture = tuple(dict.fromkeys((*recapture, *pending_source_owner_ids)))
                if self._pending_source_control_ids and not recapture:
                    raise EnvironmentalMixedOwnerControlError(
                        "touching source admission lacks same-timestamp recapture authority"
                    )

            for started in avoidance:
                guard = started.guarded_session
                if guard.next_review_time is not None and guard.next_review_time <= self.world.time + _EPS:
                    guard._advance_review_after_current()  # noqa: SLF001
            # A route owner that lost canonical fleet clearance at this boundary must
            # be recaptured before any persistent same-route command write.  Keep the
            # review due when recapture is pending so preserved route owners can consume
            # the same-timestamp review after A120 adopts the replacement owner set.
            if route_review_due and not recapture:
                if observation is None:
                    raise EnvironmentalMixedOwnerControlError(
                        "route fleet review lacks canonical fleet observation"
                    )
                self._apply_persistent_same_route_route_supervision(
                    route_review_due, observation
                )
                self._advance_route_reviews()

            self._boundaries.append(
                EnvironmentalMixedOwnerBoundary(
                    self.world.time,
                    tuple(execution_due),
                    route_due,
                    tuple(avoidance_review_due) + tuple(route_review_due),
                    membership_count,
                    removed,
                    observation,
                    recapture,
                )
            )

            if recapture:
                return self._result(
                    EnvironmentalMixedOwnerStopReason.RECAPTURE_REQUIRED,
                    recapture_observation=observation,
                    recapture_owner_ids=recapture,
                )

            current = self._drain_current()
            if current is not None:
                return current

            if self.world.time >= end_time - _EPS:
                return self._result(EnvironmentalMixedOwnerStopReason.END_TIME)

        raise EnvironmentalMixedOwnerControlError(
            f"a119 exceeded max_boundaries_per_run={self.config.max_boundaries_per_run}"
        )
