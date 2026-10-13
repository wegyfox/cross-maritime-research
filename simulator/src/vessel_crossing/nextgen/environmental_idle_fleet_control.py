"""Canonical idle/unowned-fleet supervision and automatic fresh owner bootstrap.

A123 extends the persistent A122 mixed-stage stack across the final ownership gap.
A122 deliberately stops when no avoidance or route owner remains.  The physical fleet,
however, can continue moving under its last water-relative commands, can receive caller
command events, and can gain/lose vessels.  A123 therefore keeps the *same*
``EnvironmentalWorld2D`` clock and the *same* ``EncounterRegistry`` alive while the
fleet is unowned.

Idle supervision is event-driven.  Periodic idle reviews, queued command events and
membership events share one calendar.  At every real idle boundary the coordinator
commits at most one canonical fleet observation.  If that dispatch contains active
PAIR/JOINT owners, all of them are planned read-only and their replacement commands are
published in one same-timestamp atomic batch before fresh a102/a113 guarded sessions are
attached.  No caller-side restart and no duplicate registry observation are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .encounter_registry import EncounterRegistry
from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_fleet_guarded_joint_control import EnvironmentalFleetGuardedJointControlSession
from .environmental_fleet_ingress import (
    EnvironmentalFleetIngressAdmission,
    EnvironmentalFleetIngressError,
    EnvironmentalFleetIngressRecord,
)
from .environmental_fleet_incarnation import (
    EnvironmentalFleetIncarnationAdmission,
    EnvironmentalFleetIncarnationAmendment,
    EnvironmentalFleetIncarnationCancellation,
    EnvironmentalFleetIncarnationEpisode,
    EnvironmentalFleetIncarnationError,
)
from .environmental_fleet_egress import (
    EnvironmentalFleetEgressAdmission,
    EnvironmentalFleetEgressCompletion,
    EnvironmentalFleetEgressError,
    EnvironmentalFleetEgressRecord,
    EnvironmentalFleetEgressSource,
)
from .environmental_fleet_guarded_pair_control import EnvironmentalFleetGuardedPairControlSession
from .environmental_fleet_multi_owner_recapture import (
    EnvironmentalFleetMultiOwnerApplyInvariantViolation,
    EnvironmentalFleetMultiOwnerRecaptureConfig,
    EnvironmentalFleetMultiOwnerRecaptureError,
    EnvironmentalPreparedJointOwner,
    EnvironmentalPreparedPairOwner,
    EnvironmentalStartedJointOwner,
    EnvironmentalStartedOwner,
    EnvironmentalStartedPairOwner,
    PendingEnvironmentalMultiOwnerCommandEvent,
    _admit_pair_targets,
    _pair_commands,
    _prepare_joint_owner,
    _prepare_pair_owner,
)
from .environmental_fleet_ownership import EnvironmentalFleetOwnershipObservation
from .environmental_joint_control_cycle import EnvironmentalJointControlSession
from .environmental_joint_execution import AppliedEnvironmentalJointPlan
from .environmental_joint_plan import validate_environmental_joint_execution_plan
from .environmental_mixed_owner_control import (
    EnvironmentalMixedRouteStage,
)
from .environmental_multi_owner_control import (
    EnvironmentalMultiOwnerMembershipApplication,
    EnvironmentalMultiOwnerMembershipApplicationOutcome,
    EnvironmentalMultiOwnerMembershipEventKind,
)
from .environmental_pair_control_cycle import EnvironmentalPairControlSession
from .environmental_pair_execution import AppliedEnvironmentalPairPlan
from .environmental_pair_plan import validate_environmental_pair_execution_plan
from .environmental_persistent_mixed_control import (
    EnvironmentalPersistentMixedControlResult,
    EnvironmentalPersistentMixedControlSession,
    EnvironmentalPersistentMixedStopReason,
)
from .environmental_planning_context import build_environmental_avoidance_planning_context
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .environmental_world import EnvironmentalWorldSnapshot
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalIdleFleetControlError(RuntimeError):
    """Base error for invalid A123 idle supervision/bootstrap."""


class EnvironmentalIdleFleetReason(str, Enum):
    INITIAL = "initial"
    CONTROL_COMPLETE = "control_complete"
    PERIODIC_REVIEW = "periodic_review"
    COMMAND_EVENT = "command_event"
    VESSEL_MEMBERSHIP = "vessel_membership"


class EnvironmentalPersistentFleetStopReason(str, Enum):
    END_TIME = "end_time"
    COMMAND_OVERRIDDEN = "command_overridden"


@dataclass(frozen=True, slots=True)
class EnvironmentalIdleFleetControlConfig:
    idle_review_interval_min: float = 1.0
    max_idle_boundaries_per_run: int = 8192
    max_control_idle_transitions_per_run: int = 256

    def __post_init__(self) -> None:
        if not isfinite(self.idle_review_interval_min) or self.idle_review_interval_min <= 0.0:
            raise ValueError("idle_review_interval_min must be finite and positive")
        if self.max_idle_boundaries_per_run < 1:
            raise ValueError("max_idle_boundaries_per_run must be at least 1")
        if self.max_control_idle_transitions_per_run < 1:
            raise ValueError("max_control_idle_transitions_per_run must be at least 1")


@dataclass(frozen=True, slots=True)
class EnvironmentalIdleFleetBootstrapResult:
    observation: EnvironmentalFleetOwnershipObservation
    snapshot_before: EnvironmentalWorldSnapshot
    snapshot_after: EnvironmentalWorldSnapshot
    commands_before: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    commands_after: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    started_owners: tuple[EnvironmentalStartedOwner, ...]

    @property
    def owner_ids(self) -> tuple[str, ...]:
        return tuple(item.owner.component_id for item in self.started_owners)


@dataclass(frozen=True, slots=True)
class EnvironmentalIdleFleetBoundary:
    time_min: float
    reasons: tuple[EnvironmentalIdleFleetReason, ...]
    membership_event_count: int
    removed_vessel_ids: tuple[str, ...]
    observation: EnvironmentalFleetOwnershipObservation
    bootstrapped_owner_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvironmentalPersistentFleetControlResult:
    stop_reason: EnvironmentalPersistentFleetStopReason
    snapshot: EnvironmentalWorldSnapshot
    terminal_persistent_result: EnvironmentalPersistentMixedControlResult | None
    idle_boundaries: tuple[EnvironmentalIdleFleetBoundary, ...]
    bootstrap_results: tuple[EnvironmentalIdleFleetBootstrapResult, ...]
    control_idle_transition_count: int


def _validate_all_owner_route_targets(
    owners: tuple[FleetCommandOwner, ...],
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
) -> None:
    for owner in owners:
        for vessel_id in owner.vessel_ids:
            try:
                target = route_targets[vessel_id]
            except KeyError as exc:
                raise EnvironmentalIdleFleetControlError(
                    f"missing route target for newly supervised vessel {vessel_id!r}"
                ) from exc
            if not isinstance(target, EnvironmentalRouteHandoffTarget):
                raise TypeError(
                    f"route target for {vessel_id!r} must be EnvironmentalRouteHandoffTarget"
                )
            if target.vessel_id != vessel_id:
                raise EnvironmentalIdleFleetControlError(
                    f"route target key {vessel_id!r} disagrees with target vessel id {target.vessel_id!r}"
                )


def _start_prepared_owners(
    registry: EncounterRegistry,
    persistent: EnvironmentalPersistentMixedControlSession,
    observation: EnvironmentalFleetOwnershipObservation,
    prepared: tuple[EnvironmentalPreparedPairOwner | EnvironmentalPreparedJointOwner, ...],
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None,
) -> EnvironmentalIdleFleetBootstrapResult:
    """Publish one prepared fleet batch and attach fresh guarded local sessions."""

    world = persistent.world
    command_batch: list[tuple[str, WaterRelativeManeuverCommand]] = []
    for item in prepared:
        if isinstance(item, EnvironmentalPreparedPairOwner):
            validate_environmental_pair_execution_plan(item.plan, world, profile_catalog=profile_catalog)
            command_batch.extend(_pair_commands(item.plan))
        else:
            validate_environmental_joint_execution_plan(item.plan, world, profile_catalog=profile_catalog)
            command_batch.extend(item.plan.selected_commands)

    ids = [vessel_id for vessel_id, _ in command_batch]
    if len(ids) != len(set(ids)):
        raise EnvironmentalIdleFleetControlError(
            "idle bootstrap command batch contains duplicate vessel channels"
        )
    expected_ids = [
        vessel_id
        for item in prepared
        for vessel_id in item.owner.vessel_ids
    ]
    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise EnvironmentalIdleFleetControlError(
            "idle bootstrap command batch does not cover every canonical owner vessel exactly once"
        )

    pending = world.next_command_time()
    if pending is not None and pending <= world.time + _EPS:
        raise PendingEnvironmentalMultiOwnerCommandEvent(
            "cannot bootstrap idle fleet while a command event is pending at current time"
        )

    snapshot_before = world.snapshot()
    commands_before = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in ids)
    applied_at = world.time
    world.schedule_commands_atomic(at=applied_at, commands=tuple(command_batch))
    world.run_until(applied_at)
    snapshot_after = world.snapshot()
    commands_after = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in ids)
    if snapshot_after != snapshot_before or world.time != applied_at:
        raise EnvironmentalFleetMultiOwnerApplyInvariantViolation(
            "same-timestamp idle bootstrap changed physical state or world time"
        )
    if commands_after != tuple(command_batch):
        raise EnvironmentalFleetMultiOwnerApplyInvariantViolation(
            "same-timestamp idle bootstrap did not install the complete selected batch"
        )

    before_map = dict(commands_before)
    after_map = dict(commands_after)
    started: list[EnvironmentalStartedOwner] = []

    for item in prepared:
        owner = item.owner
        if isinstance(item, EnvironmentalPreparedPairOwner):
            applied = AppliedEnvironmentalPairPlan(
                plan=item.plan,
                applied_at_time=applied_at,
                snapshot_before=snapshot_before,
                snapshot_after=snapshot_after,
                commands_before=(before_map[item.plan.first_id], before_map[item.plan.second_id]),
                commands_after=(after_map[item.plan.first_id], after_map[item.plan.second_id]),
            )
            holder: dict[str, EnvironmentalFleetGuardedPairControlSession] = {}

            def pair_factory(current_world, _policy, *, _owner=owner, _holder=holder):
                guard = _holder.get("guard")
                if guard is None or not guard.fleet_observations:
                    raise EnvironmentalIdleFleetControlError(
                        "PAIR planning factory used without current fleet ownership evidence"
                    )
                latest = guard.fleet_observations[-1]
                if abs(latest.time_min - current_world.time) > _EPS:
                    raise EnvironmentalIdleFleetControlError(
                        "PAIR replanning lacks same-timestamp fleet ownership evidence"
                    )
                matches = tuple(
                    candidate
                    for candidate in latest.dispatch.owners
                    if candidate.backend is FleetSupervisorBackend.PAIR
                    and candidate.component_id == _owner.component_id
                    and candidate.vessel_ids == _owner.vessel_ids
                    and candidate.pair_keys == _owner.pair_keys
                )
                if len(matches) != 1:
                    raise EnvironmentalIdleFleetControlError(
                        "current fleet observation no longer contains the exact idle-bootstrapped PAIR owner"
                    )
                return build_environmental_avoidance_planning_context(
                    current_world,
                    latest.dispatch,
                    matches[0],
                    profile_catalog=profile_catalog,
                    route_targets=persistent.route_targets,
                    same_route_following=config.same_route_following,
                )

            pair_cycle = EnvironmentalPairControlSession(
                world,
                applied,
                item.lifecycle,
                item.decision,
                item.route_targets,
                pair_factory,
                config=config.pair_cycle,
                profile_catalog=profile_catalog,
            )
            guarded = EnvironmentalFleetGuardedPairControlSession(
                registry,
                pair_cycle,
                config=config.pair_fleet_guard,
                initial_observation=observation,
            )
            holder["guard"] = guarded
            initial = guarded.start()
            if guarded.recapture is not None:
                raise EnvironmentalIdleFleetControlError(
                    f"fresh pair guard rejected idle bootstrap owner {owner.component_id!r}: {initial!r}"
                )
            started.append(EnvironmentalStartedPairOwner(item, applied, pair_cycle, guarded))
            continue

        applied_j = AppliedEnvironmentalJointPlan(
            plan=item.plan,
            applied_at_time=applied_at,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
            commands_before=tuple((vessel_id, before_map[vessel_id]) for vessel_id in item.plan.vessel_ids),
            commands_after=tuple((vessel_id, after_map[vessel_id]) for vessel_id in item.plan.vessel_ids),
        )
        holder_j: dict[str, EnvironmentalFleetGuardedJointControlSession] = {}

        def joint_factory(current_world, _policy, *, _owner=owner, _holder=holder_j):
            guard = _holder.get("guard")
            if guard is None or not guard.fleet_observations:
                raise EnvironmentalIdleFleetControlError(
                    "JOINT planning factory used without current fleet ownership evidence"
                )
            latest = guard.fleet_observations[-1]
            if abs(latest.time_min - current_world.time) > _EPS:
                raise EnvironmentalIdleFleetControlError(
                    "JOINT replanning lacks same-timestamp fleet ownership evidence"
                )
            matches = tuple(
                candidate
                for candidate in latest.dispatch.owners
                if candidate.backend is FleetSupervisorBackend.JOINT
                and candidate.component_id == _owner.component_id
                and candidate.vessel_ids == _owner.vessel_ids
                and candidate.pair_keys == _owner.pair_keys
            )
            if len(matches) != 1:
                raise EnvironmentalIdleFleetControlError(
                    "current fleet observation no longer contains the exact idle-bootstrapped JOINT owner"
                )
            return build_environmental_avoidance_planning_context(
                current_world,
                latest.dispatch,
                matches[0],
                profile_catalog=profile_catalog,
            )

        joint_cycle = EnvironmentalJointControlSession(
            world,
            applied_j,
            item.lifecycles,
            item.decisions,
            joint_factory,
            config=config.joint_cycle,
            profile_catalog=profile_catalog,
        )
        guarded_j = EnvironmentalFleetGuardedJointControlSession(
            registry,
            joint_cycle,
            config=config.joint_fleet_guard,
            initial_observation=observation,
        )
        holder_j["guard"] = guarded_j
        initial_j = guarded_j.start()
        if guarded_j.recapture is not None:
            raise EnvironmentalIdleFleetControlError(
                f"fresh joint guard rejected idle bootstrap owner {owner.component_id!r}: {initial_j!r}"
            )
        started.append(EnvironmentalStartedJointOwner(item, applied_j, joint_cycle, guarded_j))

    return EnvironmentalIdleFleetBootstrapResult(
        observation=observation,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        commands_before=commands_before,
        commands_after=commands_after,
        started_owners=tuple(started),
    )


def bootstrap_environmental_idle_fleet_owners(
    persistent: EnvironmentalPersistentMixedControlSession,
    observation: EnvironmentalFleetOwnershipObservation,
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalIdleFleetBootstrapResult:
    """Atomically bootstrap every canonical owner from one committed idle observation."""

    if not isinstance(persistent, EnvironmentalPersistentMixedControlSession):
        raise TypeError("persistent must be EnvironmentalPersistentMixedControlSession")
    if not isinstance(observation, EnvironmentalFleetOwnershipObservation):
        raise TypeError("observation must be EnvironmentalFleetOwnershipObservation")
    if not isinstance(config, EnvironmentalFleetMultiOwnerRecaptureConfig):
        raise TypeError("config must be EnvironmentalFleetMultiOwnerRecaptureConfig")

    mixed = persistent.mixed_session
    coordinator = mixed.coordinator
    registry = coordinator.registry
    world = coordinator.world
    if abs(observation.time_min - world.time) > _EPS:
        raise EnvironmentalIdleFleetControlError(
            "idle bootstrap observation must match the current shared-clock timestamp"
        )
    if not coordinator.fleet_observations or coordinator.fleet_observations[-1] != observation:
        raise EnvironmentalIdleFleetControlError(
            "idle bootstrap requires the coordinator's latest committed canonical observation"
        )
    if mixed._active_owner_ids() != ((), ()):  # noqa: SLF001
        raise EnvironmentalIdleFleetControlError(
            "idle bootstrap requires zero active avoidance and route owners"
        )
    owners = observation.dispatch.owners
    if not owners:
        raise EnvironmentalIdleFleetControlError("idle bootstrap requires at least one canonical owner")

    _validate_all_owner_route_targets(owners, persistent.route_targets)

    admitted_targets: dict[str, tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget]] = {}
    for owner in owners:
        if owner.backend is FleetSupervisorBackend.PAIR:
            admitted_targets[owner.component_id] = _admit_pair_targets(owner, persistent.route_targets)

    prepared: list[EnvironmentalPreparedPairOwner | EnvironmentalPreparedJointOwner] = []
    for owner in owners:
        if owner.backend is FleetSupervisorBackend.PAIR:
            prepared.append(
                _prepare_pair_owner(
                    registry,
                    world,
                    owner,
                    admitted_targets[owner.component_id],
                    persistent.route_targets,
                    observation.dispatch,
                    config=config,
                    profile_catalog=profile_catalog,
                )
            )
        elif owner.backend is FleetSupervisorBackend.JOINT:
            prepared.append(
                _prepare_joint_owner(
                    registry,
                    world,
                    owner,
                    persistent.route_targets,
                    observation.dispatch,
                    config=config,
                    profile_catalog=profile_catalog,
                )
            )
        else:  # pragma: no cover
            raise AssertionError(owner.backend)

    result = _start_prepared_owners(
        registry,
        persistent,
        observation,
        tuple(prepared),
        config=config,
        profile_catalog=profile_catalog,
    )

    # Runtime adoption occurs only after the complete same-time command transaction has
    # succeeded.  Historical COMPLETE route owners carry no authority and are purged so
    # a future component may legitimately reuse the same deterministic owner id.
    new_vessels = {v for item in result.started_owners for v in item.owner.vessel_ids}
    for owner_id, route_owner in tuple(mixed._route_owners.items()):  # noqa: SLF001
        if route_owner.stage is EnvironmentalMixedRouteStage.COMPLETE and new_vessels.intersection(route_owner.vessel_ids):
            mixed._route_owners.pop(owner_id)  # noqa: SLF001
    for item in result.started_owners:
        mixed._retired_joint_owner_ids.discard(item.owner.component_id)  # noqa: SLF001

    coordinator._adopt_replacement_owner_set(  # noqa: SLF001
        result.started_owners,
        committed_observation=observation,
    )
    mixed._managed_started_owner_ids = {  # noqa: SLF001
        item.owner.component_id for item in result.started_owners
    }
    mixed._validate_disjoint_channels()  # noqa: SLF001
    return result


class EnvironmentalPersistentFleetControlSession:
    """Keep A122 alive through ownerless intervals and bootstrap new encounters."""

    def __init__(
        self,
        persistent: EnvironmentalPersistentMixedControlSession,
        *,
        recapture_config: EnvironmentalFleetMultiOwnerRecaptureConfig,
        config: EnvironmentalIdleFleetControlConfig = EnvironmentalIdleFleetControlConfig(),
        profile_catalog: VesselProfileCatalog | None = None,
    ) -> None:
        if not isinstance(persistent, EnvironmentalPersistentMixedControlSession):
            raise TypeError("persistent must be EnvironmentalPersistentMixedControlSession")
        if not isinstance(recapture_config, EnvironmentalFleetMultiOwnerRecaptureConfig):
            raise TypeError("recapture_config must be EnvironmentalFleetMultiOwnerRecaptureConfig")
        if not isinstance(config, EnvironmentalIdleFleetControlConfig):
            raise TypeError("config must be EnvironmentalIdleFleetControlConfig")
        self.persistent = persistent
        self.mixed_session = persistent.mixed_session
        self.coordinator = self.mixed_session.coordinator
        self.world = persistent.world
        self.registry = self.coordinator.registry
        self.route_targets = persistent.route_targets
        self.recapture_config = recapture_config
        self.config = config
        self.profile_catalog = profile_catalog
        self._idle_boundaries: list[EnvironmentalIdleFleetBoundary] = []
        self._bootstrap_results: list[EnvironmentalIdleFleetBootstrapResult] = []
        self._next_idle_review_time: float | None = None
        self._last_persistent_result: EnvironmentalPersistentMixedControlResult | None = None
        self._idle_active = False
        self._ingress_admissions: list[EnvironmentalFleetIngressAdmission] = []
        self._egress_admissions: list[EnvironmentalFleetEgressAdmission] = []
        self._egress_completions: list[EnvironmentalFleetEgressCompletion] = []
        self._pending_egress_admissions: dict[tuple[float, str], EnvironmentalFleetEgressAdmission] = {}
        self._incarnation_admissions: list[EnvironmentalFleetIncarnationAdmission] = []
        self._incarnation_cancellations: list[EnvironmentalFleetIncarnationCancellation] = []
        self._incarnation_amendments: list[EnvironmentalFleetIncarnationAmendment] = []
        self._pending_incarnations: dict[str, list[EnvironmentalFleetIncarnationAdmission]] = {}
        self._next_incarnation_by_vessel: dict[str, int] = {}
        self._processed_membership_application_count = len(self.coordinator.membership_applications)
        self._processed_disposition_count = len(self.persistent.disposition_results)
        self.coordinator.add_membership_application_hook(self._on_membership_application)

    @property
    def idle_boundaries(self) -> tuple[EnvironmentalIdleFleetBoundary, ...]:
        return tuple(self._idle_boundaries)

    @property
    def bootstrap_results(self) -> tuple[EnvironmentalIdleFleetBootstrapResult, ...]:
        return tuple(self._bootstrap_results)

    @property
    def ingress_admissions(self) -> tuple[EnvironmentalFleetIngressAdmission, ...]:
        return tuple(self._ingress_admissions)

    @property
    def egress_admissions(self) -> tuple[EnvironmentalFleetEgressAdmission, ...]:
        return tuple(self._egress_admissions)

    @property
    def egress_completions(self) -> tuple[EnvironmentalFleetEgressCompletion, ...]:
        return tuple(self._egress_completions)

    @property
    def incarnation_admissions(self) -> tuple[EnvironmentalFleetIncarnationAdmission, ...]:
        return tuple(self._incarnation_admissions)

    @property
    def incarnation_cancellations(self) -> tuple[EnvironmentalFleetIncarnationCancellation, ...]:
        return tuple(self._incarnation_cancellations)

    @property
    def incarnation_amendments(self) -> tuple[EnvironmentalFleetIncarnationAmendment, ...]:
        return tuple(self._incarnation_amendments)

    @property
    def membership_applications(self) -> tuple[EnvironmentalMultiOwnerMembershipApplication, ...]:
        return self.coordinator.membership_applications

    def _ensure_runtime_route_target_catalog(self) -> dict[str, EnvironmentalRouteHandoffTarget]:
        if isinstance(self.route_targets, dict):
            return self.route_targets
        targets = dict(self.route_targets)
        self.route_targets = targets
        self.persistent.route_targets = targets
        self.mixed_session.route_targets = targets
        return targets

    def _ensure_runtime_profile_catalog(self) -> VesselProfileCatalog:
        if self.profile_catalog is None:
            catalog = VesselProfileCatalog()
            self.profile_catalog = catalog
            self.persistent.profile_catalog = catalog
            self.mixed_session.profile_catalog = catalog
        return self.profile_catalog

    def _validate_ingress_world_constraints(
        self,
        ingress: EnvironmentalFleetIngressRecord,
        *,
        error_type: type[RuntimeError] = EnvironmentalFleetIngressError,
    ) -> tuple[ManeuverLimits | None, Vector2 | None]:
        """Read-only world compatibility check shared by A125 and A127 admission."""

        effective_limits = ingress.effective_limits
        vessel_id = ingress.vessel_id
        limits_store = self.world._limits  # noqa: SLF001 - admission preflight only
        if isinstance(limits_store, dict):
            if effective_limits is None:
                raise error_type(
                    f"per-vessel world requires maneuver limits for ingress vessel {vessel_id!r}"
                )
        elif effective_limits is not None and effective_limits != limits_store:
            raise error_type(
                f"ingress maneuver limits for {vessel_id!r} disagree with shared world limits"
            )

        drift_store = self.world._drifts  # noqa: SLF001
        drift = ingress.environmental_drift_velocity
        if isinstance(drift_store, dict):
            if drift is None:
                raise error_type(
                    f"per-vessel world requires environmental drift for ingress vessel {vessel_id!r}"
                )
        elif drift is not None and tuple(float(value) for value in drift) != drift_store:
            raise error_type(
                f"ingress environmental drift for {vessel_id!r} disagrees with shared world drift"
            )
        return effective_limits, drift

    def _publish_ingress_metadata(self, ingress: EnvironmentalFleetIngressRecord) -> tuple[bool, bool]:
        """Publish one already-validated mission record into the live ID-keyed catalogues."""

        vessel_id = ingress.vessel_id
        targets = self._ensure_runtime_route_target_catalog()
        existing_target = targets.get(vessel_id)
        route_target_replaced = existing_target is not None and existing_target != ingress.route_target
        targets[vessel_id] = ingress.route_target

        profile_assigned = ingress.profile is not None
        if self.profile_catalog is not None:
            existing_profile = self.profile_catalog.profile_for(vessel_id)
            if existing_profile is not None and existing_profile != ingress.profile:
                self.profile_catalog.remove(vessel_id)
        if ingress.profile is not None:
            self._ensure_runtime_profile_catalog().assign(vessel_id, ingress.profile)
        return route_target_replaced, profile_assigned

    def schedule_ingress(
        self,
        ingress: EnvironmentalFleetIngressRecord,
    ) -> EnvironmentalFleetIngressAdmission:
        """Validate and queue one future vessel together with its route/profile metadata."""

        if not isinstance(ingress, EnvironmentalFleetIngressRecord):
            raise TypeError("ingress must be EnvironmentalFleetIngressRecord")
        self.coordinator._validate_future_membership_time(ingress.at)  # noqa: SLF001
        vessel_id = ingress.vessel_id
        if any(item.vessel_id == vessel_id for item in self.world.snapshot().vessels):
            raise EnvironmentalFleetIngressError(
                f"cannot schedule ingress for active vessel {vessel_id!r}"
            )
        for event in self.coordinator._membership_events:  # noqa: SLF001 - shared membership authority
            event_vessel_id = (
                event.vessel.vessel_id if event.vessel is not None else event.vessel_id
            )
            if event_vessel_id == vessel_id:
                raise EnvironmentalFleetIngressError(
                    f"future membership event already scheduled for vessel {vessel_id!r}"
                )

        effective_limits, drift = self._validate_ingress_world_constraints(ingress)

        # Queue membership only after every mission/world constraint has been validated.
        # Metadata publication follows immediately and cannot leave a queued vessel without
        # its route target because all remaining operations are deterministic catalogue writes.
        self.persistent.schedule_arrival(
            at=ingress.at,
            vessel=ingress.vessel,
            limits=effective_limits,
            environmental_drift_velocity=drift,
        )
        route_target_replaced, profile_assigned = self._publish_ingress_metadata(ingress)

        admission = EnvironmentalFleetIngressAdmission(
            ingress,
            route_target_replaced=route_target_replaced,
            profile_assigned=profile_assigned,
        )
        self._ingress_admissions.append(admission)
        return admission

    def apply_current_ingress_batch(
        self,
        ingress_records: tuple[EnvironmentalFleetIngressRecord, ...],
    ) -> tuple[EnvironmentalFleetIngressAdmission, ...]:
        """Atomically admit a fully validated current-time ingress batch before observation.

        A151 uses this transaction only from the opt-in FIFO source extension.  It is
        deliberately separate from ``schedule_ingress``: future A125 membership events
        remain strictly future, while source admission is committed on an already reached
        physical timestamp.  The whole batch is validated first, mission metadata is
        published before physical membership, and membership-application evidence is
        emitted synchronously before any canonical fleet observation can be requested.
        """

        records = tuple(ingress_records)
        if any(not isinstance(item, EnvironmentalFleetIngressRecord) for item in records):
            raise TypeError("ingress_records must contain EnvironmentalFleetIngressRecord values")
        if not records:
            return ()
        if any(abs(item.at - self.world.time) > _EPS for item in records):
            raise EnvironmentalFleetIngressError(
                "current ingress batch must be bound to the exact current world time"
            )

        ids = tuple(item.vessel_id for item in records)
        if len(ids) != len(set(ids)):
            raise EnvironmentalFleetIngressError("current ingress batch vessel ids must be unique")
        active = {item.vessel_id for item in self.world.snapshot().vessels}
        overlap = active.intersection(ids)
        if overlap:
            raise EnvironmentalFleetIngressError(
                f"cannot admit already active source vessels: {sorted(overlap)!r}"
            )
        pending_ids = {
            event.vessel.vessel_id if event.vessel is not None else event.vessel_id
            for event in self.coordinator._membership_events  # noqa: SLF001 - shared authority preflight
        }
        overlap = pending_ids.intersection(ids)
        if overlap:
            raise EnvironmentalFleetIngressError(
                f"future membership already scheduled for source vessels: {sorted(overlap)!r}"
            )

        validated = tuple(
            (item, *self._validate_ingress_world_constraints(item)) for item in records
        )

        # Pre-compute audit flags from the pre-publication catalogues.  IDs are unique,
        # so later batch items cannot change the answer for an earlier one.
        targets = self._ensure_runtime_route_target_catalog()
        audit_flags = tuple(
            (
                item.vessel_id in targets and targets[item.vessel_id] != item.route_target,
                item.profile is not None,
            )
            for item in records
        )

        # All fallible world/identity checks are complete.  Metadata must become visible
        # before physical membership so the one subsequent registry/bootstrap observation
        # can never see a vessel without its mission definition.
        for item, _, _ in validated:
            self._publish_ingress_metadata(item)

        admissions: list[EnvironmentalFleetIngressAdmission] = []
        for (item, effective_limits, drift), (route_replaced, profile_assigned) in zip(
            validated, audit_flags, strict=True
        ):
            self.world.add_vessel(
                item.vessel,
                limits=effective_limits,
                environmental_drift_velocity=drift,
            )
            self.coordinator._active_incarnations[item.vessel_id] = None  # noqa: SLF001
            self.coordinator._record_membership_application(  # noqa: SLF001
                EnvironmentalMultiOwnerMembershipApplication(
                    self.world.time,
                    EnvironmentalMultiOwnerMembershipEventKind.VESSEL_ARRIVAL,
                    item.vessel_id,
                    None,
                    EnvironmentalMultiOwnerMembershipApplicationOutcome.APPLIED,
                )
            )
            admission = EnvironmentalFleetIngressAdmission(
                item,
                route_target_replaced=route_replaced,
                profile_assigned=profile_assigned,
            )
            self._ingress_admissions.append(admission)
            admissions.append(admission)
        return tuple(admissions)

    def admit_current_clear_route_targets(
        self,
        vessel_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Activate nominal route control for clear channels at a committed boundary.

        This A137 composition hook does not observe the registry or mutate a command.
        The underlying A119 admission binds the current snapshot/command tuple; when at
        least one route owner is installed, A123 leaves idle mode so A119 can execute
        that owner on the existing shared clock.
        """

        owner_ids = self.mixed_session.admit_current_clear_route_targets(vessel_ids)
        if owner_ids:
            self._idle_active = False
        return owner_ids

    def _pending_membership_events_for(self, vessel_id: str):
        return tuple(
            event
            for event in self.coordinator._membership_events  # noqa: SLF001
            if (event.vessel.vessel_id if event.vessel is not None else event.vessel_id) == vessel_id
        )

    def _activate_next_incarnation_metadata(self, vessel_id: str) -> bool:
        pending = self._pending_incarnations.get(vessel_id, [])
        if not pending:
            return False
        self._publish_ingress_metadata(pending[0].episode.ingress)
        return True

    def schedule_incarnation_episode(
        self,
        episode: EnvironmentalFleetIncarnationEpisode,
    ) -> EnvironmentalFleetIncarnationAdmission:
        """Queue one reusable-ID generation without exposing later mission metadata early."""

        if not isinstance(episode, EnvironmentalFleetIncarnationEpisode):
            raise TypeError("episode must be EnvironmentalFleetIncarnationEpisode")
        ingress = episode.ingress
        vessel_id = episode.vessel_id
        self.coordinator._validate_future_membership_time(ingress.at)  # noqa: SLF001
        if episode.egress_at is not None:
            self.coordinator._validate_future_membership_time(episode.egress_at)  # noqa: SLF001

        active_ids = {item.vessel_id for item in self.world.snapshot().vessels}
        pending = self._pending_incarnations.setdefault(vessel_id, [])
        existing_events = self._pending_membership_events_for(vessel_id)

        if not pending:
            # A127 owns the complete future timeline for an identifier once the first
            # episode is accepted.  Mixing it with raw A117/A125/A126 events would make
            # generation ownership ambiguous, so reject that combination fail-closed.
            if vessel_id in active_ids:
                raise EnvironmentalFleetIncarnationError(
                    f"cannot begin future incarnation timeline while vessel {vessel_id!r} is active"
                )
            if existing_events:
                last_assigned = self._next_incarnation_by_vessel.get(vessel_id, 0)
                if any(
                    event.incarnation is None or event.incarnation > last_assigned
                    for event in existing_events
                ):
                    raise EnvironmentalFleetIncarnationError(
                        f"non-incarnation membership event already scheduled for vessel {vessel_id!r}"
                    )
                # Only stale generation-tagged events from an already retired incarnation
                # remain.  They are intentionally allowed to coexist with the new episode;
                # the A117 generation guard will skip them if they later become due.
        else:
            expected_incarnations = {item.incarnation for item in pending}
            if any(event.incarnation not in expected_incarnations for event in existing_events):
                raise EnvironmentalFleetIncarnationError(
                    f"membership timeline for vessel {vessel_id!r} contains an unowned event"
                )
            previous = pending[-1]
            if previous.egress_at is None:
                raise EnvironmentalFleetIncarnationError(
                    f"incarnation {previous.incarnation} for vessel {vessel_id!r} has no scheduled egress"
                )
            if ingress.at < previous.egress_at - _EPS:
                raise EnvironmentalFleetIncarnationError(
                    f"incarnation ingress for vessel {vessel_id!r} overlaps the previous generation"
                )

        effective_limits, drift = self._validate_ingress_world_constraints(
            ingress,
            error_type=EnvironmentalFleetIncarnationError,
        )
        incarnation = self._next_incarnation_by_vessel.get(vessel_id, 0) + 1

        # The coordinator publishes both generation-tagged events as one heap replacement,
        # so an exception cannot leave an arrival without its paired removal.
        self.coordinator.schedule_incarnation_episode(
            at=ingress.at,
            vessel=ingress.vessel,
            incarnation=incarnation,
            limits=effective_limits,
            environmental_drift_velocity=drift,
            remove_at=episode.egress_at,
        )

        metadata_activated = not pending
        admission = EnvironmentalFleetIncarnationAdmission(
            episode,
            incarnation,
            metadata_activated=metadata_activated,
        )
        pending.append(admission)
        self._next_incarnation_by_vessel[vessel_id] = incarnation
        self._incarnation_admissions.append(admission)
        if metadata_activated:
            self._publish_ingress_metadata(ingress)
        return admission

    def _pending_incarnation_index(self, vessel_id: str, incarnation: int) -> tuple[list[EnvironmentalFleetIncarnationAdmission], int]:
        pending = self._pending_incarnations.get(vessel_id, [])
        for index, admission in enumerate(pending):
            if admission.incarnation == incarnation:
                return pending, index
        raise EnvironmentalFleetIncarnationError(
            f"no pending incarnation {incarnation} for vessel {vessel_id!r}"
        )

    def _clear_live_mission_metadata(self, vessel_id: str) -> None:
        self._ensure_runtime_route_target_catalog().pop(vessel_id, None)
        if self.profile_catalog is not None:
            self.profile_catalog.remove(vessel_id)

    def cancel_incarnation_episode(
        self,
        vessel_id: str,
        incarnation: int,
    ) -> EnvironmentalFleetIncarnationCancellation:
        """Cancel one not-yet-arrived generation without reusing its token."""

        if not vessel_id or not vessel_id.strip():
            raise ValueError("vessel_id must be non-empty")
        vessel_id = vessel_id.strip()
        if not isinstance(incarnation, int) or isinstance(incarnation, bool) or incarnation < 1:
            raise ValueError("incarnation must be a positive integer")
        pending, index = self._pending_incarnation_index(vessel_id, incarnation)
        if vessel_id in {item.vessel_id for item in self.world.snapshot().vessels}:
            try:
                active = self.coordinator.active_incarnation_for(vessel_id)
            except KeyError:  # pragma: no cover - guarded by snapshot membership
                active = None
            if active == incarnation:
                raise EnvironmentalFleetIncarnationError(
                    f"cannot cancel active incarnation {incarnation} of vessel {vessel_id!r}"
                )

        admission = pending[index]
        expected_events = 1 + int(admission.egress_at is not None)
        matching = tuple(
            event
            for event in self._pending_membership_events_for(vessel_id)
            if event.incarnation == incarnation
        )
        if len(matching) != expected_events:
            raise EnvironmentalFleetIncarnationError(
                f"pending calendar for incarnation {incarnation} of vessel {vessel_id!r} is incomplete"
            )

        removed_count = self.coordinator.cancel_incarnation_events(
            vessel_id=vessel_id,
            incarnation=incarnation,
        )
        pending.pop(index)
        if not pending:
            self._pending_incarnations.pop(vessel_id, None)

        metadata_reassigned = index == 0
        if metadata_reassigned:
            self._clear_live_mission_metadata(vessel_id)
            self._activate_next_incarnation_metadata(vessel_id)

        result = EnvironmentalFleetIncarnationCancellation(
            vessel_id,
            incarnation,
            admission.episode,
            removed_count,
            metadata_reassigned,
        )
        self._incarnation_cancellations.append(result)
        return result

    def amend_incarnation_episode(
        self,
        vessel_id: str,
        incarnation: int,
        amended: EnvironmentalFleetIncarnationEpisode,
    ) -> EnvironmentalFleetIncarnationAmendment:
        """Replace one not-yet-arrived generation in place, preserving its token."""

        if not vessel_id or not vessel_id.strip():
            raise ValueError("vessel_id must be non-empty")
        vessel_id = vessel_id.strip()
        if not isinstance(incarnation, int) or isinstance(incarnation, bool) or incarnation < 1:
            raise ValueError("incarnation must be a positive integer")
        if not isinstance(amended, EnvironmentalFleetIncarnationEpisode):
            raise TypeError("amended must be EnvironmentalFleetIncarnationEpisode")
        if amended.vessel_id != vessel_id:
            raise EnvironmentalFleetIncarnationError(
                "amended episode vessel id must match the requested timeline vessel"
            )
        pending, index = self._pending_incarnation_index(vessel_id, incarnation)
        if vessel_id in {item.vessel_id for item in self.world.snapshot().vessels}:
            try:
                active = self.coordinator.active_incarnation_for(vessel_id)
            except KeyError:  # pragma: no cover
                active = None
            if active == incarnation:
                raise EnvironmentalFleetIncarnationError(
                    f"cannot amend active incarnation {incarnation} of vessel {vessel_id!r}"
                )

        self.coordinator._validate_future_membership_time(amended.ingress_at)  # noqa: SLF001
        if amended.egress_at is not None:
            self.coordinator._validate_future_membership_time(amended.egress_at)  # noqa: SLF001
        if index > 0:
            previous = pending[index - 1]
            if previous.egress_at is None or amended.ingress_at < previous.egress_at - _EPS:
                raise EnvironmentalFleetIncarnationError(
                    f"amended incarnation {incarnation} overlaps the previous generation"
                )
        if index + 1 < len(pending):
            following = pending[index + 1]
            if amended.egress_at is None or following.ingress_at < amended.egress_at - _EPS:
                raise EnvironmentalFleetIncarnationError(
                    f"amended incarnation {incarnation} overlaps the following generation"
                )

        effective_limits, drift = self._validate_ingress_world_constraints(
            amended.ingress,
            error_type=EnvironmentalFleetIncarnationError,
        )
        old = pending[index]
        expected_events = 1 + int(old.egress_at is not None)
        matching = tuple(
            event
            for event in self._pending_membership_events_for(vessel_id)
            if event.incarnation == incarnation
        )
        if len(matching) != expected_events:
            raise EnvironmentalFleetIncarnationError(
                f"pending calendar for incarnation {incarnation} of vessel {vessel_id!r} is incomplete"
            )

        self.coordinator.amend_incarnation_episode_events(
            vessel=amended.ingress.vessel,
            incarnation=incarnation,
            at=amended.ingress_at,
            limits=effective_limits,
            environmental_drift_velocity=drift,
            remove_at=amended.egress_at,
        )
        metadata_republished = index == 0
        replacement = EnvironmentalFleetIncarnationAdmission(
            amended,
            incarnation,
            metadata_activated=metadata_republished,
        )
        pending[index] = replacement
        if metadata_republished:
            self._publish_ingress_metadata(amended.ingress)

        result = EnvironmentalFleetIncarnationAmendment(
            vessel_id,
            incarnation,
            old.episode,
            amended,
            metadata_republished,
        )
        self._incarnation_amendments.append(result)
        return result

    def schedule_egress(
        self,
        egress: EnvironmentalFleetEgressRecord,
    ) -> EnvironmentalFleetEgressAdmission:
        """Validate and queue one future removal with mission-retirement evidence."""

        if not isinstance(egress, EnvironmentalFleetEgressRecord):
            raise TypeError("egress must be EnvironmentalFleetEgressRecord")
        self.coordinator._validate_future_membership_time(egress.at)  # noqa: SLF001
        vessel_id = egress.vessel_id
        if vessel_id not in {item.vessel_id for item in self.world.snapshot().vessels}:
            raise EnvironmentalFleetEgressError(
                f"cannot schedule egress for inactive vessel {vessel_id!r}"
            )
        for event in self.coordinator._membership_events:  # noqa: SLF001 - shared membership authority
            event_vessel_id = (
                event.vessel.vessel_id if event.vessel is not None else event.vessel_id
            )
            if event_vessel_id == vessel_id:
                raise EnvironmentalFleetEgressError(
                    f"future membership event already scheduled for vessel {vessel_id!r}"
                )

        targets = self._ensure_runtime_route_target_catalog()
        route_target = targets.get(vessel_id)
        profile = None if self.profile_catalog is None else self.profile_catalog.profile_for(vessel_id)
        admission = EnvironmentalFleetEgressAdmission(egress, route_target, profile)

        self.persistent.schedule_removal(at=egress.at, vessel_id=vessel_id)
        self._egress_admissions.append(admission)
        self._pending_egress_admissions[(float(egress.at), vessel_id)] = admission
        return admission

    def _retire_removed_mission_metadata(
        self,
        vessel_id: str,
        *,
        at: float,
        source: EnvironmentalFleetEgressSource,
        incarnation: int | None = None,
    ) -> EnvironmentalFleetEgressCompletion:
        targets = self._ensure_runtime_route_target_catalog()
        retired_target = targets.pop(vessel_id, None)
        retired_profile = None
        if self.profile_catalog is not None:
            retired_profile = self.profile_catalog.remove(vessel_id)
        admission = self._pending_egress_admissions.pop((float(at), vessel_id), None)

        if incarnation is not None:
            pending = self._pending_incarnations.get(vessel_id, [])
            if pending and pending[0].incarnation == incarnation:
                pending.pop(0)
                if not pending:
                    self._pending_incarnations.pop(vessel_id, None)
            elif any(item.incarnation == incarnation for item in pending):
                raise EnvironmentalIdleFleetControlError(
                    f"incarnation {incarnation} for vessel {vessel_id!r} retired out of order"
                )

        completion = EnvironmentalFleetEgressCompletion(
            vessel_id,
            at,
            source,
            admission,
            retired_target,
            retired_profile,
            incarnation,
        )
        self._egress_completions.append(completion)
        self._activate_next_incarnation_metadata(vessel_id)
        return completion

    def _handle_membership_application(
        self,
        item: EnvironmentalMultiOwnerMembershipApplication,
    ) -> None:
        if not item.applied:
            return
        if item.kind is EnvironmentalMultiOwnerMembershipEventKind.VESSEL_REMOVAL:
            self._retire_removed_mission_metadata(
                item.vessel_id,
                at=item.time_min,
                source=EnvironmentalFleetEgressSource.MEMBERSHIP_REMOVAL,
                incarnation=item.incarnation,
            )
            return
        if item.incarnation is None:
            return
        pending = self._pending_incarnations.get(item.vessel_id, [])
        if not pending or pending[0].incarnation != item.incarnation:
            # Generation-tagged events may be used directly by lower-level callers.
            # A127 only owns metadata for admissions present in its own pending ledger.
            return
        # Re-publish the already validated mission metadata at physical ingress.  This
        # makes the ordering robust even if an inactive-ID catalogue was externally
        # edited between scheduling and the arrival boundary, and it happens before
        # A117 can request registry observation/rebootstrap at this timestamp.
        self._publish_ingress_metadata(pending[0].episode.ingress)

    def _on_membership_application(
        self,
        item: EnvironmentalMultiOwnerMembershipApplication,
    ) -> None:
        # Called synchronously by A117 immediately after physical membership commit and
        # before that boundary can trigger registry observation/rebootstrap.  This is the
        # key ordering guarantee that swaps mission metadata between same-ID generations
        # before any new owner is planned.
        self._handle_membership_application(item)
        self._processed_membership_application_count = len(self.coordinator.membership_applications)

    def _consume_membership_application_evidence(self) -> None:
        applications = self.coordinator.membership_applications
        for item in applications[self._processed_membership_application_count :]:
            self._handle_membership_application(item)
        self._processed_membership_application_count = len(applications)

    def _consume_active_removal_evidence(self) -> None:
        self._consume_membership_application_evidence()

        dispositions = self.persistent.disposition_results
        for result in dispositions[self._processed_disposition_count :]:
            for vessel_id in result.removed_vessel_ids:
                try:
                    incarnation = self.coordinator.active_incarnation_for(vessel_id)
                except KeyError:
                    # A121 has already physically removed the vessel.  The coordinator's
                    # generation ledger still contains the pre-removal identity until this
                    # runtime-owned disposition evidence retires it.
                    incarnation = self.coordinator._active_incarnations.get(vessel_id)  # noqa: SLF001
                self.coordinator.note_external_vessel_removal(vessel_id)
                self._retire_removed_mission_metadata(
                    vessel_id,
                    at=result.snapshot_after.time,
                    source=EnvironmentalFleetEgressSource.ROUTE_END_REMOVE,
                    incarnation=incarnation,
                )
        self._processed_disposition_count = len(dispositions)

    def schedule_arrival(self, **kwargs) -> None:
        self.persistent.schedule_arrival(**kwargs)

    def schedule_removal(self, **kwargs) -> None:
        self.persistent.schedule_removal(**kwargs)

    def _purge_completed_route_owners(self) -> None:
        for owner_id, owner in tuple(self.mixed_session._route_owners.items()):  # noqa: SLF001
            if owner.stage is EnvironmentalMixedRouteStage.COMPLETE:
                self.mixed_session._route_owners.pop(owner_id)  # noqa: SLF001

    def _current_observation(self) -> EnvironmentalFleetOwnershipObservation:
        if self.coordinator.fleet_observations and abs(
            self.coordinator.fleet_observations[-1].time_min - self.world.time
        ) <= _EPS:
            return self.coordinator.fleet_observations[-1]
        return self.coordinator._shared_observation()  # noqa: SLF001

    def _bootstrap_if_needed(
        self,
        observation: EnvironmentalFleetOwnershipObservation,
    ) -> tuple[str, ...]:
        if not observation.dispatch.owners:
            return ()
        result = bootstrap_environmental_idle_fleet_owners(
            self.persistent,
            observation,
            config=self.recapture_config,
            profile_catalog=self.profile_catalog,
        )
        self._bootstrap_results.append(result)
        self._next_idle_review_time = None
        return result.owner_ids

    def _record_idle_boundary(
        self,
        *,
        reasons: tuple[EnvironmentalIdleFleetReason, ...],
        membership_event_count: int,
        removed_vessel_ids: tuple[str, ...],
        observation: EnvironmentalFleetOwnershipObservation,
        bootstrapped_owner_ids: tuple[str, ...] = (),
    ) -> None:
        self._idle_boundaries.append(
            EnvironmentalIdleFleetBoundary(
                self.world.time,
                reasons,
                membership_event_count,
                removed_vessel_ids,
                observation,
                bootstrapped_owner_ids,
            )
        )

    def _enter_idle(self) -> bool:
        self._purge_completed_route_owners()
        observation = self._current_observation()
        owner_ids = self._bootstrap_if_needed(observation)
        self._record_idle_boundary(
            reasons=(EnvironmentalIdleFleetReason.CONTROL_COMPLETE,),
            membership_event_count=0,
            removed_vessel_ids=(),
            observation=observation,
            bootstrapped_owner_ids=owner_ids,
        )
        if owner_ids:
            self._idle_active = False
            return False
        self._idle_active = True
        if self._next_idle_review_time is None or self._next_idle_review_time <= self.world.time + _EPS:
            self._next_idle_review_time = self.world.time + self.config.idle_review_interval_min
        return True

    def _run_idle_until(self, end_time: float) -> bool:
        """Run idle calendar. Return False after fresh owners were bootstrapped."""

        if self._next_idle_review_time is None:
            self._next_idle_review_time = self.world.time + self.config.idle_review_interval_min

        for _ in range(self.config.max_idle_boundaries_per_run):
            if self.world.time >= end_time - _EPS:
                return True
            membership_time = self.coordinator._next_membership_time()  # noqa: SLF001
            command_time = self.world.next_command_time()
            candidates = [end_time, self._next_idle_review_time]
            if membership_time is not None:
                candidates.append(membership_time)
            if command_time is not None:
                candidates.append(command_time)
            native_boundary = min(candidates)

            source_boundary = self.mixed_session._source_boundary_time(native_boundary)  # noqa: SLF001
            if source_boundary is not None:
                if source_boundary < self.world.time - _EPS:
                    raise EnvironmentalIdleFleetControlError(
                        "source extension proposed an idle boundary in the past"
                    )
                if source_boundary > native_boundary + _EPS:
                    raise EnvironmentalIdleFleetControlError(
                        "source extension proposed an idle boundary beyond its frozen horizon"
                    )
            boundary = (
                source_boundary
                if source_boundary is not None and source_boundary < native_boundary - _EPS
                else native_boundary
            )
            source_boundary_due = (
                source_boundary is not None and abs(source_boundary - boundary) <= _EPS
            )
            native_reached = abs(native_boundary - boundary) <= _EPS
            review_due = native_reached and abs(native_boundary - self._next_idle_review_time) <= _EPS
            command_due = (
                native_reached
                and command_time is not None
                and abs(native_boundary - command_time) <= _EPS
            )
            membership_due = (
                native_reached
                and membership_time is not None
                and abs(native_boundary - membership_time) <= _EPS
            )
            native_boundary_due = review_due or command_due or membership_due

            self.world.run_until(boundary)
            if not native_boundary_due and not source_boundary_due:
                return True

            membership_count, removed = self.coordinator._apply_due_membership()  # noqa: SLF001
            self._consume_membership_application_evidence()

            source_vessel_ids = self.mixed_session._apply_source_pre_observation_phase(  # noqa: SLF001
                native_boundary_due=native_boundary_due,
                source_boundary_due=source_boundary_due,
            )
            membership_count += len(source_vessel_ids)
            self._consume_membership_application_evidence()

            reasons: list[EnvironmentalIdleFleetReason] = []
            if review_due:
                reasons.append(EnvironmentalIdleFleetReason.PERIODIC_REVIEW)
                while self._next_idle_review_time <= self.world.time + _EPS:
                    self._next_idle_review_time += self.config.idle_review_interval_min
            if command_due:
                reasons.append(EnvironmentalIdleFleetReason.COMMAND_EVENT)
            if membership_count:
                reasons.append(EnvironmentalIdleFleetReason.VESSEL_MEMBERSHIP)

            # A source-only wake that remains blocked is an event-planning boundary,
            # not a navigation observation.  Do not fabricate a registry sample merely
            # because A150 asked to re-evaluate the FIFO head.
            if source_boundary_due and not native_boundary_due and not source_vessel_ids:
                continue

            observation = self.coordinator._shared_observation()  # noqa: SLF001
            owner_ids = self._bootstrap_if_needed(observation)
            clear_route_owner_ids = self.admit_current_clear_route_targets(source_vessel_ids)
            self._record_idle_boundary(
                reasons=tuple(reasons),
                membership_event_count=membership_count,
                removed_vessel_ids=removed,
                observation=observation,
                bootstrapped_owner_ids=owner_ids,
            )
            if owner_ids or clear_route_owner_ids:
                self._idle_active = False
                return False
            if self.world.time >= end_time - _EPS:
                return True

        raise EnvironmentalIdleFleetControlError(
            f"a123 exceeded max_idle_boundaries_per_run={self.config.max_idle_boundaries_per_run}"
        )

    def _result(
        self,
        reason: EnvironmentalPersistentFleetStopReason,
        transitions: int,
    ) -> EnvironmentalPersistentFleetControlResult:
        return EnvironmentalPersistentFleetControlResult(
            stop_reason=reason,
            snapshot=self.world.snapshot(),
            terminal_persistent_result=self._last_persistent_result,
            idle_boundaries=self.idle_boundaries,
            bootstrap_results=self.bootstrap_results,
            control_idle_transition_count=transitions,
        )

    def run_until(self, end_time: float) -> EnvironmentalPersistentFleetControlResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")

        transitions = 0
        while transitions <= self.config.max_control_idle_transitions_per_run:
            if self._idle_active:
                reached_end = self._run_idle_until(end_time)
                if reached_end:
                    return self._result(EnvironmentalPersistentFleetStopReason.END_TIME, transitions)
                transitions += 1
                # A fresh idle bootstrap reactivated mixed control at the same timestamp.

            result = self.persistent.run_until(end_time)
            self._last_persistent_result = result
            self._consume_active_removal_evidence()
            if result.stop_reason is EnvironmentalPersistentMixedStopReason.COMMAND_OVERRIDDEN:
                return self._result(EnvironmentalPersistentFleetStopReason.COMMAND_OVERRIDDEN, transitions)
            if result.stop_reason is EnvironmentalPersistentMixedStopReason.END_TIME:
                return self._result(EnvironmentalPersistentFleetStopReason.END_TIME, transitions)
            if result.stop_reason is not EnvironmentalPersistentMixedStopReason.ALL_CONTROL_COMPLETE:
                raise AssertionError(result.stop_reason)

            transitions += 1
            idle = self._enter_idle()
            if not idle:
                # A current-time canonical dispatch already contained fresh owners.
                continue
            reached_end = self._run_idle_until(end_time)
            if reached_end:
                return self._result(EnvironmentalPersistentFleetStopReason.END_TIME, transitions)
            transitions += 1
            # Fresh owners were bootstrapped from an idle boundary; resume A122.

        raise EnvironmentalIdleFleetControlError(
            "a123 exceeded max_control_idle_transitions_per_run before reaching caller boundary"
        )
