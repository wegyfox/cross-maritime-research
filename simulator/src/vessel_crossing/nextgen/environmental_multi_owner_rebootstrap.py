"""Generic owner-set rebootstrap from one committed a117 recapture boundary.

A118 consumes the single canonical fleet observation already produced by the shared
A117 coordinator.  It never re-observes the registry at that timestamp.  Every new
PAIR/JOINT owner touching the old coordinator command set is fully preflighted against
that same observation and physical snapshot.  Only after all plans validate is one
combined water-relative command batch published.  Fresh a102/a113 guards are then
attached to the already-committed observation and the A117 coordinator adopts the new
owner set without a second registry update.

Released vessels that are no longer part of any active replacement owner are reported
but deliberately remain outside A118 route-handoff authority.  Mixed avoidance/route
ownership is the next layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .encounter_registry import EncounterRegistry
from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_fleet_guarded_joint_control import EnvironmentalFleetGuardedJointControlSession
from .environmental_fleet_guarded_pair_control import EnvironmentalFleetGuardedPairControlSession
from .environmental_fleet_multi_owner_recapture import (
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
from .environmental_joint_control_cycle import EnvironmentalJointControlSession
from .environmental_joint_execution import AppliedEnvironmentalJointPlan
from .environmental_joint_plan import validate_environmental_joint_execution_plan
from .environmental_multi_owner_control import (
    EnvironmentalMultiOwnerControlResult,
    EnvironmentalMultiOwnerControlSession,
    EnvironmentalMultiOwnerControlStopReason,
)
from .environmental_pair_control_cycle import EnvironmentalPairControlSession
from .environmental_pair_execution import AppliedEnvironmentalPairPlan
from .environmental_pair_plan import validate_environmental_pair_execution_plan
from .environmental_planning_context import build_environmental_avoidance_planning_context
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalMultiOwnerRebootstrapError(RuntimeError):
    """Base error for invalid A118 owner-set replacement."""


class EnvironmentalMultiOwnerRebootstrapAction(str, Enum):
    OWNER_SET_REPLACED = "owner_set_replaced"
    NO_ACTIVE_REPLACEMENT_OWNER = "no_active_replacement_owner"


@dataclass(frozen=True, slots=True)
class EnvironmentalMultiOwnerRebootstrapResult:
    source: EnvironmentalMultiOwnerControlResult
    action: EnvironmentalMultiOwnerRebootstrapAction
    committed_observation_time: float
    replacement_owners: tuple[FleetCommandOwner, ...]
    released_vessel_ids: tuple[str, ...]
    removed_vessel_ids: tuple[str, ...]
    snapshot_before: EnvironmentalWorldSnapshot
    snapshot_after: EnvironmentalWorldSnapshot
    commands_before: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    commands_after: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    started_owners: tuple[EnvironmentalStartedOwner, ...]
    released_commands: tuple[tuple[str, WaterRelativeManeuverCommand], ...] = ()

    @property
    def owner_ids(self) -> tuple[str, ...]:
        return tuple(item.owner.component_id for item in self.started_owners)

    @property
    def replaced(self) -> bool:
        return self.action is EnvironmentalMultiOwnerRebootstrapAction.OWNER_SET_REPLACED


def _validate_boundary(
    coordinator: EnvironmentalMultiOwnerControlSession,
    source: EnvironmentalMultiOwnerControlResult,
) -> None:
    if not isinstance(coordinator, EnvironmentalMultiOwnerControlSession):
        raise TypeError("coordinator must be EnvironmentalMultiOwnerControlSession")
    if not isinstance(source, EnvironmentalMultiOwnerControlResult):
        raise TypeError("source must be EnvironmentalMultiOwnerControlResult")
    if source.stop_reason is not EnvironmentalMultiOwnerControlStopReason.RECAPTURE_REQUIRED:
        raise EnvironmentalMultiOwnerRebootstrapError(
            "a118 requires an a117 RECAPTURE_REQUIRED boundary"
        )
    if source.recapture_observation is None or not source.recapture_owner_ids:
        raise EnvironmentalMultiOwnerRebootstrapError(
            "a117 recapture result has no committed ownership observation"
        )
    if source.source != coordinator.source:
        raise EnvironmentalMultiOwnerRebootstrapError(
            "a117 recapture result does not belong to this coordinator"
        )
    if coordinator.world.snapshot() != source.snapshot:
        raise EnvironmentalMultiOwnerRebootstrapError(
            "world no longer matches the a117 recapture boundary"
        )
    if abs(coordinator.world.time - source.recapture_observation.time_min) > _EPS:
        raise EnvironmentalMultiOwnerRebootstrapError(
            "committed a117 observation timestamp does not match current world time"
        )
    observations = coordinator.fleet_observations
    if not observations or observations[-1] != source.recapture_observation:
        raise EnvironmentalMultiOwnerRebootstrapError(
            "a117 recapture observation is not the coordinator's latest canonical observation"
        )


def _select_replacement_owners(
    coordinator: EnvironmentalMultiOwnerControlSession,
    source: EnvironmentalMultiOwnerControlResult,
) -> tuple[tuple[FleetCommandOwner, ...], tuple[str, ...], tuple[str, ...]]:
    old_ids = {
        vessel_id
        for started in coordinator.started_owners
        for vessel_id in started.owner.vessel_ids
    }
    owners = tuple(
        owner
        for owner in source.recapture_observation.dispatch.owners  # type: ignore[union-attr]
        if old_ids.intersection(owner.vessel_ids)
    )

    claimed: set[str] = set()
    for owner in owners:
        overlap = claimed.intersection(owner.vessel_ids)
        if overlap:
            raise EnvironmentalMultiOwnerRebootstrapError(
                f"canonical replacement owners overlap command channels: {sorted(overlap)!r}"
            )
        claimed.update(owner.vessel_ids)

    current_ids = {vessel.vessel_id for vessel in coordinator.world.snapshot().vessels}
    released = tuple(sorted((old_ids - claimed).intersection(current_ids)))
    removed = tuple(sorted(old_ids - current_ids))
    return owners, released, removed


def _attach_pair(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    item: EnvironmentalPreparedPairOwner,
    snapshot_before: EnvironmentalWorldSnapshot,
    snapshot_after: EnvironmentalWorldSnapshot,
    before_map: dict[str, WaterRelativeManeuverCommand],
    after_map: dict[str, WaterRelativeManeuverCommand],
    observation,
    all_route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None,
) -> EnvironmentalStartedPairOwner:
    owner = item.owner
    applied_at = world.time
    applied = AppliedEnvironmentalPairPlan(
        plan=item.plan,
        applied_at_time=applied_at,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        commands_before=(before_map[item.plan.first_id], before_map[item.plan.second_id]),
        commands_after=(after_map[item.plan.first_id], after_map[item.plan.second_id]),
    )
    holder: dict[str, EnvironmentalFleetGuardedPairControlSession] = {}

    def planning_context_factory(current_world, _policy):
        guard = holder.get("guard")
        if guard is None or not guard.fleet_observations:
            raise EnvironmentalMultiOwnerRebootstrapError(
                "replacement PAIR planning factory lacks current a102 ownership evidence"
            )
        latest = guard.fleet_observations[-1]
        if abs(latest.time_min - current_world.time) > _EPS:
            raise EnvironmentalMultiOwnerRebootstrapError(
                "replacement PAIR replanning lacks same-timestamp fleet observation"
            )
        matches = tuple(
            candidate
            for candidate in latest.dispatch.owners
            if candidate.backend is FleetSupervisorBackend.PAIR
            and candidate.vessel_ids == owner.vessel_ids
            and candidate.pair_keys == owner.pair_keys
        )
        if len(matches) != 1:
            raise EnvironmentalMultiOwnerRebootstrapError(
                "current fleet observation no longer contains exact replacement PAIR owner"
            )
        return build_environmental_avoidance_planning_context(
            current_world,
            latest.dispatch,
            matches[0],
            profile_catalog=profile_catalog,
            route_targets=all_route_targets,
            same_route_following=config.same_route_following,
        )

    pair_cycle = EnvironmentalPairControlSession(
        world,
        applied,
        item.lifecycle,
        item.decision,
        item.route_targets,
        planning_context_factory,
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
        raise EnvironmentalMultiOwnerRebootstrapError(
            f"fresh a102 guard rejected replacement PAIR owner {owner.component_id!r}: {initial!r}"
        )
    return EnvironmentalStartedPairOwner(item, applied, pair_cycle, guarded)


def _attach_joint(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    item: EnvironmentalPreparedJointOwner,
    snapshot_before: EnvironmentalWorldSnapshot,
    snapshot_after: EnvironmentalWorldSnapshot,
    before_map: dict[str, WaterRelativeManeuverCommand],
    after_map: dict[str, WaterRelativeManeuverCommand],
    observation,
    all_route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None,
) -> EnvironmentalStartedJointOwner:
    owner = item.owner
    applied_at = world.time
    applied = AppliedEnvironmentalJointPlan(
        plan=item.plan,
        applied_at_time=applied_at,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        commands_before=tuple((vessel_id, before_map[vessel_id]) for vessel_id in item.plan.vessel_ids),
        commands_after=tuple((vessel_id, after_map[vessel_id]) for vessel_id in item.plan.vessel_ids),
    )
    holder: dict[str, EnvironmentalFleetGuardedJointControlSession] = {}

    def planning_context_factory(current_world, _policy):
        guard = holder.get("guard")
        if guard is None or not guard.fleet_observations:
            raise EnvironmentalMultiOwnerRebootstrapError(
                "replacement JOINT planning factory lacks current a113 ownership evidence"
            )
        latest = guard.fleet_observations[-1]
        if abs(latest.time_min - current_world.time) > _EPS:
            raise EnvironmentalMultiOwnerRebootstrapError(
                "replacement JOINT replanning lacks same-timestamp fleet observation"
            )
        matches = tuple(
            candidate
            for candidate in latest.dispatch.owners
            if candidate.backend is FleetSupervisorBackend.JOINT
            and candidate.vessel_ids == owner.vessel_ids
            and candidate.pair_keys == owner.pair_keys
        )
        if len(matches) != 1:
            raise EnvironmentalMultiOwnerRebootstrapError(
                "current fleet observation no longer contains exact replacement JOINT owner"
            )
        return build_environmental_avoidance_planning_context(
            current_world,
            latest.dispatch,
            matches[0],
            profile_catalog=profile_catalog,
            route_targets=all_route_targets,
            same_route_following=config.same_route_following,
        )

    joint_cycle = EnvironmentalJointControlSession(
        world,
        applied,
        item.lifecycles,
        item.decisions,
        planning_context_factory,
        config=config.joint_cycle,
        profile_catalog=profile_catalog,
    )
    guarded = EnvironmentalFleetGuardedJointControlSession(
        registry,
        joint_cycle,
        config=config.joint_fleet_guard,
        initial_observation=observation,
    )
    holder["guard"] = guarded
    initial = guarded.start()
    if guarded.recapture is not None:
        raise EnvironmentalMultiOwnerRebootstrapError(
            f"fresh a113 guard rejected replacement JOINT owner {owner.component_id!r}: {initial!r}"
        )
    return EnvironmentalStartedJointOwner(item, applied, joint_cycle, guarded)


def rebootstrap_environmental_multi_owner_control(
    registry: EncounterRegistry,
    coordinator: EnvironmentalMultiOwnerControlSession,
    source: EnvironmentalMultiOwnerControlResult,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalMultiOwnerRebootstrapResult:
    """Replace the current A117 owner set using its already-committed recapture evidence."""

    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(route_targets, Mapping):
        raise TypeError("route_targets must be a mapping")
    if not isinstance(config, EnvironmentalFleetMultiOwnerRecaptureConfig):
        raise TypeError("config must be EnvironmentalFleetMultiOwnerRecaptureConfig")

    _validate_boundary(coordinator, source)
    world = coordinator.world
    observation = source.recapture_observation
    assert observation is not None
    owners, released, removed = _select_replacement_owners(coordinator, source)
    snapshot_before = world.snapshot()
    released_commands = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in released)

    if not owners:
        return EnvironmentalMultiOwnerRebootstrapResult(
            source,
            EnvironmentalMultiOwnerRebootstrapAction.NO_ACTIVE_REPLACEMENT_OWNER,
            observation.time_min,
            (),
            released,
            removed,
            snapshot_before,
            snapshot_before,
            (),
            (),
            (),
            released_commands,
        )

    admitted_targets: dict[
        str, tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget]
    ] = {}
    for owner in owners:
        if owner.backend is FleetSupervisorBackend.PAIR:
            try:
                admitted_targets[owner.component_id] = _admit_pair_targets(owner, route_targets)
            except EnvironmentalFleetMultiOwnerRecaptureError as exc:
                raise EnvironmentalMultiOwnerRebootstrapError(str(exc)) from exc

    prepared: list[EnvironmentalPreparedPairOwner | EnvironmentalPreparedJointOwner] = []
    try:
        for owner in owners:
            if owner.backend is FleetSupervisorBackend.PAIR:
                prepared.append(
                    _prepare_pair_owner(
                        registry,
                        world,
                        owner,
                        admitted_targets[owner.component_id],
                        route_targets,
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
                        route_targets,
                        observation.dispatch,
                        config=config,
                        profile_catalog=profile_catalog,
                    )
                )
            else:  # pragma: no cover
                raise AssertionError(owner.backend)
    except EnvironmentalFleetMultiOwnerRecaptureError as exc:
        raise EnvironmentalMultiOwnerRebootstrapError(str(exc)) from exc

    command_batch: list[tuple[str, WaterRelativeManeuverCommand]] = []
    for item in prepared:
        if isinstance(item, EnvironmentalPreparedPairOwner):
            validate_environmental_pair_execution_plan(
                item.plan, world, profile_catalog=profile_catalog
            )
            command_batch.extend(_pair_commands(item.plan))
        else:
            validate_environmental_joint_execution_plan(
                item.plan, world, profile_catalog=profile_catalog
            )
            command_batch.extend(item.plan.selected_commands)

    ids = [vessel_id for vessel_id, _ in command_batch]
    if len(ids) != len(set(ids)):
        raise EnvironmentalMultiOwnerRebootstrapError(
            "replacement command transaction contains duplicate vessel channels"
        )
    expected = [vessel_id for owner in owners for vessel_id in owner.vessel_ids]
    if set(ids) != set(expected):
        raise EnvironmentalMultiOwnerRebootstrapError(
            "replacement command transaction does not cover every replacement owner vessel exactly once"
        )

    pending = world.next_command_time()
    if pending is not None and pending <= world.time + _EPS:
        raise PendingEnvironmentalMultiOwnerCommandEvent(
            "cannot rebootstrap owner set while a command event is pending at current time"
        )

    commands_before = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in ids)
    applied_at = world.time
    world.schedule_commands_atomic(at=applied_at, commands=tuple(command_batch))
    world.run_until(applied_at)
    snapshot_after = world.snapshot()
    commands_after = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in ids)
    if snapshot_after != snapshot_before or abs(world.time - applied_at) > _EPS:
        raise EnvironmentalMultiOwnerRebootstrapError(
            "same-timestamp owner-set replacement changed physical state or world time"
        )
    if commands_after != tuple(command_batch):
        raise EnvironmentalMultiOwnerRebootstrapError(
            "same-timestamp owner-set replacement did not publish the full command transaction"
        )

    before_map = dict(commands_before)
    after_map = dict(commands_after)
    started: list[EnvironmentalStartedOwner] = []
    for item in prepared:
        if isinstance(item, EnvironmentalPreparedPairOwner):
            started.append(
                _attach_pair(
                    registry,
                    world,
                    item,
                    snapshot_before,
                    snapshot_after,
                    before_map,
                    after_map,
                    observation,
                    route_targets,
                    config=config,
                    profile_catalog=profile_catalog,
                )
            )
        else:
            started.append(
                _attach_joint(
                    registry,
                    world,
                    item,
                    snapshot_before,
                    snapshot_after,
                    before_map,
                    after_map,
                    observation,
                    route_targets,
                    config=config,
                    profile_catalog=profile_catalog,
                )
            )

    started_tuple = tuple(started)
    coordinator._adopt_replacement_owner_set(  # noqa: SLF001 - A118 is the owner-set transaction layer
        started_tuple,
        committed_observation=observation,
    )

    return EnvironmentalMultiOwnerRebootstrapResult(
        source,
        EnvironmentalMultiOwnerRebootstrapAction.OWNER_SET_REPLACED,
        observation.time_min,
        owners,
        released,
        removed,
        snapshot_before,
        snapshot_after,
        commands_before,
        commands_after,
        started_tuple,
        released_commands,
    )
