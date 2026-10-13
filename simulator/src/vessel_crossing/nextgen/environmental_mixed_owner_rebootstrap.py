"""Atomic mixed-stage fleet recapture from one committed A119 observation.

A120 consumes the canonical ``RECAPTURE_REQUIRED`` boundary emitted by A119.  It does
not observe the fleet registry again.  Instead it treats the already committed dispatch
as the sole ownership decision for the timestamp, fully preflights every replacement
PAIR/JOINT avoidance owner, and publishes all replacement avoidance commands in one
same-timestamp atomic batch.

Route authority is changed only after that batch succeeds.  Route channels absorbed by
a replacement avoidance owner are retired; removed channels disappear; still-clear
channels preserve their route stage and progress.  A multi-vessel route owner may be
split so one recaptured vessel cannot revoke an unrelated clear route channel.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_fleet_multi_owner_recapture import (
    EnvironmentalFleetMultiOwnerRecaptureConfig,
    EnvironmentalFleetMultiOwnerRecaptureError,
    EnvironmentalPreparedJointOwner,
    EnvironmentalPreparedPairOwner,
    EnvironmentalStartedOwner,
    PendingEnvironmentalMultiOwnerCommandEvent,
    _admit_pair_targets,
    _pair_commands,
    _prepare_joint_owner,
    _prepare_pair_owner,
)
from .environmental_joint_plan import validate_environmental_joint_execution_plan
from .environmental_mixed_owner_control import (
    EnvironmentalMixedOwnerControlResult,
    EnvironmentalMixedOwnerControlSession,
    EnvironmentalMixedOwnerStopReason,
    EnvironmentalMixedRouteStage,
    _RouteOwner,
)
from .environmental_multi_owner_rebootstrap import _attach_joint, _attach_pair
from .environmental_nominal_route_session import EnvironmentalNominalRouteSession
from .environmental_pair_plan import validate_environmental_pair_execution_plan
from .environmental_route_handoff import EnvironmentalRouteHandoffBinding, EnvironmentalRouteHandoffTarget
from .environmental_route_handoff_session import (
    EnvironmentalRouteHandoffSession,
    EnvironmentalRouteHandoffSessionRunResult,
)
from .environmental_world import EnvironmentalWorldSnapshot
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalMixedOwnerRebootstrapError(RuntimeError):
    """Base error for invalid A120 mixed-stage rebootstrap."""


class EnvironmentalMixedOwnerRebootstrapAction(str, Enum):
    MIXED_OWNER_SET_REPLACED = "mixed_owner_set_replaced"
    NO_ACTIVE_REPLACEMENT_AVOIDANCE_OWNER = "no_active_replacement_avoidance_owner"


@dataclass(frozen=True, slots=True)
class EnvironmentalMixedOwnerRebootstrapResult:
    source: EnvironmentalMixedOwnerControlResult
    action: EnvironmentalMixedOwnerRebootstrapAction
    committed_observation_time: float
    replacement_owners: tuple[FleetCommandOwner, ...]
    retained_avoidance_owner_ids: tuple[str, ...]
    started_replacement_owners: tuple[EnvironmentalStartedOwner, ...]
    active_avoidance_owner_ids: tuple[str, ...]
    retired_route_owner_ids: tuple[str, ...]
    preserved_route_owner_ids: tuple[str, ...]
    preserved_route_vessel_ids: tuple[str, ...]
    released_vessel_ids: tuple[str, ...]
    released_commands: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    removed_vessel_ids: tuple[str, ...]
    snapshot_before: EnvironmentalWorldSnapshot
    snapshot_after: EnvironmentalWorldSnapshot
    commands_before: tuple[tuple[str, WaterRelativeManeuverCommand], ...]
    commands_after: tuple[tuple[str, WaterRelativeManeuverCommand], ...]

    @property
    def replaced(self) -> bool:
        return self.action is EnvironmentalMixedOwnerRebootstrapAction.MIXED_OWNER_SET_REPLACED


def _validate_boundary(
    session: EnvironmentalMixedOwnerControlSession,
    source: EnvironmentalMixedOwnerControlResult,
) -> None:
    if not isinstance(session, EnvironmentalMixedOwnerControlSession):
        raise TypeError("session must be EnvironmentalMixedOwnerControlSession")
    if not isinstance(source, EnvironmentalMixedOwnerControlResult):
        raise TypeError("source must be EnvironmentalMixedOwnerControlResult")
    if source.stop_reason is not EnvironmentalMixedOwnerStopReason.RECAPTURE_REQUIRED:
        raise EnvironmentalMixedOwnerRebootstrapError(
            "a120 requires an a119 RECAPTURE_REQUIRED boundary"
        )
    observation = source.recapture_observation
    if observation is None or not source.recapture_owner_ids:
        raise EnvironmentalMixedOwnerRebootstrapError(
            "a119 recapture result lacks committed ownership evidence"
        )
    if source.snapshot != session.world.snapshot():
        raise EnvironmentalMixedOwnerRebootstrapError(
            "world no longer matches the a119 recapture boundary"
        )
    if abs(observation.time_min - session.world.time) > _EPS:
        raise EnvironmentalMixedOwnerRebootstrapError(
            "a119 recapture observation timestamp no longer matches world"
        )
    if source.boundaries != session.boundaries:
        raise EnvironmentalMixedOwnerRebootstrapError(
            "a119 result does not belong to the current mixed-session boundary history"
        )
    if source.fleet_observations != session.fleet_observations:
        raise EnvironmentalMixedOwnerRebootstrapError(
            "a119 result does not match the current mixed-session fleet evidence"
        )
    if not session.fleet_observations or session.fleet_observations[-1] != observation:
        raise EnvironmentalMixedOwnerRebootstrapError(
            "a119 recapture observation is not the latest canonical shared observation"
        )


def _affected_seed_vessels(
    session: EnvironmentalMixedOwnerControlSession,
    source: EnvironmentalMixedOwnerControlResult,
) -> set[str]:
    seed: set[str] = set()
    avoidance = {item.owner.component_id: item for item in session._avoidance_started()}  # noqa: SLF001
    routes = session._route_owners  # noqa: SLF001
    pending_sources = {
        owner_id: vessel_id
        for vessel_id, owner_id in session._pending_source_control_ids.items()  # noqa: SLF001
    }
    for owner_id in source.recapture_owner_ids:
        if owner_id in avoidance:
            seed.update(avoidance[owner_id].owner.vessel_ids)
            continue
        if owner_id in routes:
            seed.update(routes[owner_id].vessel_ids)
            continue
        if owner_id in pending_sources:
            seed.add(pending_sources[owner_id])
            continue
        raise EnvironmentalMixedOwnerRebootstrapError(
            f"a119 recapture owner {owner_id!r} is no longer active in mixed runtime"
        )
    return seed


def _replacement_owners(
    source: EnvironmentalMixedOwnerControlResult,
    seed: set[str],
) -> tuple[FleetCommandOwner, ...]:
    observation = source.recapture_observation
    assert observation is not None
    owners = tuple(
        owner for owner in observation.dispatch.owners if seed.intersection(owner.vessel_ids)
    )
    claimed: set[str] = set()
    for owner in owners:
        overlap = claimed.intersection(owner.vessel_ids)
        if overlap:
            raise EnvironmentalMixedOwnerRebootstrapError(
                f"canonical mixed replacement owners overlap channels: {sorted(overlap)!r}"
            )
        claimed.update(owner.vessel_ids)
    return owners


def _subset_binding(
    binding: EnvironmentalRouteHandoffBinding,
    vessel_ids: tuple[str, ...],
) -> EnvironmentalRouteHandoffBinding:
    wanted = set(vessel_ids)
    vessels = tuple(item for item in binding.vessels if item.vessel_id in wanted)
    if tuple(item.vessel_id for item in vessels) != vessel_ids:
        # Preserve caller order even if the historical binding had another order.
        vessels = tuple(binding.for_vessel(vessel_id) for vessel_id in vessel_ids)
    return EnvironmentalRouteHandoffBinding(
        source_policy=binding.source_policy,
        snapshot=binding.snapshot,
        vessels=vessels,
        profile_catalog_bound=binding.profile_catalog_bound,
    )


def _subset_handoff_result(
    result: EnvironmentalRouteHandoffSessionRunResult,
    vessel_ids: tuple[str, ...],
) -> EnvironmentalRouteHandoffSessionRunResult:
    wanted = set(vessel_ids)
    return EnvironmentalRouteHandoffSessionRunResult(
        snapshot=result.snapshot,
        vessels=tuple(result.by_id(vessel_id) for vessel_id in vessel_ids),
        steps=tuple(step for step in result.steps if step.vessel_id in wanted),
        stop_reason=result.stop_reason,
        overridden_vessel_ids=tuple(v for v in result.overridden_vessel_ids if v in wanted),
    )


def _split_route_owner(
    session: EnvironmentalMixedOwnerControlSession,
    owner: _RouteOwner,
    vessel_ids: tuple[str, ...],
) -> _RouteOwner:
    """Preserve one clear subset of a route owner without resetting route progress."""

    if vessel_ids == owner.vessel_ids:
        return owner
    if not vessel_ids:
        raise ValueError("route-owner subset must not be empty")

    new_id = f"clear:{owner.owner_id}:{'|'.join(vessel_ids)}"
    binding = _subset_binding(owner.binding, vessel_ids)
    handoff = EnvironmentalRouteHandoffSession(
        session.world,
        binding,
        config=owner.handoff_session.config,
        profile_catalog=owner.handoff_session.profile_catalog,
        before_command_write=lambda owner_id=new_id: session._route_write_guard(owner_id),  # noqa: SLF001
        command_transform=owner.handoff_session.command_transform,
    )
    handoff._cursors = {v: copy(owner.handoff_session._cursors[v]) for v in vessel_ids}  # noqa: SLF001
    handoff._steps = [  # noqa: SLF001
        step for step in owner.handoff_session._steps if step.vessel_id in set(vessel_ids)  # noqa: SLF001
    ]
    handoff._started = owner.handoff_session._started  # noqa: SLF001
    handoff._terminal_reason = owner.handoff_session._terminal_reason  # noqa: SLF001
    handoff._overridden_ids = tuple(  # noqa: SLF001
        v for v in owner.handoff_session._overridden_ids if v in set(vessel_ids)  # noqa: SLF001
    )

    nominal = None
    if owner.stage is EnvironmentalMixedRouteStage.NOMINAL_ROUTE:
        if owner.nominal_session is None:
            raise EnvironmentalMixedOwnerRebootstrapError(
                f"route owner {owner.owner_id!r} has NOMINAL_ROUTE stage without nominal session"
            )
        old_nominal = owner.nominal_session
        handoff_result = _subset_handoff_result(old_nominal.handoff_result, vessel_ids)
        nominal = EnvironmentalNominalRouteSession(
            session.world,
            binding,
            handoff_result,
            config=old_nominal.config,
            profile_catalog=old_nominal.profile_catalog,
            before_command_write=lambda owner_id=new_id: session._route_write_guard(owner_id),  # noqa: SLF001
            command_transform=old_nominal.command_transform,
        )
        nominal._cursors = {v: copy(old_nominal._cursors[v]) for v in vessel_ids}  # noqa: SLF001
        nominal._steps = [  # noqa: SLF001
            step for step in old_nominal._steps if step.vessel_id in set(vessel_ids)  # noqa: SLF001
        ]
        nominal._started = old_nominal._started  # noqa: SLF001
        nominal._terminal_reason = old_nominal._terminal_reason  # noqa: SLF001
        nominal._overridden_ids = tuple(  # noqa: SLF001
            v for v in old_nominal._overridden_ids if v in set(vessel_ids)  # noqa: SLF001
        )

    return _RouteOwner(
        owner_id=new_id,
        vessel_ids=vessel_ids,
        binding=binding,
        handoff_session=handoff,
        stage=owner.stage,
        nominal_session=nominal,
        pair_source=None,
        next_fleet_review_time=owner.next_fleet_review_time,
    )


def rebootstrap_environmental_mixed_owner_control(
    session: EnvironmentalMixedOwnerControlSession,
    source: EnvironmentalMixedOwnerControlResult,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    *,
    config: EnvironmentalFleetMultiOwnerRecaptureConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalMixedOwnerRebootstrapResult:
    """Atomically recapture affected mixed-stage channels from A119's committed evidence."""

    if not isinstance(route_targets, Mapping):
        raise TypeError("route_targets must be a mapping")
    if not isinstance(config, EnvironmentalFleetMultiOwnerRecaptureConfig):
        raise TypeError("config must be EnvironmentalFleetMultiOwnerRecaptureConfig")

    _validate_boundary(session, source)
    observation = source.recapture_observation
    assert observation is not None
    world = session.world
    registry = session.registry
    snapshot_before = world.snapshot()

    seed = _affected_seed_vessels(session, source)
    replacement = _replacement_owners(source, seed)
    replacement_ids = {v for owner in replacement for v in owner.vessel_ids}
    current_ids = {v.vessel_id for v in snapshot_before.vessels}

    old_avoidance = session._avoidance_started()  # noqa: SLF001
    recapture_ids = set(source.recapture_owner_ids)
    retained_avoidance = tuple(
        item
        for item in old_avoidance
        if item.owner.component_id not in recapture_ids
        and not replacement_ids.intersection(item.owner.vessel_ids)
    )
    affected_old_avoidance_ids = {
        v
        for item in old_avoidance
        if item not in retained_avoidance
        for v in item.owner.vessel_ids
    }
    released = tuple(sorted((affected_old_avoidance_ids - replacement_ids).intersection(current_ids)))
    removed = tuple(sorted(seed - current_ids))
    released_commands = tuple((vessel_id, world.command_for(vessel_id)) for vessel_id in released)

    # Plan every replacement avoidance owner without mutating world or mixed runtime.
    admitted_targets: dict[str, tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget]] = {}
    for owner in replacement:
        if owner.backend is FleetSupervisorBackend.PAIR:
            try:
                admitted_targets[owner.component_id] = _admit_pair_targets(owner, route_targets)
            except EnvironmentalFleetMultiOwnerRecaptureError as exc:
                raise EnvironmentalMixedOwnerRebootstrapError(str(exc)) from exc

    prepared: list[EnvironmentalPreparedPairOwner | EnvironmentalPreparedJointOwner] = []
    try:
        for owner in replacement:
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
        raise EnvironmentalMixedOwnerRebootstrapError(str(exc)) from exc

    command_batch: list[tuple[str, WaterRelativeManeuverCommand]] = []
    for item in prepared:
        if isinstance(item, EnvironmentalPreparedPairOwner):
            validate_environmental_pair_execution_plan(item.plan, world, profile_catalog=profile_catalog)
            command_batch.extend(_pair_commands(item.plan))
        else:
            validate_environmental_joint_execution_plan(item.plan, world, profile_catalog=profile_catalog)
            command_batch.extend(item.plan.selected_commands)

    command_ids = [v for v, _ in command_batch]
    if len(command_ids) != len(set(command_ids)):
        raise EnvironmentalMixedOwnerRebootstrapError(
            "mixed replacement transaction contains duplicate command channels"
        )
    if set(command_ids) != replacement_ids:
        raise EnvironmentalMixedOwnerRebootstrapError(
            "mixed replacement transaction does not cover every replacement avoidance channel"
        )

    pending = world.next_command_time()
    if pending is not None and pending <= world.time + _EPS:
        raise PendingEnvironmentalMultiOwnerCommandEvent(
            "cannot rebootstrap mixed owners while a command event is pending at current time"
        )

    commands_before = tuple((v, world.command_for(v)) for v in command_ids)
    if command_batch:
        applied_at = world.time
        world.schedule_commands_atomic(at=applied_at, commands=tuple(command_batch))
        world.run_until(applied_at)
    snapshot_after = world.snapshot()
    commands_after = tuple((v, world.command_for(v)) for v in command_ids)
    if snapshot_after != snapshot_before or abs(world.time - observation.time_min) > _EPS:
        raise EnvironmentalMixedOwnerRebootstrapError(
            "same-timestamp mixed recapture changed physical state or world time"
        )
    if commands_after != tuple(command_batch):
        raise EnvironmentalMixedOwnerRebootstrapError(
            "same-timestamp mixed recapture did not publish the complete avoidance command batch"
        )

    before_map = dict(commands_before)
    after_map = dict(commands_after)
    started_new: list[EnvironmentalStartedOwner] = []
    for item in prepared:
        if isinstance(item, EnvironmentalPreparedPairOwner):
            started_new.append(
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
            started_new.append(
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

    # Build the post-transaction route owner set only after replacement planning/write
    # succeeded.  Clear subsets retain their stage/cursors instead of restarting routes.
    new_route_owners: dict[str, _RouteOwner] = {}
    retired_route_ids: list[str] = []
    preserved_route_vessels: list[str] = []
    for owner_id, route_owner in session._route_owners.items():  # noqa: SLF001
        if route_owner.stage is EnvironmentalMixedRouteStage.COMPLETE:
            new_route_owners[owner_id] = route_owner
            continue
        clear_ids = tuple(
            v for v in route_owner.vessel_ids if v in current_ids and v not in replacement_ids
        )
        if clear_ids == route_owner.vessel_ids:
            new_route_owners[owner_id] = route_owner
            preserved_route_vessels.extend(clear_ids)
            continue
        retired_route_ids.append(owner_id)
        if clear_ids:
            split = _split_route_owner(session, route_owner, clear_ids)
            new_route_owners[split.owner_id] = split
            preserved_route_vessels.extend(clear_ids)

    active_avoidance = retained_avoidance + tuple(started_new)
    session.coordinator._adopt_replacement_owner_set(  # noqa: SLF001 - A120 owns mixed runtime adoption
        active_avoidance,
        committed_observation=observation,
    )
    session._route_owners = new_route_owners  # noqa: SLF001
    session._managed_started_owner_ids = {  # noqa: SLF001
        item.owner.component_id for item in active_avoidance
    }
    session._retired_joint_owner_ids.clear()  # noqa: SLF001 - historical JOINT route wrappers are now explicit route owners
    session._current_write_observation = None  # noqa: SLF001
    session._validate_disjoint_channels()  # noqa: SLF001

    active_ids = tuple(item.owner.component_id for item in active_avoidance)
    preserved_route_ids = tuple(
        owner.owner_id
        for owner in new_route_owners.values()
        if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
    )
    action = (
        EnvironmentalMixedOwnerRebootstrapAction.MIXED_OWNER_SET_REPLACED
        if replacement
        else EnvironmentalMixedOwnerRebootstrapAction.NO_ACTIVE_REPLACEMENT_AVOIDANCE_OWNER
    )
    return EnvironmentalMixedOwnerRebootstrapResult(
        source=source,
        action=action,
        committed_observation_time=observation.time_min,
        replacement_owners=replacement,
        retained_avoidance_owner_ids=tuple(item.owner.component_id for item in retained_avoidance),
        started_replacement_owners=tuple(started_new),
        active_avoidance_owner_ids=active_ids,
        retired_route_owner_ids=tuple(retired_route_ids),
        preserved_route_owner_ids=preserved_route_ids,
        preserved_route_vessel_ids=tuple(preserved_route_vessels),
        released_vessel_ids=released,
        released_commands=released_commands,
        removed_vessel_ids=removed,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        commands_before=commands_before,
        commands_after=commands_after,
    )
