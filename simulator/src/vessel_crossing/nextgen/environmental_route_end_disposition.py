"""Fleet-safe environmental route-end disposition from one committed A119 boundary.

A121 closes the command-authority gap at environmental route completion.  A119 now
stops as soon as one or more nominal-route channels cross their finish line and exposes
the exact canonical fleet observation plus the exact pre-disposition command tuple.
A121 consumes that evidence without observing the registry again.

Disposition is explicit per route target:

- ``CONTINUE`` relinquishes route authority while leaving the current water-relative
  command unchanged;
- ``STOP`` relinquishes route authority and atomically replaces the current command by
  a zero through-water-speed command at the same target heading (the vessel may still
  drift over ground under environmental current);
- ``REMOVE`` relinquishes route authority and removes the vessel from the physical
  environmental world at the same timestamp.

Every completed channel is preflighted before the first mutation.  STOP command writes
and REMOVE operations are then committed through one policy-free current-time world
transaction.  A multi-vessel route owner is split when only a subset has completed, so
unfinished route channels retain their existing cursor/session progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .environmental_dynamics import WaterRelativeManeuverCommand, WaterRelativeVesselState
from .environmental_mixed_owner_control import (
    EnvironmentalMixedOwnerControlResult,
    EnvironmentalMixedOwnerControlSession,
    EnvironmentalMixedOwnerStopReason,
    EnvironmentalMixedRouteStage,
)
from .environmental_mixed_owner_rebootstrap import _split_route_owner
from .environmental_nominal_route_session import (
    EnvironmentalNominalRouteMode,
    EnvironmentalNominalRouteStopReason,
)
from .environmental_route_handoff_session import EnvironmentalRouteHandoffSessionMode
from .environmental_pair_control_cycle import (
    EnvironmentalPairControlStage,
    EnvironmentalPairControlStopReason,
)
from .environmental_world import EnvironmentalWorldSnapshot
from .route_exit import RouteEndAction

_EPS = 1e-10


class EnvironmentalRouteEndDispositionError(RuntimeError):
    """Base error for invalid A121 route-end disposition."""


class StaleEnvironmentalRouteEndDisposition(EnvironmentalRouteEndDispositionError):
    """Raised when A119 completion evidence no longer matches the live world/runtime."""


class EnvironmentalRouteEndDispositionAction(str, Enum):
    DISPOSITIONS_APPLIED = "dispositions_applied"


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteEndVesselDisposition:
    owner_id: str
    vessel_id: str
    action: RouteEndAction
    endpoint_state: WaterRelativeVesselState
    command_before: WaterRelativeManeuverCommand
    command_after: WaterRelativeManeuverCommand | None
    removed: bool


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteEndDispositionResult:
    source: EnvironmentalMixedOwnerControlResult
    action: EnvironmentalRouteEndDispositionAction
    committed_observation_time: float
    dispositions: tuple[EnvironmentalRouteEndVesselDisposition, ...]
    retired_route_owner_ids: tuple[str, ...]
    preserved_route_owner_ids: tuple[str, ...]
    active_route_owner_ids: tuple[str, ...]
    continued_vessel_ids: tuple[str, ...]
    stopped_vessel_ids: tuple[str, ...]
    removed_vessel_ids: tuple[str, ...]
    snapshot_before: EnvironmentalWorldSnapshot
    snapshot_after: EnvironmentalWorldSnapshot


def _validate_boundary(
    session: EnvironmentalMixedOwnerControlSession,
    source: EnvironmentalMixedOwnerControlResult,
) -> None:
    if not isinstance(session, EnvironmentalMixedOwnerControlSession):
        raise TypeError("session must be EnvironmentalMixedOwnerControlSession")
    if not isinstance(source, EnvironmentalMixedOwnerControlResult):
        raise TypeError("source must be EnvironmentalMixedOwnerControlResult")
    if source.stop_reason is not EnvironmentalMixedOwnerStopReason.ROUTE_DISPOSITION_REQUIRED:
        raise EnvironmentalRouteEndDispositionError(
            "a121 requires an a119 ROUTE_DISPOSITION_REQUIRED boundary"
        )
    observation = source.route_disposition_observation
    if observation is None or not source.route_completions:
        raise EnvironmentalRouteEndDispositionError(
            "a119 route-disposition result lacks committed completion evidence"
        )
    if source.snapshot != session.world.snapshot():
        raise StaleEnvironmentalRouteEndDisposition(
            "world no longer matches the a119 route-completion boundary"
        )
    if abs(observation.time_min - session.world.time) > _EPS:
        raise StaleEnvironmentalRouteEndDisposition(
            "a119 route-disposition observation timestamp no longer matches world"
        )
    if source.boundaries != session.boundaries:
        raise EnvironmentalRouteEndDispositionError(
            "a119 route-disposition result does not belong to this mixed-session boundary history"
        )
    if source.fleet_observations != session.fleet_observations:
        raise EnvironmentalRouteEndDispositionError(
            "a119 route-disposition result does not match current shared fleet evidence"
        )
    if not session.fleet_observations or session.fleet_observations[-1] != observation:
        raise EnvironmentalRouteEndDispositionError(
            "a119 route-disposition observation is not the latest canonical shared observation"
        )


def apply_environmental_route_end_dispositions(
    session: EnvironmentalMixedOwnerControlSession,
    source: EnvironmentalMixedOwnerControlResult,
) -> EnvironmentalRouteEndDispositionResult:
    """Apply all A119 route-end actions from its already-committed fleet evidence."""

    _validate_boundary(session, source)
    world = session.world
    observation = source.route_disposition_observation
    assert observation is not None
    snapshot_before = world.snapshot()

    expected_by_id: dict[str, WaterRelativeManeuverCommand] = {}
    owner_completed: dict[str, tuple[str, ...]] = {}
    owner_order: list[str] = []
    for completion in source.route_completions:
        if completion.owner_id in owner_completed:
            raise EnvironmentalRouteEndDispositionError(
                f"duplicate route completion owner {completion.owner_id!r}"
            )
        owner = session._route_owners.get(completion.owner_id)  # noqa: SLF001
        if owner is None or owner.stage not in {
            EnvironmentalMixedRouteStage.ROUTE_HANDOFF,
            EnvironmentalMixedRouteStage.NOMINAL_ROUTE,
        }:
            raise EnvironmentalRouteEndDispositionError(
                f"route owner {completion.owner_id!r} is no longer in active route ownership"
            )
        if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF:
            if not owner.handoff_session._started:  # noqa: SLF001
                raise EnvironmentalRouteEndDispositionError(
                    f"route owner {completion.owner_id!r} lacks an active handoff session"
                )
        elif owner.nominal_session is None or not owner.nominal_session._started:  # noqa: SLF001
            raise EnvironmentalRouteEndDispositionError(
                f"route owner {completion.owner_id!r} lacks an active nominal session"
            )
        if not completion.vessel_ids:
            raise EnvironmentalRouteEndDispositionError("route completion vessel set must be non-empty")
        if not set(completion.vessel_ids).issubset(owner.vessel_ids):
            raise EnvironmentalRouteEndDispositionError(
                f"route completion vessels do not belong to owner {completion.owner_id!r}"
            )
        if tuple(v for v, _ in completion.expected_commands) != completion.vessel_ids:
            raise EnvironmentalRouteEndDispositionError(
                f"route completion command tuple disagrees with vessel order for {completion.owner_id!r}"
            )
        for vessel_id, command in completion.expected_commands:
            if vessel_id in expected_by_id:
                raise EnvironmentalRouteEndDispositionError(
                    f"route completion duplicates command channel {vessel_id!r}"
                )
            if owner.stage is EnvironmentalMixedRouteStage.ROUTE_HANDOFF:
                cursor = owner.handoff_session._cursors[vessel_id]  # noqa: SLF001
                complete_mode = cursor.mode is EnvironmentalRouteHandoffSessionMode.ROUTE_COMPLETE
            else:
                assert owner.nominal_session is not None
                cursor = owner.nominal_session._cursors[vessel_id]  # noqa: SLF001
                complete_mode = cursor.mode is EnvironmentalNominalRouteMode.ROUTE_COMPLETE
            if not complete_mode or not cursor.follower_state.completed:
                raise EnvironmentalRouteEndDispositionError(
                    f"route vessel {vessel_id!r} is not complete at the a119 disposition boundary"
                )
            expected_by_id[vessel_id] = command
        owner_completed[completion.owner_id] = completion.vessel_ids
        owner_order.append(completion.owner_id)

    completed_ids = tuple(expected_by_id)
    touching = tuple(
        owner
        for owner in observation.dispatch.owners
        if set(owner.vessel_ids).intersection(completed_ids)
    )
    if touching:
        raise EnvironmentalRouteEndDispositionError(
            "canonical fleet dispatch owns a route-complete channel; recapture must precede disposition"
        )

    pending = world.next_command_time()
    if pending is not None and pending <= world.time + _EPS:
        raise StaleEnvironmentalRouteEndDisposition(
            "pending same-timestamp command event exists before route-end disposition"
        )

    plans: list[
        tuple[
            str,
            str,
            RouteEndAction,
            WaterRelativeVesselState,
            WaterRelativeManeuverCommand,
            WaterRelativeManeuverCommand | None,
        ]
    ] = []
    stop_commands: list[tuple[str, WaterRelativeManeuverCommand]] = []
    removals: list[str] = []
    for owner_id in owner_order:
        owner = session._route_owners[owner_id]  # noqa: SLF001
        for vessel_id in owner_completed[owner_id]:
            try:
                state = snapshot_before.by_id(vessel_id)
                current = world.command_for(vessel_id)
            except KeyError as exc:
                raise StaleEnvironmentalRouteEndDisposition(
                    f"route-complete vessel {vessel_id!r} is no longer active"
                ) from exc
            expected = expected_by_id[vessel_id]
            if current != expected:
                raise StaleEnvironmentalRouteEndDisposition(
                    f"active command changed for route-complete vessel {vessel_id!r}"
                )
            action = owner.binding.for_vessel(vessel_id).target.end_action
            if not isinstance(action, RouteEndAction):
                raise TypeError("route target end_action must be RouteEndAction")
            command_after: WaterRelativeManeuverCommand | None
            if action is RouteEndAction.CONTINUE:
                command_after = current
            elif action is RouteEndAction.STOP:
                command_after = WaterRelativeManeuverCommand(
                    current.target_through_water_heading_deg,
                    0.0,
                )
                stop_commands.append((vessel_id, command_after))
            elif action is RouteEndAction.REMOVE:
                command_after = None
                removals.append(vessel_id)
            else:  # pragma: no cover - enum exhaustiveness
                raise AssertionError(action)
            plans.append((owner_id, vessel_id, action, state, current, command_after))

    # One policy-free physical transaction: all stale/ownership/action checks above must
    # succeed before either the first STOP command or the first REMOVE becomes visible.
    world.apply_current_transaction_atomic(
        commands=tuple(stop_commands),
        removals=tuple(removals),
    )

    # Change route ownership only after the world transaction succeeded.  Incomplete
    # siblings keep their exact route cursor/session progress under a split owner.
    retired_owner_ids: list[str] = []
    preserved_owner_ids: list[str] = []
    for owner_id in owner_order:
        owner = session._route_owners[owner_id]  # noqa: SLF001
        completed = set(owner_completed[owner_id])
        remaining = tuple(v for v in owner.vessel_ids if v not in completed)
        if remaining:
            split = _split_route_owner(session, owner, remaining)
            session._route_owners.pop(owner_id)  # noqa: SLF001
            session._route_owners[split.owner_id] = split  # noqa: SLF001
            retired_owner_ids.append(owner_id)
            preserved_owner_ids.append(split.owner_id)
            if owner.pair_source is not None:
                session._managed_started_owner_ids.discard(  # noqa: SLF001
                    owner.pair_source.owner.component_id
                )
        else:
            owner.stage = EnvironmentalMixedRouteStage.COMPLETE
            if owner_id not in session._completed_route_owner_ids:  # noqa: SLF001
                session._completed_route_owner_ids.append(owner_id)  # noqa: SLF001
            retired_owner_ids.append(owner_id)
            if owner.pair_source is not None:
                cycle = owner.pair_source.pair_cycle
                cycle._stage = EnvironmentalPairControlStage.COMPLETE  # noqa: SLF001
                cycle._terminal_reason = EnvironmentalPairControlStopReason.ROUTE_COMPLETE  # noqa: SLF001
                if owner.nominal_session is not None:
                    cycle._nominal_result = owner.nominal_session.result(  # noqa: SLF001
                        EnvironmentalNominalRouteStopReason.ROUTE_COMPLETE
                    )
                session._managed_started_owner_ids.discard(  # noqa: SLF001
                    owner.pair_source.owner.component_id
                )

    session._current_write_observation = None  # noqa: SLF001
    session._validate_disjoint_channels()  # noqa: SLF001

    dispositions = tuple(
        EnvironmentalRouteEndVesselDisposition(
            owner_id=owner_id,
            vessel_id=vessel_id,
            action=action,
            endpoint_state=state,
            command_before=command_before,
            command_after=command_after,
            removed=(action is RouteEndAction.REMOVE),
        )
        for owner_id, vessel_id, action, state, command_before, command_after in plans
    )
    active_route_ids = tuple(
        owner.owner_id
        for owner in session._route_owners.values()  # noqa: SLF001
        if owner.stage is not EnvironmentalMixedRouteStage.COMPLETE
    )
    return EnvironmentalRouteEndDispositionResult(
        source=source,
        action=EnvironmentalRouteEndDispositionAction.DISPOSITIONS_APPLIED,
        committed_observation_time=observation.time_min,
        dispositions=dispositions,
        retired_route_owner_ids=tuple(retired_owner_ids),
        preserved_route_owner_ids=tuple(preserved_owner_ids),
        active_route_owner_ids=active_route_ids,
        continued_vessel_ids=tuple(
            item.vessel_id for item in dispositions if item.action is RouteEndAction.CONTINUE
        ),
        stopped_vessel_ids=tuple(
            item.vessel_id for item in dispositions if item.action is RouteEndAction.STOP
        ),
        removed_vessel_ids=tuple(
            item.vessel_id for item in dispositions if item.action is RouteEndAction.REMOVE
        ),
        snapshot_before=snapshot_before,
        snapshot_after=world.snapshot(),
    )
