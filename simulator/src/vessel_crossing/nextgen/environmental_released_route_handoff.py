"""Route-handoff admission for vessels released by canonical fleet ownership.

A119 needs a route binding even when a vessel did not arrive through the pair-local
A95 handoff path (for example a JOINT component clears, or an A118 recapture leaves a
surviving vessel outside every avoidance owner).  The existing A98/A99 executors are
already vessel-iterative, so this module supplies the missing admission contract:

- one already-committed environmental fleet observation is the ownership evidence;
- every admitted vessel must be untouched by every active PAIR/JOINT dispatch owner;
- the exact physical snapshot and current water-relative command tuple are stale-bound;
- route targets, drift, maneuver limits and optional vessel profiles are captured before
  the first route command can be written.

No registry observation, world advancement or command mutation occurs here.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .environmental_dynamics import WaterRelativeManeuverCommand
from .environmental_fleet_ownership import EnvironmentalFleetOwnershipObservation
from .environmental_route_handoff import (
    EnvironmentalRouteHandoffBinding,
    EnvironmentalRouteHandoffError,
    EnvironmentalRouteHandoffTarget,
    EnvironmentalRouteHandoffVesselBinding,
    StaleEnvironmentalRouteHandoff,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalReleasedRouteHandoffError(EnvironmentalRouteHandoffError):
    """Base error for A119 fleet-released route admission."""


class EnvironmentalReleasedRouteOwnershipConflict(EnvironmentalReleasedRouteHandoffError):
    """Raised when canonical dispatch still owns an admitted route vessel."""


def build_environmental_released_route_handoff_binding(
    world: EnvironmentalWorld2D,
    observation: EnvironmentalFleetOwnershipObservation,
    source_snapshot: EnvironmentalWorldSnapshot,
    targets: Iterable[EnvironmentalRouteHandoffTarget],
    expected_commands: Mapping[str, WaterRelativeManeuverCommand],
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalRouteHandoffBinding:
    """Bind fleet-released vessel channels to the existing A98/A99 route executors."""

    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(observation, EnvironmentalFleetOwnershipObservation):
        raise TypeError("observation must be EnvironmentalFleetOwnershipObservation")
    if not isinstance(source_snapshot, EnvironmentalWorldSnapshot):
        raise TypeError("source_snapshot must be EnvironmentalWorldSnapshot")
    if not isinstance(expected_commands, Mapping):
        raise TypeError("expected_commands must be a mapping")
    if abs(observation.time_min - world.time) > _EPS:
        raise StaleEnvironmentalRouteHandoff(
            "committed fleet observation timestamp no longer matches environmental world"
        )
    if source_snapshot != world.snapshot() or abs(source_snapshot.time - world.time) > _EPS:
        raise StaleEnvironmentalRouteHandoff(
            "environmental world no longer matches the released-route source snapshot"
        )

    items = tuple(targets)
    if not items:
        raise ValueError("released route handoff requires at least one route target")
    if any(not isinstance(item, EnvironmentalRouteHandoffTarget) for item in items):
        raise TypeError("targets must contain EnvironmentalRouteHandoffTarget objects")
    ids = tuple(item.vessel_id for item in items)
    if len(ids) != len(set(ids)):
        raise ValueError("released route handoff target ids must be unique")
    if set(expected_commands) != set(ids):
        raise EnvironmentalReleasedRouteHandoffError(
            "expected command map must cover every released route vessel exactly once"
        )

    touching = tuple(
        owner
        for owner in observation.dispatch.owners
        if set(owner.vessel_ids).intersection(ids)
    )
    if touching:
        raise EnvironmentalReleasedRouteOwnershipConflict(
            "released route admission requires no canonical avoidance owner touching its vessels"
        )

    snapshot = world.snapshot()
    bindings: list[EnvironmentalRouteHandoffVesselBinding] = []
    for target in items:
        vessel_id = target.vessel_id
        try:
            state = snapshot.by_id(vessel_id)
            drift = world.drift_for(vessel_id)
            limits = world.limits_for(vessel_id)
            command = world.command_for(vessel_id)
        except KeyError as exc:
            raise StaleEnvironmentalRouteHandoff(
                f"released route vessel {vessel_id!r} is no longer active"
            ) from exc
        if command != expected_commands[vessel_id]:
            raise StaleEnvironmentalRouteHandoff(
                f"active command changed for released route vessel {vessel_id!r}"
            )
        profile = None if profile_catalog is None else profile_catalog.profile_for(vessel_id)
        if profile is not None and profile.maneuver_limits != limits:
            raise EnvironmentalReleasedRouteHandoffError(
                f"profile maneuver limits disagree with world for {vessel_id!r}"
            )
        bindings.append(
            EnvironmentalRouteHandoffVesselBinding(
                target=target,
                state=state,
                environmental_drift_velocity=drift,
                maneuver_limits=limits,
                command_before=command,
                profile=profile,
            )
        )

    return EnvironmentalRouteHandoffBinding(
        source_policy=None,
        snapshot=snapshot,
        vessels=tuple(bindings),
        profile_catalog_bound=profile_catalog is not None,
    )
