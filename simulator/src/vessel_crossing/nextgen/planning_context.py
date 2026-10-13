"""Snapshot-bound planning inputs for one fleet command owner.

The avoidance evaluators deliberately accept explicit state/limit/domain inputs.  This
module provides one narrow adapter from a live ``World2D`` + supervision dispatch into
those inputs without changing replanner signatures or introducing hidden global model
state.  A context is valid only for the exact world snapshot and relevant profile
bindings from which it was built.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dynamics import ManeuverLimits, Vessel2DState
from .encounter_registry import PairKey
from .fleet_supervision import (
    FleetCommandOwner,
    FleetSupervisionDispatch,
    FleetSupervisorBackend,
)
from .ship_domain import ShipDomain
from .vessel_classes import VesselClassProfile, VesselProfileCatalog
from .world import World2D, WorldSnapshot


class PlanningContextError(RuntimeError):
    """Base error for invalid or stale avoidance planning context usage."""


class PlanningContextOwnerMismatch(PlanningContextError):
    """Raised when dispatch/owner/world membership does not describe one component."""


class StalePlanningContext(PlanningContextError):
    """Raised when world state, limits, or relevant profile bindings changed."""


@dataclass(frozen=True, slots=True)
class VesselPlanningInputs:
    """Captured model inputs for one vessel at a planning boundary."""

    vessel_id: str
    state: Vessel2DState
    maneuver_limits: ManeuverLimits
    profile: VesselClassProfile | None
    domain: ShipDomain | None

    def __post_init__(self) -> None:
        if self.state.vessel_id != self.vessel_id:
            raise ValueError("state.vessel_id does not match vessel_id")
        expected_domain = None if self.profile is None else self.profile.safety_domain
        if self.domain != expected_domain:
            raise ValueError("domain must match the captured profile safety_domain")


@dataclass(frozen=True, slots=True)
class AvoidancePlanningContext:
    """Immutable owner-scoped planning evidence bound to one exact world snapshot."""

    planning_snapshot: WorldSnapshot
    component_id: str
    backend: FleetSupervisorBackend
    vessel_ids: tuple[str, ...]
    pair_keys: tuple[PairKey, ...]
    vessels: tuple[VesselPlanningInputs, ...]
    profile_catalog_bound: bool

    def __post_init__(self) -> None:
        if tuple(v.vessel_id for v in self.vessels) != self.vessel_ids:
            raise ValueError("vessels must exactly follow vessel_ids order")
        if len(set(self.vessel_ids)) != len(self.vessel_ids):
            raise ValueError("duplicate vessel_id in planning context")
        if self.component_id != "|".join(self.vessel_ids):
            raise ValueError("component_id does not match vessel_ids")

    @property
    def time_min(self) -> float:
        return self.planning_snapshot.time

    @property
    def states(self) -> dict[str, Vessel2DState]:
        return {item.vessel_id: item.state for item in self.vessels}

    @property
    def limits(self) -> dict[str, ManeuverLimits]:
        return {item.vessel_id: item.maneuver_limits for item in self.vessels}

    @property
    def profiles(self) -> dict[str, VesselClassProfile | None]:
        return {item.vessel_id: item.profile for item in self.vessels}

    @property
    def domains(self) -> dict[str, ShipDomain | None]:
        return {item.vessel_id: item.domain for item in self.vessels}

    @property
    def profile_complete(self) -> bool:
        return all(item.profile is not None for item in self.vessels)

    @property
    def domain_complete(self) -> bool:
        return all(item.domain is not None for item in self.vessels)

    def for_vessel(self, vessel_id: str) -> VesselPlanningInputs:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)

    def validate(
        self,
        world: World2D,
        *,
        profile_catalog: VesselProfileCatalog | None = None,
    ) -> None:
        """Fail closed unless all captured owner-relevant planning evidence is current."""

        current_snapshot = world.snapshot()
        if current_snapshot != self.planning_snapshot:
            raise StalePlanningContext("world snapshot changed after planning context creation")

        if self.profile_catalog_bound != (profile_catalog is not None):
            raise StalePlanningContext("profile catalog binding mode changed")

        for item in self.vessels:
            try:
                current_limits = world.limits_for(item.vessel_id)
            except KeyError as exc:
                raise StalePlanningContext(
                    f"planning vessel {item.vessel_id!r} is no longer active"
                ) from exc
            if current_limits != item.maneuver_limits:
                raise StalePlanningContext(
                    f"maneuver limits changed for planning vessel {item.vessel_id!r}"
                )

            current_profile = (
                None if profile_catalog is None else profile_catalog.profile_for(item.vessel_id)
            )
            if current_profile != item.profile:
                raise StalePlanningContext(
                    f"profile binding changed for planning vessel {item.vessel_id!r}"
                )


def build_avoidance_planning_context(
    world: World2D,
    dispatch: FleetSupervisionDispatch,
    owner: FleetCommandOwner,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> AvoidancePlanningContext:
    """Capture current planning evidence for exactly one dispatched command owner."""

    if dispatch.time_min != world.time:
        raise PlanningContextOwnerMismatch(
            "dispatch timestamp does not match the current world timestamp"
        )
    if owner not in dispatch.owners:
        raise PlanningContextOwnerMismatch("owner is not part of the supplied dispatch")

    snapshot = world.snapshot()
    snapshot_ids = {state.vessel_id for state in snapshot.vessels}
    missing = set(owner.vessel_ids) - snapshot_ids
    if missing:
        raise PlanningContextOwnerMismatch(
            f"owner references vessels absent from world snapshot: {sorted(missing)}"
        )

    inputs: list[VesselPlanningInputs] = []
    for vessel_id in owner.vessel_ids:
        profile = None if profile_catalog is None else profile_catalog.profile_for(vessel_id)
        maneuver_limits = world.limits_for(vessel_id)
        if profile is not None and profile.maneuver_limits != maneuver_limits:
            raise PlanningContextOwnerMismatch(
                f"profile maneuver limits disagree with World2D for {vessel_id!r}"
            )
        state = snapshot.by_id(vessel_id)
        if (
            profile is not None
            and profile.speed_envelope is not None
            and not profile.speed_envelope.permits(state.speed)
        ):
            raise PlanningContextOwnerMismatch(
                f"world speed exceeds class speed envelope for {vessel_id!r}"
            )
        inputs.append(
            VesselPlanningInputs(
                vessel_id=vessel_id,
                state=state,
                maneuver_limits=maneuver_limits,
                profile=profile,
                domain=None if profile is None else profile.safety_domain,
            )
        )

    return AvoidancePlanningContext(
        planning_snapshot=snapshot,
        component_id=owner.component_id,
        backend=owner.backend,
        vessel_ids=owner.vessel_ids,
        pair_keys=owner.pair_keys,
        vessels=tuple(inputs),
        profile_catalog_bound=profile_catalog is not None,
    )


def bind_planning_context_replanner(
    world: World2D,
    replanner,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
):
    """Adapt a context-aware replanner to the existing fleet-controller signature.

    The returned callable still accepts ``(snapshot, registry_update, owner)``.  Before
    delegating, it derives the dispatch from that exact update and captures an
    ``AvoidancePlanningContext``.  The context is validated again after the callback so
    same-time world/profile mutation during planning cannot silently leak into plan
    application.
    """

    if not callable(replanner):
        raise ValueError("replanner must be callable")

    # Local import avoids coupling the core context dataclasses to review-session types.
    from .fleet_supervision import build_fleet_supervision_dispatch

    def adapted(snapshot, update, owner):
        if world.snapshot() != snapshot:
            raise StalePlanningContext(
                "shared planning snapshot does not match the current World2D snapshot"
            )
        dispatch = build_fleet_supervision_dispatch(update)
        context = build_avoidance_planning_context(
            world,
            dispatch,
            owner,
            profile_catalog=profile_catalog,
        )
        if context.planning_snapshot != snapshot:
            raise StalePlanningContext("planning context snapshot differs from controller snapshot")
        result = replanner(context, update, owner)
        context.validate(world, profile_catalog=profile_catalog)
        return result

    return adapted
