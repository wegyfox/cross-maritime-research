"""Snapshot-bound planning evidence for ``EnvironmentalWorld2D`` command owners.

A83 establishes the physical evidence boundary required before environmental encounter
intents may enter avoidance candidate physics.  Unlike the historical planning context,
this module never coerces water-relative state plus current into ``Vessel2DState``:

- hull / controlled heading and speed remain water-relative state;
- environmental drift is explicit;
- ground velocity / COG are captured as derived motion evidence;
- maneuver limits, class profile and ship-domain bindings are snapshotted separately.

No candidate command is generated and no command ownership is executed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .environmental_candidate_transition import (
    EnvironmentalMotionBackend,
    EnvironmentalMotionState,
    validate_environmental_motion_state,
)
from .environmental_dynamics import WaterRelativeVesselState
from .environmental_evidence import (
    EnvironmentalVesselMotionEvidence,
    environmental_vessel_motion_evidence,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .encounter_registry import PairKey
from .fleet_supervision import FleetCommandOwner, FleetSupervisionDispatch, FleetSupervisorBackend
from .planning_context import PlanningContextOwnerMismatch, StalePlanningContext
from .ship_domain import ShipDomain
from .same_route_following import SameRouteFollowingConstraint
from .vessel_classes import VesselClassProfile, VesselProfileCatalog

if TYPE_CHECKING:
    from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
    from .same_route_following import SameRouteFollowingSupervisor


@dataclass(frozen=True, slots=True)
class EnvironmentalVesselPlanningInputs:
    """Captured environmental planning evidence for one owned vessel."""

    vessel_id: str
    state: WaterRelativeVesselState
    environmental_drift_velocity: Vector2
    motion: EnvironmentalVesselMotionEvidence
    maneuver_limits: ManeuverLimits
    profile: VesselClassProfile | None
    domain: ShipDomain | None
    motion_state: EnvironmentalMotionState | None = None

    def __post_init__(self) -> None:
        if self.state.vessel_id != self.vessel_id:
            raise ValueError("state.vessel_id does not match vessel_id")
        if self.motion.vessel_id != self.vessel_id:
            raise ValueError("motion.vessel_id does not match vessel_id")
        if self.motion.position != self.state.position:
            raise ValueError("motion position does not match captured state")
        if self.motion.hull_heading_deg != self.state.through_water_heading_deg:
            raise ValueError("motion hull heading does not match captured water-relative state")
        if self.motion.environmental_drift_velocity != self.environmental_drift_velocity:
            raise ValueError("motion drift does not match captured environmental drift")
        expected_domain = None if self.profile is None else self.profile.safety_domain
        if self.domain != expected_domain:
            raise ValueError("domain must match the captured profile safety_domain")


@dataclass(frozen=True, slots=True)
class EnvironmentalAvoidancePlanningContext:
    """Immutable owner-scoped evidence bound to one environmental world snapshot."""

    planning_snapshot: EnvironmentalWorldSnapshot
    component_id: str
    backend: FleetSupervisorBackend
    vessel_ids: tuple[str, ...]
    pair_keys: tuple[PairKey, ...]
    vessels: tuple[EnvironmentalVesselPlanningInputs, ...]
    profile_catalog_bound: bool
    same_route_following_constraints: tuple[SameRouteFollowingConstraint, ...] = ()
    delegated_same_route_pair_keys: tuple[PairKey, ...] = ()
    motion_backend: EnvironmentalMotionBackend | None = None

    def __post_init__(self) -> None:
        if tuple(item.vessel_id for item in self.vessels) != self.vessel_ids:
            raise ValueError("vessels must exactly follow vessel_ids order")
        if len(set(self.vessel_ids)) != len(self.vessel_ids):
            raise ValueError("duplicate vessel_id in planning context")
        if self.component_id != "|".join(self.vessel_ids):
            raise ValueError("component_id does not match vessel_ids")
        follower_ids: set[str] = set()
        for item in self.same_route_following_constraints:
            if not isinstance(item, SameRouteFollowingConstraint):
                raise TypeError("same_route_following_constraints must contain SameRouteFollowingConstraint")
            if item.follower_id not in self.vessel_ids:
                raise ValueError("following constraint follower must belong to planning owner")
            if item.follower_id in follower_ids:
                raise ValueError("at most one nearest-leader following constraint is allowed per owner vessel")
            follower_ids.add(item.follower_id)
        if tuple(sorted(set(self.delegated_same_route_pair_keys))) != self.delegated_same_route_pair_keys:
            raise ValueError("delegated_same_route_pair_keys must be unique and sorted")
        for key in self.delegated_same_route_pair_keys:
            if key != tuple(sorted(key)) or not set(key).issubset(set(self.vessel_ids)):
                raise ValueError("delegated same-route pair must be canonical and inside owner")
        if self.motion_backend is not None and not isinstance(
            self.motion_backend, EnvironmentalMotionBackend
        ):
            raise TypeError("motion_backend must implement EnvironmentalMotionBackend")
        if self.motion_backend is None:
            if any(item.motion_state is not None for item in self.vessels):
                raise ValueError("motion_state requires an explicit motion_backend")
        else:
            for item in self.vessels:
                validate_environmental_motion_state(
                    self.motion_backend,
                    item.motion_state,
                    boundary=f"planning context {item.vessel_id!r}",
                )

    @property
    def time_min(self) -> float:
        return self.planning_snapshot.time

    @property
    def states(self) -> dict[str, WaterRelativeVesselState]:
        return {item.vessel_id: item.state for item in self.vessels}

    @property
    def drifts(self) -> dict[str, Vector2]:
        return {item.vessel_id: item.environmental_drift_velocity for item in self.vessels}

    @property
    def motion_evidence(self) -> dict[str, EnvironmentalVesselMotionEvidence]:
        return {item.vessel_id: item.motion for item in self.vessels}

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

    def for_vessel(self, vessel_id: str) -> EnvironmentalVesselPlanningInputs:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)

    def following_constraint_for(self, vessel_id: str) -> SameRouteFollowingConstraint | None:
        for item in self.same_route_following_constraints:
            if item.follower_id == vessel_id:
                return item
        return None

    def same_route_pair_delegated(self, first_id: str, second_id: str) -> bool:
        key = tuple(sorted((first_id, second_id)))
        return key in self.delegated_same_route_pair_keys

    def validate(
        self,
        world: EnvironmentalWorld2D,
        *,
        profile_catalog: VesselProfileCatalog | None = None,
    ) -> None:
        """Fail closed unless all owner-relevant environmental evidence is unchanged."""

        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if world.snapshot() != self.planning_snapshot:
            raise StalePlanningContext(
                "environmental world snapshot changed after planning context creation"
            )
        if self.profile_catalog_bound != (profile_catalog is not None):
            raise StalePlanningContext("profile catalog binding mode changed")
        if world.motion_backend is not self.motion_backend:
            raise StalePlanningContext("environmental motion backend changed after planning context creation")

        for item in self.vessels:
            try:
                current_limits = world.limits_for(item.vessel_id)
                current_drift = world.drift_for(item.vessel_id)
            except KeyError as exc:
                raise StalePlanningContext(
                    f"planning vessel {item.vessel_id!r} is no longer active"
                ) from exc
            if current_limits != item.maneuver_limits:
                raise StalePlanningContext(
                    f"maneuver limits changed for planning vessel {item.vessel_id!r}"
                )
            if current_drift != item.environmental_drift_velocity:
                raise StalePlanningContext(
                    f"environmental drift changed for planning vessel {item.vessel_id!r}"
                )
            if world.motion_state_for(item.vessel_id) != item.motion_state:
                raise StalePlanningContext(
                    f"motion backend state changed for planning vessel {item.vessel_id!r}"
                )
            current_profile = (
                None if profile_catalog is None else profile_catalog.profile_for(item.vessel_id)
            )
            if current_profile != item.profile:
                raise StalePlanningContext(
                    f"profile binding changed for planning vessel {item.vessel_id!r}"
                )


def build_environmental_avoidance_planning_context(
    world: EnvironmentalWorld2D,
    dispatch: FleetSupervisionDispatch,
    owner: FleetCommandOwner,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
    route_targets: Mapping[str, "EnvironmentalRouteHandoffTarget"] | None = None,
    same_route_following: "SameRouteFollowingSupervisor | None" = None,
) -> EnvironmentalAvoidancePlanningContext:
    """Capture environmental planning evidence for exactly one dispatched owner.

    A class ``speed_envelope`` constrains the controlled **through-water** speed in this
    environment-aware path.  Current may increase or decrease speed over ground without
    violating that propulsion/control envelope.
    """

    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if dispatch.time_min != world.time:
        raise PlanningContextOwnerMismatch(
            "dispatch timestamp does not match the current environmental world timestamp"
        )
    if owner not in dispatch.owners:
        raise PlanningContextOwnerMismatch("owner is not part of the supplied dispatch")

    snapshot = world.snapshot()
    snapshot_ids = {state.vessel_id for state in snapshot.vessels}
    missing = set(owner.vessel_ids) - snapshot_ids
    if missing:
        raise PlanningContextOwnerMismatch(
            f"owner references vessels absent from environmental snapshot: {sorted(missing)}"
        )

    inputs: list[EnvironmentalVesselPlanningInputs] = []
    for vessel_id in owner.vessel_ids:
        profile = None if profile_catalog is None else profile_catalog.profile_for(vessel_id)
        limits = world.limits_for(vessel_id)
        if profile is not None and profile.maneuver_limits != limits:
            raise PlanningContextOwnerMismatch(
                f"profile maneuver limits disagree with EnvironmentalWorld2D for {vessel_id!r}"
            )
        state = snapshot.by_id(vessel_id)
        if (
            profile is not None
            and profile.speed_envelope is not None
            and not profile.speed_envelope.permits(state.through_water_speed)
        ):
            raise PlanningContextOwnerMismatch(
                f"water-relative speed exceeds class speed envelope for {vessel_id!r}"
            )
        drift = world.drift_for(vessel_id)
        motion = environmental_vessel_motion_evidence(state, drift)
        inputs.append(
            EnvironmentalVesselPlanningInputs(
                vessel_id=vessel_id,
                state=state,
                environmental_drift_velocity=drift,
                motion=motion,
                maneuver_limits=limits,
                profile=profile,
                domain=None if profile is None else profile.safety_domain,
                motion_state=world.motion_state_for(vessel_id),
            )
        )

    following_constraints: tuple[SameRouteFollowingConstraint, ...] = ()
    delegated_same_route_pair_keys: tuple[PairKey, ...] = ()
    if same_route_following is not None:
        if route_targets is None:
            raise PlanningContextOwnerMismatch(
                "same-route following requires the active route-target mapping"
            )
        following_constraints = same_route_following.constraints_for(
            world, owner.vessel_ids, route_targets, profile_catalog
        )
        # Only pairs for which the following layer has an explicit current constraint
        # are delegated out of A104 physical pair screening.  Other same-route pairs
        # remain excluded from COLREG semantics by the registry delegation resolver,
        # but they still need transition/domain clearance when JOINT commands can move
        # them laterally into one another.
        owner_vessel_id_set = set(owner.vessel_ids)
        delegated_same_route_pair_keys = tuple(
            sorted(
                {
                    item.pair_key
                    for item in following_constraints
                    if set(item.pair_key).issubset(owner_vessel_id_set)
                }
            )
        )

    return EnvironmentalAvoidancePlanningContext(
        planning_snapshot=snapshot,
        component_id=owner.component_id,
        backend=owner.backend,
        vessel_ids=owner.vessel_ids,
        pair_keys=owner.pair_keys,
        vessels=tuple(inputs),
        profile_catalog_bound=profile_catalog is not None,
        same_route_following_constraints=following_constraints,
        delegated_same_route_pair_keys=delegated_same_route_pair_keys,
        motion_backend=world.motion_backend,
    )
