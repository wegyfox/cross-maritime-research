"""Environmental route-handoff admission and exact velocity-frame translation.

A97 deliberately does not reuse the historical ``FleetRouteHandoffSession`` execution
state machine under non-zero current.  That older controller assumes one heading/speed
pair describes both vessel orientation and ground translation.  In the environmental
model those frames are distinct.

This module therefore establishes two prerequisites for a later environmental recovery
session:

1. bind one a95 ``HANDOFF_ELIGIBLE`` boundary to exact current world/route evidence;
2. translate a desired Earth-fixed ground course/speed into the water-relative command
   whose vector sum with the captured drift produces that desired ground velocity.

No route recovery, world advancement or command mutation occurs in a97.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from ..kinematics import Vector2
from .dynamics import ManeuverLimits
from .environmental_dynamics import WaterRelativeManeuverCommand, WaterRelativeVesselState
from .environmental_pair_execution_policy import (
    EnvironmentalPairExecutionPolicyAction,
    EnvironmentalPairExecutionPolicyDecision,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .follower import RouteFollowerState
from .navigator import RouteNavigator
from .route_exit import RouteEndAction
from .velocity_frames import (
    compose_ground_velocity,
    course_of_velocity,
    infer_through_water_velocity,
    speed_of_velocity,
    velocity_from_course_speed,
)
from .vessel_classes import VesselClassProfile, VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalRouteHandoffError(RuntimeError):
    """Base error for a97 handoff-admission failures."""


class EnvironmentalRouteHandoffNotReady(EnvironmentalRouteHandoffError):
    """Raised when a95 has not declared the pair handoff-eligible."""


class StaleEnvironmentalRouteHandoff(EnvironmentalRouteHandoffError):
    """Raised when world/control/profile evidence changed after binding."""


class EnvironmentalGroundVelocityUnavailable(EnvironmentalRouteHandoffError):
    """Raised when a desired ground velocity cannot be expressed by model metadata."""


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteHandoffTarget:
    vessel_id: str
    navigator: RouteNavigator
    follower_state: RouteFollowerState = RouteFollowerState()
    end_action: RouteEndAction = RouteEndAction.CONTINUE

    def __post_init__(self) -> None:
        if not self.vessel_id:
            raise ValueError("vessel_id must be non-empty")
        if not isinstance(self.end_action, RouteEndAction):
            raise TypeError("end_action must be RouteEndAction")


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteHandoffVesselBinding:
    target: EnvironmentalRouteHandoffTarget
    state: WaterRelativeVesselState
    environmental_drift_velocity: Vector2
    maneuver_limits: ManeuverLimits
    command_before: WaterRelativeManeuverCommand
    profile: VesselClassProfile | None

    @property
    def vessel_id(self) -> str:
        return self.target.vessel_id


@dataclass(frozen=True, slots=True)
class EnvironmentalRouteHandoffBinding:
    source_policy: EnvironmentalPairExecutionPolicyDecision | None
    snapshot: EnvironmentalWorldSnapshot
    vessels: tuple[EnvironmentalRouteHandoffVesselBinding, ...]
    profile_catalog_bound: bool

    @property
    def time_min(self) -> float:
        return self.snapshot.time

    @property
    def vessel_ids(self) -> tuple[str, ...]:
        return tuple(item.vessel_id for item in self.vessels)

    def for_vessel(self, vessel_id: str) -> EnvironmentalRouteHandoffVesselBinding:
        for item in self.vessels:
            if item.vessel_id == vessel_id:
                return item
        raise KeyError(vessel_id)


@dataclass(frozen=True, slots=True)
class EnvironmentalGroundVelocityCommand:
    desired_ground_heading_deg: float
    desired_ground_speed: float
    desired_ground_velocity: Vector2
    required_through_water_velocity: Vector2
    command: WaterRelativeManeuverCommand

    @property
    def achieved_ground_velocity(self) -> Vector2:
        tw = velocity_from_course_speed(
            course_deg=self.command.target_through_water_heading_deg,
            speed=self.command.target_through_water_speed,
        )
        drift = (
            self.desired_ground_velocity[0] - self.required_through_water_velocity[0],
            self.desired_ground_velocity[1] - self.required_through_water_velocity[1],
        )
        return compose_ground_velocity(tw, drift)


def command_for_desired_ground_velocity(
    state: WaterRelativeVesselState,
    environmental_drift_velocity: Vector2,
    *,
    desired_ground_heading_deg: float,
    desired_ground_speed: float,
    profile: VesselClassProfile | None = None,
) -> EnvironmentalGroundVelocityCommand:
    """Return the exact static water-relative command for one ground-velocity target.

    At zero required through-water speed, water-relative heading is physically
    underdetermined; the current hull heading is retained to avoid inventing a turn.
    """

    if not isinstance(state, WaterRelativeVesselState):
        raise TypeError("state must be WaterRelativeVesselState")
    if not isfinite(desired_ground_heading_deg):
        raise ValueError("desired_ground_heading_deg must be finite")
    if not isfinite(desired_ground_speed) or desired_ground_speed < 0.0:
        raise ValueError("desired_ground_speed must be finite and non-negative")
    if profile is not None and not isinstance(profile, VesselClassProfile):
        raise TypeError("profile must be VesselClassProfile or None")

    desired = velocity_from_course_speed(
        course_deg=desired_ground_heading_deg,
        speed=desired_ground_speed,
    )
    required = infer_through_water_velocity(desired, environmental_drift_velocity)
    required_speed = speed_of_velocity(required)
    required_course = course_of_velocity(required)
    heading = state.through_water_heading_deg if required_course is None else required_course

    if (
        profile is not None
        and profile.speed_envelope is not None
        and not profile.speed_envelope.permits(required_speed)
    ):
        raise EnvironmentalGroundVelocityUnavailable(
            "desired ground velocity requires through-water speed above vessel profile envelope"
        )

    return EnvironmentalGroundVelocityCommand(
        desired_ground_heading_deg=desired_ground_heading_deg % 360.0,
        desired_ground_speed=desired_ground_speed,
        desired_ground_velocity=desired,
        required_through_water_velocity=required,
        command=WaterRelativeManeuverCommand(heading, required_speed),
    )


def build_environmental_route_handoff_binding(
    world: EnvironmentalWorld2D,
    policy: EnvironmentalPairExecutionPolicyDecision,
    targets: Iterable[EnvironmentalRouteHandoffTarget],
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalRouteHandoffBinding:
    """Bind one handoff-eligible a95 boundary to exact current environmental evidence."""

    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(policy, EnvironmentalPairExecutionPolicyDecision):
        raise TypeError("policy must be EnvironmentalPairExecutionPolicyDecision")
    if policy.action is not EnvironmentalPairExecutionPolicyAction.HANDOFF_ELIGIBLE:
        raise EnvironmentalRouteHandoffNotReady(
            f"a95 action {policy.action.value!r} is not handoff eligible"
        )
    if policy.source.decision.responsibility_active:
        raise EnvironmentalRouteHandoffNotReady(
            "handoff binding requires inactive encounter responsibility"
        )
    if abs(policy.time_min - world.time) > _EPS:
        raise StaleEnvironmentalRouteHandoff(
            "a95 handoff boundary timestamp does not match environmental world time"
        )

    items = tuple(targets)
    if len(items) != 2:
        raise ValueError("environmental pair handoff requires exactly two route targets")
    if len({item.vessel_id for item in items}) != 2:
        raise ValueError("environmental route handoff target ids must be unique")

    execution = policy.source.execution
    expected_ids = (execution.first_state.vessel_id, execution.second_state.vessel_id)
    if tuple(item.vessel_id for item in items) != expected_ids:
        raise EnvironmentalRouteHandoffError(
            "route handoff target order does not match a95 execution pair"
        )

    snapshot = world.snapshot()
    expected_states = (execution.first_state, execution.second_state)
    bindings: list[EnvironmentalRouteHandoffVesselBinding] = []
    for target, expected_state in zip(items, expected_states):
        try:
            state = snapshot.by_id(target.vessel_id)
            drift = world.drift_for(target.vessel_id)
            limits = world.limits_for(target.vessel_id)
            command = world.command_for(target.vessel_id)
        except KeyError as exc:
            raise StaleEnvironmentalRouteHandoff(
                f"handoff vessel {target.vessel_id!r} is no longer active"
            ) from exc
        if state != expected_state:
            raise StaleEnvironmentalRouteHandoff(
                f"world state for {target.vessel_id!r} no longer matches a95 boundary evidence"
            )
        profile = None if profile_catalog is None else profile_catalog.profile_for(target.vessel_id)
        if profile is not None and profile.maneuver_limits != limits:
            raise EnvironmentalRouteHandoffError(
                f"profile maneuver limits disagree with world for {target.vessel_id!r}"
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
        source_policy=policy,
        snapshot=snapshot,
        vessels=(bindings[0], bindings[1]),
        profile_catalog_bound=profile_catalog is not None,
    )


def validate_environmental_route_handoff_binding(
    binding: EnvironmentalRouteHandoffBinding,
    world: EnvironmentalWorld2D,
    *,
    profile_catalog: VesselProfileCatalog | None = None,
) -> None:
    """Fail closed unless physical/control/drift/limits/profile evidence is unchanged."""

    if not isinstance(binding, EnvironmentalRouteHandoffBinding):
        raise TypeError("binding must be EnvironmentalRouteHandoffBinding")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if abs(world.time - binding.time_min) > _EPS:
        raise StaleEnvironmentalRouteHandoff("world time changed after route-handoff binding")
    if binding.profile_catalog_bound != (profile_catalog is not None):
        raise StaleEnvironmentalRouteHandoff("profile-catalog binding changed after handoff binding")

    snapshot = world.snapshot()
    for item in binding.vessels:
        try:
            if snapshot.by_id(item.vessel_id) != item.state:
                raise StaleEnvironmentalRouteHandoff(
                    f"physical state changed for handoff vessel {item.vessel_id!r}"
                )
            if world.drift_for(item.vessel_id) != item.environmental_drift_velocity:
                raise StaleEnvironmentalRouteHandoff(
                    f"environmental drift changed for handoff vessel {item.vessel_id!r}"
                )
            if world.limits_for(item.vessel_id) != item.maneuver_limits:
                raise StaleEnvironmentalRouteHandoff(
                    f"maneuver limits changed for handoff vessel {item.vessel_id!r}"
                )
            if world.command_for(item.vessel_id) != item.command_before:
                raise StaleEnvironmentalRouteHandoff(
                    f"active command changed for handoff vessel {item.vessel_id!r}"
                )
        except KeyError as exc:
            raise StaleEnvironmentalRouteHandoff(
                f"handoff vessel {item.vessel_id!r} is no longer active"
            ) from exc
        current_profile = None if profile_catalog is None else profile_catalog.profile_for(item.vessel_id)
        if current_profile != item.profile:
            raise StaleEnvironmentalRouteHandoff(
                f"profile binding changed for handoff vessel {item.vessel_id!r}"
            )
