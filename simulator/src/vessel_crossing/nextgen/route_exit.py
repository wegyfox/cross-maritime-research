"""Explicit route-end behavior for nominal route schedulers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dynamics import ManeuverCommand, Vessel2DState
from .follower import RouteGuidance
from .world import World2D


class RouteEndAction(str, Enum):
    """What the physical world should do when route guidance reaches its endpoint."""

    CONTINUE = "continue"
    STOP = "stop"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class RouteEndDisposition:
    vessel_id: str
    action: RouteEndAction
    endpoint_state: Vessel2DState
    removed: bool


def apply_route_end_action(
    world: World2D,
    *,
    vessel_id: str,
    guidance: RouteGuidance,
    action: RouteEndAction,
) -> RouteEndDisposition:
    """Apply one explicit route-end action at the current world timestamp."""
    if not guidance.follower_state.completed:
        raise ValueError("route-end action requires completed route guidance")
    state = world.snapshot().by_id(vessel_id)

    if action is RouteEndAction.CONTINUE:
        return RouteEndDisposition(vessel_id, action, state, False)

    if action is RouteEndAction.STOP:
        world.schedule_command(
            at=world.time,
            vessel_id=vessel_id,
            command=ManeuverCommand(guidance.command.target_heading_deg, 0.0),
        )
        world.run_until(world.time)
        return RouteEndDisposition(vessel_id, action, state, False)

    if action is RouteEndAction.REMOVE:
        removed = world.remove_vessel(vessel_id)
        return RouteEndDisposition(vessel_id, action, removed, True)

    raise ValueError(f"unsupported route-end action: {action!r}")
