"""Route navigation that combines nominal event guidance with exact recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .follower import RouteFollower, RouteFollowerState, RouteGuidance
from .recovery import CrossTrackRecovery, CrossTrackRecoveryGuidance
from .world import World2D, WorldSnapshot

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class NavigationRunResult:
    snapshot: WorldSnapshot
    follower_state: RouteFollowerState
    nominal_steps: tuple[RouteGuidance, ...]
    recovery_steps: tuple[CrossTrackRecoveryGuidance, ...]
    recovery_count: int


@dataclass(frozen=True, slots=True)
class RouteNavigator:
    follower: RouteFollower
    recovery: CrossTrackRecovery

    def __post_init__(self) -> None:
        if self.follower.route != self.recovery.route:
            raise ValueError("follower and recovery must use the same Route2D")

    def run_world_until(
        self,
        world: World2D,
        *,
        vessel_id: str,
        end_time: float,
        follower_state: RouteFollowerState | None = None,
        max_navigation_events: int = 256,
    ) -> NavigationRunResult:
        state = follower_state or RouteFollowerState()
        nominal: list[RouteGuidance] = []
        recovery_steps: list[CrossTrackRecoveryGuidance] = []
        recovery_count = 0

        for _ in range(max_navigation_events):
            if state.completed or world.time >= end_time - _EPS:
                break
            vessel = world.snapshot().by_id(vessel_id)
            limits = world.limits_for(vessel_id)
            guidance = self.follower.guidance(vessel, state, limits)
            nominal.append(guidance)
            state = guidance.follower_state

            if guidance.next_review_after is None and not state.completed:
                recovered = self.recovery.run_world_until_recovered(
                    world,
                    vessel_id=vessel_id,
                    min_along_track=state.along_track,
                )
                recovery_steps.extend(recovered.guidance_steps)
                recovery_count += 1
                state = replace(
                    state,
                    along_track=max(state.along_track, recovered.final_along_track),
                    active_turn_waypoint=None,
                )
                continue

            world.schedule_command(at=world.time, vessel_id=vessel_id, command=guidance.command)
            if state.completed:
                world.run_until(world.time)
                break
            assert guidance.next_review_after is not None
            if guidance.next_review_after <= _EPS:
                world.run_until(world.time)
                continue
            world.run_until(min(end_time, world.time + guidance.next_review_after))
        else:
            raise RuntimeError("route navigation exceeded max_navigation_events")

        return NavigationRunResult(
            snapshot=world.snapshot(),
            follower_state=state,
            nominal_steps=tuple(nominal),
            recovery_steps=tuple(recovery_steps),
            recovery_count=recovery_count,
        )
