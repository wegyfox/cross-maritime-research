"""Legacy-oriented maneuver and lost-time summaries for trajectory runs."""

from __future__ import annotations

from dataclasses import dataclass

from .experiments import ExperimentResult


@dataclass(frozen=True, slots=True)
class LegacyTrajectoryMetrics:
    """Metrics explicitly requested by the 2020 numerical-experiment plan."""

    maneuver_count: int
    reduction_maneuver_count: int
    restore_maneuver_count: int
    stop_command_count: int
    total_route_delay: float
    mean_route_delay: float | None
    total_source_waiting_delay: float
    mean_source_waiting_delay: float | None
    completed_vessels: int


def legacy_trajectory_metrics(result: ExperimentResult) -> LegacyTrajectoryMetrics:
    """Summarize commanded maneuvers and lost time from one trajectory run.

    Route delay is measured against free-flow travel from *actual activation*,
    so entrance/source waiting remains separate and is not double-counted.
    """

    changes = result.effective_target_changes
    reductions = tuple(item for item in changes if item.is_reduction)
    restores = tuple(item for item in changes if item.is_restore)
    stops = tuple(item for item in changes if item.new_target <= 1e-12)

    by_id = {item.vessel_id: item for item in result.final_vessels}
    route_delays: list[float] = []
    source_waits: list[float] = []
    for timing in result.vessel_timings:
        vessel = by_id[timing.vessel_id]
        if timing.waiting_delay is not None:
            source_waits.append(max(0.0, timing.waiting_delay))
        if timing.route_duration is None or vessel.desired_speed <= 0:
            continue
        free_flow = vessel.route.length / vessel.desired_speed
        route_delays.append(max(0.0, timing.route_duration - free_flow))

    return LegacyTrajectoryMetrics(
        maneuver_count=len(changes),
        reduction_maneuver_count=len(reductions),
        restore_maneuver_count=len(restores),
        stop_command_count=len(stops),
        total_route_delay=sum(route_delays),
        mean_route_delay=(sum(route_delays) / len(route_delays) if route_delays else None),
        total_source_waiting_delay=sum(source_waits),
        mean_source_waiting_delay=(sum(source_waits) / len(source_waits) if source_waits else None),
        completed_vessels=len(route_delays),
    )
