"""Lifecycle timing metrics derived directly from immutable vessel state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .vessel import Vessel, VesselStatus
from .zones import AxisAlignedConflictZone, ConflictZonePassage


@dataclass(frozen=True, slots=True)
class VesselLifecycleTiming:
    """Auditable timestamps and delays for one vessel.

    ``arrival_time`` is when the traffic process presents the vessel at the
    modeled route entrance. ``activated_at`` is when the simulation actually
    admits it onto the route.  They are deliberately different because source
    queueing may make a vessel wait before entering the controlled approach.
    """

    vessel_id: str
    arrival_time: float
    activated_at: float | None
    completed_at: float | None
    waiting_delay: float | None
    route_duration: float | None
    total_system_time: float | None


@dataclass(frozen=True, slots=True)
class ConflictZoneTiming:
    """Actual passage versus unconstrained desired-speed zone timing.

    The free-flow baseline starts at the vessel's *actual* route admission
    (``activated_at``) and then assumes constant ``desired_speed`` from route
    progress zero. Source-queue waiting is therefore not counted again here.
    """

    vessel_id: str
    expected_entry_at: float | None
    entered_at: float | None
    entry_delay: float | None
    expected_exit_at: float | None
    exited_at: float | None
    exit_delay: float | None
    occupancy_time: float | None


@dataclass(frozen=True, slots=True)
class RouteApproachBacklog:
    """Exact active pre-zone occupancy summary for one route.

    Only vessels already admitted to the modeled route are counted. Source
    queue ``WAITING`` time remains a separate lifecycle stage.
    """

    route_name: str
    mean_count: float | None
    peak_count: int
    vessel_minutes: float


@dataclass(frozen=True, slots=True)
class ApproachBacklogMetrics:
    """Exact occupancy of the active approach before conflict-zone entry."""

    observation_start: float
    observation_end: float
    mean_count: float | None
    peak_count: int
    vessel_minutes: float
    by_route: tuple[RouteApproachBacklog, ...]


@dataclass(frozen=True, slots=True)
class RouteServiceMetrics:
    """Realized per-route service diagnostics inside one observation window."""

    route_name: str
    generated: int
    activated: int
    conflict_zone_exits_in_window: int
    service_fraction_in_window: float | None
    crossing_throughput_per_hour: float | None
    realized_demand_share: float | None
    crossing_exit_share: float | None
    exit_share_minus_demand_share: float | None
    mean_waiting_delay: float | None
    p95_waiting_delay: float | None
    mean_conflict_zone_entry_delay: float | None
    p95_conflict_zone_entry_delay: float | None
    starved_in_window: bool


@dataclass(frozen=True, slots=True)
class RouteFairnessMetrics:
    """Compact fairness indicators backed by explicit per-route evidence."""

    service_fraction_jain_index: float | None
    starved_route_count_in_window: int
    max_abs_exit_share_error: float | None
    by_route: tuple[RouteServiceMetrics, ...]


def _jain_index(values: list[float]) -> float | None:
    if not values:
        return None
    denominator = len(values) * sum(value * value for value in values)
    if denominator <= 0:
        return None
    total = sum(values)
    return total * total / denominator


def route_fairness_metrics(
    vessels: tuple[Vessel, ...] | list[Vessel],
    passages: tuple[ConflictZonePassage, ...] | list[ConflictZonePassage],
    zone: AxisAlignedConflictZone,
    *,
    observation_start: float,
    observation_end: float,
) -> RouteFairnessMetrics:
    """Return realized route service/fairness diagnostics.

    Fairness is defined only for routes with realized demand in this cohort.
    The Jain index is applied to per-route ``exits / generated`` service
    fractions, not to raw exit counts, so unequal demand volumes are not
    automatically labeled unfair. A route is flagged as starved when it has
    realized demand but zero conflict-zone exits in the observation window.
    """

    if observation_start < 0 or observation_end < observation_start:
        raise ValueError("invalid observation window")

    passage_by_id: dict[str, ConflictZonePassage] = {}
    for passage in passages:
        if passage.vessel_id in passage_by_id:
            raise ValueError("conflict-zone passage IDs must be unique")
        passage_by_id[passage.vessel_id] = passage

    eligible = [vessel for vessel in vessels if zone.route_interval(vessel.route) is not None]
    by_route_vessels: dict[str, list[Vessel]] = {}
    for vessel in eligible:
        if vessel.vessel_id not in passage_by_id:
            raise ValueError(
                f"missing conflict-zone passage for eligible vessel {vessel.vessel_id}"
            )
        by_route_vessels.setdefault(vessel.route.name, []).append(vessel)

    total_generated = len(eligible)
    duration = observation_end - observation_start
    exits_by_route: dict[str, int] = {}
    total_exits = 0
    for route_name, route_vessels in by_route_vessels.items():
        exit_count = 0
        for vessel in route_vessels:
            exited_at = passage_by_id[vessel.vessel_id].exited_at
            if (
                exited_at is not None
                and observation_start - 1e-12 <= exited_at <= observation_end + 1e-12
            ):
                exit_count += 1
        exits_by_route[route_name] = exit_count
        total_exits += exit_count

    route_metrics: list[RouteServiceMetrics] = []
    service_fractions: list[float] = []
    share_errors: list[float] = []
    for route_name in sorted(by_route_vessels):
        route_vessels = by_route_vessels[route_name]
        generated = len(route_vessels)
        activated = sum(vessel.activated_at is not None for vessel in route_vessels)
        exit_count = exits_by_route[route_name]
        service_fraction = exit_count / generated if generated > 0 else None
        if service_fraction is not None:
            service_fractions.append(service_fraction)

        demand_share = generated / total_generated if total_generated > 0 else None
        exit_share = exit_count / total_exits if total_exits > 0 else None
        share_error = (
            exit_share - demand_share
            if exit_share is not None and demand_share is not None
            else None
        )
        if share_error is not None:
            share_errors.append(share_error)

        waits = [
            timing.waiting_delay
            for timing in (lifecycle_timing(vessel) for vessel in route_vessels)
            if timing.waiting_delay is not None
        ]
        entry_delays: list[float] = []
        for vessel in route_vessels:
            timing = conflict_zone_timing(vessel, passage_by_id[vessel.vessel_id], zone)
            if timing.entry_delay is not None:
                entry_delays.append(timing.entry_delay)

        route_metrics.append(
            RouteServiceMetrics(
                route_name=route_name,
                generated=generated,
                activated=activated,
                conflict_zone_exits_in_window=exit_count,
                service_fraction_in_window=service_fraction,
                crossing_throughput_per_hour=(
                    exit_count * 60.0 / duration if duration > 0 else None
                ),
                realized_demand_share=demand_share,
                crossing_exit_share=exit_share,
                exit_share_minus_demand_share=share_error,
                mean_waiting_delay=_mean([float(value) for value in waits]),
                p95_waiting_delay=_linear_percentile(
                    [float(value) for value in waits], 95.0
                ),
                mean_conflict_zone_entry_delay=_mean(entry_delays),
                p95_conflict_zone_entry_delay=_linear_percentile(entry_delays, 95.0),
                starved_in_window=(generated > 0 and exit_count == 0),
            )
        )

    return RouteFairnessMetrics(
        service_fraction_jain_index=_jain_index(service_fractions),
        starved_route_count_in_window=sum(item.starved_in_window for item in route_metrics),
        max_abs_exit_share_error=(
            max(abs(value) for value in share_errors) if share_errors else None
        ),
        by_route=tuple(route_metrics),
    )


def _interval_overlap(
    start: float,
    end: float,
    window_start: float,
    window_end: float,
) -> tuple[float, float] | None:
    clipped_start = max(start, window_start)
    clipped_end = min(end, window_end)
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def _peak_half_open_intervals(intervals: list[tuple[float, float]]) -> int:
    """Peak concurrency for half-open ``[start, end)`` intervals.

    End events are processed before start events at an equal timestamp, so a
    vessel entering the zone exactly when another is activated does not create
    a fictitious one-instant overlap.
    """

    events: list[tuple[float, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))  # -1 before +1

    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        if current < 0:  # defensive: valid clipped intervals cannot do this.
            raise RuntimeError("approach interval sweep produced a negative count")
        peak = max(peak, current)
    return peak


def approach_backlog_metrics(
    vessels: tuple[Vessel, ...] | list[Vessel],
    passages: tuple[ConflictZonePassage, ...] | list[ConflictZonePassage],
    zone: AxisAlignedConflictZone,
    *,
    observation_start: float,
    observation_end: float,
) -> ApproachBacklogMetrics:
    """Measure exact active pre-zone occupancy without time-step sampling.

    Each eligible vessel contributes the half-open interval
    ``[activated_at, entered_at)``. If zone entry has not happened by the end
    of a partial run, the interval is right-censored at ``observation_end``.
    Source-queue waiting before ``activated_at`` is intentionally excluded.
    """

    if observation_start < 0 or observation_end < observation_start:
        raise ValueError("invalid observation window")

    passage_by_id: dict[str, ConflictZonePassage] = {}
    for passage in passages:
        if passage.vessel_id in passage_by_id:
            raise ValueError("conflict-zone passage IDs must be unique")
        passage_by_id[passage.vessel_id] = passage

    intervals: list[tuple[float, float]] = []
    intervals_by_route: dict[str, list[tuple[float, float]]] = {}
    eligible_route_names: set[str] = set()

    for vessel in vessels:
        if zone.route_interval(vessel.route) is None:
            continue
        eligible_route_names.add(vessel.route.name)
        passage = passage_by_id.get(vessel.vessel_id)
        if passage is None:
            raise ValueError(
                f"missing conflict-zone passage for eligible vessel {vessel.vessel_id}"
            )
        if vessel.activated_at is None:
            continue

        end = passage.entered_at if passage.entered_at is not None else observation_end
        if end < vessel.activated_at - 1e-12:
            raise ValueError("conflict-zone entry cannot precede route activation")
        overlap = _interval_overlap(
            vessel.activated_at,
            end,
            observation_start,
            observation_end,
        )
        if overlap is None:
            continue
        intervals.append(overlap)
        intervals_by_route.setdefault(vessel.route.name, []).append(overlap)

    duration = observation_end - observation_start
    vessel_minutes = sum(end - start for start, end in intervals)
    mean_count = vessel_minutes / duration if duration > 0 else None

    by_route: list[RouteApproachBacklog] = []
    for route_name in sorted(eligible_route_names):
        route_intervals = intervals_by_route.get(route_name, [])
        route_vessel_minutes = sum(end - start for start, end in route_intervals)
        by_route.append(
            RouteApproachBacklog(
                route_name=route_name,
                mean_count=(route_vessel_minutes / duration if duration > 0 else None),
                peak_count=_peak_half_open_intervals(route_intervals),
                vessel_minutes=route_vessel_minutes,
            )
        )

    return ApproachBacklogMetrics(
        observation_start=observation_start,
        observation_end=observation_end,
        mean_count=mean_count,
        peak_count=_peak_half_open_intervals(intervals),
        vessel_minutes=vessel_minutes,
        by_route=tuple(by_route),
    )


def conflict_zone_timing(
    vessel: Vessel,
    passage: ConflictZonePassage,
    zone: AxisAlignedConflictZone,
) -> ConflictZoneTiming:
    if passage.vessel_id != vessel.vessel_id:
        raise ValueError("passage and vessel IDs must match")
    interval = zone.route_interval(vessel.route)
    if interval is None:
        raise ValueError("vessel route does not intersect the configured conflict zone")

    expected_entry_at = None
    expected_exit_at = None
    if vessel.activated_at is not None and vessel.desired_speed > 0:
        expected_entry_at = vessel.activated_at + interval.entry_progress / vessel.desired_speed
        expected_exit_at = vessel.activated_at + interval.exit_progress / vessel.desired_speed

    entry_delay = None
    if passage.entered_at is not None and expected_entry_at is not None:
        entry_delay = passage.entered_at - expected_entry_at

    exit_delay = None
    if passage.exited_at is not None and expected_exit_at is not None:
        exit_delay = passage.exited_at - expected_exit_at

    values = [
        value
        for value in (
            expected_entry_at,
            passage.entered_at,
            entry_delay,
            expected_exit_at,
            passage.exited_at,
            exit_delay,
            passage.occupancy_time,
        )
        if value is not None
    ]
    if not all(isfinite(value) for value in values):
        raise ValueError("conflict-zone timing values must be finite")

    return ConflictZoneTiming(
        vessel_id=vessel.vessel_id,
        expected_entry_at=expected_entry_at,
        entered_at=passage.entered_at,
        entry_delay=entry_delay,
        expected_exit_at=expected_exit_at,
        exited_at=passage.exited_at,
        exit_delay=exit_delay,
        occupancy_time=passage.occupancy_time,
    )


def lifecycle_timing(vessel: Vessel) -> VesselLifecycleTiming:
    """Return timing metrics without reconstructing the event log."""

    activated_at = vessel.activated_at
    completed_at = vessel.completed_at

    waiting_delay = None
    if activated_at is not None:
        waiting_delay = activated_at - vessel.arrival_time

    route_duration = None
    total_system_time = None
    if completed_at is not None:
        if activated_at is None:  # defensive; Vessel validates this relation.
            raise ValueError("completed vessel timing requires activated_at")
        route_duration = completed_at - activated_at
        total_system_time = completed_at - vessel.arrival_time

    values = [vessel.arrival_time]
    values.extend(
        value
        for value in (activated_at, completed_at, waiting_delay, route_duration, total_system_time)
        if value is not None
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("lifecycle timing values must be finite")

    return VesselLifecycleTiming(
        vessel_id=vessel.vessel_id,
        arrival_time=vessel.arrival_time,
        activated_at=activated_at,
        completed_at=completed_at,
        waiting_delay=waiting_delay,
        route_duration=route_duration,
        total_system_time=total_system_time,
    )

@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Small policy-comparison summary for one simulation run."""

    generated: int
    scheduled: int
    waiting: int
    active: int
    completed: int
    completed_in_window: int
    completion_fraction_in_window: float | None
    observation_start: float
    observation_end: float
    throughput_per_hour: float | None
    mean_waiting_delay: float | None
    median_waiting_delay: float | None
    p95_waiting_delay: float | None
    max_waiting_delay: float | None
    mean_route_duration: float | None
    mean_total_system_time: float | None
    post_window_clearance_time: float | None
    conflict_zone_exits_in_window: int | None = None
    crossing_service_fraction_in_window: float | None = None
    crossing_throughput_per_hour: float | None = None
    mean_conflict_zone_occupancy_time: float | None = None
    post_window_conflict_zone_clearance_time: float | None = None
    mean_conflict_zone_entry_delay: float | None = None
    median_conflict_zone_entry_delay: float | None = None
    p95_conflict_zone_entry_delay: float | None = None
    max_conflict_zone_entry_delay: float | None = None
    mean_conflict_zone_exit_delay: float | None = None
    median_conflict_zone_exit_delay: float | None = None
    p95_conflict_zone_exit_delay: float | None = None
    max_conflict_zone_exit_delay: float | None = None
    mean_active_approach_count: float | None = None
    peak_active_approach_backlog: int | None = None
    approach_backlog_by_route: tuple[RouteApproachBacklog, ...] = ()
    service_fraction_jain_index: float | None = None
    starved_route_count_in_window: int | None = None
    max_abs_exit_share_error: float | None = None
    route_service_by_route: tuple[RouteServiceMetrics, ...] = ()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _linear_percentile(values: list[float], percentile: float) -> float | None:
    """Return a deterministic linearly interpolated sample percentile."""
    if not values:
        return None
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def aggregate_metrics(
    vessels: tuple[Vessel, ...] | list[Vessel],
    *,
    observation_start: float,
    observation_end: float,
    zone_passages: tuple[ConflictZonePassage, ...] | list[ConflictZonePassage] | None = None,
    conflict_zone: AxisAlignedConflictZone | None = None,
) -> AggregateMetrics:
    """Aggregate lifecycle metrics while preserving incomplete-run counts."""

    if observation_start < 0 or observation_end < observation_start:
        raise ValueError("invalid observation window")

    timings = [lifecycle_timing(vessel) for vessel in vessels]
    waiting_delays = [
        timing.waiting_delay for timing in timings if timing.waiting_delay is not None
    ]
    route_durations = [
        timing.route_duration for timing in timings if timing.route_duration is not None
    ]
    total_times = [
        timing.total_system_time
        for timing in timings
        if timing.total_system_time is not None
    ]
    completion_times = [
        timing.completed_at for timing in timings if timing.completed_at is not None
    ]

    counts = {status: 0 for status in VesselStatus}
    for vessel in vessels:
        counts[vessel.status] += 1

    duration = observation_end - observation_start
    completed_in_window = sum(
        1
        for timing in timings
        if timing.completed_at is not None
        and observation_start - 1e-12 <= timing.completed_at <= observation_end + 1e-12
    )

    throughput = None
    completion_fraction = None
    if len(vessels) > 0:
        completion_fraction = completed_in_window / len(vessels)
    if duration > 0:
        throughput = completed_in_window * 60.0 / duration

    post_window_clearance_time = None
    if completion_times:
        post_window_clearance_time = max(0.0, max(completion_times) - observation_end)

    zone_exits_in_window = None
    crossing_service_fraction = None
    crossing_throughput = None
    mean_zone_occupancy = None
    zone_clearance = None
    zone_entry_delays: list[float] = []
    zone_exit_delays: list[float] = []
    approach_metrics: ApproachBacklogMetrics | None = None
    fairness_metrics: RouteFairnessMetrics | None = None
    if zone_passages is not None:
        passages = tuple(zone_passages)
        exit_times = [
            passage.exited_at for passage in passages if passage.exited_at is not None
        ]
        occupancies = [
            passage.occupancy_time
            for passage in passages
            if passage.occupancy_time is not None
        ]
        zone_exits_in_window = sum(
            1
            for exited_at in exit_times
            if observation_start - 1e-12 <= exited_at <= observation_end + 1e-12
        )
        if len(passages) > 0:
            crossing_service_fraction = zone_exits_in_window / len(passages)
        if duration > 0:
            crossing_throughput = zone_exits_in_window * 60.0 / duration
        mean_zone_occupancy = _mean([float(value) for value in occupancies])
        if exit_times:
            zone_clearance = max(0.0, max(exit_times) - observation_end)

        if conflict_zone is not None:
            by_id = {vessel.vessel_id: vessel for vessel in vessels}
            for passage in passages:
                timing = conflict_zone_timing(
                    by_id[passage.vessel_id],
                    passage,
                    conflict_zone,
                )
                if timing.entry_delay is not None:
                    zone_entry_delays.append(timing.entry_delay)
                if timing.exit_delay is not None:
                    zone_exit_delays.append(timing.exit_delay)
            approach_metrics = approach_backlog_metrics(
                vessels,
                passages,
                conflict_zone,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            fairness_metrics = route_fairness_metrics(
                vessels,
                passages,
                conflict_zone,
                observation_start=observation_start,
                observation_end=observation_end,
            )

    return AggregateMetrics(
        generated=len(vessels),
        scheduled=counts[VesselStatus.SCHEDULED],
        waiting=counts[VesselStatus.WAITING],
        active=counts[VesselStatus.ACTIVE],
        completed=counts[VesselStatus.COMPLETED],
        completed_in_window=completed_in_window,
        completion_fraction_in_window=completion_fraction,
        observation_start=observation_start,
        observation_end=observation_end,
        throughput_per_hour=throughput,
        mean_waiting_delay=_mean(waiting_delays),
        median_waiting_delay=_median(waiting_delays),
        p95_waiting_delay=_linear_percentile(waiting_delays, 95.0),
        max_waiting_delay=max(waiting_delays) if waiting_delays else None,
        mean_route_duration=_mean(route_durations),
        mean_total_system_time=_mean(total_times),
        post_window_clearance_time=post_window_clearance_time,
        conflict_zone_exits_in_window=zone_exits_in_window,
        crossing_service_fraction_in_window=crossing_service_fraction,
        crossing_throughput_per_hour=crossing_throughput,
        mean_conflict_zone_occupancy_time=mean_zone_occupancy,
        post_window_conflict_zone_clearance_time=zone_clearance,
        mean_conflict_zone_entry_delay=_mean(zone_entry_delays),
        median_conflict_zone_entry_delay=_median(zone_entry_delays),
        p95_conflict_zone_entry_delay=_linear_percentile(zone_entry_delays, 95.0),
        max_conflict_zone_entry_delay=max(zone_entry_delays) if zone_entry_delays else None,
        mean_conflict_zone_exit_delay=_mean(zone_exit_delays),
        median_conflict_zone_exit_delay=_median(zone_exit_delays),
        p95_conflict_zone_exit_delay=_linear_percentile(zone_exit_delays, 95.0),
        max_conflict_zone_exit_delay=max(zone_exit_delays) if zone_exit_delays else None,
        mean_active_approach_count=(
            approach_metrics.mean_count if approach_metrics is not None else None
        ),
        peak_active_approach_backlog=(
            approach_metrics.peak_count if approach_metrics is not None else None
        ),
        approach_backlog_by_route=(
            approach_metrics.by_route if approach_metrics is not None else ()
        ),
        service_fraction_jain_index=(
            fairness_metrics.service_fraction_jain_index
            if fairness_metrics is not None
            else None
        ),
        starved_route_count_in_window=(
            fairness_metrics.starved_route_count_in_window if fairness_metrics is not None else None
        ),
        max_abs_exit_share_error=(
            fairness_metrics.max_abs_exit_share_error
            if fairness_metrics is not None
            else None
        ),
        route_service_by_route=(
            fairness_metrics.by_route if fairness_metrics is not None else ()
        ),
    )
