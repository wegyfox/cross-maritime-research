"""Exploratory paired experiments for the recovered two-stream service model."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import t as student_t

from .experiments import replication_seeds
from .legacy_queueing import (
    FCFSIntersectionServer,
    PriorityReservationIntersectionServer,
    SingleServerRun,
    crossing_service_time_minutes,
    generate_poisson_service_requests,
    reconfigure_service_requests,
)


@dataclass(frozen=True, slots=True)
class QueueObservationMetrics:
    generated: int
    completed_in_window: int
    completion_fraction_in_window: float | None
    throughput_per_hour: float
    mean_waiting_time: float | None
    p95_waiting_time: float | None
    max_waiting_time: float | None
    mean_system_time: float | None
    unfinished_at_window_end: int
    post_window_clearance_time: float
    server_busy_fraction: float


@dataclass(frozen=True, slots=True)
class LegacyQueuePairResult:
    total_rate_per_minute: float
    seed: int
    generated: int
    fcfs: QueueObservationMetrics
    priority: QueueObservationMetrics


@dataclass(frozen=True, slots=True)
class QueueScalarSummary:
    n: int
    mean: float | None
    sample_std: float | None
    standard_error: float | None
    confidence_level: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass(frozen=True, slots=True)
class LegacyQueueSweepPoint:
    total_rate_per_minute: float
    replications: int
    fcfs_throughput_per_hour: QueueScalarSummary
    priority_throughput_per_hour: QueueScalarSummary
    throughput_difference_fcfs_minus_priority: QueueScalarSummary
    fcfs_mean_waiting_time: QueueScalarSummary
    priority_mean_waiting_time: QueueScalarSummary
    fcfs_clearance_time: QueueScalarSummary
    priority_clearance_time: QueueScalarSummary


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def queue_observation_metrics(run: SingleServerRun, *, observation_end: float) -> QueueObservationMetrics:
    """Summarize one finite arrival cohort against a fixed observation window."""

    if observation_end <= 0:
        raise ValueError("observation_end must be positive")
    intervals = tuple(item for item in run.intervals if item.requested_at < observation_end)
    generated = len(intervals)
    completed = sum(item.service_completed_at <= observation_end for item in intervals)
    completion_fraction = None if generated == 0 else completed / generated
    waits = [item.waiting_time for item in intervals]
    systems = [item.system_time for item in intervals]
    unfinished = generated - completed
    final_completion = max((item.service_completed_at for item in intervals), default=observation_end)
    clearance = max(0.0, final_completion - observation_end)
    busy = 0.0
    for item in intervals:
        overlap_start = max(0.0, item.service_started_at)
        overlap_end = min(observation_end, item.service_completed_at)
        if overlap_end > overlap_start:
            busy += overlap_end - overlap_start

    return QueueObservationMetrics(
        generated=generated,
        completed_in_window=completed,
        completion_fraction_in_window=completion_fraction,
        throughput_per_hour=completed * 60.0 / observation_end,
        mean_waiting_time=(None if not waits else sum(waits) / len(waits)),
        p95_waiting_time=_percentile(waits, 95.0),
        max_waiting_time=(None if not waits else max(waits)),
        mean_system_time=(None if not systems else sum(systems) / len(systems)),
        unfinished_at_window_end=unfinished,
        post_window_clearance_time=clearance,
        server_busy_fraction=min(1.0, busy / observation_end),
    )


def run_legacy_two_stream_pair(
    *,
    total_rate_per_minute: float,
    horizon: float,
    seed: int,
    cpa_limit_nm: float = 1.0,
    speed_knots: float = 6.0,
    privileged_fraction: float = 0.5,
    ordinary_multiplier: float = 3.0,
    privileged_multiplier: float = 4.5,
) -> LegacyQueuePairResult:
    """Run identical equal-split Poisson approach streams under both disciplines."""

    if total_rate_per_minute < 0:
        raise ValueError("total_rate_per_minute must be non-negative")
    if not 0.0 <= privileged_fraction <= 1.0:
        raise ValueError("privileged_fraction must be between 0 and 1")
    ordinary_service = crossing_service_time_minutes(
        cpa_limit_nm=cpa_limit_nm,
        speed_knots=speed_knots,
        occupation_multiplier=ordinary_multiplier,
    )
    privileged_service = crossing_service_time_minutes(
        cpa_limit_nm=cpa_limit_nm,
        speed_knots=speed_knots,
        occupation_multiplier=privileged_multiplier,
    )
    base = generate_poisson_service_requests(
        stream_rates={
            "A": total_rate_per_minute * (1.0 - privileged_fraction),
            "B": total_rate_per_minute * privileged_fraction,
        },
        horizon=horizon,
        service_time=1.0,
        rng=np.random.default_rng(seed),
    )
    fcfs_requests = reconfigure_service_requests(
        base,
        service_time_by_stream={"A": ordinary_service, "B": ordinary_service},
    )
    priority_requests = reconfigure_service_requests(
        base,
        service_time_by_stream={"A": ordinary_service, "B": privileged_service},
        priority_class_by_stream={"B": 1},
    )
    fcfs_run = FCFSIntersectionServer().schedule(fcfs_requests)
    priority_run = PriorityReservationIntersectionServer().schedule(priority_requests)
    return LegacyQueuePairResult(
        total_rate_per_minute=total_rate_per_minute,
        seed=seed,
        generated=len(base),
        fcfs=queue_observation_metrics(fcfs_run, observation_end=horizon),
        priority=queue_observation_metrics(priority_run, observation_end=horizon),
    )


def _summary(values: list[float], *, confidence_level: float = 0.95) -> QueueScalarSummary:
    n = len(values)
    if n == 0:
        return QueueScalarSummary(0, None, None, None)
    mean = sum(values) / n
    if n == 1:
        return QueueScalarSummary(1, mean, None, None)
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    sample_std = sqrt(variance)
    standard_error = sample_std / sqrt(n)
    critical = float(student_t.ppf((1.0 + confidence_level) / 2.0, df=n - 1))
    half_width = critical * standard_error
    return QueueScalarSummary(
        n=n,
        mean=mean,
        sample_std=sample_std,
        standard_error=standard_error,
        confidence_level=confidence_level,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
    )


def summarize_queue_scalars(
    values: list[float] | tuple[float, ...],
    *,
    confidence_level: float = 0.95,
) -> QueueScalarSummary:
    """Public scalar-summary helper used by frozen queue executors."""

    return _summary([float(value) for value in values], confidence_level=confidence_level)


def run_legacy_two_stream_sweep(
    *,
    total_rates_per_minute: tuple[float, ...] | list[float],
    horizon: float,
    base_seed: int,
    replications: int,
    cpa_limit_nm: float = 1.0,
    speed_knots: float = 6.0,
    privileged_fraction: float = 0.5,
    ordinary_multiplier: float = 3.0,
    privileged_multiplier: float = 4.5,
) -> tuple[LegacyQueueSweepPoint, ...]:
    """Small paired exploratory sweep using common random numbers per demand point."""

    if replications <= 0:
        raise ValueError("replications must be positive")
    rates = tuple(float(rate) for rate in total_rates_per_minute)
    if len(rates) != len(set(rates)):
        raise ValueError("demand rates must be unique")
    point_seeds = replication_seeds(base_seed=base_seed, count=len(rates))
    output: list[LegacyQueueSweepPoint] = []
    for rate, point_seed in zip(rates, point_seeds):
        seeds = replication_seeds(base_seed=point_seed, count=replications)
        pairs = [
            run_legacy_two_stream_pair(
                total_rate_per_minute=rate,
                horizon=horizon,
                seed=seed,
                cpa_limit_nm=cpa_limit_nm,
                speed_knots=speed_knots,
                privileged_fraction=privileged_fraction,
                ordinary_multiplier=ordinary_multiplier,
                privileged_multiplier=privileged_multiplier,
            )
            for seed in seeds
        ]
        fcfs_throughput = [pair.fcfs.throughput_per_hour for pair in pairs]
        priority_throughput = [pair.priority.throughput_per_hour for pair in pairs]
        difference = [left - right for left, right in zip(fcfs_throughput, priority_throughput)]
        fcfs_wait = [pair.fcfs.mean_waiting_time or 0.0 for pair in pairs]
        priority_wait = [pair.priority.mean_waiting_time or 0.0 for pair in pairs]
        output.append(
            LegacyQueueSweepPoint(
                total_rate_per_minute=rate,
                replications=replications,
                fcfs_throughput_per_hour=_summary(fcfs_throughput),
                priority_throughput_per_hour=_summary(priority_throughput),
                throughput_difference_fcfs_minus_priority=_summary(difference),
                fcfs_mean_waiting_time=_summary(fcfs_wait),
                priority_mean_waiting_time=_summary(priority_wait),
                fcfs_clearance_time=_summary([pair.fcfs.post_window_clearance_time for pair in pairs]),
                priority_clearance_time=_summary([pair.priority.post_window_clearance_time for pair in pairs]),
            )
        )
    return tuple(output)
