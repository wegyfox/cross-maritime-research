"""Transparent engineering classification of demand-sweep congestion states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .experiments import DemandSweepPointResult


class SaturationLevel(str, Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    STABLE = "stable"
    STRAINED = "strained"
    SATURATED = "saturated"
    BREAKDOWN = "breakdown"


class SaturationReason(str, Enum):
    MISSING_REQUIRED_METRIC = "missing_required_metric"
    LOW_CROSSING_SERVICE_FRACTION = "low_crossing_service_fraction"
    HIGH_MEAN_APPROACH_BACKLOG = "high_mean_approach_backlog"
    HIGH_MEAN_ENTRY_DELAY = "high_mean_entry_delay"
    LONG_ZONE_CLEARANCE = "long_zone_clearance"
    IN_WINDOW_STARVATION = "in_window_starvation"
    THROUGHPUT_PLATEAU = "throughput_plateau"
    THROUGHPUT_DECLINE = "throughput_decline"


@dataclass(frozen=True, slots=True)
class SaturationThresholds:
    """Explicit research/engineering thresholds; deliberately no hidden defaults."""

    min_crossing_service_fraction: float
    max_mean_active_approach_count: float
    max_mean_conflict_zone_entry_delay: float
    max_post_window_conflict_zone_clearance_time: float
    max_starved_route_count_in_window: float
    plateau_relative_throughput_gain: float
    decline_relative_tolerance: float = 0.0
    min_congestion_reasons_for_saturation: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_crossing_service_fraction <= 1.0:
            raise ValueError("min_crossing_service_fraction must be in [0, 1]")
        nonnegative = (
            self.max_mean_active_approach_count,
            self.max_mean_conflict_zone_entry_delay,
            self.max_post_window_conflict_zone_clearance_time,
            self.max_starved_route_count_in_window,
            self.plateau_relative_throughput_gain,
            self.decline_relative_tolerance,
        )
        if any(not isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("saturation thresholds must be finite and non-negative")
        if self.decline_relative_tolerance > 1.0:
            raise ValueError("decline_relative_tolerance must not exceed 1")
        if self.min_congestion_reasons_for_saturation < 1:
            raise ValueError("min_congestion_reasons_for_saturation must be >= 1")


@dataclass(frozen=True, slots=True)
class SaturationInputPoint:
    """Minimal auditable metric vector used by the classifier."""

    name: str
    total_demand_rate_per_minute: float
    crossing_throughput_per_hour: float | None
    crossing_service_fraction_in_window: float | None
    mean_active_approach_count: float | None
    mean_conflict_zone_entry_delay: float | None
    post_window_conflict_zone_clearance_time: float | None
    starved_route_count_in_window: float | None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("saturation point name must be non-empty")
        if not isfinite(self.total_demand_rate_per_minute) or self.total_demand_rate_per_minute < 0:
            raise ValueError("total demand rate must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DemandPointSaturationAssessment:
    """One demand point plus every reason behind its classification."""

    point: SaturationInputPoint
    level: SaturationLevel
    reasons: tuple[SaturationReason, ...]
    congestion_reasons: tuple[SaturationReason, ...]
    relative_throughput_gain_from_previous: float | None


_REQUIRED_FIELDS = (
    "crossing_throughput_per_hour",
    "crossing_service_fraction_in_window",
    "mean_active_approach_count",
    "mean_conflict_zone_entry_delay",
    "post_window_conflict_zone_clearance_time",
    "starved_route_count_in_window",
)


_CONGESTION_REASON_ORDER = (
    SaturationReason.LOW_CROSSING_SERVICE_FRACTION,
    SaturationReason.HIGH_MEAN_APPROACH_BACKLOG,
    SaturationReason.HIGH_MEAN_ENTRY_DELAY,
    SaturationReason.LONG_ZONE_CLEARANCE,
    SaturationReason.IN_WINDOW_STARVATION,
)


def _relative_throughput_gain(previous: float, current: float) -> float | None:
    if previous < 0 or current < 0:
        raise ValueError("throughput cannot be negative")
    if previous == 0:
        if current == 0:
            return 0.0
        return None  # No finite relative gain denominator exists.
    return (current - previous) / previous


def classify_saturation_curve(
    points: tuple[SaturationInputPoint, ...] | list[SaturationInputPoint],
    thresholds: SaturationThresholds,
) -> tuple[DemandPointSaturationAssessment, ...]:
    """Classify a strictly increasing total-demand curve with explicit reasons.

    ``SATURATED`` is an engineering label, not an inferred universal capacity:
    it requires either (a) a throughput plateau plus at least one within-point
    congestion threshold breach or (b) at least the configured number of
    within-point congestion breaches. ``BREAKDOWN`` requires throughput decline
    beyond tolerance plus at least one congestion breach.
    """

    ordered = tuple(sorted(points, key=lambda point: point.total_demand_rate_per_minute))
    if len({point.name for point in ordered}) != len(ordered):
        raise ValueError("saturation point names must be unique")
    rates = [point.total_demand_rate_per_minute for point in ordered]
    if any(right <= left for left, right in zip(rates, rates[1:])):
        raise ValueError("saturation points must have strictly increasing total demand")

    output: list[DemandPointSaturationAssessment] = []
    previous_throughput: float | None = None

    for point in ordered:
        missing = [field for field in _REQUIRED_FIELDS if getattr(point, field) is None]
        if missing:
            output.append(
                DemandPointSaturationAssessment(
                    point=point,
                    level=SaturationLevel.INSUFFICIENT_DATA,
                    reasons=(SaturationReason.MISSING_REQUIRED_METRIC,),
                    congestion_reasons=(),
                    relative_throughput_gain_from_previous=None,
                )
            )
            current = point.crossing_throughput_per_hour
            previous_throughput = float(current) if current is not None else None
            continue

        throughput = float(point.crossing_throughput_per_hour)
        service_fraction = float(point.crossing_service_fraction_in_window)
        approach = float(point.mean_active_approach_count)
        entry_delay = float(point.mean_conflict_zone_entry_delay)
        clearance = float(point.post_window_conflict_zone_clearance_time)
        starved = float(point.starved_route_count_in_window)
        values = (throughput, service_fraction, approach, entry_delay, clearance, starved)
        if not all(isfinite(value) for value in values):
            raise ValueError("saturation input metrics must be finite when present")

        congestion: list[SaturationReason] = []
        if service_fraction < thresholds.min_crossing_service_fraction:
            congestion.append(SaturationReason.LOW_CROSSING_SERVICE_FRACTION)
        if approach > thresholds.max_mean_active_approach_count:
            congestion.append(SaturationReason.HIGH_MEAN_APPROACH_BACKLOG)
        if entry_delay > thresholds.max_mean_conflict_zone_entry_delay:
            congestion.append(SaturationReason.HIGH_MEAN_ENTRY_DELAY)
        if clearance > thresholds.max_post_window_conflict_zone_clearance_time:
            congestion.append(SaturationReason.LONG_ZONE_CLEARANCE)
        if starved > thresholds.max_starved_route_count_in_window:
            congestion.append(SaturationReason.IN_WINDOW_STARVATION)

        relative_gain = None
        trend_reason: SaturationReason | None = None
        if previous_throughput is not None:
            relative_gain = _relative_throughput_gain(previous_throughput, throughput)
            if relative_gain is not None:
                if relative_gain < -thresholds.decline_relative_tolerance:
                    trend_reason = SaturationReason.THROUGHPUT_DECLINE
                elif relative_gain <= thresholds.plateau_relative_throughput_gain:
                    trend_reason = SaturationReason.THROUGHPUT_PLATEAU

        reasons = list(congestion)
        if trend_reason is not None:
            reasons.append(trend_reason)

        if trend_reason is SaturationReason.THROUGHPUT_DECLINE and congestion:
            level = SaturationLevel.BREAKDOWN
        elif (
            trend_reason is SaturationReason.THROUGHPUT_PLATEAU and congestion
        ) or len(congestion) >= thresholds.min_congestion_reasons_for_saturation:
            level = SaturationLevel.SATURATED
        elif reasons:
            level = SaturationLevel.STRAINED
        else:
            level = SaturationLevel.STABLE

        ordered_congestion = tuple(
            reason for reason in _CONGESTION_REASON_ORDER if reason in congestion
        )
        output.append(
            DemandPointSaturationAssessment(
                point=point,
                level=level,
                reasons=tuple(reasons),
                congestion_reasons=ordered_congestion,
                relative_throughput_gain_from_previous=relative_gain,
            )
        )
        previous_throughput = throughput

    return tuple(output)


def assess_demand_sweep_saturation(
    results: tuple[DemandSweepPointResult, ...] | list[DemandSweepPointResult],
    *,
    policy_name: str,
    thresholds: SaturationThresholds,
) -> tuple[DemandPointSaturationAssessment, ...]:
    """Adapt demand-sweep replication summaries into the pure classifier."""

    points: list[SaturationInputPoint] = []
    for result in results:
        matches = [summary for summary in result.policy_summaries if summary.policy_name == policy_name]
        if len(matches) != 1:
            raise ValueError(f"policy {policy_name!r} must exist exactly once at every demand point")
        summary = matches[0]
        total_rate = sum(
            demand.rate_per_minute for demand in result.point.scenario_config.route_demands
        )
        points.append(
            SaturationInputPoint(
                name=result.point.name,
                total_demand_rate_per_minute=total_rate,
                crossing_throughput_per_hour=summary.crossing_throughput_per_hour.mean,
                crossing_service_fraction_in_window=(
                    summary.crossing_service_fraction_in_window.mean
                ),
                mean_active_approach_count=summary.mean_active_approach_count.mean,
                mean_conflict_zone_entry_delay=summary.mean_conflict_zone_entry_delay.mean,
                post_window_conflict_zone_clearance_time=(
                    summary.post_window_conflict_zone_clearance_time.mean
                ),
                starved_route_count_in_window=summary.starved_route_count_in_window.mean,
            )
        )
    return classify_saturation_curve(points, thresholds)
