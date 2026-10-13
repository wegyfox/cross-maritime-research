"""Capacity arithmetic with explicit exact-vs-source-display calculation modes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .r2_s1_pair_semantics import AS, HS, GUO_2024_RUNNING_EXAMPLE_S1, SourcePairIntervalMatrix


class CapacityCalculationMode(str, Enum):
    EXACT = "EXACT"
    SOURCE_DISPLAY_REPRODUCTION = "SOURCE_DISPLAY_REPRODUCTION"


@dataclass(frozen=True, slots=True)
class RunningExample50PercentReport:
    alternating_mean_interval_min: float
    clustered_mean_interval_min: float
    alternating_capacity_exact_ships_per_day: float
    clustered_capacity_exact_ships_per_day: float
    exact_gain_percent: float
    alternating_capacity_source_display: float
    clustered_capacity_source_display: float
    source_display_gain_percent: float


def reciprocal_mean_interval_capacity_ships_per_day(mean_interval_min: float) -> float:
    if not isfinite(mean_interval_min) or mean_interval_min <= 0.0:
        raise ValueError("mean_interval_min must be finite and positive")
    return 1440.0 / mean_interval_min


def percent_gain(*, baseline: float, improved: float) -> float:
    if not isfinite(baseline) or not isfinite(improved) or baseline <= 0.0:
        raise ValueError("baseline must be finite and positive and improved must be finite")
    return 100.0 * (improved - baseline) / baseline


def running_example_50_percent_report(
    matrix: SourcePairIntervalMatrix = GUO_2024_RUNNING_EXAMPLE_S1,
) -> RunningExample50PercentReport:
    alternating_mean = 0.5 * (
        matrix.interval_for_source_roles(follower_type=AS, leader_type=HS)
        + matrix.interval_for_source_roles(follower_type=HS, leader_type=AS)
    )
    clustered_mean = 0.5 * (
        matrix.interval_for_source_roles(follower_type=AS, leader_type=AS)
        + matrix.interval_for_source_roles(follower_type=HS, leader_type=HS)
    )
    alternating_exact = reciprocal_mean_interval_capacity_ships_per_day(alternating_mean)
    clustered_exact = reciprocal_mean_interval_capacity_ships_per_day(clustered_mean)
    alternating_display = round(alternating_exact, 0)
    clustered_display = round(clustered_exact, 1)
    return RunningExample50PercentReport(
        alternating_mean_interval_min=alternating_mean,
        clustered_mean_interval_min=clustered_mean,
        alternating_capacity_exact_ships_per_day=alternating_exact,
        clustered_capacity_exact_ships_per_day=clustered_exact,
        exact_gain_percent=percent_gain(baseline=alternating_exact, improved=clustered_exact),
        alternating_capacity_source_display=alternating_display,
        clustered_capacity_source_display=clustered_display,
        source_display_gain_percent=percent_gain(baseline=alternating_display, improved=clustered_display),
    )
