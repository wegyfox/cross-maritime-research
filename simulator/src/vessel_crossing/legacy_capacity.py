"""Analytical capacity references recovered from the 2020 thesis.

The formulas in this module are deliberately separate from stochastic queue
simulation.  They reconstruct the *idealized capacity arithmetic* used in the
thesis chapter for two perpendicular one-way channels with equal vessel speed.

The no-priority server uses ``t2 = 3*CPA/v`` per vessel.  The priority chapter
uses a longer privileged interval ``t1 = 4.5*CPA/v`` and, for the tabulated
50/50 two-stream limiting case, treats one privileged plus one ordinary vessel
as a two-vessel service cycle.  That arithmetic implies a priority capacity
that is exactly 80% of the no-priority capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .legacy_queueing import crossing_service_time_minutes


@dataclass(frozen=True, slots=True)
class LegacyCapacityReference:
    """Idealized two-stream capacity reference under explicit service mix."""

    cpa_limit_nm: float
    speed_knots: float
    ordinary_multiplier: float
    privileged_multiplier: float
    privileged_fraction: float
    ordinary_service_minutes: float
    privileged_service_minutes: float
    fcfs_capacity_per_hour: float
    priority_nominal_capacity_per_hour: float
    priority_shortfall_fraction_of_fcfs: float
    fcfs_advantage_fraction_of_priority: float


def legacy_capacity_reference(
    *,
    cpa_limit_nm: float,
    speed_knots: float,
    privileged_fraction: float = 0.5,
    ordinary_multiplier: float = 3.0,
    privileged_multiplier: float = 4.5,
) -> LegacyCapacityReference:
    """Return the idealized service-time capacity arithmetic.

    ``priority_nominal_capacity_per_hour`` is a service-mix upper reference:
    it assumes no additional idle fragments beyond the explicit service times.
    The stochastic reservation scheduler can be lower because an ordinary job
    may not fit into a short gap before a privileged reservation.
    """

    if not isfinite(privileged_fraction) or not 0.0 <= privileged_fraction <= 1.0:
        raise ValueError("privileged_fraction must be between 0 and 1")

    ordinary = crossing_service_time_minutes(
        cpa_limit_nm=cpa_limit_nm,
        speed_knots=speed_knots,
        occupation_multiplier=ordinary_multiplier,
    )
    privileged = crossing_service_time_minutes(
        cpa_limit_nm=cpa_limit_nm,
        speed_knots=speed_knots,
        occupation_multiplier=privileged_multiplier,
    )
    fcfs_capacity = 60.0 / ordinary
    mean_priority_service = (
        (1.0 - privileged_fraction) * ordinary
        + privileged_fraction * privileged
    )
    priority_capacity = 60.0 / mean_priority_service
    shortfall = (fcfs_capacity - priority_capacity) / fcfs_capacity
    advantage = (fcfs_capacity - priority_capacity) / priority_capacity
    return LegacyCapacityReference(
        cpa_limit_nm=cpa_limit_nm,
        speed_knots=speed_knots,
        ordinary_multiplier=ordinary_multiplier,
        privileged_multiplier=privileged_multiplier,
        privileged_fraction=privileged_fraction,
        ordinary_service_minutes=ordinary,
        privileged_service_minutes=privileged,
        fcfs_capacity_per_hour=fcfs_capacity,
        priority_nominal_capacity_per_hour=priority_capacity,
        priority_shortfall_fraction_of_fcfs=shortfall,
        fcfs_advantage_fraction_of_priority=advantage,
    )


def thesis_priority_table_capacity_per_hour(*, cpa_limit_nm: float, speed_knots: float) -> float:
    """Reproduce Table 2's equal-stream two-vessel-cycle capacity.

    For a 50/50 mix this is algebraically identical to
    ``legacy_capacity_reference(..., privileged_fraction=0.5)``.
    """

    reference = legacy_capacity_reference(
        cpa_limit_nm=cpa_limit_nm,
        speed_knots=speed_knots,
        privileged_fraction=0.5,
    )
    return reference.priority_nominal_capacity_per_hour


def mm1_wait_to_service_ratio(utilization: float) -> float:
    """Historical M/M/1 queue-wait divided by deterministic service time."""

    if utilization < 0 or utilization >= 1:
        raise ValueError("utilization must satisfy 0 <= rho < 1")
    return utilization / (1.0 - utilization)


def md1_wait_to_service_ratio(utilization: float) -> float:
    """Correct M/D/1 queue-wait divided by deterministic service time."""

    if utilization < 0 or utilization >= 1:
        raise ValueError("utilization must satisfy 0 <= rho < 1")
    return utilization / (2.0 * (1.0 - utilization))


def utilization_when_wait_equals_service(*, deterministic_service: bool) -> float:
    """Return rho where mean queue waiting equals one service interval.

    This is an operational heuristic, not a queue-stability boundary.  The
    historical M/M/1 expression reaches it at rho=0.5; the mathematically
    matching M/D/1 expression reaches it at rho=2/3.
    """

    return 2.0 / 3.0 if deterministic_service else 0.5
