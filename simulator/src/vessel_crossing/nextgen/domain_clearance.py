"""Conservative continuous-time clearance certification for oriented ship domains.

The a40 :mod:`ship_domain` layer evaluates one instantaneous pair geometry.  Avoidance
screening needs a stronger question: did two moving, possibly turning vessels keep their
oriented domains disjoint over an entire maneuver interval?

This module answers that question without introducing a fixed simulation step.  It uses
exact state queries supplied by the caller, adaptive interval subdivision, and a
conservative Lipschitz bound for the *domain margin*::

    margin = actual centre distance - directional required domain distance

The bound covers relative translation, change of the line-of-sight direction, and the
bounded heading rates of both vessels.  If an interval cannot be certified safe before
the configured subdivision limit, its conservative lower bound remains non-positive and
the caller must fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, pi
from typing import Callable, Protocol
from .ship_domain import ShipDomain, evaluate_pair_ship_domain

_EPS = 1e-12


class DomainClearanceState(Protocol):
    """Minimal structural state required by the continuous domain certificate."""

    position: tuple[float, float]
    heading_deg: float


@dataclass(frozen=True, slots=True)
class DomainClearanceCertification:
    """Continuous-time lower-bound certificate for one pair-domain interval."""

    evaluated_duration_min: float
    sampled_min_margin_nm: float
    conservative_lower_bound_nm: float
    sampled_min_separation_nm: float
    sampled_max_required_distance_nm: float
    state_evaluations: int

    @property
    def certified_clear(self) -> bool:
        """Whether the entire interval is proven strictly outside both domains."""

        return self.conservative_lower_bound_nm > _EPS


def _max_extent(domain: ShipDomain) -> float:
    return max(domain.ahead_nm, domain.astern_nm, domain.port_nm, domain.starboard_nm)


def _angular_sensitivity_per_rad(domain: ShipDomain) -> float:
    """Global |dr/dangle| bound for the piecewise quadrant-ellipse radial function.

    In one quadrant the radial extent is
    ``r=(cos²/a² + sin²/b²)^(-1/2)``.  Its derivative magnitude is bounded by
    ``0.5 * max(a,b)^3 * |1/a² - 1/b²|``.  Taking the maximum over the four
    longitudinal/transverse extent combinations covers quadrant changes as a global
    Lipschitz bound; the radial function itself is continuous at the quadrant axes.
    """

    best = 0.0
    for longitudinal in (domain.ahead_nm, domain.astern_nm):
        for transverse in (domain.port_nm, domain.starboard_nm):
            radius = max(longitudinal, transverse)
            sensitivity = 0.5 * radius**3 * abs(
                1.0 / longitudinal**2 - 1.0 / transverse**2
            )
            best = max(best, sensitivity)
    return best


def certify_pair_domain_clearance(
    *,
    first_state_at: Callable[[float], DomainClearanceState],
    second_state_at: Callable[[float], DomainClearanceState],
    first_domain: ShipDomain,
    second_domain: ShipDomain,
    duration_min: float,
    first_speed_bound_nm_per_min: float,
    second_speed_bound_nm_per_min: float,
    first_turn_rate_bound_deg_per_min: float,
    second_turn_rate_bound_deg_per_min: float,
    time_tolerance_min: float,
    margin_tolerance_nm: float,
    max_subdivisions: int,
) -> DomainClearanceCertification:
    """Conservatively certify oriented pair-domain clearance over ``[0, duration]``.

    ``first_state_at`` and ``second_state_at`` must return exact deterministic states for
    the queried relative time.  Speed and turn-rate bounds may be loose but must bound
    the whole interval.  Loose bounds reduce certification power, never safety.
    """

    numeric = (
        duration_min,
        first_speed_bound_nm_per_min,
        second_speed_bound_nm_per_min,
        first_turn_rate_bound_deg_per_min,
        second_turn_rate_bound_deg_per_min,
        time_tolerance_min,
        margin_tolerance_nm,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("domain-clearance numerical inputs must be finite")
    if duration_min < 0.0:
        raise ValueError("duration_min must be non-negative")
    if first_speed_bound_nm_per_min < 0.0 or second_speed_bound_nm_per_min < 0.0:
        raise ValueError("speed bounds must be non-negative")
    if first_turn_rate_bound_deg_per_min < 0.0 or second_turn_rate_bound_deg_per_min < 0.0:
        raise ValueError("turn-rate bounds must be non-negative")
    if time_tolerance_min <= 0.0:
        raise ValueError("time_tolerance_min must be positive")
    if margin_tolerance_nm < 0.0:
        raise ValueError("margin_tolerance_nm must be non-negative")
    if max_subdivisions < 1:
        raise ValueError("max_subdivisions must be at least 1")

    relative_speed_bound = first_speed_bound_nm_per_min + second_speed_bound_nm_per_min
    first_turn_rad = first_turn_rate_bound_deg_per_min * pi / 180.0
    second_turn_rad = second_turn_rate_bound_deg_per_min * pi / 180.0
    first_sensitivity = _angular_sensitivity_per_rad(first_domain)
    second_sensitivity = _angular_sensitivity_per_rad(second_domain)
    coincident_required = _max_extent(first_domain) + _max_extent(second_domain)

    cache: dict[float, tuple[float, float, float]] = {}

    def sample(time_min: float) -> tuple[float, float, float]:
        """Return (margin, actual separation, required domain distance)."""

        key = round(time_min, 14)
        if key not in cache:
            first = first_state_at(time_min)
            second = second_state_at(time_min)
            relative = (
                second.position[0] - first.position[0],
                second.position[1] - first.position[1],
            )
            actual = hypot(relative[0], relative[1])
            if actual <= _EPS:
                required = coincident_required
                margin = -required
            else:
                clearance = evaluate_pair_ship_domain(
                    first_heading_deg=first.heading_deg,
                    second_heading_deg=second.heading_deg,
                    relative_position=relative,
                    first_domain=first_domain,
                    second_domain=second_domain,
                )
                required = clearance.required_distance_nm
                margin = actual - required
            cache[key] = (margin, actual, required)
        return cache[key]

    first_sample = sample(0.0)
    if duration_min <= _EPS:
        return DomainClearanceCertification(
            evaluated_duration_min=0.0,
            sampled_min_margin_nm=first_sample[0],
            conservative_lower_bound_nm=first_sample[0],
            sampled_min_separation_nm=first_sample[1],
            sampled_max_required_distance_nm=first_sample[2],
            state_evaluations=len(cache),
        )

    end_sample = sample(duration_min)
    best_margin = [min(first_sample[0], end_sample[0])]
    best_distance = [min(first_sample[1], end_sample[1])]
    max_required = [max(first_sample[2], end_sample[2])]

    def recurse(a: float, b: float, depth: int) -> float:
        width = b - a
        mid = 0.5 * (a + b)
        sa = sample(a)
        sm = sample(mid)
        sb = sample(b)
        best_margin[0] = min(best_margin[0], sa[0], sm[0], sb[0])
        best_distance[0] = min(best_distance[0], sa[1], sm[1], sb[1])
        max_required[0] = max(max_required[0], sa[2], sm[2], sb[2])

        # A lower bound on centre separation anywhere in this interval.  If it can
        # reach zero, line-of-sight angle is not Lipschitz; fail closed by using a
        # deliberately non-certifying lower bound and subdividing as far as allowed.
        half = 0.5 * width
        distance_floor = max(0.0, sm[1] - relative_speed_bound * half)
        if distance_floor <= _EPS:
            lower_bound = -coincident_required
        else:
            line_of_sight_rate_bound = relative_speed_bound / distance_floor
            required_rate_bound = (
                first_sensitivity * (line_of_sight_rate_bound + first_turn_rad)
                + second_sensitivity * (line_of_sight_rate_bound + second_turn_rad)
            )
            margin_rate_bound = relative_speed_bound + required_rate_bound
            lower_bound = max(
                sa[0] - margin_rate_bound * width,
                sb[0] - margin_rate_bound * width,
                sm[0] - margin_rate_bound * half,
            )

        # Positive lower bound is a complete safety certificate for this interval.
        if lower_bound > _EPS:
            return lower_bound
        # Otherwise continue until the numerical lower bound is already within the
        # requested margin resolution of the best sampled value or subdivision caps.
        if lower_bound >= best_margin[0] - margin_tolerance_nm:
            return lower_bound
        if width <= time_tolerance_min or depth >= max_subdivisions:
            return lower_bound
        return min(recurse(a, mid, depth + 1), recurse(mid, b, depth + 1))

    conservative_lower = recurse(0.0, duration_min, 0)
    return DomainClearanceCertification(
        evaluated_duration_min=duration_min,
        sampled_min_margin_nm=best_margin[0],
        conservative_lower_bound_nm=conservative_lower,
        sampled_min_separation_nm=best_distance[0],
        sampled_max_required_distance_nm=max_required[0],
        state_evaluations=len(cache),
    )
