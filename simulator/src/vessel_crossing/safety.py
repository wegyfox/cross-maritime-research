"""Geometric encounter-risk classification built on analytical CPA/TCPA."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import inf, isinf, sqrt

from .cpa import ClosestApproach, closest_approach
from .kinematics import MotionState, dot, subtract

_EPSILON = 1e-10


class RiskReason(str, Enum):
    """Why an encounter was classified the way it was."""

    CURRENT_DOMAIN_VIOLATION = "current_domain_violation"
    PREDICTED_CONFLICT = "predicted_conflict"
    DIVERGING = "diverging"
    BEYOND_HORIZON = "beyond_horizon"
    CONSTANT_SEPARATION = "constant_separation"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class CircularSafetyDomain:
    """First safety model: radial minimum separation plus TCPA horizon.

    This is deliberately simpler than a final ship-domain model.  It provides
    a transparent baseline and reproduces the diploma's scalar CPA threshold
    in a mathematically explicit form.
    """

    min_dcpa: float
    tcpa_horizon: float | None = None

    def __post_init__(self) -> None:
        if self.min_dcpa < 0:
            raise ValueError("min_dcpa must be non-negative")
        if self.tcpa_horizon is not None and self.tcpa_horizon < 0:
            raise ValueError("tcpa_horizon must be non-negative")


@dataclass(frozen=True, slots=True)
class EncounterAssessment:
    dangerous: bool
    reason: RiskReason
    closest_approach: ClosestApproach


def assess_encounter(
    first: MotionState,
    second: MotionState,
    domain: CircularSafetyDomain,
) -> EncounterAssessment:
    """Classify a pair using current separation, DCPA and future TCPA."""

    cpa = closest_approach(first, second)

    if cpa.current_distance <= domain.min_dcpa:
        return EncounterAssessment(True, RiskReason.CURRENT_DOMAIN_VIOLATION, cpa)

    if isinf(cpa.raw_tcpa):
        return EncounterAssessment(False, RiskReason.CONSTANT_SEPARATION, cpa)

    if cpa.raw_tcpa < 0:
        return EncounterAssessment(False, RiskReason.DIVERGING, cpa)

    if (
        domain.tcpa_horizon is not None
        and cpa.raw_tcpa > domain.tcpa_horizon + _EPSILON
    ):
        return EncounterAssessment(False, RiskReason.BEYOND_HORIZON, cpa)

    if cpa.dcpa <= domain.min_dcpa:
        return EncounterAssessment(True, RiskReason.PREDICTED_CONFLICT, cpa)

    return EncounterAssessment(False, RiskReason.CLEAR, cpa)


def time_to_exit_circular_domain(
    first: MotionState,
    second: MotionState,
    radius: float,
) -> float:
    """Return future time when pair exits a circular separation boundary.

    The function solves ``|r + v t| = radius`` and returns the larger positive
    root, i.e. the outbound crossing after closest approach.  It returns
    ``math.inf`` when relative motion never produces a future exit crossing.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative")

    r = subtract(second.position, first.position)
    v = subtract(second.velocity, first.velocity)
    vv = dot(v, v)
    if vv <= _EPSILON:
        return inf

    rv = dot(r, v)
    rr_minus_radius = dot(r, r) - radius**2
    discriminant = rv**2 - vv * rr_minus_radius
    if discriminant < -_EPSILON:
        return inf
    discriminant = max(0.0, discriminant)

    root = sqrt(discriminant)
    later = (-rv + root) / vv
    return later if later > _EPSILON else inf
