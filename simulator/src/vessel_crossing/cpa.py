"""Closest-point-of-approach calculations for constant-velocity segments."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .kinematics import MotionState, dot, norm, subtract

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ClosestApproach:
    """CPA result over the selected prediction interval.

    ``raw_tcpa`` is the unconstrained mathematical minimum.  It may be
    negative when the vessels are already moving away from their historic
    closest point.

    ``tcpa`` is the time actually used after clamping to the prediction
    interval [0, horizon].  ``dcpa`` is the separation at that time.
    """

    current_distance: float
    relative_speed: float
    raw_tcpa: float
    tcpa: float
    dcpa: float


def closest_approach(
    first: MotionState,
    second: MotionState,
    *,
    horizon: float | None = None,
) -> ClosestApproach:
    """Compute CPA/TCPA between two constant-velocity point targets.

    Parameters
    ----------
    first, second:
        Current motion states. Positions are expected in NM and velocities in
        NM/minute when using the project's standard units.
    horizon:
        Optional non-negative prediction horizon in minutes.  If omitted, the
        future interval is unbounded.  The returned ``tcpa`` is always
        clamped to the future interval, so a negative mathematical TCPA
        becomes 0.

    Notes
    -----
    With relative position ``r = p2 - p1`` and relative velocity
    ``v = v2 - v1``, separation squared is ``|r + vt|^2``. Differentiating
    with respect to time gives the unconstrained minimum

        ``t* = -dot(r, v) / dot(v, v)``.

    When relative velocity is zero, separation is constant; the function
    reports ``raw_tcpa = inf`` and uses the current instant (tcpa=0) as the
    representative closest point.
    """

    if horizon is not None and horizon < 0:
        raise ValueError("horizon must be non-negative")

    relative_position = subtract(second.position, first.position)
    relative_velocity = subtract(second.velocity, first.velocity)

    current_distance = norm(relative_position)
    relative_speed_sq = dot(relative_velocity, relative_velocity)
    relative_speed = relative_speed_sq**0.5

    if relative_speed_sq <= _EPSILON:
        raw_tcpa = inf
        tcpa = 0.0
        dcpa = current_distance
    else:
        raw_tcpa = -dot(relative_position, relative_velocity) / relative_speed_sq
        tcpa = max(0.0, raw_tcpa)
        if horizon is not None:
            tcpa = min(tcpa, horizon)

        future_relative_position = (
            relative_position[0] + relative_velocity[0] * tcpa,
            relative_position[1] + relative_velocity[1] * tcpa,
        )
        dcpa = norm(future_relative_position)

    return ClosestApproach(
        current_distance=current_distance,
        relative_speed=relative_speed,
        raw_tcpa=raw_tcpa,
        tcpa=tcpa,
        dcpa=dcpa,
    )
