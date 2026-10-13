"""Physical progress capability for persisted Rule-13 overtaking duties.

Rule-13 lifecycle release is deliberately stricter than mere lateral separation: the
saved overtaker must become *past and clear*.  A course-only maneuver at equal or lower
forward speed can diverge safely forever without ever becoming past.  This module
therefore answers a narrower physical question for a chosen command pair: after both
commands reach their target states, is the saved overtaker making positive relative
progress along the current heading axis of the overtaken vessel?

The assessment is a capability screen, not a finite-time release guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .dynamics import Vessel2DState, heading_unit_vector
from .maneuver_intent import ManeuverIntent
from .responsibility import ResponsibilityBasis, ResponsibilityRole

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class Rule13ProgressAssessment:
    overtaking_vessel_id: str
    overtaken_vessel_id: str
    relative_forward_progress_rate_nm_per_min: float
    minimum_required_rate_nm_per_min: float
    capable: bool


def assess_rule13_past_progress(
    first: Vessel2DState,
    second: Vessel2DState,
    first_intent: ManeuverIntent,
    second_intent: ManeuverIntent,
    *,
    minimum_required_rate_nm_per_min: float = 0.0,
) -> Rule13ProgressAssessment | None:
    """Assess post-command ability to progress toward Rule-13 ``past`` geometry.

    Returns ``None`` when neither intent requires Rule-13 past progress.  When enabled,
    exactly one give-way Rule-13 intent must carry the requirement.  ``0.0`` still
    means *strictly positive* relative progress; a positive configured minimum imposes
    that stronger threshold.
    """

    if not isfinite(minimum_required_rate_nm_per_min) or minimum_required_rate_nm_per_min < 0.0:
        raise ValueError("minimum_required_rate_nm_per_min must be finite and non-negative")

    state_by_id = {first.vessel_id: first, second.vessel_id: second}
    if len(state_by_id) != 2:
        raise ValueError("Rule-13 progress assessment requires two distinct vessel IDs")

    required = tuple(
        intent
        for intent in (first_intent, second_intent)
        if intent.rule13_past_progress_required
    )
    if not required:
        return None
    if len(required) != 1:
        raise ValueError("exactly one pair intent may require Rule-13 past progress")

    intent = required[0]
    if intent.basis is not ResponsibilityBasis.RULE_13_OVERTAKING:
        raise ValueError("Rule-13 past-progress requirement is only valid for Rule-13 intent")
    if intent.role is not ResponsibilityRole.GIVE_WAY:
        raise ValueError("Rule-13 past-progress requirement must belong to give-way overtaker")
    if intent.vessel_id not in state_by_id or intent.counterpart_id not in state_by_id:
        raise ValueError("Rule-13 intent vessel IDs do not match supplied pair states")

    overtaker = state_by_id[intent.vessel_id]
    overtaken = state_by_id[intent.counterpart_id]
    ux, uy = heading_unit_vector(overtaken.heading_deg)
    ovx, ovy = overtaker.velocity
    lvx, lvy = overtaken.velocity
    rate = (ovx - lvx) * ux + (ovy - lvy) * uy
    capable = rate > _EPS and rate + _EPS >= minimum_required_rate_nm_per_min
    return Rule13ProgressAssessment(
        overtaking_vessel_id=overtaker.vessel_id,
        overtaken_vessel_id=overtaken.vessel_id,
        relative_forward_progress_rate_nm_per_min=rate,
        minimum_required_rate_nm_per_min=minimum_required_rate_nm_per_min,
        capable=capable,
    )
