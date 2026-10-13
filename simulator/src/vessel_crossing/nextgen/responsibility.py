"""COLREG Section-II responsibility scaffold for the next-generation simulator.

This module deliberately does **not** decide whether collision risk exists and does not
issue manoeuvre commands.  It maps an already-classified geometric encounter to the
conditional responsibility structure that would apply to the narrow baseline of two
power-driven vessels in sight of one another, with no Rule-18 precedence or other
routing-rule override.

A later risk gate decides whether the conditional structure becomes operationally
active; later control-policy code decides what action is actually commanded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dynamics import Vessel2DState
from .encounter_classification import EncounterClassification, EncounterType, classify_encounter

_EPS = 1e-12


class ResponsibilityRole(str, Enum):
    NONE = "none"
    GIVE_WAY = "give_way"
    STAND_ON = "stand_on"
    MUTUAL_ACTION = "mutual_action"


class ResponsibilityBasis(str, Enum):
    NONE = "none"
    RULE_13_OVERTAKING = "rule_13_overtaking"
    RULE_14_HEAD_ON = "rule_14_head_on"
    RULE_15_CROSSING = "rule_15_crossing"
    MODELED_MUTUAL_KEEP_CLEAR = "modeled_mutual_keep_clear"


@dataclass(frozen=True, slots=True)
class ResponsibilityContext:
    """Scope declaration for this deliberately narrow Section-II scaffold."""

    in_sight: bool = True
    both_power_driven: bool = True
    rule18_precedence_applies: bool = False
    narrow_channel_or_tss_override: bool = False

    @property
    def supported(self) -> bool:
        return (
            self.in_sight
            and self.both_power_driven
            and not self.rule18_precedence_applies
            and not self.narrow_channel_or_tss_override
        )


@dataclass(frozen=True, slots=True)
class ResponsibilityScaffold:
    first_id: str
    second_id: str
    first_role: ResponsibilityRole
    second_role: ResponsibilityRole
    basis: ResponsibilityBasis
    encounter_type: EncounterType
    requires_collision_risk_gate: bool
    mutual_action_direction: str | None = None

    def role_for(self, vessel_id: str) -> ResponsibilityRole:
        if vessel_id == self.first_id:
            return self.first_role
        if vessel_id == self.second_id:
            return self.second_role
        raise KeyError(vessel_id)


class ResponsibilityOutOfScope(ValueError):
    """The requested encounter falls outside the assumptions of the a8 scaffold."""


def _none_ids(first_id: str, second_id: str, encounter_type: EncounterType) -> ResponsibilityScaffold:
    return ResponsibilityScaffold(
        first_id=first_id,
        second_id=second_id,
        first_role=ResponsibilityRole.NONE,
        second_role=ResponsibilityRole.NONE,
        basis=ResponsibilityBasis.NONE,
        encounter_type=encounter_type,
        requires_collision_risk_gate=False,
    )


def _is_starboard_side(relative_bearing_deg: float) -> bool:
    """True for a target strictly on the observer's starboard half-plane."""
    return relative_bearing_deg > _EPS and relative_bearing_deg < 180.0 - _EPS


def assign_responsibility_from_classification(
    first_id: str,
    second_id: str,
    *,
    classification: EncounterClassification,
    context: ResponsibilityContext | None = None,
) -> ResponsibilityScaffold:
    """Map an already-neutral classification to Rule-13/14/15 responsibility roles.

    A77 makes the actual dependency explicit: responsibility mapping needs vessel
    identities plus the neutral classification, not a particular physical state class.
    This lets environmental and recorded evidence reuse the same scaffold without
    fabricating ``Vessel2DState`` objects.
    """
    if not first_id or not second_id:
        raise ValueError("vessel ids must be non-empty")
    if first_id == second_id:
        raise ValueError("responsibility requires two distinct vessel ids")
    if not isinstance(classification, EncounterClassification):
        raise TypeError("classification must be an EncounterClassification")
    ctx = context or ResponsibilityContext()
    if not ctx.supported:
        raise ResponsibilityOutOfScope(
            "a8 responsibility scaffold supports only two power-driven vessels in sight, "
            "without Rule-18 precedence or narrow-channel/TSS override"
        )

    result = classification
    if {result.prediction.first_id, result.prediction.second_id} != {first_id, second_id}:
        raise ValueError("classification vessel ids do not match supplied vessel ids")

    encounter_type = result.encounter_type
    if encounter_type is EncounterType.OVERTAKING:
        if not result.overtaking_vessel_id or not result.overtaken_vessel_id:
            raise ValueError("overtaking classification lacks actor identities")
        if {result.overtaking_vessel_id, result.overtaken_vessel_id} != {first_id, second_id}:
            raise ValueError("overtaking actor ids do not match supplied vessel ids")
        first_role = (
            ResponsibilityRole.GIVE_WAY
            if first_id == result.overtaking_vessel_id
            else ResponsibilityRole.STAND_ON
        )
        second_role = (
            ResponsibilityRole.GIVE_WAY
            if second_id == result.overtaking_vessel_id
            else ResponsibilityRole.STAND_ON
        )
        return ResponsibilityScaffold(
            first_id,
            second_id,
            first_role,
            second_role,
            ResponsibilityBasis.RULE_13_OVERTAKING,
            encounter_type,
            requires_collision_risk_gate=False,
        )

    if encounter_type is EncounterType.HEAD_ON:
        return ResponsibilityScaffold(
            first_id,
            second_id,
            ResponsibilityRole.MUTUAL_ACTION,
            ResponsibilityRole.MUTUAL_ACTION,
            ResponsibilityBasis.RULE_14_HEAD_ON,
            encounter_type,
            requires_collision_risk_gate=True,
            mutual_action_direction="starboard",
        )

    if encounter_type is EncounterType.CROSSING:
        # Prediction bearings are ordered by prediction.first_id/second_id, which may be
        # the reverse of the supplied IDs. Resolve by identity rather than position.
        bearing_by_observer = {
            result.prediction.first_id: result.prediction.first_relative_bearing_deg,
            result.prediction.second_id: result.prediction.second_relative_bearing_deg,
        }
        first_sees_second_starboard = _is_starboard_side(bearing_by_observer[first_id])
        second_sees_first_starboard = _is_starboard_side(bearing_by_observer[second_id])
        if first_sees_second_starboard == second_sees_first_starboard:
            raise ValueError(
                "crossing classification does not yield a unique starboard-side give-way vessel"
            )
        if first_sees_second_starboard:
            first_role, second_role = ResponsibilityRole.GIVE_WAY, ResponsibilityRole.STAND_ON
        else:
            first_role, second_role = ResponsibilityRole.STAND_ON, ResponsibilityRole.GIVE_WAY
        return ResponsibilityScaffold(
            first_id,
            second_id,
            first_role,
            second_role,
            ResponsibilityBasis.RULE_15_CROSSING,
            encounter_type,
            requires_collision_risk_gate=True,
        )

    return _none_ids(first_id, second_id, encounter_type)


def assign_responsibility_scaffold(
    first: Vessel2DState,
    second: Vessel2DState,
    *,
    classification: EncounterClassification | None = None,
    context: ResponsibilityContext | None = None,
) -> ResponsibilityScaffold:
    """Historical state-based wrapper around the a77 ID+classification mapper."""
    result = classification or classify_encounter(first, second)
    return assign_responsibility_from_classification(
        first.vessel_id,
        second.vessel_id,
        classification=result,
        context=context,
    )
