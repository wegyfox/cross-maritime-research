"""Declarative COLREG maneuver intents layered above the encounter lifecycle.

Intents describe what kind of action is required/permitted by the current simplified
Rule-13/14/15/16/17 semantics.  They are deliberately **not** physical
``ManeuverCommand`` objects: no heading magnitude, speed target, execution timing, or
route-feasibility choice is made here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .lifecycle import EncounterLifecycleDecision
from .responsibility import ResponsibilityBasis, ResponsibilityRole


class StandOnStage(str, Enum):
    """Stand-on action stage used by the simplified compliance/coordination layer.

    ``MULTI_ENCOUNTER_CONFLICT`` is deliberately not a factual Rule-17 claim about
    the counterpart. It is an explicit coordination override: another active encounter
    in the same connected component already requires this stand-on vessel to maneuver,
    so literal keep-course/speed cannot be satisfied simultaneously.
    """

    NORMAL = "normal"
    GIVE_WAY_NOT_ACTING = "give_way_not_acting"
    COLLISION_NOT_AVOIDABLE_BY_GIVE_WAY_ALONE = "collision_not_avoidable_by_give_way_alone"
    MULTI_ENCOUNTER_CONFLICT = "multi_encounter_conflict"


class Rule13LivenessPolicy(str, Enum):
    """Engineering liveness policy layered below the persisted Rule-13 duty."""

    REQUIRE_UNTIL_PAST = "require_until_past"
    KEEP_CLEAR_WHEN_ALREADY_CLEAR = "keep_clear_when_already_clear"


class ManeuverIntentKind(str, Enum):
    NONE = "none"
    KEEP_WELL_CLEAR = "keep_well_clear"
    ALTER_COURSE_STARBOARD = "alter_course_starboard"
    STAND_ON_KEEP_COURSE_SPEED = "stand_on_keep_course_speed"
    STAND_ON_MAY_TAKE_AVOIDING_ACTION = "stand_on_may_take_avoiding_action"
    STAND_ON_MUST_TAKE_AVOIDING_ACTION = "stand_on_must_take_avoiding_action"


@dataclass(frozen=True, slots=True)
class ManeuverIntent:
    vessel_id: str
    counterpart_id: str
    role: ResponsibilityRole
    basis: ResponsibilityBasis
    kind: ManeuverIntentKind
    action_required: bool
    action_permitted: bool
    early_and_substantial: bool = False
    keep_well_clear: bool = False
    maintain_course_and_speed: bool = False
    avoid_crossing_ahead: bool = False
    required_course_direction: str | None = None
    avoid_port_alteration_for_target_on_port: bool = False
    rule13_past_progress_required: bool = False


@dataclass(frozen=True, slots=True)
class PairManeuverIntents:
    first: ManeuverIntent
    second: ManeuverIntent

    def for_vessel(self, vessel_id: str) -> ManeuverIntent:
        if self.first.vessel_id == vessel_id:
            return self.first
        if self.second.vessel_id == vessel_id:
            return self.second
        raise KeyError(vessel_id)


def _none(vessel_id: str, counterpart_id: str) -> ManeuverIntent:
    return ManeuverIntent(
        vessel_id=vessel_id,
        counterpart_id=counterpart_id,
        role=ResponsibilityRole.NONE,
        basis=ResponsibilityBasis.NONE,
        kind=ManeuverIntentKind.NONE,
        action_required=False,
        action_permitted=False,
    )


def _stand_on_intent(
    *,
    vessel_id: str,
    counterpart_id: str,
    basis: ResponsibilityBasis,
    stage: StandOnStage,
) -> ManeuverIntent:
    if stage is StandOnStage.NORMAL:
        return ManeuverIntent(
            vessel_id=vessel_id,
            counterpart_id=counterpart_id,
            role=ResponsibilityRole.STAND_ON,
            basis=basis,
            kind=ManeuverIntentKind.STAND_ON_KEEP_COURSE_SPEED,
            action_required=True,
            action_permitted=False,
            maintain_course_and_speed=True,
        )

    if stage in {StandOnStage.GIVE_WAY_NOT_ACTING, StandOnStage.MULTI_ENCOUNTER_CONFLICT}:
        return ManeuverIntent(
            vessel_id=vessel_id,
            counterpart_id=counterpart_id,
            role=ResponsibilityRole.STAND_ON,
            basis=basis,
            kind=ManeuverIntentKind.STAND_ON_MAY_TAKE_AVOIDING_ACTION,
            action_required=False,
            action_permitted=True,
            avoid_port_alteration_for_target_on_port=(basis is ResponsibilityBasis.RULE_15_CROSSING),
        )

    return ManeuverIntent(
        vessel_id=vessel_id,
        counterpart_id=counterpart_id,
        role=ResponsibilityRole.STAND_ON,
        basis=basis,
        kind=ManeuverIntentKind.STAND_ON_MUST_TAKE_AVOIDING_ACTION,
        action_required=True,
        action_permitted=True,
    )


def select_pair_maneuver_intents(
    decision: EncounterLifecycleDecision,
    *,
    stand_on_stage: StandOnStage = StandOnStage.NORMAL,
    rule13_liveness_policy: Rule13LivenessPolicy = Rule13LivenessPolicy.REQUIRE_UNTIL_PAST,
) -> PairManeuverIntents:
    """Translate an *active lifecycle responsibility* into declarative pair intents.

    The function consumes ``effective_scaffold`` rather than instantaneous encounter
    classification, so a maneuver does not reclassify its own responsibility away.
    """

    if not isinstance(rule13_liveness_policy, Rule13LivenessPolicy):
        raise TypeError("rule13_liveness_policy must be Rule13LivenessPolicy")
    first_id = decision.current_scaffold.first_id
    second_id = decision.current_scaffold.second_id
    scaffold = decision.effective_scaffold
    if scaffold is None:
        return PairManeuverIntents(_none(first_id, second_id), _none(second_id, first_id))

    if scaffold.basis is ResponsibilityBasis.MODELED_MUTUAL_KEEP_CLEAR:
        if stand_on_stage is not StandOnStage.NORMAL:
            raise ValueError("stand_on_stage is not applicable to modeled mutual keep-clear fallback")
        first = ManeuverIntent(
            vessel_id=scaffold.first_id,
            counterpart_id=scaffold.second_id,
            role=ResponsibilityRole.MUTUAL_ACTION,
            basis=scaffold.basis,
            kind=ManeuverIntentKind.KEEP_WELL_CLEAR,
            action_required=True,
            action_permitted=True,
            early_and_substantial=True,
            keep_well_clear=True,
        )
        second = ManeuverIntent(
            vessel_id=scaffold.second_id,
            counterpart_id=scaffold.first_id,
            role=ResponsibilityRole.MUTUAL_ACTION,
            basis=scaffold.basis,
            kind=ManeuverIntentKind.KEEP_WELL_CLEAR,
            action_required=True,
            action_permitted=True,
            early_and_substantial=True,
            keep_well_clear=True,
        )
        return PairManeuverIntents(first, second)

    if scaffold.basis is ResponsibilityBasis.RULE_14_HEAD_ON:
        if stand_on_stage is not StandOnStage.NORMAL:
            raise ValueError("stand_on_stage is not applicable to symmetric Rule-14 mutual action")
        first = ManeuverIntent(
            vessel_id=scaffold.first_id,
            counterpart_id=scaffold.second_id,
            role=ResponsibilityRole.MUTUAL_ACTION,
            basis=scaffold.basis,
            kind=ManeuverIntentKind.ALTER_COURSE_STARBOARD,
            action_required=True,
            action_permitted=True,
            required_course_direction="starboard",
        )
        second = ManeuverIntent(
            vessel_id=scaffold.second_id,
            counterpart_id=scaffold.first_id,
            role=ResponsibilityRole.MUTUAL_ACTION,
            basis=scaffold.basis,
            kind=ManeuverIntentKind.ALTER_COURSE_STARBOARD,
            action_required=True,
            action_permitted=True,
            required_course_direction="starboard",
        )
        return PairManeuverIntents(first, second)

    intents: list[ManeuverIntent] = []
    for vessel_id, counterpart_id in (
        (scaffold.first_id, scaffold.second_id),
        (scaffold.second_id, scaffold.first_id),
    ):
        role = scaffold.role_for(vessel_id)
        if role is ResponsibilityRole.GIVE_WAY:
            intents.append(
                ManeuverIntent(
                    vessel_id=vessel_id,
                    counterpart_id=counterpart_id,
                    role=role,
                    basis=scaffold.basis,
                    kind=ManeuverIntentKind.KEEP_WELL_CLEAR,
                    action_required=True,
                    action_permitted=True,
                    early_and_substantial=True,
                    keep_well_clear=True,
                    avoid_crossing_ahead=(scaffold.basis is ResponsibilityBasis.RULE_15_CROSSING),
                    rule13_past_progress_required=(
                        scaffold.basis is ResponsibilityBasis.RULE_13_OVERTAKING
                        and not decision.overtaking_past
                        and not (
                            rule13_liveness_policy is Rule13LivenessPolicy.KEEP_CLEAR_WHEN_ALREADY_CLEAR
                            and decision.overtaking_clear
                        )
                    ),
                )
            )
        elif role is ResponsibilityRole.STAND_ON:
            intents.append(
                _stand_on_intent(
                    vessel_id=vessel_id,
                    counterpart_id=counterpart_id,
                    basis=scaffold.basis,
                    stage=stand_on_stage,
                )
            )
        else:
            raise ValueError(f"unsupported active role {role!r} for basis {scaffold.basis.value}")

    return PairManeuverIntents(intents[0], intents[1])
