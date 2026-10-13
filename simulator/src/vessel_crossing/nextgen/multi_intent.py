"""Aggregate pairwise maneuver intents inside one active encounter component.

This layer checks declarative compatibility only. It does not generate headings/speeds or
claim that a compatible intent set has a physically safe joint maneuver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from .encounter_components import ActiveEncounterComponent, EncounterComponentBatch
from .encounter_registry import PairKey, canonical_pair_key
from .supervision_admission import SupervisionAdmissionDecision
from .maneuver_intent import (
    ManeuverIntent,
    ManeuverIntentKind,
    PairManeuverIntents,
    Rule13LivenessPolicy,
    StandOnStage,
    select_pair_maneuver_intents,
)
from .lifecycle import EncounterLifecycleDecision




class MultiIntentResolutionPolicy(str, Enum):
    """Explicit policy for resolving declarative conflicts before physical planning."""

    STRICT = "strict"
    RELAX_STAND_ON_FOR_REQUIRED_AVOIDANCE = "relax_stand_on_for_required_avoidance"


class VesselIntentCompatibility(str, Enum):
    COMPATIBLE = "compatible"
    CONFLICTING = "conflicting"


class VesselIntentConflict(str, Enum):
    MAINTAIN_VS_AVOID = "maintain_course_speed_vs_required_avoiding_action"
    COURSE_DIRECTION_CONFLICT = "conflicting_required_course_directions"


@dataclass(frozen=True, slots=True)
class ComponentPairIntents:
    key: PairKey
    intents: PairManeuverIntents


@dataclass(frozen=True, slots=True)
class VesselIntentAggregate:
    vessel_id: str
    intents: tuple[ManeuverIntent, ...]
    compatibility: VesselIntentCompatibility
    conflicts: tuple[VesselIntentConflict, ...]
    required_course_direction: str | None
    maintain_course_and_speed: bool
    avoiding_action_required: bool
    keep_well_clear_targets: tuple[str, ...]
    avoid_crossing_ahead_targets: tuple[str, ...]
    early_and_substantial: bool

    @property
    def compatible(self) -> bool:
        return self.compatibility is VesselIntentCompatibility.COMPATIBLE


@dataclass(frozen=True, slots=True)
class ComponentIntentAggregation:
    component_id: str
    pair_intents: tuple[ComponentPairIntents, ...]
    vessels: tuple[VesselIntentAggregate, ...]

    @property
    def compatible(self) -> bool:
        return all(v.compatible for v in self.vessels)

    def for_vessel(self, vessel_id: str) -> VesselIntentAggregate:
        for vessel in self.vessels:
            if vessel.vessel_id == vessel_id:
                return vessel
        raise KeyError(vessel_id)


@dataclass(frozen=True, slots=True)
class ComponentIntentProjection:
    """One active encounter component paired with its declarative intent aggregate."""

    component: ActiveEncounterComponent
    aggregation: ComponentIntentAggregation


@dataclass(frozen=True, slots=True)
class ComponentIntentProjectionBatch:
    """State-independent intent projection for one component observation batch.

    The batch is intentionally declarative only. It does not resolve multi-encounter
    conflicts, generate physical candidates, arbitrate commands, or own execution.
    """

    time_min: float
    components: tuple[ComponentIntentProjection, ...]
    observation_only: tuple[SupervisionAdmissionDecision, ...]

    @property
    def compatible(self) -> bool:
        return all(item.aggregation.compatible for item in self.components)

    def for_component(self, component_id: str) -> ComponentIntentProjection:
        for item in self.components:
            if item.component.component_id == component_id:
                return item
        raise KeyError(component_id)


def project_component_intents(
    batch: EncounterComponentBatch,
    *,
    stand_on_stages: Mapping[PairKey, StandOnStage] | None = None,
) -> ComponentIntentProjectionBatch:
    """Project every active component into existing declarative maneuver intents.

    ``EncounterComponentBatch`` is already state-independent, so this function is the
    common semantic boundary for legacy-ground and environmental observation paths.
    The supplied stage map is validated against the *entire* batch before any
    component is projected. No conflict-resolution policy is applied implicitly.
    """

    if not isinstance(batch, EncounterComponentBatch):
        raise TypeError("batch must be EncounterComponentBatch")
    stage_by_pair = dict(stand_on_stages or {})
    active_pairs = {key for component in batch.components for key in component.pair_keys}
    unknown = set(stage_by_pair) - active_pairs
    if unknown:
        raise ValueError(f"stand_on_stages contains pairs outside batch: {sorted(unknown)}")

    projected: list[ComponentIntentProjection] = []
    for component in batch.components:
        local_stages = {
            key: stage_by_pair[key]
            for key in component.pair_keys
            if key in stage_by_pair
        }
        aggregation = aggregate_component_intents(
            component,
            stand_on_stages=local_stages,
        )
        projected.append(ComponentIntentProjection(component, aggregation))

    return ComponentIntentProjectionBatch(
        batch.time_min,
        tuple(projected),
        batch.observation_only,
    )


_AVOIDING_REQUIRED_KINDS = {
    ManeuverIntentKind.KEEP_WELL_CLEAR,
    ManeuverIntentKind.ALTER_COURSE_STARBOARD,
    ManeuverIntentKind.STAND_ON_MUST_TAKE_AVOIDING_ACTION,
}


def _aggregate_pair_intent_results(
    component_id: str,
    vessel_ids: tuple[str, ...],
    pair_results: tuple[ComponentPairIntents, ...],
) -> ComponentIntentAggregation:
    """Build per-vessel compatibility from already-selected pair intents."""

    by_vessel: dict[str, list[ManeuverIntent]] = {vessel_id: [] for vessel_id in vessel_ids}
    for pair_item in pair_results:
        if pair_item.intents.first.vessel_id not in by_vessel or pair_item.intents.second.vessel_id not in by_vessel:
            raise ValueError(f"pair {pair_item.key!r} contains vessel outside component")
        by_vessel[pair_item.intents.first.vessel_id].append(pair_item.intents.first)
        by_vessel[pair_item.intents.second.vessel_id].append(pair_item.intents.second)

    aggregates: list[VesselIntentAggregate] = []
    for vessel_id in vessel_ids:
        intents = tuple(sorted(by_vessel[vessel_id], key=lambda item: item.counterpart_id))
        directions = tuple(sorted({i.required_course_direction for i in intents if i.required_course_direction}))
        maintain = any(i.maintain_course_and_speed for i in intents)
        avoid_required = any(i.action_required and i.kind in _AVOIDING_REQUIRED_KINDS for i in intents)
        conflicts: list[VesselIntentConflict] = []
        if maintain and avoid_required:
            conflicts.append(VesselIntentConflict.MAINTAIN_VS_AVOID)
        if len(directions) > 1:
            conflicts.append(VesselIntentConflict.COURSE_DIRECTION_CONFLICT)
        compatibility = (
            VesselIntentCompatibility.COMPATIBLE
            if not conflicts
            else VesselIntentCompatibility.CONFLICTING
        )
        aggregates.append(
            VesselIntentAggregate(
                vessel_id=vessel_id,
                intents=intents,
                compatibility=compatibility,
                conflicts=tuple(conflicts),
                required_course_direction=directions[0] if len(directions) == 1 else None,
                maintain_course_and_speed=maintain,
                avoiding_action_required=avoid_required,
                keep_well_clear_targets=tuple(sorted(i.counterpart_id for i in intents if i.keep_well_clear)),
                avoid_crossing_ahead_targets=tuple(sorted(i.counterpart_id for i in intents if i.avoid_crossing_ahead)),
                early_and_substantial=any(i.early_and_substantial for i in intents),
            )
        )

    return ComponentIntentAggregation(
        component_id,
        tuple(sorted(pair_results, key=lambda item: item.key)),
        tuple(aggregates),
    )


def aggregate_lifecycle_decisions(
    component_id: str,
    vessel_ids: tuple[str, ...],
    decisions: Mapping[PairKey, EncounterLifecycleDecision],
    *,
    stand_on_stages: Mapping[PairKey, StandOnStage] | None = None,
    rule13_liveness_policy: Rule13LivenessPolicy = Rule13LivenessPolicy.REQUIRE_UNTIL_PAST,
) -> ComponentIntentAggregation:
    """Aggregate a caller-owned set of lifecycle decisions without a registry wrapper.

    This is useful when command ownership already belongs to a persistent controller and
    a fresh maneuver must be planned from its synchronized local lifecycle decisions.
    Inactive decisions deliberately remain as ``NONE`` intents so the original semantic
    edge set is preserved while no new duty is invented.
    """

    if not component_id or component_id != "|".join(vessel_ids):
        raise ValueError("component_id must match vessel_ids")
    if len(vessel_ids) < 2 or len(set(vessel_ids)) != len(vessel_ids):
        raise ValueError("vessel_ids must contain distinct component vessels")

    normalized: dict[PairKey, EncounterLifecycleDecision] = {}
    for raw_key, decision in decisions.items():
        if not isinstance(decision, EncounterLifecycleDecision):
            raise TypeError(f"decision for {raw_key!r} must be EncounterLifecycleDecision")
        key = canonical_pair_key(*raw_key)
        if key != raw_key:
            raise ValueError(f"decision key {raw_key!r} must be canonical")
        if key in normalized:
            raise ValueError(f"duplicate lifecycle decision for pair {key!r}")
        if set(key) - set(vessel_ids):
            raise ValueError(f"pair {key!r} contains vessel outside component")
        observed = canonical_pair_key(decision.observation.first_id, decision.observation.second_id)
        if observed != key:
            raise ValueError(f"decision observation does not match pair {key!r}")
        normalized[key] = decision

    stages = dict(stand_on_stages or {})
    unknown = set(stages) - set(normalized)
    if unknown:
        raise ValueError(f"stand_on_stages contains pairs outside decisions: {sorted(unknown)}")
    if any(not isinstance(stage, StandOnStage) for stage in stages.values()):
        raise TypeError("stand_on_stages values must be StandOnStage")

    pair_results = tuple(
        ComponentPairIntents(
            key,
            select_pair_maneuver_intents(
                normalized[key],
                stand_on_stage=stages.get(key, StandOnStage.NORMAL),
                rule13_liveness_policy=rule13_liveness_policy,
            ),
        )
        for key in sorted(normalized)
    )
    aggregation = _aggregate_pair_intent_results(component_id, vessel_ids, pair_results)

    # A retained original edge may have cleared while another edge in the same local
    # JOINT component remains active.  A vessel whose synchronized local decisions are
    # all NONE has no avoidance maneuver authority at this boundary; represent that as
    # hold-only rather than letting a downstream candidate template move it freely.
    held_vessels = tuple(
        replace(vessel, maintain_course_and_speed=True)
        if not vessel.intents or all(intent.kind is ManeuverIntentKind.NONE for intent in vessel.intents)
        else vessel
        for vessel in aggregation.vessels
    )
    return replace(aggregation, vessels=held_vessels)


def aggregate_component_intents(
    component: ActiveEncounterComponent,
    *,
    stand_on_stages: Mapping[PairKey, StandOnStage] | None = None,
    rule13_liveness_policy: Rule13LivenessPolicy = Rule13LivenessPolicy.REQUIRE_UNTIL_PAST,
) -> ComponentIntentAggregation:
    """Aggregate active pair intents and detect declarative per-vessel contradictions."""

    stage_by_pair = stand_on_stages or {}
    unknown = set(stage_by_pair) - set(component.pair_keys)
    if unknown:
        raise ValueError(f"stand_on_stages contains pairs outside component: {sorted(unknown)}")

    pair_results = tuple(
        ComponentPairIntents(
            decision.key,
            select_pair_maneuver_intents(
                decision.entry.decision,
                stand_on_stage=stage_by_pair.get(decision.key, StandOnStage.NORMAL),
                rule13_liveness_policy=rule13_liveness_policy,
            ),
        )
        for decision in component.decisions
    )
    return _aggregate_pair_intent_results(component.component_id, component.vessel_ids, pair_results)


def _resolve_stages_from_aggregation(
    initial: ComponentIntentAggregation,
    stages: dict[PairKey, StandOnStage],
    policy: MultiIntentResolutionPolicy,
) -> dict[PairKey, StandOnStage]:
    if policy is MultiIntentResolutionPolicy.STRICT:
        return stages
    for vessel in initial.vessels:
        if VesselIntentConflict.MAINTAIN_VS_AVOID not in vessel.conflicts:
            continue
        for pair_item in initial.pair_intents:
            if vessel.vessel_id not in pair_item.key:
                continue
            intent = pair_item.intents.for_vessel(vessel.vessel_id)
            if not intent.maintain_course_and_speed:
                continue
            current_stage = stages.get(pair_item.key, StandOnStage.NORMAL)
            if current_stage is StandOnStage.NORMAL:
                stages[pair_item.key] = StandOnStage.MULTI_ENCOUNTER_CONFLICT
    return stages


def resolve_component_stand_on_stages(
    component: ActiveEncounterComponent,
    *,
    stand_on_stages: Mapping[PairKey, StandOnStage] | None = None,
    policy: MultiIntentResolutionPolicy = MultiIntentResolutionPolicy.STRICT,
    rule13_liveness_policy: Rule13LivenessPolicy = Rule13LivenessPolicy.REQUIRE_UNTIL_PAST,
) -> dict[PairKey, StandOnStage]:
    """Return pair stages after one explicit multi-encounter compatibility policy.

    The relaxed policy resolves only ``MAINTAIN_VS_AVOID``: when a vessel is NORMAL
    stand-on in one pair but is simultaneously required to take avoiding action in
    another active pair, that stand-on pair is changed to
    ``MULTI_ENCOUNTER_CONFLICT``. The pairwise intent then permits maneuvering without
    asserting Rule-17 evidence about the counterpart. Course-direction conflicts and
    any conflict not removed by this transformation remain conflicting and therefore
    fail closed in candidate generation.
    """

    if not isinstance(policy, MultiIntentResolutionPolicy):
        raise TypeError("policy must be MultiIntentResolutionPolicy")
    stages = dict(stand_on_stages or {})
    unknown = set(stages) - set(component.pair_keys)
    if unknown:
        raise ValueError(f"stand_on_stages contains pairs outside component: {sorted(unknown)}")
    initial = aggregate_component_intents(
        component, stand_on_stages=stages, rule13_liveness_policy=rule13_liveness_policy
    )
    return _resolve_stages_from_aggregation(initial, stages, policy)


def resolve_lifecycle_stand_on_stages(
    component_id: str,
    vessel_ids: tuple[str, ...],
    decisions: Mapping[PairKey, EncounterLifecycleDecision],
    *,
    stand_on_stages: Mapping[PairKey, StandOnStage] | None = None,
    policy: MultiIntentResolutionPolicy = MultiIntentResolutionPolicy.STRICT,
    rule13_liveness_policy: Rule13LivenessPolicy = Rule13LivenessPolicy.REQUIRE_UNTIL_PAST,
) -> dict[PairKey, StandOnStage]:
    """Apply the same explicit multi-intent policy to controller-owned lifecycle state.

    Persistent environmental JOINT replanning no longer has the original registry
    ``ActiveEncounterComponent`` object; it owns synchronized per-pair lifecycle
    decisions instead.  This helper keeps conflict resolution identical at initial
    capture and later replans without rediscovering topology or inventing evidence.
    """

    if not isinstance(policy, MultiIntentResolutionPolicy):
        raise TypeError("policy must be MultiIntentResolutionPolicy")
    stages = dict(stand_on_stages or {})
    unknown = set(stages) - set(decisions)
    if unknown:
        raise ValueError(f"stand_on_stages contains pairs outside decisions: {sorted(unknown)}")
    initial = aggregate_lifecycle_decisions(
        component_id,
        vessel_ids,
        decisions,
        stand_on_stages=stages,
        rule13_liveness_policy=rule13_liveness_policy,
    )
    return _resolve_stages_from_aggregation(initial, stages, policy)
