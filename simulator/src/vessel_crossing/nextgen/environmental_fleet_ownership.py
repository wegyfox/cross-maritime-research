'''Current-time environmental fleet ownership and pair-cycle recapture gate.

A101 deliberately owns no simulation clock. It converts one environmental registry
update into the existing fleet dispatch and decides whether one already-running a100
PAIR cycle may keep its local command ownership at that exact timestamp.

The gate is stage-aware:
- while a100 is in AVOIDANCE, the exact isolated PAIR owner may continue; an empty fleet
  owner set is also allowed so the pair's own lifecycle/policy can release into handoff;
- after a100 has entered ROUTE_HANDOFF or NOMINAL_ROUTE, any newly active encounter owner
  touching either controlled vessel requires fleet recapture;
- unrelated active components never steal the pair's command channels.

No command is applied, no lifecycle is updated twice, and no world time is advanced.
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .encounter_classification import EncounterClassificationConfig
from .encounter_registry import EncounterRegistry, EncounterRegistryUpdate, PairKey, canonical_pair_key
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_pair_control_cycle import EnvironmentalPairControlStage
from .environmental_pair_execution_policy import EnvironmentalPairExecutionPolicyAction
from .environmental_registry import update_environmental_registry
from .environmental_world import EnvironmentalWorld2D
from .fleet_supervision import (
    FleetCommandOwner,
    FleetSupervisionDispatch,
    FleetSupervisorBackend,
    build_fleet_supervision_dispatch,
)
from .responsibility import ResponsibilityContext


class EnvironmentalPairOwnershipAction(str, Enum):
    CONTINUE_LOCAL_CYCLE = 'continue_local_cycle'
    RECAPTURE_REQUIRED = 'recapture_required'
    CYCLE_COMPLETE = 'cycle_complete'


class EnvironmentalPairOwnershipReason(str, Enum):
    EXACT_PAIR_OWNER = 'exact_pair_owner'
    NO_ACTIVE_OWNER = 'no_active_owner'
    REPLAN_WITHOUT_ACTIVE_OWNER = 'replan_without_active_owner'
    UNRELATED_ACTIVE_COMPONENTS_ONLY = 'unrelated_active_components_only'
    CONTROLLED_VESSEL_OWNED_BY_CHANGED_COMPONENT = 'controlled_vessel_owned_by_changed_component'
    POST_AVOIDANCE_NEW_ENCOUNTER_OWNER = 'post_avoidance_new_encounter_owner'
    LOCAL_CYCLE_COMPLETE = 'local_cycle_complete'


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetOwnershipObservation:
    registry_update: EncounterRegistryUpdate
    dispatch: FleetSupervisionDispatch

    def __post_init__(self) -> None:
        if self.registry_update.time_min != self.dispatch.time_min:
            raise ValueError('registry update and dispatch must share one timestamp')

    @property
    def time_min(self) -> float:
        return self.dispatch.time_min


@dataclass(frozen=True, slots=True)
class EnvironmentalPairOwnershipDecision:
    time_min: float
    stage: EnvironmentalPairControlStage
    pair_key: PairKey
    action: EnvironmentalPairOwnershipAction
    reason: EnvironmentalPairOwnershipReason
    touching_owners: tuple[FleetCommandOwner, ...]

    @property
    def recapture_required(self) -> bool:
        return self.action is EnvironmentalPairOwnershipAction.RECAPTURE_REQUIRED


def observe_environmental_fleet_ownership(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    *,
    uncertain_pairs: Iterable[PairKey] = (),
    responsibility_context: ResponsibilityContext | None = None,
    classification_config: EncounterClassificationConfig | None = None,
    direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
) -> EnvironmentalFleetOwnershipObservation:
    '''Assimilate one environmental registry boundary and derive unique fleet owners.'''

    update = update_environmental_registry(
        registry,
        world,
        uncertain_pairs=uncertain_pairs,
        responsibility_context=responsibility_context,
        classification_config=classification_config,
        direction_basis=direction_basis,
    )
    return EnvironmentalFleetOwnershipObservation(
        registry_update=update,
        dispatch=build_fleet_supervision_dispatch(update),
    )


def assess_environmental_pair_cycle_ownership(
    stage: EnvironmentalPairControlStage,
    first_id: str,
    second_id: str,
    dispatch: FleetSupervisionDispatch,
    *,
    requested_policy_action: EnvironmentalPairExecutionPolicyAction | None = None,
) -> EnvironmentalPairOwnershipDecision:
    '''Decide whether one a100 pair-cycle may keep local ownership at this dispatch.'''

    if not isinstance(stage, EnvironmentalPairControlStage):
        raise TypeError('stage must be EnvironmentalPairControlStage')
    if not isinstance(dispatch, FleetSupervisionDispatch):
        raise TypeError('dispatch must be FleetSupervisionDispatch')
    if requested_policy_action is not None and not isinstance(
        requested_policy_action, EnvironmentalPairExecutionPolicyAction
    ):
        raise TypeError(
            'requested_policy_action must be EnvironmentalPairExecutionPolicyAction or None'
        )
    pair_key = canonical_pair_key(first_id, second_id)
    pair_set = set(pair_key)

    touching = tuple(
        owner for owner in dispatch.owners if pair_set.intersection(owner.vessel_ids)
    )

    if stage is EnvironmentalPairControlStage.COMPLETE:
        return EnvironmentalPairOwnershipDecision(
            dispatch.time_min,
            stage,
            pair_key,
            EnvironmentalPairOwnershipAction.CYCLE_COMPLETE,
            EnvironmentalPairOwnershipReason.LOCAL_CYCLE_COMPLETE,
            touching,
        )

    if stage is EnvironmentalPairControlStage.AVOIDANCE:
        exact = tuple(
            owner
            for owner in touching
            if owner.backend is FleetSupervisorBackend.PAIR
            and set(owner.vessel_ids) == pair_set
            and owner.pair_keys == (pair_key,)
        )
        if len(exact) == 1 and len(touching) == 1:
            return EnvironmentalPairOwnershipDecision(
                dispatch.time_min,
                stage,
                pair_key,
                EnvironmentalPairOwnershipAction.CONTINUE_LOCAL_CYCLE,
                EnvironmentalPairOwnershipReason.EXACT_PAIR_OWNER,
                touching,
            )
        if not touching:
            # Ownerless avoidance may preserve an already-selected command while local
            # release hysteresis settles, but it cannot authorize a *fresh* avoidance
            # plan.  REPLAN_REQUIRED therefore yields to fleet recapture/release when
            # the canonical dispatch has already dropped this PAIR owner.
            if requested_policy_action is EnvironmentalPairExecutionPolicyAction.REPLAN_REQUIRED:
                return EnvironmentalPairOwnershipDecision(
                    dispatch.time_min,
                    stage,
                    pair_key,
                    EnvironmentalPairOwnershipAction.RECAPTURE_REQUIRED,
                    EnvironmentalPairOwnershipReason.REPLAN_WITHOUT_ACTIVE_OWNER,
                    touching,
                )
            reason = (
                EnvironmentalPairOwnershipReason.UNRELATED_ACTIVE_COMPONENTS_ONLY
                if dispatch.owners
                else EnvironmentalPairOwnershipReason.NO_ACTIVE_OWNER
            )
            return EnvironmentalPairOwnershipDecision(
                dispatch.time_min,
                stage,
                pair_key,
                EnvironmentalPairOwnershipAction.CONTINUE_LOCAL_CYCLE,
                reason,
                touching,
            )
        return EnvironmentalPairOwnershipDecision(
            dispatch.time_min,
            stage,
            pair_key,
            EnvironmentalPairOwnershipAction.RECAPTURE_REQUIRED,
            EnvironmentalPairOwnershipReason.CONTROLLED_VESSEL_OWNED_BY_CHANGED_COMPONENT,
            touching,
        )

    if stage in {
        EnvironmentalPairControlStage.ROUTE_HANDOFF,
        EnvironmentalPairControlStage.NOMINAL_ROUTE,
    }:
        if not touching:
            reason = (
                EnvironmentalPairOwnershipReason.UNRELATED_ACTIVE_COMPONENTS_ONLY
                if dispatch.owners
                else EnvironmentalPairOwnershipReason.NO_ACTIVE_OWNER
            )
            return EnvironmentalPairOwnershipDecision(
                dispatch.time_min,
                stage,
                pair_key,
                EnvironmentalPairOwnershipAction.CONTINUE_LOCAL_CYCLE,
                reason,
                touching,
            )
        return EnvironmentalPairOwnershipDecision(
            dispatch.time_min,
            stage,
            pair_key,
            EnvironmentalPairOwnershipAction.RECAPTURE_REQUIRED,
            EnvironmentalPairOwnershipReason.POST_AVOIDANCE_NEW_ENCOUNTER_OWNER,
            touching,
        )

    raise AssertionError(f'unknown pair-control stage: {stage!r}')
