'''Fresh environmental dispatcher for one a102 fleet recapture boundary.

A103 converts the canonical fleet dispatch returned by a102 into the next explicit
control action without silently reusing the historical ground-referenced JOINT backend.

For exactly one touching isolated PAIR owner it:
- clones the registry lifecycle at the current fleet boundary so fleet observations and
  local a100 observations cannot double-write one lifecycle history;
- captures fresh a83 environmental planning evidence from the current owner;
- rebuilds the existing a87--a92 pair pipeline and atomically applies the selected pair
  commands at the same timestamp;
- starts a new a100 pair cycle and wraps it in a102 using the already-produced fleet
  observation, so the registry is not observed twice at the delegation timestamp.

If the recaptured vessels belong to a JOINT component, multiple independent owners, or
no current owner, a103 returns an explicit non-mutating disposition.  JOINT execution
remains unavailable until a water-relative multi-vessel physical backend exists.
'''

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from .environmental_candidate_generation import (
    EnvironmentalPairCommandGrid,
    generate_environmental_pair_command_grid,
)
from .environmental_fleet_guarded_pair_control import (
    EnvironmentalFleetGuardedPairConfig,
    EnvironmentalFleetGuardedPairControlSession,
    EnvironmentalFleetGuardedPairResult,
    EnvironmentalFleetGuardedPairStopReason,
)
from .environmental_fleet_ownership import EnvironmentalFleetOwnershipObservation
from .environmental_pair_arbiter import (
    EnvironmentalPairArbitrationResult,
    arbitrate_environmental_pair_rows,
)
from .environmental_pair_control_cycle import (
    EnvironmentalPairControlCycleConfig,
    EnvironmentalPairControlSession,
)
from .environmental_pair_execution import (
    AppliedEnvironmentalPairPlan,
    apply_environmental_pair_execution_plan,
)
from .environmental_pair_grid_evaluation import (
    EnvironmentalPairGridEvaluationBatch,
    evaluate_environmental_pair_command_grid,
)
from .environmental_pair_plan import (
    EnvironmentalPairExecutionPlan,
    build_environmental_pair_execution_plan,
)
from .environmental_planning_context import (
    EnvironmentalAvoidancePlanningContext,
    build_environmental_avoidance_planning_context,
)
from .environmental_route_handoff import EnvironmentalRouteHandoffTarget
from .environmental_rule13_progress import (
    EnvironmentalRule13ProgressBatch,
    filter_environmental_rule13_progress,
)
from .environmental_world import EnvironmentalWorld2D
from .fleet_supervision import FleetCommandOwner, FleetSupervisionDispatch, FleetSupervisorBackend
from .maneuver_intent import PairManeuverIntents, select_pair_maneuver_intents
from .encounter_registry import EncounterRegistry
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalFleetRecaptureDispatchError(RuntimeError):
    '''Base error for invalid a103 recapture-dispatch requests.'''


class EnvironmentalFleetRecaptureDispatchAction(str, Enum):
    PAIR_CYCLE_STARTED = 'pair_cycle_started'
    ENVIRONMENTAL_JOINT_BACKEND_REQUIRED = 'environmental_joint_backend_required'
    MULTI_OWNER_COORDINATION_REQUIRED = 'multi_owner_coordination_required'
    NO_ACTIVE_OWNER = 'no_active_owner'
    CONTROLLED_VESSEL_REMOVED = 'controlled_vessel_removed'


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRecaptureDispatchConfig:
    pair_cycle: EnvironmentalPairControlCycleConfig
    fleet_guard: EnvironmentalFleetGuardedPairConfig

    def __post_init__(self) -> None:
        if not isinstance(self.pair_cycle, EnvironmentalPairControlCycleConfig):
            raise TypeError('pair_cycle must be EnvironmentalPairControlCycleConfig')
        if not isinstance(self.fleet_guard, EnvironmentalFleetGuardedPairConfig):
            raise TypeError('fleet_guard must be EnvironmentalFleetGuardedPairConfig')


@dataclass(frozen=True, slots=True)
class EnvironmentalFreshPairBootstrap:
    owner: FleetCommandOwner
    decision: EncounterLifecycleDecision
    lifecycle: EncounterLifecycle
    planning_context: EnvironmentalAvoidancePlanningContext
    intents: PairManeuverIntents
    command_grid: EnvironmentalPairCommandGrid
    evaluated: EnvironmentalPairGridEvaluationBatch
    progress: EnvironmentalRule13ProgressBatch
    arbitration: EnvironmentalPairArbitrationResult
    plan: EnvironmentalPairExecutionPlan
    applied: AppliedEnvironmentalPairPlan
    pair_cycle: EnvironmentalPairControlSession
    guarded_session: EnvironmentalFleetGuardedPairControlSession


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetRecaptureDispatchResult:
    source: EnvironmentalFleetGuardedPairResult
    action: EnvironmentalFleetRecaptureDispatchAction
    touching_owners: tuple[FleetCommandOwner, ...]
    selected_owner: FleetCommandOwner | None = None
    pair_bootstrap: EnvironmentalFreshPairBootstrap | None = None

    @property
    def pair_started(self) -> bool:
        return self.action is EnvironmentalFleetRecaptureDispatchAction.PAIR_CYCLE_STARTED



def _route_targets_for_owner(
    owner: FleetCommandOwner,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    source: EnvironmentalFleetGuardedPairResult,
) -> tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget]:
    try:
        ordered = tuple(route_targets[vessel_id] for vessel_id in owner.vessel_ids)
    except KeyError as exc:
        raise EnvironmentalFleetRecaptureDispatchError(
            f'missing route target for recaptured pair vessel {exc.args[0]!r}'
        ) from exc
    if len(ordered) != 2 or any(item.vessel_id != vessel_id for item, vessel_id in zip(ordered, owner.vessel_ids)):
        raise EnvironmentalFleetRecaptureDispatchError(
            'route target mapping does not match recaptured PAIR owner vessel ids'
        )

    latest_states = {}
    pair_result = source.pair_result
    if pair_result.nominal_result is not None:
        latest_states.update(
            (item.vessel_id, item.follower_state) for item in pair_result.nominal_result.vessels
        )
    elif pair_result.handoff_result is not None:
        latest_states.update(
            (item.vessel_id, item.follower_state) for item in pair_result.handoff_result.vessels
        )
    ordered = tuple(
        replace(item, follower_state=latest_states.get(item.vessel_id, item.follower_state))
        for item in ordered
    )
    return ordered  # type: ignore[return-value]



def _bootstrap_pair_owner(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    observation: EnvironmentalFleetOwnershipObservation,
    owner: FleetCommandOwner,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    source: EnvironmentalFleetGuardedPairResult,
    *,
    config: EnvironmentalFleetRecaptureDispatchConfig,
    profile_catalog: VesselProfileCatalog | None,
) -> EnvironmentalFreshPairBootstrap:
    dispatch = observation.dispatch
    if owner.backend is not FleetSupervisorBackend.PAIR or len(owner.vessel_ids) != 2:
        raise EnvironmentalFleetRecaptureDispatchError('fresh pair bootstrap requires one PAIR owner')
    if len(owner.component.decisions) != 1:
        raise EnvironmentalFleetRecaptureDispatchError('PAIR owner must contain exactly one admission decision')

    admission = owner.component.decisions[0]
    decision = admission.entry.decision
    if not decision.responsibility_active:
        raise EnvironmentalFleetRecaptureDispatchError('PAIR owner decision has no active responsibility')
    lifecycle = registry.clone_lifecycle_for(*owner.vessel_ids)
    if not lifecycle.observations or lifecycle.observations[-1] != decision.observation:
        raise EnvironmentalFleetRecaptureDispatchError(
            'cloned registry lifecycle is not aligned with the PAIR owner decision'
        )

    # Validate route-control admission before any a92 command mutation.  A missing or
    # mismatched route target must not leave the fleet half-transitioned into a new PAIR
    # controller whose eventual handoff cannot be represented.
    targets = _route_targets_for_owner(owner, route_targets, source)

    context = build_environmental_avoidance_planning_context(
        world,
        dispatch,
        owner,
        profile_catalog=profile_catalog,
    )
    replanning = config.pair_cycle.replanning
    intents = select_pair_maneuver_intents(
        decision,
        stand_on_stage=replanning.stand_on_stage,
    )
    grid = generate_environmental_pair_command_grid(
        context,
        intents,
        policy=replanning.candidate_policy,
    )
    evaluated = evaluate_environmental_pair_command_grid(
        context,
        grid,
        config=replanning.candidate_config,
    )
    progress = filter_environmental_rule13_progress(
        context,
        evaluated,
        minimum_required_rate_nm_per_min=replanning.minimum_rule13_progress_rate_nm_per_min,
    )
    arbitration = arbitrate_environmental_pair_rows(
        progress,
        config=replanning.arbitration_config,
    )
    plan = build_environmental_pair_execution_plan(
        world,
        context,
        arbitration,
        profile_catalog=profile_catalog,
    )
    applied = apply_environmental_pair_execution_plan(
        world,
        plan,
        profile_catalog=profile_catalog,
    )

    # The a102 pre-write ownership guard is the current-time topology authority.  Once
    # it has admitted the same isolated pair, a100 replanning needs only fresh physical
    # a83 evidence for the already-owned pair; re-observing the fleet registry here
    # would create a duplicate same-timestamp registry update.
    def planning_context_factory(current_world, _policy):
        current_dispatch = replace(dispatch, time_min=current_world.time)
        return build_environmental_avoidance_planning_context(
            current_world,
            current_dispatch,
            owner,
            profile_catalog=profile_catalog,
        )

    pair_cycle = EnvironmentalPairControlSession(
        world,
        applied,
        lifecycle,
        decision,
        targets,
        planning_context_factory,
        config=config.pair_cycle,
        profile_catalog=profile_catalog,
    )
    guarded = EnvironmentalFleetGuardedPairControlSession(
        registry,
        pair_cycle,
        config=config.fleet_guard,
        initial_observation=observation,
    )
    guarded.start()
    return EnvironmentalFreshPairBootstrap(
        owner=owner,
        decision=decision,
        lifecycle=lifecycle,
        planning_context=context,
        intents=intents,
        command_grid=grid,
        evaluated=evaluated,
        progress=progress,
        arbitration=arbitration,
        plan=plan,
        applied=applied,
        pair_cycle=pair_cycle,
        guarded_session=guarded,
    )



def dispatch_environmental_fleet_recapture(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    source: EnvironmentalFleetGuardedPairResult,
    route_targets: Mapping[str, EnvironmentalRouteHandoffTarget],
    *,
    config: EnvironmentalFleetRecaptureDispatchConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalFleetRecaptureDispatchResult:
    '''Dispatch one a102 ``RECAPTURE_REQUIRED`` result without guessing a JOINT backend.'''

    if not isinstance(registry, EncounterRegistry):
        raise TypeError('registry must be EncounterRegistry')
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError('world must be EnvironmentalWorld2D')
    if not isinstance(source, EnvironmentalFleetGuardedPairResult):
        raise TypeError('source must be EnvironmentalFleetGuardedPairResult')
    if not isinstance(config, EnvironmentalFleetRecaptureDispatchConfig):
        raise TypeError('config must be EnvironmentalFleetRecaptureDispatchConfig')
    if source.stop_reason is not EnvironmentalFleetGuardedPairStopReason.RECAPTURE_REQUIRED or source.recapture is None:
        raise EnvironmentalFleetRecaptureDispatchError('a103 requires an a102 RECAPTURE_REQUIRED result')
    if abs(source.snapshot.time - world.time) > _EPS or source.snapshot != world.snapshot():
        raise EnvironmentalFleetRecaptureDispatchError('world no longer matches the a102 recapture boundary')

    recapture = source.recapture
    observation = recapture.fleet_observation
    if abs(observation.time_min - world.time) > _EPS:
        raise EnvironmentalFleetRecaptureDispatchError('fleet observation time does not match current world time')

    old_pair = {
        source.pair_result.current_applied.plan.first_id,
        source.pair_result.current_applied.plan.second_id,
    }
    if recapture.removed_controlled_vessel_ids:
        return EnvironmentalFleetRecaptureDispatchResult(
            source=source,
            action=EnvironmentalFleetRecaptureDispatchAction.CONTROLLED_VESSEL_REMOVED,
            touching_owners=(),
        )

    touching = tuple(
        owner for owner in observation.dispatch.owners if old_pair.intersection(owner.vessel_ids)
    )
    if not touching:
        return EnvironmentalFleetRecaptureDispatchResult(
            source=source,
            action=EnvironmentalFleetRecaptureDispatchAction.NO_ACTIVE_OWNER,
            touching_owners=(),
        )
    if len(touching) > 1:
        return EnvironmentalFleetRecaptureDispatchResult(
            source=source,
            action=EnvironmentalFleetRecaptureDispatchAction.MULTI_OWNER_COORDINATION_REQUIRED,
            touching_owners=touching,
        )

    owner = touching[0]
    if owner.backend is FleetSupervisorBackend.JOINT:
        return EnvironmentalFleetRecaptureDispatchResult(
            source=source,
            action=EnvironmentalFleetRecaptureDispatchAction.ENVIRONMENTAL_JOINT_BACKEND_REQUIRED,
            touching_owners=touching,
            selected_owner=owner,
        )

    # The current fleet observation was already committed by a102. Reuse it as the
    # a102 cache for the newly delegated local pair cycle so registry time is advanced
    # exactly once at this physical timestamp.
    bootstrap = _bootstrap_pair_owner(
        registry,
        world,
        observation,
        owner,
        route_targets,
        source,
        config=config,
        profile_catalog=profile_catalog,
    )
    return EnvironmentalFleetRecaptureDispatchResult(
        source=source,
        action=EnvironmentalFleetRecaptureDispatchAction.PAIR_CYCLE_STARTED,
        touching_owners=touching,
        selected_owner=owner,
        pair_bootstrap=bootstrap,
    )
