"""Fresh environmental dispatcher for one a113 JOINT recapture boundary.

A114 consumes the canonical fleet observation already committed by a113.  It never
re-observes the registry at the delegation timestamp.  If exactly one touching owner is
still a JOINT component, A114 rebuilds the existing environmental JOINT planning chain,
atomically installs the selected water-relative tuple, attaches a fresh a112 cycle and
wraps that cycle in a new a113 guard reusing the committed observation.

A recapture that shrinks to an isolated PAIR, splits into multiple owners, loses all
active ownership, or removes a controlled vessel remains explicit and non-mutating here.
Those cases require their dedicated transfer path rather than hidden sequential writers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .encounter_registry import EncounterRegistry, PairKey
from .environmental_fleet_guarded_joint_control import (
    EnvironmentalFleetGuardedJointConfig,
    EnvironmentalFleetGuardedJointControlSession,
    EnvironmentalFleetGuardedJointResult,
    EnvironmentalFleetGuardedJointStopReason,
)
from .environmental_joint_arbiter import EnvironmentalJointArbitrationResult
from .environmental_joint_exact_search import EnvironmentalJointExactSearchResult
from .environmental_joint_control_cycle import (
    EnvironmentalJointControlCycleConfig,
    EnvironmentalJointControlSession,
)
from .environmental_joint_execution import (
    AppliedEnvironmentalJointPlan,
    apply_environmental_joint_execution_plan,
)
from .environmental_joint_plan import (
    EnvironmentalJointExecutionPlan,
    build_environmental_joint_execution_plan,
)
from .environmental_joint_semantics import EnvironmentalJointSemanticGrid
from .environmental_joint_selection import select_environmental_joint
from .environmental_planning_context import (
    EnvironmentalAvoidancePlanningContext,
    build_environmental_avoidance_planning_context,
)
from .environmental_world import EnvironmentalWorld2D
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .multi_intent import ComponentIntentAggregation, aggregate_component_intents, resolve_component_stand_on_stages
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalFleetJointRecaptureDispatchError(RuntimeError):
    """Base error for invalid a114 JOINT recapture delegation."""


class EnvironmentalFleetJointRecaptureDispatchAction(str, Enum):
    JOINT_CYCLE_STARTED = "joint_cycle_started"
    PAIR_BACKEND_REQUIRED = "pair_backend_required"
    MULTI_OWNER_COORDINATION_REQUIRED = "multi_owner_coordination_required"
    NO_ACTIVE_OWNER = "no_active_owner"
    CONTROLLED_VESSEL_REMOVED = "controlled_vessel_removed"


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetJointRecaptureDispatchConfig:
    joint_cycle: EnvironmentalJointControlCycleConfig
    fleet_guard: EnvironmentalFleetGuardedJointConfig

    def __post_init__(self) -> None:
        if not isinstance(self.joint_cycle, EnvironmentalJointControlCycleConfig):
            raise TypeError("joint_cycle must be EnvironmentalJointControlCycleConfig")
        if not isinstance(self.fleet_guard, EnvironmentalFleetGuardedJointConfig):
            raise TypeError("fleet_guard must be EnvironmentalFleetGuardedJointConfig")


@dataclass(frozen=True, slots=True)
class EnvironmentalFreshJointBootstrap:
    owner: FleetCommandOwner
    lifecycles: dict[PairKey, EncounterLifecycle]
    decisions: dict[PairKey, EncounterLifecycleDecision]
    planning_context: EnvironmentalAvoidancePlanningContext
    aggregation: ComponentIntentAggregation
    semantic_grid: EnvironmentalJointSemanticGrid | None
    exact_search: EnvironmentalJointExactSearchResult | None
    arbitration: EnvironmentalJointArbitrationResult
    plan: EnvironmentalJointExecutionPlan
    applied: AppliedEnvironmentalJointPlan
    joint_cycle: EnvironmentalJointControlSession
    guarded_session: EnvironmentalFleetGuardedJointControlSession


@dataclass(frozen=True, slots=True)
class EnvironmentalFleetJointRecaptureDispatchResult:
    source: EnvironmentalFleetGuardedJointResult
    action: EnvironmentalFleetJointRecaptureDispatchAction
    touching_owners: tuple[FleetCommandOwner, ...]
    selected_owner: FleetCommandOwner | None = None
    joint_bootstrap: EnvironmentalFreshJointBootstrap | None = None

    @property
    def joint_started(self) -> bool:
        return self.action is EnvironmentalFleetJointRecaptureDispatchAction.JOINT_CYCLE_STARTED


def _component_decisions(owner: FleetCommandOwner) -> dict[PairKey, EncounterLifecycleDecision]:
    decisions = {item.key: item.entry.decision for item in owner.component.decisions}
    if tuple(decisions) != owner.pair_keys:
        raise EnvironmentalFleetJointRecaptureDispatchError(
            "JOINT owner admission decisions do not exactly match owner pair order"
        )
    if not all(item.responsibility_active for item in decisions.values()):
        raise EnvironmentalFleetJointRecaptureDispatchError(
            "JOINT owner contains an inactive responsibility decision"
        )
    return decisions


def _clone_component_lifecycles(
    registry: EncounterRegistry,
    owner: FleetCommandOwner,
    decisions: dict[PairKey, EncounterLifecycleDecision],
) -> dict[PairKey, EncounterLifecycle]:
    result: dict[PairKey, EncounterLifecycle] = {}
    for key in owner.pair_keys:
        lifecycle = registry.clone_lifecycle_for(*key)
        if not lifecycle.observations or lifecycle.observations[-1] != decisions[key].observation:
            raise EnvironmentalFleetJointRecaptureDispatchError(
                f"cloned registry lifecycle for {key!r} is not aligned with the committed fleet boundary"
            )
        result[key] = lifecycle
    return result


def _bootstrap_joint_owner(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    source: EnvironmentalFleetGuardedJointResult,
    owner: FleetCommandOwner,
    *,
    config: EnvironmentalFleetJointRecaptureDispatchConfig,
    profile_catalog: VesselProfileCatalog | None,
) -> EnvironmentalFreshJointBootstrap:
    recapture = source.recapture
    assert recapture is not None
    observation = recapture.fleet_observation
    dispatch = observation.dispatch
    if owner.backend is not FleetSupervisorBackend.JOINT or len(owner.vessel_ids) < 3:
        raise EnvironmentalFleetJointRecaptureDispatchError(
            "fresh JOINT bootstrap requires exactly one JOINT owner"
        )

    decisions = _component_decisions(owner)
    lifecycles = _clone_component_lifecycles(registry, owner, decisions)
    context = build_environmental_avoidance_planning_context(
        world,
        dispatch,
        owner,
        profile_catalog=profile_catalog,
    )
    replanning = config.joint_cycle.replanning
    effective_stages = resolve_component_stand_on_stages(
        owner.component,
        stand_on_stages=replanning.stand_on_stages,
        policy=replanning.multi_intent_resolution,
    )
    aggregation = aggregate_component_intents(
        owner.component,
        stand_on_stages=effective_stages,
    )
    selection = select_environmental_joint(
        context,
        aggregation,
        semantic_config=replanning.semantic_config,
        arbitration_config=replanning.arbitration_config,
        exact_search_config=replanning.exact_search_config,
    )
    semantic_grid = selection.semantic_grid
    exact_search = selection.exact_search
    arbitration = selection.arbitration
    plan = build_environmental_joint_execution_plan(
        world,
        context,
        arbitration,
        profile_catalog=profile_catalog,
    )
    applied = apply_environmental_joint_execution_plan(
        world,
        plan,
        profile_catalog=profile_catalog,
    )

    holder: dict[str, EnvironmentalFleetGuardedJointControlSession] = {}

    def planning_context_factory(current_world, _policy):
        guard = holder.get("guard")
        if guard is None:
            raise EnvironmentalFleetJointRecaptureDispatchError(
                "fresh JOINT planning factory used before a113 guard attachment"
            )
        if not guard.fleet_observations:
            raise EnvironmentalFleetJointRecaptureDispatchError(
                "a113 has no canonical fleet observation for current JOINT replanning"
            )
        latest = guard.fleet_observations[-1]
        if abs(latest.time_min - current_world.time) > _EPS:
            raise EnvironmentalFleetJointRecaptureDispatchError(
                "current JOINT replanning lacks a same-timestamp a113 fleet observation"
            )
        matches = tuple(
            item
            for item in latest.dispatch.owners
            if item.backend is FleetSupervisorBackend.JOINT
            and item.component_id == owner.component_id
            and item.vessel_ids == owner.vessel_ids
            and item.pair_keys == owner.pair_keys
        )
        if len(matches) != 1:
            raise EnvironmentalFleetJointRecaptureDispatchError(
                "a113 fleet observation no longer contains the exact delegated JOINT owner"
            )
        return build_environmental_avoidance_planning_context(
            current_world,
            latest.dispatch,
            matches[0],
            profile_catalog=profile_catalog,
        )

    joint_cycle = EnvironmentalJointControlSession(
        world,
        applied,
        lifecycles,
        decisions,
        planning_context_factory,
        config=config.joint_cycle,
        profile_catalog=profile_catalog,
    )
    guarded = EnvironmentalFleetGuardedJointControlSession(
        registry,
        joint_cycle,
        config=config.fleet_guard,
        initial_observation=observation,
    )
    holder["guard"] = guarded
    guarded.start()

    return EnvironmentalFreshJointBootstrap(
        owner,
        lifecycles,
        decisions,
        context,
        aggregation,
        semantic_grid,
        exact_search,
        arbitration,
        plan,
        applied,
        joint_cycle,
        guarded,
    )


def dispatch_environmental_joint_recapture(
    registry: EncounterRegistry,
    world: EnvironmentalWorld2D,
    source: EnvironmentalFleetGuardedJointResult,
    *,
    config: EnvironmentalFleetJointRecaptureDispatchConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalFleetJointRecaptureDispatchResult:
    """Delegate one a113 recapture without overlapping or guessing command writers."""

    if not isinstance(registry, EncounterRegistry):
        raise TypeError("registry must be EncounterRegistry")
    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(source, EnvironmentalFleetGuardedJointResult):
        raise TypeError("source must be EnvironmentalFleetGuardedJointResult")
    if not isinstance(config, EnvironmentalFleetJointRecaptureDispatchConfig):
        raise TypeError("config must be EnvironmentalFleetJointRecaptureDispatchConfig")
    if source.stop_reason is not EnvironmentalFleetGuardedJointStopReason.RECAPTURE_REQUIRED or source.recapture is None:
        raise EnvironmentalFleetJointRecaptureDispatchError(
            "a114 requires an a113 RECAPTURE_REQUIRED result"
        )
    if abs(source.snapshot.time - world.time) > _EPS or source.snapshot != world.snapshot():
        raise EnvironmentalFleetJointRecaptureDispatchError(
            "world no longer matches the a113 recapture boundary"
        )

    recapture = source.recapture
    observation = recapture.fleet_observation
    if abs(observation.time_min - world.time) > _EPS:
        raise EnvironmentalFleetJointRecaptureDispatchError(
            "fleet observation time does not match current environmental world time"
        )

    if recapture.removed_controlled_vessel_ids:
        return EnvironmentalFleetJointRecaptureDispatchResult(
            source,
            EnvironmentalFleetJointRecaptureDispatchAction.CONTROLLED_VESSEL_REMOVED,
            (),
        )

    old_ids = set(source.joint_result.current_applied.plan.vessel_ids)
    touching = tuple(owner for owner in observation.dispatch.owners if old_ids.intersection(owner.vessel_ids))
    if not touching:
        return EnvironmentalFleetJointRecaptureDispatchResult(
            source,
            EnvironmentalFleetJointRecaptureDispatchAction.NO_ACTIVE_OWNER,
            (),
        )
    if len(touching) > 1:
        return EnvironmentalFleetJointRecaptureDispatchResult(
            source,
            EnvironmentalFleetJointRecaptureDispatchAction.MULTI_OWNER_COORDINATION_REQUIRED,
            touching,
        )

    owner = touching[0]
    if owner.backend is FleetSupervisorBackend.PAIR:
        return EnvironmentalFleetJointRecaptureDispatchResult(
            source,
            EnvironmentalFleetJointRecaptureDispatchAction.PAIR_BACKEND_REQUIRED,
            touching,
            selected_owner=owner,
        )

    bootstrap = _bootstrap_joint_owner(
        registry,
        world,
        source,
        owner,
        config=config,
        profile_catalog=profile_catalog,
    )
    return EnvironmentalFleetJointRecaptureDispatchResult(
        source,
        EnvironmentalFleetJointRecaptureDispatchAction.JOINT_CYCLE_STARTED,
        touching,
        selected_owner=owner,
        joint_bootstrap=bootstrap,
    )
