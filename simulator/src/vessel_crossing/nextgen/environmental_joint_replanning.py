"""Stale-bound environmental JOINT replanning after an a110 policy boundary.

A111 closes the explicit replacement-plan gap after a110.  When synchronized a109
execution/lifecycle evidence yields ``REPLAN_REQUIRED``, the caller supplies a fresh a83
JOINT planning context for the *same controlled component*.  This module then rebuilds
exactly the existing environmental JOINT pipeline:

    current local pair decisions -> declarative component aggregation
    -> a105 semantic option generation / a104 all-pairs physical safety
    -> a106 explicit arbitration -> a107 immutable plan binding / atomic apply.

The replanner deliberately does not rediscover fleet topology, create a new command
owner, advance the world, run another execution session, or perform route handoff.
Original semantic edges whose local lifecycle has cleared remain represented by NONE
intents; they create no maneuver authority, while a104 still checks every physical pair
inside the supplied JOINT vessel set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .encounter_registry import PairKey
from .environmental_joint_arbiter import EnvironmentalJointArbitrationResult
from .environmental_joint_exact_search import (
    EnvironmentalJointExactSearchConfig,
    EnvironmentalJointExactSearchResult,
)
from .environmental_joint_execution import (
    AppliedEnvironmentalJointPlan,
    apply_environmental_joint_execution_plan,
)
from .environmental_joint_execution_policy import (
    EnvironmentalJointExecutionPolicyAction,
    EnvironmentalJointExecutionPolicyDecision,
)
from .environmental_joint_plan import (
    EnvironmentalJointExecutionPlan,
    build_environmental_joint_execution_plan,
)
from .environmental_joint_semantics import (
    EnvironmentalJointSemanticConfig,
    EnvironmentalJointSemanticGrid,
)
from .environmental_joint_selection import select_environmental_joint
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .environmental_world import EnvironmentalWorld2D
from .fleet_supervision import FleetSupervisorBackend
from .joint_arbiter import JointArbitrationConfig
from .maneuver_intent import Rule13LivenessPolicy, StandOnStage
from .multi_intent import (
    ComponentIntentAggregation,
    MultiIntentResolutionPolicy,
    aggregate_lifecycle_decisions,
    resolve_lifecycle_stand_on_stages,
)
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalJointReplanningError(RuntimeError):
    """Base error for invalid a111 JOINT replanning requests."""


class EnvironmentalJointReplanningNotRequired(EnvironmentalJointReplanningError):
    """Raised when a110 did not request JOINT replanning at this boundary."""


class EnvironmentalJointReplanningUnavailable(EnvironmentalJointReplanningError):
    """Raised when no current component responsibility remains to replan."""


class EnvironmentalJointReplanningMismatch(EnvironmentalJointReplanningError):
    """Raised when fresh caller evidence does not describe the same JOINT owner."""


@dataclass(frozen=True, slots=True)
class EnvironmentalJointReplanningConfig:
    semantic_config: EnvironmentalJointSemanticConfig
    arbitration_config: JointArbitrationConfig
    stand_on_stages: Mapping[PairKey, StandOnStage] | None = None
    multi_intent_resolution: MultiIntentResolutionPolicy = MultiIntentResolutionPolicy.STRICT
    rule13_liveness_policy: Rule13LivenessPolicy = Rule13LivenessPolicy.REQUIRE_UNTIL_PAST
    exact_search_config: EnvironmentalJointExactSearchConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_config, EnvironmentalJointSemanticConfig):
            raise TypeError("semantic_config must be EnvironmentalJointSemanticConfig")
        if not isinstance(self.arbitration_config, JointArbitrationConfig):
            raise TypeError("arbitration_config must be JointArbitrationConfig")
        if not isinstance(self.multi_intent_resolution, MultiIntentResolutionPolicy):
            raise TypeError("multi_intent_resolution must be MultiIntentResolutionPolicy")
        if not isinstance(self.rule13_liveness_policy, Rule13LivenessPolicy):
            raise TypeError("rule13_liveness_policy must be Rule13LivenessPolicy")
        if self.exact_search_config is not None and not isinstance(
            self.exact_search_config, EnvironmentalJointExactSearchConfig
        ):
            raise TypeError(
                "exact_search_config must be EnvironmentalJointExactSearchConfig or None"
            )
        if self.stand_on_stages is not None:
            for key, stage in self.stand_on_stages.items():
                if len(key) != 2:
                    raise ValueError("stand_on_stages keys must be pair keys")
                if not isinstance(stage, StandOnStage):
                    raise TypeError("stand_on_stages values must be StandOnStage")


@dataclass(frozen=True, slots=True)
class EnvironmentalJointReplanningResult:
    source_policy: EnvironmentalJointExecutionPolicyDecision
    planning_context: EnvironmentalAvoidancePlanningContext
    aggregation: ComponentIntentAggregation
    semantic_grid: EnvironmentalJointSemanticGrid | None
    exact_search: EnvironmentalJointExactSearchResult | None
    arbitration: EnvironmentalJointArbitrationResult
    plan: EnvironmentalJointExecutionPlan
    applied: AppliedEnvironmentalJointPlan

    @property
    def replanned_at_time(self) -> float:
        return self.applied.applied_at_time

    @property
    def selected_command_index(self) -> int:
        return self.plan.selected_command_index


def replan_environmental_joint(
    world: EnvironmentalWorld2D,
    policy: EnvironmentalJointExecutionPolicyDecision,
    planning_context: EnvironmentalAvoidancePlanningContext,
    *,
    config: EnvironmentalJointReplanningConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalJointReplanningResult:
    """Rebuild and atomically apply one current environmental JOINT plan.

    The caller remains responsible for current fleet topology and command ownership.  A
    fresh context is accepted only when it still names the exact vessel set and original
    semantic pair set represented by the synchronized a109/a110 boundary.  This prevents
    a local replanner from silently absorbing a newly appeared/removed encounter edge.
    """

    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(policy, EnvironmentalJointExecutionPolicyDecision):
        raise TypeError("policy must be EnvironmentalJointExecutionPolicyDecision")
    if not isinstance(planning_context, EnvironmentalAvoidancePlanningContext):
        raise TypeError("planning_context must be EnvironmentalAvoidancePlanningContext")
    if not isinstance(config, EnvironmentalJointReplanningConfig):
        raise TypeError("config must be EnvironmentalJointReplanningConfig")

    if policy.action is not EnvironmentalJointExecutionPolicyAction.REPLAN_REQUIRED:
        raise EnvironmentalJointReplanningNotRequired(
            f"a110 action {policy.action.value!r} does not request JOINT replanning"
        )
    if not policy.responsibility_active_pair_keys:
        raise EnvironmentalJointReplanningUnavailable(
            "JOINT replanning requires at least one active component responsibility"
        )
    if abs(policy.time_min - world.time) > _EPS:
        raise EnvironmentalJointReplanningMismatch(
            "a110 policy boundary timestamp does not match current environmental world time"
        )

    source_vessel_ids = tuple(item.vessel_id for item in policy.source.execution.vessels)
    source_pair_keys = tuple(item.key for item in policy.source.pair_decisions)
    if planning_context.backend is not FleetSupervisorBackend.JOINT:
        raise EnvironmentalJointReplanningMismatch("fresh planning context is not a JOINT owner")
    if planning_context.vessel_ids != source_vessel_ids:
        raise EnvironmentalJointReplanningMismatch(
            "fresh planning-context vessel order does not match the a110 JOINT execution component"
        )
    if planning_context.component_id != "|".join(source_vessel_ids):
        raise EnvironmentalJointReplanningMismatch(
            "fresh planning-context component id does not match the a110 JOINT component"
        )
    if tuple(planning_context.pair_keys) != source_pair_keys:
        raise EnvironmentalJointReplanningMismatch(
            "fresh planning-context semantic pair set does not match the a110 component"
        )

    stage_keys = set(config.stand_on_stages or {})
    unknown_stage_keys = stage_keys - set(source_pair_keys)
    if unknown_stage_keys:
        raise EnvironmentalJointReplanningMismatch(
            f"stand_on_stages contains pairs outside the a110 component: {sorted(unknown_stage_keys)}"
        )

    # Validate all caller-supplied a83 physical/ownership evidence before generating any
    # replacement commands.  Every later stage before a107 apply is pure.
    planning_context.validate(world, profile_catalog=profile_catalog)

    decisions = {item.key: item.decision for item in policy.source.pair_decisions}
    effective_stages = resolve_lifecycle_stand_on_stages(
        planning_context.component_id,
        planning_context.vessel_ids,
        decisions,
        stand_on_stages=config.stand_on_stages,
        policy=config.multi_intent_resolution,
        rule13_liveness_policy=config.rule13_liveness_policy,
    )
    aggregation = aggregate_lifecycle_decisions(
        planning_context.component_id,
        planning_context.vessel_ids,
        decisions,
        stand_on_stages=effective_stages,
        rule13_liveness_policy=config.rule13_liveness_policy,
    )
    selection = select_environmental_joint(
        planning_context,
        aggregation,
        semantic_config=config.semantic_config,
        arbitration_config=config.arbitration_config,
        exact_search_config=config.exact_search_config,
    )
    semantic_grid = selection.semantic_grid
    exact_search = selection.exact_search
    arbitration = selection.arbitration
    plan = build_environmental_joint_execution_plan(
        world,
        planning_context,
        arbitration,
        profile_catalog=profile_catalog,
    )
    applied = apply_environmental_joint_execution_plan(
        world,
        plan,
        profile_catalog=profile_catalog,
    )
    return EnvironmentalJointReplanningResult(
        policy,
        planning_context,
        aggregation,
        semantic_grid,
        exact_search,
        arbitration,
        plan,
        applied,
    )
