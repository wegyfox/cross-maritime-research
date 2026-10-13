"""Caller-supplied environmental PAIR replanning after an a95 policy boundary.

A96 closes exactly one mutation gap: when a95 returns ``REPLAN_REQUIRED`` and the
pair still has active encounter responsibility, the caller may supply a *fresh* a83
planning context for the current world timestamp and ask this module to rebuild the
existing environmental pair pipeline end-to-end:

    lifecycle decision -> intents -> a87 grid -> a88 safety -> a89 Rule-13 progress
    -> a90 arbitration -> a91 immutable plan -> a92 atomic apply.

The module does not rediscover fleet topology, create command ownership, advance the
world, run a review loop, or perform route handoff.  Ownership remains caller-supplied
through the fresh planning context.  All stages before the final a92 apply are pure, so
any validation/generation/safety/arbitration failure leaves the world unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .candidate_generation import IntentDrivenCandidatePolicy
from .environmental_candidate_generation import (
    EnvironmentalPairCommandGrid,
    generate_environmental_pair_command_grid,
)
from .environmental_pair_arbiter import (
    EnvironmentalPairArbitrationResult,
    arbitrate_environmental_pair_rows,
)
from .environmental_pair_candidate import EnvironmentalPairCandidateConfig
from .environmental_pair_execution import (
    AppliedEnvironmentalPairPlan,
    apply_environmental_pair_execution_plan,
)
from .environmental_pair_execution_policy import (
    EnvironmentalPairExecutionPolicyAction,
    EnvironmentalPairExecutionPolicyDecision,
)
from .environmental_pair_grid_evaluation import (
    EnvironmentalPairGridEvaluationBatch,
    evaluate_environmental_pair_command_grid,
)
from .environmental_pair_plan import (
    EnvironmentalPairExecutionPlan,
    build_environmental_pair_execution_plan,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .environmental_rule13_progress import (
    EnvironmentalRule13ProgressBatch,
    filter_environmental_rule13_progress,
)
from .environmental_world import EnvironmentalWorld2D
from .maneuver_intent import PairManeuverIntents, Rule13LivenessPolicy, StandOnStage, select_pair_maneuver_intents
from .pair_arbiter import PairArbitrationConfig
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalPairReplanningError(RuntimeError):
    """Base error for invalid a96 replanning requests."""


class EnvironmentalPairReplanningNotRequired(EnvironmentalPairReplanningError):
    """Raised when a95 did not request pair replanning at this boundary."""


class EnvironmentalPairReplanningUnavailable(EnvironmentalPairReplanningError):
    """Raised when pair-avoidance replanning has no active responsibility to plan."""


class EnvironmentalPairReplanningMismatch(EnvironmentalPairReplanningError):
    """Raised when supplied current planning evidence does not match the a95 pair."""


@dataclass(frozen=True, slots=True)
class EnvironmentalPairReplanningConfig:
    candidate_policy: IntentDrivenCandidatePolicy
    candidate_config: EnvironmentalPairCandidateConfig
    arbitration_config: PairArbitrationConfig
    stand_on_stage: StandOnStage = StandOnStage.NORMAL
    rule13_liveness_policy: Rule13LivenessPolicy = Rule13LivenessPolicy.REQUIRE_UNTIL_PAST
    minimum_rule13_progress_rate_nm_per_min: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_policy, IntentDrivenCandidatePolicy):
            raise TypeError("candidate_policy must be IntentDrivenCandidatePolicy")
        if not isinstance(self.candidate_config, EnvironmentalPairCandidateConfig):
            raise TypeError("candidate_config must be EnvironmentalPairCandidateConfig")
        if not isinstance(self.arbitration_config, PairArbitrationConfig):
            raise TypeError("arbitration_config must be PairArbitrationConfig")
        if not isinstance(self.stand_on_stage, StandOnStage):
            raise TypeError("stand_on_stage must be StandOnStage")
        if not isinstance(self.rule13_liveness_policy, Rule13LivenessPolicy):
            raise TypeError("rule13_liveness_policy must be Rule13LivenessPolicy")
        if (
            not isfinite(self.minimum_rule13_progress_rate_nm_per_min)
            or self.minimum_rule13_progress_rate_nm_per_min < 0.0
        ):
            raise ValueError(
                "minimum_rule13_progress_rate_nm_per_min must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class EnvironmentalPairReplanningResult:
    source_policy: EnvironmentalPairExecutionPolicyDecision
    planning_context: EnvironmentalAvoidancePlanningContext
    intents: PairManeuverIntents
    command_grid: EnvironmentalPairCommandGrid
    evaluated: EnvironmentalPairGridEvaluationBatch
    progress: EnvironmentalRule13ProgressBatch
    arbitration: EnvironmentalPairArbitrationResult
    plan: EnvironmentalPairExecutionPlan
    applied: AppliedEnvironmentalPairPlan

    @property
    def replanned_at_time(self) -> float:
        return self.applied.applied_at_time

    @property
    def selected_row_index(self) -> int:
        return self.plan.selected_row_index


def replan_environmental_pair(
    world: EnvironmentalWorld2D,
    policy: EnvironmentalPairExecutionPolicyDecision,
    planning_context: EnvironmentalAvoidancePlanningContext,
    *,
    config: EnvironmentalPairReplanningConfig,
    profile_catalog: VesselProfileCatalog | None = None,
) -> EnvironmentalPairReplanningResult:
    """Rebuild and atomically apply one current environmental PAIR avoidance plan.

    The caller owns fleet topology/ownership discovery and must supply a fresh a83
    planning context for ``world.time``.  This function deliberately does not silently
    turn ``HANDOFF_ELIGIBLE`` or ``CONTINUE_SELECTED_COMMAND`` into replanning.
    """

    if not isinstance(world, EnvironmentalWorld2D):
        raise TypeError("world must be EnvironmentalWorld2D")
    if not isinstance(policy, EnvironmentalPairExecutionPolicyDecision):
        raise TypeError("policy must be EnvironmentalPairExecutionPolicyDecision")
    if not isinstance(planning_context, EnvironmentalAvoidancePlanningContext):
        raise TypeError("planning_context must be EnvironmentalAvoidancePlanningContext")
    if not isinstance(config, EnvironmentalPairReplanningConfig):
        raise TypeError("config must be EnvironmentalPairReplanningConfig")

    if policy.action is not EnvironmentalPairExecutionPolicyAction.REPLAN_REQUIRED:
        raise EnvironmentalPairReplanningNotRequired(
            f"a95 action {policy.action.value!r} does not request pair replanning"
        )

    current_decision = policy.source.decision
    if not current_decision.responsibility_active or current_decision.effective_scaffold is None:
        raise EnvironmentalPairReplanningUnavailable(
            "pair-avoidance replanning requires active encounter responsibility"
        )
    if abs(policy.time_min - world.time) > _EPS:
        raise EnvironmentalPairReplanningMismatch(
            "a95 policy boundary timestamp does not match current environmental world time"
        )

    execution = policy.source.execution
    source_pair = (execution.first_state.vessel_id, execution.second_state.vessel_id)
    if planning_context.vessel_ids != source_pair:
        raise EnvironmentalPairReplanningMismatch(
            "fresh planning-context vessel order does not match the a95 execution pair"
        )

    # Validate all caller-supplied a83 ownership/state/drift/limits/profile evidence
    # before generating any replacement commands.
    planning_context.validate(world, profile_catalog=profile_catalog)

    intents = select_pair_maneuver_intents(
        current_decision,
        stand_on_stage=config.stand_on_stage,
        rule13_liveness_policy=config.rule13_liveness_policy,
    )
    grid = generate_environmental_pair_command_grid(
        planning_context,
        intents,
        policy=config.candidate_policy,
    )
    evaluated = evaluate_environmental_pair_command_grid(
        planning_context,
        grid,
        config=config.candidate_config,
    )
    progress = filter_environmental_rule13_progress(
        planning_context,
        evaluated,
        minimum_required_rate_nm_per_min=config.minimum_rule13_progress_rate_nm_per_min,
    )
    arbitration = arbitrate_environmental_pair_rows(
        progress,
        config=config.arbitration_config,
    )
    plan = build_environmental_pair_execution_plan(
        world,
        planning_context,
        arbitration,
        profile_catalog=profile_catalog,
    )
    applied = apply_environmental_pair_execution_plan(
        world,
        plan,
        profile_catalog=profile_catalog,
    )
    return EnvironmentalPairReplanningResult(
        source_policy=policy,
        planning_context=planning_context,
        intents=intents,
        command_grid=grid,
        evaluated=evaluated,
        progress=progress,
        arbitration=arbitration,
        plan=plan,
        applied=applied,
    )
