"""Profile-aware fleet review controller factory.

This layer packages the a44 -> a45 -> a46 path behind the existing mission controller
factory interface.  Callers still own candidate/command option generation and explicit
arbitration policy, but they no longer need to branch on PAIR versus JOINT inside every
fleet-review callback or manually rebuild profile-aware planning context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Mapping

from .avoidance import AvoidanceCandidateConfig
from .coupled_avoidance import CoupledAvoidanceConfig
from .candidate_generation import (
    IntentDrivenCandidatePolicy,
    pair_candidate_configs_from_intents,
    joint_command_options_from_intents,
)
from .dynamics import ManeuverCommand
from .encounter_registry import EncounterRegistryUpdate
from .lifecycle import LifecycleRiskBasis
from .fleet_review_session import FleetReviewController
from .fleet_supervision import FleetCommandOwner, FleetSupervisorBackend, build_fleet_supervision_dispatch
from .joint_arbiter import JointArbitrationConfig
from .joint_avoidance import JointAvoidanceConfig
from .maneuver_intent import StandOnStage
from .multi_intent import MultiIntentResolutionPolicy, resolve_component_stand_on_stages
from .owner_execution import OwnerExecutionPlan, build_owner_execution_plan
from .pair_arbiter import PairArbitrationConfig
from .planning_context import AvoidancePlanningContext, StalePlanningContext, build_avoidance_planning_context
from .rule13_progress_coordination import Rule13ProgressCoordinationPolicy
from .planning_helpers import (
    PlanningSafetyGeometry,
    plan_joint_owner_avoidance,
    plan_pair_owner_avoidance,
)
from .vessel_classes import VesselProfileCatalog
from .world import World2D, WorldSnapshot


class ControllerFactoryError(RuntimeError):
    """Base error for invalid profile-aware controller factory configuration."""


class MissingControllerPlanningProvider(ControllerFactoryError):
    """Raised when the active backend has no explicit candidate/command provider."""




class OwnerReplanPolicy(str, Enum):
    """Explicit review-time command refresh policy.

    ``ALWAYS`` preserves the historical a47-a53 behavior.
    ``FRESH_ACTION_EVIDENCE`` keeps the previously accepted absolute command through
    persistence-only lifecycle reviews and replans only when new collision-risk evidence
    or a materially different responsibility scaffold appears.
    """

    ALWAYS = "always"
    FRESH_ACTION_EVIDENCE = "fresh_action_evidence"


def _scaffold_signature(decision):
    scaffold = decision.effective_scaffold
    if scaffold is None:
        return None
    return (
        scaffold.basis,
        scaffold.first_role,
        scaffold.second_role,
        scaffold.mutual_action_direction,
    )


def _fresh_action_evidence_replan(
    previous_update: EncounterRegistryUpdate,
    current_update: EncounterRegistryUpdate,
    owner: FleetCommandOwner,
) -> bool:
    fresh_risk = {
        LifecycleRiskBasis.SNAPSHOT,
        LifecycleRiskBasis.STEADY_BEARING,
        LifecycleRiskBasis.SNAPSHOT_AND_STEADY_BEARING,
    }
    for first_id, second_id in owner.pair_keys:
        try:
            previous = previous_update.by_key(first_id, second_id).decision
            current = current_update.by_key(first_id, second_id).decision
        except KeyError:
            return True
        if current.risk_basis in fresh_risk:
            return True
        if _scaffold_signature(previous) != _scaffold_signature(current):
            return True
        if (not previous.rule13_persisted) and current.rule13_persisted:
            return True
    return False

PairCandidateConfigProvider = Callable[
    [AvoidancePlanningContext, FleetCommandOwner],
    Mapping[str, AvoidanceCandidateConfig],
]
JointCommandOptionsProvider = Callable[
    [AvoidancePlanningContext, FleetCommandOwner],
    Mapping[str, tuple[ManeuverCommand, ...]],
]
ReviewIntervalProvider = Callable[[FleetCommandOwner], float]


@dataclass(frozen=True, slots=True)
class ProfileAwareControllerPolicy:
    """Explicit planning policy shared by dynamically created fleet controllers."""

    review_interval_min: float | ReviewIntervalProvider
    coupled_config: CoupledAvoidanceConfig
    pair_arbitration_config: PairArbitrationConfig
    joint_config: JointAvoidanceConfig
    joint_arbitration_config: JointArbitrationConfig
    pair_candidate_configs: PairCandidateConfigProvider | None = None
    joint_command_options: JointCommandOptionsProvider | None = None
    candidate_generation_policy: IntentDrivenCandidatePolicy | None = None
    pair_stand_on_stage: StandOnStage = StandOnStage.NORMAL
    joint_stand_on_stages: Mapping[tuple[str, str], StandOnStage] | None = None
    multi_intent_resolution: MultiIntentResolutionPolicy = MultiIntentResolutionPolicy.STRICT
    rule13_progress_coordination: Rule13ProgressCoordinationPolicy | None = None
    safety_geometry: PlanningSafetyGeometry = PlanningSafetyGeometry.PROFILE_DOMAINS_REQUIRED
    replan_policy: OwnerReplanPolicy = OwnerReplanPolicy.ALWAYS

    def interval_for(self, owner: FleetCommandOwner) -> float:
        value = (
            self.review_interval_min(owner)
            if callable(self.review_interval_min)
            else self.review_interval_min
        )
        if not isfinite(value) or value <= 0.0:
            raise ControllerFactoryError("review interval must be finite and positive")
        return value


def build_profile_aware_controller_factory(
    world: World2D,
    *,
    profile_catalog: VesselProfileCatalog | None,
    policy: ProfileAwareControllerPolicy,
):
    """Return a ``MissionControllerFactory`` producing a46 owner execution plans."""

    if not isinstance(policy, ProfileAwareControllerPolicy):
        raise TypeError("policy must be ProfileAwareControllerPolicy")

    def factory(owner: FleetCommandOwner) -> FleetReviewController:
        interval = policy.interval_for(owner)
        owner_signature = (owner.component_id, owner.backend, owner.vessel_ids, owner.pair_keys)

        def replanner(
            snapshot: WorldSnapshot,
            update: EncounterRegistryUpdate,
            callback_owner: FleetCommandOwner,
        ) -> OwnerExecutionPlan:
            callback_signature = (
                callback_owner.component_id,
                callback_owner.backend,
                callback_owner.vessel_ids,
                callback_owner.pair_keys,
            )
            if callback_signature != owner_signature:
                raise ControllerFactoryError("controller invoked for a different command owner")
            if world.snapshot() != snapshot:
                raise StalePlanningContext(
                    "fleet review snapshot does not match current world before planning"
                )
            dispatch = build_fleet_supervision_dispatch(update)
            context = build_avoidance_planning_context(
                world,
                dispatch,
                callback_owner,
                profile_catalog=profile_catalog,
            )
            if context.planning_snapshot != snapshot:
                raise StalePlanningContext(
                    "derived planning context differs from shared fleet-review snapshot"
                )

            if callback_owner.backend is FleetSupervisorBackend.PAIR:
                if policy.pair_candidate_configs is not None:
                    candidate_configs = policy.pair_candidate_configs(context, callback_owner)
                elif policy.candidate_generation_policy is not None:
                    candidate_configs = pair_candidate_configs_from_intents(
                        context,
                        callback_owner,
                        policy=policy.candidate_generation_policy,
                        stand_on_stage=policy.pair_stand_on_stage,
                    )
                else:
                    raise MissingControllerPlanningProvider(
                        "PAIR owner requires pair_candidate_configs provider or candidate_generation_policy"
                    )
                planning = plan_pair_owner_avoidance(
                    context,
                    callback_owner,
                    candidate_configs=candidate_configs,
                    coupled_config=policy.coupled_config,
                    arbitration_config=policy.pair_arbitration_config,
                    stand_on_stage=policy.pair_stand_on_stage,
                    safety_geometry=policy.safety_geometry,
                )
            elif callback_owner.backend is FleetSupervisorBackend.JOINT:
                effective_stages = resolve_component_stand_on_stages(
                    callback_owner.component,
                    stand_on_stages=policy.joint_stand_on_stages,
                    policy=policy.multi_intent_resolution,
                )
                if policy.joint_command_options is not None:
                    command_options = policy.joint_command_options(context, callback_owner)
                elif policy.candidate_generation_policy is not None:
                    command_options = joint_command_options_from_intents(
                        context,
                        callback_owner,
                        policy=policy.candidate_generation_policy,
                        stand_on_stages=effective_stages,
                        rule13_progress_coordination=policy.rule13_progress_coordination,
                    )
                else:
                    raise MissingControllerPlanningProvider(
                        "JOINT owner requires joint_command_options provider or candidate_generation_policy"
                    )
                planning = plan_joint_owner_avoidance(
                    context,
                    callback_owner,
                    command_options=command_options,
                    joint_config=policy.joint_config,
                    arbitration_config=policy.joint_arbitration_config,
                    stand_on_stages=effective_stages,
                    safety_geometry=policy.safety_geometry,
                    rule13_progress_coordination=policy.rule13_progress_coordination,
                )
            else:  # pragma: no cover - enum exhaustiveness guard
                raise AssertionError(f"unsupported fleet backend {callback_owner.backend!r}")

            # Revalidate after option generation + physical screening + arbitration.  A
            # provider is allowed to compute, not to mutate the planning evidence.
            context.validate(world, profile_catalog=profile_catalog)
            return build_owner_execution_plan(planning, callback_owner)

        replan_decider = (
            _fresh_action_evidence_replan
            if policy.replan_policy is OwnerReplanPolicy.FRESH_ACTION_EVIDENCE
            else None
        )
        return FleetReviewController(
            component_id=owner.component_id,
            backend=owner.backend,
            review_interval_min=interval,
            replanner=replanner,
            replan_decider=replan_decider,
        )

    return factory
