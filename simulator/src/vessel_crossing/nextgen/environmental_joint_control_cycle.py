"""Persistent automatic environmental JOINT avoidance/replanning cycle.

A112 composes a108--a111 without changing their semantics:

    applied a107 JOINT plan
      -> a108 physical execution boundaries
      -> a109 synchronized component-pair lifecycle assimilation
      -> a110 pure continue/replan/handoff policy
      -> a111 stale-bound same-owner JOINT replanning when required
      -> fresh a108/a109 stack on the replacement plan.

The cycle intentionally stops at local ``HANDOFF_ELIGIBLE``.  Fleet topology recapture,
route ownership and nominal-route continuation remain higher-level work.  Every replan
still requires a caller-supplied fresh a83 planning context; this orchestrator never
rediscovers ownership itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Iterable, Mapping

from .encounter_classification import EncounterClassificationConfig
from .encounter_registry import PairKey, canonical_pair_key
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_joint_execution import AppliedEnvironmentalJointPlan
from .environmental_joint_execution_policy import (
    EnvironmentalJointExecutionPolicyAction,
    EnvironmentalJointExecutionPolicyDecision,
    decide_environmental_joint_execution_policy,
)
from .environmental_joint_execution_session import (
    EnvironmentalJointExecutionSession,
    EnvironmentalJointExecutionSessionConfig,
)
from .environmental_joint_lifecycle_execution import (
    EnvironmentalJointLifecycleExecutionRecord,
    EnvironmentalJointLifecycleExecutionSession,
    EnvironmentalJointPairLifecycleDecision,
)
from .environmental_joint_replanning import (
    EnvironmentalJointReplanningConfig,
    EnvironmentalJointReplanningResult,
    replan_environmental_joint,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .responsibility import ResponsibilityContext
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalJointControlCycleError(RuntimeError):
    """Base error for a112 JOINT-cycle orchestration failures."""


class EnvironmentalJointControlReplanLimitExceeded(EnvironmentalJointControlCycleError):
    """Raised when the explicit same-owner JOINT replanning cap is exhausted."""

    def __init__(self, message: str, *, attempted_replan_number: int, max_replans: int) -> None:
        super().__init__(message)
        self.attempted_replan_number = attempted_replan_number
        self.max_replans = max_replans


class EnvironmentalJointControlStage(str, Enum):
    AVOIDANCE = "avoidance"
    HANDOFF_ELIGIBLE = "handoff_eligible"


class EnvironmentalJointControlStopReason(str, Enum):
    HANDOFF_ELIGIBLE = "handoff_eligible"
    END_TIME = "end_time"


JointPlanningContextFactory = Callable[
    [EnvironmentalWorld2D, EnvironmentalJointExecutionPolicyDecision],
    EnvironmentalAvoidancePlanningContext,
]
JointPolicyBoundaryGuard = Callable[[EnvironmentalJointExecutionPolicyDecision], None]


@dataclass(frozen=True, slots=True)
class EnvironmentalJointControlCycleConfig:
    execution: EnvironmentalJointExecutionSessionConfig
    replanning: EnvironmentalJointReplanningConfig
    max_replans: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.execution, EnvironmentalJointExecutionSessionConfig):
            raise TypeError("execution must be EnvironmentalJointExecutionSessionConfig")
        if not isinstance(self.replanning, EnvironmentalJointReplanningConfig):
            raise TypeError("replanning must be EnvironmentalJointReplanningConfig")
        if self.max_replans < 0:
            raise ValueError("max_replans must be non-negative")


@dataclass(frozen=True, slots=True)
class EnvironmentalJointControlCycleResult:
    snapshot: EnvironmentalWorldSnapshot
    stage: EnvironmentalJointControlStage
    stop_reason: EnvironmentalJointControlStopReason
    latest_pair_decisions: tuple[EnvironmentalJointPairLifecycleDecision, ...]
    current_applied: AppliedEnvironmentalJointPlan
    policy_decisions: tuple[EnvironmentalJointExecutionPolicyDecision, ...]
    replans: tuple[EnvironmentalJointReplanningResult, ...]

    @property
    def responsibility_active_pair_keys(self) -> tuple[PairKey, ...]:
        return tuple(
            item.key for item in self.latest_pair_decisions if item.decision.responsibility_active
        )


class EnvironmentalJointControlSession:
    """Persist one caller-owned environmental JOINT component until local handoff eligibility."""

    def __init__(
        self,
        world: EnvironmentalWorld2D,
        applied: AppliedEnvironmentalJointPlan,
        lifecycles: Mapping[PairKey, EncounterLifecycle],
        initial_decisions: Mapping[PairKey, EncounterLifecycleDecision],
        planning_context_factory: JointPlanningContextFactory,
        *,
        config: EnvironmentalJointControlCycleConfig,
        profile_catalog: VesselProfileCatalog | None = None,
        uncertain_pairs: Iterable[PairKey] = (),
        responsibility_context: ResponsibilityContext | None = None,
        classification_config: EncounterClassificationConfig | None = None,
        direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
        before_policy_action: JointPolicyBoundaryGuard | None = None,
    ) -> None:
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError("world must be EnvironmentalWorld2D")
        if not isinstance(applied, AppliedEnvironmentalJointPlan):
            raise TypeError("applied must be AppliedEnvironmentalJointPlan")
        if not isinstance(config, EnvironmentalJointControlCycleConfig):
            raise TypeError("config must be EnvironmentalJointControlCycleConfig")
        if not callable(planning_context_factory):
            raise TypeError("planning_context_factory must be callable")
        if responsibility_context is not None and not isinstance(
            responsibility_context, ResponsibilityContext
        ):
            raise TypeError("responsibility_context must be ResponsibilityContext or None")
        if before_policy_action is not None and not callable(before_policy_action):
            raise TypeError("before_policy_action must be callable or None")

        self.world = world
        self.lifecycles = dict(lifecycles)
        self.planning_context_factory = planning_context_factory
        self.config = config
        self.profile_catalog = profile_catalog
        self.uncertain_pairs = tuple(canonical_pair_key(*key) for key in uncertain_pairs)
        self.responsibility_context = responsibility_context
        self.classification_config = classification_config
        self.direction_basis = direction_basis
        self.before_policy_action = before_policy_action

        self._stage = EnvironmentalJointControlStage.AVOIDANCE
        self._current_applied = applied
        self._policy_history: list[EnvironmentalJointExecutionPolicyDecision] = []
        self._replans: list[EnvironmentalJointReplanningResult] = []
        self._pending_policy: EnvironmentalJointExecutionPolicyDecision | None = None
        self._execution_session: EnvironmentalJointExecutionSession
        self._lifecycle_execution: EnvironmentalJointLifecycleExecutionSession
        self._latest_pair_decisions: tuple[EnvironmentalJointPairLifecycleDecision, ...] = ()

        self._install_execution_stack(applied, initial_decisions)

    @property
    def stage(self) -> EnvironmentalJointControlStage:
        return self._stage

    @property
    def current_applied(self) -> AppliedEnvironmentalJointPlan:
        return self._current_applied

    @property
    def policy_decisions(self) -> tuple[EnvironmentalJointExecutionPolicyDecision, ...]:
        return tuple(self._policy_history)

    @property
    def replans(self) -> tuple[EnvironmentalJointReplanningResult, ...]:
        return tuple(self._replans)

    @property
    def latest_pair_decisions(self) -> tuple[EnvironmentalJointPairLifecycleDecision, ...]:
        return self._latest_pair_decisions

    def latest_decision_for(self, first_id: str, second_id: str) -> EncounterLifecycleDecision:
        key = canonical_pair_key(first_id, second_id)
        for item in self._latest_pair_decisions:
            if item.key == key:
                return item.decision
        raise KeyError(key)

    @staticmethod
    def _decision_mapping(
        decisions: Mapping[PairKey, EncounterLifecycleDecision]
        | tuple[EnvironmentalJointPairLifecycleDecision, ...],
    ) -> dict[PairKey, EncounterLifecycleDecision]:
        if isinstance(decisions, tuple):
            return {item.key: item.decision for item in decisions}
        return dict(decisions)

    def _install_execution_stack(
        self,
        applied: AppliedEnvironmentalJointPlan,
        decisions: Mapping[PairKey, EncounterLifecycleDecision]
        | tuple[EnvironmentalJointPairLifecycleDecision, ...],
    ) -> None:
        decision_map = self._decision_mapping(decisions)
        execution = EnvironmentalJointExecutionSession(
            self.world,
            applied,
            config=self.config.execution,
        )
        wrapped = EnvironmentalJointLifecycleExecutionSession(
            execution,
            self.lifecycles,
            decision_map,
            uncertain_pairs=self.uncertain_pairs,
            responsibility_context=self.responsibility_context,
            classification_config=self.classification_config,
            direction_basis=self.direction_basis,
        )
        initial_pairs = tuple(
            EnvironmentalJointPairLifecycleDecision(key, decision_map[key])
            for key in wrapped.pair_keys
        )
        initial_record = EnvironmentalJointLifecycleExecutionRecord(
            execution.history[0],
            initial_pairs,
        )
        policy = decide_environmental_joint_execution_policy(initial_record)

        self._execution_session = execution
        self._lifecycle_execution = wrapped
        self._current_applied = applied
        self._latest_pair_decisions = initial_pairs
        self._pending_policy = policy
        self._policy_history.append(policy)

    def _apply_pending_policy(self) -> None:
        policy = self._pending_policy
        if policy is None:
            return

        # A113 may install a current-time fleet/topology guard here.  The guard runs for
        # every true a110 boundary, including CONTINUE and local HANDOFF eligibility, so
        # a same-timestamp component change cannot be crossed before fleet ownership is
        # revalidated.  It runs before any a111 command write or local stage transition.
        if self.before_policy_action is not None:
            self.before_policy_action(policy)

        if policy.action is EnvironmentalJointExecutionPolicyAction.CONTINUE_SELECTED_COMMAND:
            self._pending_policy = None
            return

        if policy.action is EnvironmentalJointExecutionPolicyAction.HANDOFF_ELIGIBLE:
            self._pending_policy = None
            self._stage = EnvironmentalJointControlStage.HANDOFF_ELIGIBLE
            return

        if policy.action is EnvironmentalJointExecutionPolicyAction.REPLAN_REQUIRED:
            if len(self._replans) >= self.config.max_replans:
                raise EnvironmentalJointControlReplanLimitExceeded(
                    f"environmental JOINT cycle exceeded max_replans={self.config.max_replans}",
                    attempted_replan_number=len(self._replans) + 1,
                    max_replans=self.config.max_replans,
                )
            planning_context = self.planning_context_factory(self.world, policy)
            if not isinstance(planning_context, EnvironmentalAvoidancePlanningContext):
                raise TypeError(
                    "planning_context_factory must return EnvironmentalAvoidancePlanningContext"
                )
            # Do not consume this boundary until the complete a111 transaction succeeds.
            replanned = replan_environmental_joint(
                self.world,
                policy,
                planning_context,
                config=self.config.replanning,
                profile_catalog=self.profile_catalog,
            )
            self._pending_policy = None
            self._replans.append(replanned)
            self._install_execution_stack(replanned.applied, policy.source.pair_decisions)
            return

        raise AssertionError(f"unsupported a110 action: {policy.action!r}")

    def result(
        self,
        stop_reason: EnvironmentalJointControlStopReason,
    ) -> EnvironmentalJointControlCycleResult:
        return EnvironmentalJointControlCycleResult(
            self.world.snapshot(),
            self._stage,
            stop_reason,
            self._latest_pair_decisions,
            self._current_applied,
            self.policy_decisions,
            self.replans,
        )

    def run_until(self, end_time: float) -> EnvironmentalJointControlCycleResult:
        if not isfinite(end_time):
            raise ValueError("end_time must be finite")
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")

        while True:
            if self._stage is EnvironmentalJointControlStage.HANDOFF_ELIGIBLE:
                return self.result(EnvironmentalJointControlStopReason.HANDOFF_ELIGIBLE)

            if self._pending_policy is not None:
                self._apply_pending_policy()
                if self._stage is EnvironmentalJointControlStage.HANDOFF_ELIGIBLE:
                    return self.result(EnvironmentalJointControlStopReason.HANDOFF_ELIGIBLE)
                # A successful replan can install a new plan at the same timestamp;
                # evaluate its initial APPLIED policy before advancing physical time.
                continue

            if self.world.time >= end_time - _EPS:
                return self.result(EnvironmentalJointControlStopReason.END_TIME)

            boundary = self._execution_session.next_boundary_time()
            if boundary is None or boundary > end_time + _EPS:
                result = self._lifecycle_execution.run_until(end_time)
                self._latest_pair_decisions = result.pair_decisions
                if result.records:
                    raise EnvironmentalJointControlCycleError(
                        "a108 emitted an unexpected record before its reported next boundary"
                    )
                return self.result(EnvironmentalJointControlStopReason.END_TIME)

            result = self._lifecycle_execution.run_until(boundary)
            self._latest_pair_decisions = result.pair_decisions
            if len(result.records) != 1:
                raise EnvironmentalJointControlCycleError(
                    "automatic JOINT cycle requires exactly one a109 record per reported a108 boundary"
                )
            policy = decide_environmental_joint_execution_policy(result.records[0])
            self._policy_history.append(policy)
            self._pending_policy = policy
