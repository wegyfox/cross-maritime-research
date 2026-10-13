'''Persistent automatic environmental PAIR control cycle.

A100 composes the already-validated a92--a99 boundaries without replacing their
semantics. Starting from one already-applied environmental pair plan and the matching
already-assimilated lifecycle decision, it performs this caller-owned PAIR loop:

    applied avoidance plan
      -> a93 execution observation
      -> a94 lifecycle assimilation
      -> a95 explicit policy decision
      -> a96 replanning when required
      -> a97/a98 route handoff when responsibility clears
      -> a99 nominal environmental route continuation.

The orchestrator never discovers fleet topology, changes responsibility rules, weakens
candidate safety, ranks candidates itself, or invents route targets. Replanning still
requires a caller-supplied *fresh* a83 planning context at every REPLAN_REQUIRED
boundary. The cycle is persistent across arbitrary run_until horizons and evaluates
policy at every real a93 boundary before allowing physical time to advance farther.
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable

from .encounter_classification import EncounterClassificationConfig
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_nominal_route_session import (
    EnvironmentalNominalRouteRunResult,
    EnvironmentalNominalRouteSession,
    EnvironmentalNominalRouteSessionConfig,
    EnvironmentalNominalRouteStopReason,
)
from .environmental_pair_execution import AppliedEnvironmentalPairPlan
from .environmental_pair_execution_policy import (
    EnvironmentalPairExecutionPolicyAction,
    EnvironmentalPairExecutionPolicyDecision,
    decide_environmental_pair_execution_policy,
)
from .environmental_pair_execution_session import (
    EnvironmentalPairExecutionSession,
    EnvironmentalPairExecutionSessionConfig,
)
from .environmental_pair_lifecycle_execution import (
    EnvironmentalPairLifecycleExecutionRecord,
    EnvironmentalPairLifecycleExecutionSession,
)
from .environmental_pair_replanning import (
    EnvironmentalPairReplanningConfig,
    EnvironmentalPairReplanningResult,
    replan_environmental_pair,
)
from .environmental_planning_context import EnvironmentalAvoidancePlanningContext
from .environmental_route_handoff import (
    EnvironmentalRouteHandoffBinding,
    EnvironmentalRouteHandoffTarget,
    build_environmental_route_handoff_binding,
)
from .environmental_route_handoff_session import (
    EnvironmentalRouteHandoffSession,
    EnvironmentalRouteHandoffSessionConfig,
    EnvironmentalRouteHandoffSessionRunResult,
    EnvironmentalRouteHandoffSessionStopReason,
)
from .environmental_world import EnvironmentalWorld2D, EnvironmentalWorldSnapshot
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .responsibility import ResponsibilityContext
from .vessel_classes import VesselProfileCatalog

_EPS = 1e-10


class EnvironmentalPairControlCycleError(RuntimeError):
    '''Base error for a100 automatic pair-cycle orchestration failures.'''


class EnvironmentalPairControlReplanLimitExceeded(EnvironmentalPairControlCycleError):
    '''Raised when the explicit per-cycle replanning cap is exhausted.'''

    def __init__(self, message: str, *, attempted_replan_number: int, max_replans: int) -> None:
        super().__init__(message)
        self.attempted_replan_number = attempted_replan_number
        self.max_replans = max_replans


class EnvironmentalPairControlStage(str, Enum):
    AVOIDANCE = 'avoidance'
    ROUTE_HANDOFF = 'route_handoff'
    NOMINAL_ROUTE = 'nominal_route'
    COMPLETE = 'complete'


class EnvironmentalPairControlStopReason(str, Enum):
    ROUTE_COMPLETE = 'route_complete'
    END_TIME = 'end_time'
    COMMAND_OVERRIDDEN = 'command_overridden'


CommandWriteGuard = Callable[[EnvironmentalPairControlStage], None]


PlanningContextFactory = Callable[
    [EnvironmentalWorld2D, EnvironmentalPairExecutionPolicyDecision],
    EnvironmentalAvoidancePlanningContext,
]


@dataclass(frozen=True, slots=True)
class EnvironmentalPairControlCycleConfig:
    execution: EnvironmentalPairExecutionSessionConfig
    replanning: EnvironmentalPairReplanningConfig
    route_handoff: EnvironmentalRouteHandoffSessionConfig = EnvironmentalRouteHandoffSessionConfig()
    nominal_route: EnvironmentalNominalRouteSessionConfig = EnvironmentalNominalRouteSessionConfig()
    max_replans: int = 16
    max_route_events_per_run: int = 4096

    def __post_init__(self) -> None:
        if not isinstance(self.execution, EnvironmentalPairExecutionSessionConfig):
            raise TypeError('execution must be EnvironmentalPairExecutionSessionConfig')
        if not isinstance(self.replanning, EnvironmentalPairReplanningConfig):
            raise TypeError('replanning must be EnvironmentalPairReplanningConfig')
        if not isinstance(self.route_handoff, EnvironmentalRouteHandoffSessionConfig):
            raise TypeError('route_handoff must be EnvironmentalRouteHandoffSessionConfig')
        if not isinstance(self.nominal_route, EnvironmentalNominalRouteSessionConfig):
            raise TypeError('nominal_route must be EnvironmentalNominalRouteSessionConfig')
        if self.max_replans < 0:
            raise ValueError('max_replans must be non-negative')
        if self.max_route_events_per_run < 1:
            raise ValueError('max_route_events_per_run must be at least 1')


@dataclass(frozen=True, slots=True)
class EnvironmentalPairControlCycleResult:
    snapshot: EnvironmentalWorldSnapshot
    stage: EnvironmentalPairControlStage
    stop_reason: EnvironmentalPairControlStopReason
    latest_decision: EncounterLifecycleDecision
    current_applied: AppliedEnvironmentalPairPlan
    policy_decisions: tuple[EnvironmentalPairExecutionPolicyDecision, ...]
    replans: tuple[EnvironmentalPairReplanningResult, ...]
    handoff_binding: EnvironmentalRouteHandoffBinding | None
    handoff_result: EnvironmentalRouteHandoffSessionRunResult | None
    nominal_result: EnvironmentalNominalRouteRunResult | None

    @property
    def route_complete(self) -> bool:
        return self.stop_reason is EnvironmentalPairControlStopReason.ROUTE_COMPLETE


class EnvironmentalPairControlSession:
    '''Persist one caller-owned environmental pair from avoidance to route completion.'''

    def __init__(
        self,
        world: EnvironmentalWorld2D,
        applied: AppliedEnvironmentalPairPlan,
        lifecycle: EncounterLifecycle,
        initial_decision: EncounterLifecycleDecision,
        route_targets: tuple[EnvironmentalRouteHandoffTarget, EnvironmentalRouteHandoffTarget],
        planning_context_factory: PlanningContextFactory,
        *,
        config: EnvironmentalPairControlCycleConfig,
        profile_catalog: VesselProfileCatalog | None = None,
        uncertain: bool = False,
        responsibility_context: ResponsibilityContext | None = None,
        classification_config: EncounterClassificationConfig | None = None,
        direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
        before_command_write: CommandWriteGuard | None = None,
    ) -> None:
        if not isinstance(world, EnvironmentalWorld2D):
            raise TypeError('world must be EnvironmentalWorld2D')
        if not isinstance(applied, AppliedEnvironmentalPairPlan):
            raise TypeError('applied must be AppliedEnvironmentalPairPlan')
        if not isinstance(lifecycle, EncounterLifecycle):
            raise TypeError('lifecycle must be EncounterLifecycle')
        if not isinstance(initial_decision, EncounterLifecycleDecision):
            raise TypeError('initial_decision must be EncounterLifecycleDecision')
        if not isinstance(config, EnvironmentalPairControlCycleConfig):
            raise TypeError('config must be EnvironmentalPairControlCycleConfig')
        if not callable(planning_context_factory):
            raise TypeError('planning_context_factory must be callable')
        if len(route_targets) != 2 or any(
            not isinstance(item, EnvironmentalRouteHandoffTarget) for item in route_targets
        ):
            raise TypeError(
                'route_targets must contain exactly two EnvironmentalRouteHandoffTarget objects'
            )
        if responsibility_context is not None and not isinstance(
            responsibility_context, ResponsibilityContext
        ):
            raise TypeError('responsibility_context must be ResponsibilityContext or None')
        if before_command_write is not None and not callable(before_command_write):
            raise TypeError('before_command_write must be callable or None')

        pair_ids = (applied.plan.first_id, applied.plan.second_id)
        if tuple(item.vessel_id for item in route_targets) != pair_ids:
            raise ValueError('route target order must match applied pair order')
        if (
            abs(world.time - applied.applied_at_time) > _EPS
            or world.snapshot() != applied.snapshot_after
        ):
            raise EnvironmentalPairControlCycleError(
                'world no longer matches the supplied applied pair plan'
            )
        if {lifecycle.first_id, lifecycle.second_id} != set(pair_ids):
            raise EnvironmentalPairControlCycleError(
                'lifecycle pair does not match applied plan'
            )
        if not lifecycle.observations or lifecycle.observations[-1] != initial_decision.observation:
            raise EnvironmentalPairControlCycleError(
                'initial decision is not the lifecycle latest observation'
            )
        if abs(initial_decision.observation.time_min - applied.applied_at_time) > _EPS:
            raise EnvironmentalPairControlCycleError(
                'initial decision time does not match applied plan time'
            )

        self.world = world
        self.lifecycle = lifecycle
        self.route_targets = route_targets
        self.planning_context_factory = planning_context_factory
        self.config = config
        self.profile_catalog = profile_catalog
        self.uncertain = bool(uncertain)
        self.responsibility_context = responsibility_context
        self.classification_config = classification_config
        self.direction_basis = direction_basis
        self.before_command_write = before_command_write

        self._stage = EnvironmentalPairControlStage.AVOIDANCE
        self._terminal_reason: EnvironmentalPairControlStopReason | None = None
        self._current_applied = applied
        self._policy_history: list[EnvironmentalPairExecutionPolicyDecision] = []
        self._replans: list[EnvironmentalPairReplanningResult] = []
        self._handoff_binding: EnvironmentalRouteHandoffBinding | None = None
        self._handoff_session: EnvironmentalRouteHandoffSession | None = None
        self._handoff_result: EnvironmentalRouteHandoffSessionRunResult | None = None
        self._nominal_session: EnvironmentalNominalRouteSession | None = None
        self._nominal_result: EnvironmentalNominalRouteRunResult | None = None
        self._latest_decision = initial_decision
        self._execution_session: EnvironmentalPairExecutionSession
        self._lifecycle_execution: EnvironmentalPairLifecycleExecutionSession
        self._pending_policy: EnvironmentalPairExecutionPolicyDecision | None = None
        self._install_execution_stack(applied, initial_decision)

    @property
    def stage(self) -> EnvironmentalPairControlStage:
        return self._stage

    @property
    def policy_decisions(self) -> tuple[EnvironmentalPairExecutionPolicyDecision, ...]:
        return tuple(self._policy_history)

    @property
    def replans(self) -> tuple[EnvironmentalPairReplanningResult, ...]:
        return tuple(self._replans)

    @property
    def latest_decision(self) -> EncounterLifecycleDecision:
        return self._latest_decision

    def _guard_command_write(self, stage: EnvironmentalPairControlStage) -> None:
        if self.before_command_write is not None:
            self.before_command_write(stage)

    def _install_execution_stack(
        self,
        applied: AppliedEnvironmentalPairPlan,
        decision: EncounterLifecycleDecision,
    ) -> None:
        execution = EnvironmentalPairExecutionSession(
            self.world,
            applied,
            config=self.config.execution,
        )
        wrapped = EnvironmentalPairLifecycleExecutionSession(
            execution,
            self.lifecycle,
            decision,
            uncertain=self.uncertain,
            responsibility_context=self.responsibility_context,
            classification_config=self.classification_config,
            direction_basis=self.direction_basis,
        )
        initial_record = EnvironmentalPairLifecycleExecutionRecord(
            execution.history[0],
            decision,
        )
        policy = decide_environmental_pair_execution_policy(initial_record)
        self._execution_session = execution
        self._lifecycle_execution = wrapped
        self._current_applied = applied
        self._latest_decision = decision
        self._pending_policy = policy
        self._policy_history.append(policy)

    def _apply_pending_policy(self) -> None:
        policy = self._pending_policy
        if policy is None:
            return

        if policy.action is EnvironmentalPairExecutionPolicyAction.CONTINUE_SELECTED_COMMAND:
            self._pending_policy = None
            return

        if policy.action is EnvironmentalPairExecutionPolicyAction.REPLAN_REQUIRED:
            if len(self._replans) >= self.config.max_replans:
                raise EnvironmentalPairControlReplanLimitExceeded(
                    f'environmental pair cycle exceeded max_replans={self.config.max_replans}',
                    attempted_replan_number=len(self._replans) + 1,
                    max_replans=self.config.max_replans,
                )
            self._guard_command_write(EnvironmentalPairControlStage.AVOIDANCE)
            # Only consume the pending policy after the ownership guard has allowed
            # this writer to proceed. A102 recapture therefore leaves the local cycle
            # inspectable at the exact transfer boundary.
            self._pending_policy = None
            planning_context = self.planning_context_factory(self.world, policy)
            if not isinstance(planning_context, EnvironmentalAvoidancePlanningContext):
                raise TypeError(
                    'planning_context_factory must return EnvironmentalAvoidancePlanningContext'
                )
            replanned = replan_environmental_pair(
                self.world,
                policy,
                planning_context,
                config=self.config.replanning,
                profile_catalog=self.profile_catalog,
            )
            self._replans.append(replanned)
            self._install_execution_stack(replanned.applied, policy.source.decision)
            return

        if policy.action is EnvironmentalPairExecutionPolicyAction.HANDOFF_ELIGIBLE:
            self._pending_policy = None
            binding = build_environmental_route_handoff_binding(
                self.world,
                policy,
                self.route_targets,
                profile_catalog=self.profile_catalog,
            )
            self._handoff_binding = binding
            self._handoff_session = EnvironmentalRouteHandoffSession(
                self.world,
                binding,
                config=self.config.route_handoff,
                profile_catalog=self.profile_catalog,
                before_command_write=lambda: self._guard_command_write(
                    EnvironmentalPairControlStage.ROUTE_HANDOFF
                ),
            )
            self._stage = EnvironmentalPairControlStage.ROUTE_HANDOFF
            return

        raise AssertionError(f'unsupported a95 action: {policy.action!r}')

    def _run_avoidance_until(
        self,
        end_time: float,
    ) -> EnvironmentalPairControlStopReason | None:
        while self._stage is EnvironmentalPairControlStage.AVOIDANCE:
            if self._pending_policy is not None:
                self._apply_pending_policy()
                if self._stage is not EnvironmentalPairControlStage.AVOIDANCE:
                    return None
                continue

            if self.world.time >= end_time - _EPS:
                return EnvironmentalPairControlStopReason.END_TIME

            boundary = self._execution_session.next_boundary_time()
            if boundary is None or boundary > end_time + _EPS:
                result = self._lifecycle_execution.run_until(end_time)
                self._latest_decision = result.decision
                if result.records:
                    raise EnvironmentalPairControlCycleError(
                        'a93 emitted an unexpected record before its reported next boundary'
                    )
                return EnvironmentalPairControlStopReason.END_TIME

            result = self._lifecycle_execution.run_until(boundary)
            self._latest_decision = result.decision
            if len(result.records) != 1:
                raise EnvironmentalPairControlCycleError(
                    'automatic pair cycle requires exactly one a94 record per reported a93 boundary'
                )
            policy = decide_environmental_pair_execution_policy(result.records[0])
            self._policy_history.append(policy)
            self._pending_policy = policy

        return None

    def _run_handoff_until(
        self,
        end_time: float,
    ) -> EnvironmentalPairControlStopReason | None:
        assert self._handoff_session is not None
        result = self._handoff_session.run_until(
            end_time,
            max_events=self.config.max_route_events_per_run,
        )
        self._handoff_result = result
        if result.stop_reason is EnvironmentalRouteHandoffSessionStopReason.END_TIME:
            return EnvironmentalPairControlStopReason.END_TIME
        if result.stop_reason is EnvironmentalRouteHandoffSessionStopReason.COMMAND_OVERRIDDEN:
            self._terminal_reason = EnvironmentalPairControlStopReason.COMMAND_OVERRIDDEN
            return self._terminal_reason
        if result.stop_reason is EnvironmentalRouteHandoffSessionStopReason.NOMINAL_READY:
            assert self._handoff_binding is not None
            self._nominal_session = EnvironmentalNominalRouteSession(
                self.world,
                self._handoff_binding,
                result,
                config=self.config.nominal_route,
                profile_catalog=self.profile_catalog,
                before_command_write=lambda: self._guard_command_write(
                    EnvironmentalPairControlStage.NOMINAL_ROUTE
                ),
            )
            self._stage = EnvironmentalPairControlStage.NOMINAL_ROUTE
            return None
        raise AssertionError(f'unsupported a98 stop reason: {result.stop_reason!r}')

    def _run_nominal_until(
        self,
        end_time: float,
    ) -> EnvironmentalPairControlStopReason | None:
        assert self._nominal_session is not None
        result = self._nominal_session.run_until(
            end_time,
            max_events=self.config.max_route_events_per_run,
        )
        self._nominal_result = result
        if result.stop_reason is EnvironmentalNominalRouteStopReason.END_TIME:
            return EnvironmentalPairControlStopReason.END_TIME
        if result.stop_reason is EnvironmentalNominalRouteStopReason.COMMAND_OVERRIDDEN:
            self._terminal_reason = EnvironmentalPairControlStopReason.COMMAND_OVERRIDDEN
            return self._terminal_reason
        if result.stop_reason is EnvironmentalNominalRouteStopReason.ROUTE_COMPLETE:
            self._stage = EnvironmentalPairControlStage.COMPLETE
            self._terminal_reason = EnvironmentalPairControlStopReason.ROUTE_COMPLETE
            return self._terminal_reason
        raise AssertionError(f'unsupported a99 stop reason: {result.stop_reason!r}')

    def result(
        self,
        stop_reason: EnvironmentalPairControlStopReason,
    ) -> EnvironmentalPairControlCycleResult:
        return EnvironmentalPairControlCycleResult(
            snapshot=self.world.snapshot(),
            stage=self._stage,
            stop_reason=stop_reason,
            latest_decision=self._latest_decision,
            current_applied=self._current_applied,
            policy_decisions=self.policy_decisions,
            replans=self.replans,
            handoff_binding=self._handoff_binding,
            handoff_result=self._handoff_result,
            nominal_result=self._nominal_result,
        )

    def run_until(self, end_time: float) -> EnvironmentalPairControlCycleResult:
        if not isfinite(end_time):
            raise ValueError('end_time must be finite')
        if end_time < self.world.time - _EPS:
            raise ValueError('end_time must not move backwards')
        if self._terminal_reason is not None:
            return self.result(self._terminal_reason)

        while True:
            if self._stage is EnvironmentalPairControlStage.AVOIDANCE:
                stop = self._run_avoidance_until(end_time)
            elif self._stage is EnvironmentalPairControlStage.ROUTE_HANDOFF:
                stop = self._run_handoff_until(end_time)
            elif self._stage is EnvironmentalPairControlStage.NOMINAL_ROUTE:
                stop = self._run_nominal_until(end_time)
            elif self._stage is EnvironmentalPairControlStage.COMPLETE:
                self._terminal_reason = EnvironmentalPairControlStopReason.ROUTE_COMPLETE
                stop = self._terminal_reason
            else:
                raise AssertionError(f'unknown pair-control stage: {self._stage!r}')

            if stop is not None:
                return self.result(stop)
            # A stage transition occurred at the current timestamp. Continue into the
            # next stage within the same caller horizon rather than creating a fake gap.
