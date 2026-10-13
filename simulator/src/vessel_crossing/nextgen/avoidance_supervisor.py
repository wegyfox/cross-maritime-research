"""Event-oriented execution supervisor for accepted two-vessel avoidance plans.

The a12/a13 layers are safety screens; they do not mutate ``World2D``.  This module is
the explicit bridge from an externally chosen *accepted* coupled plan to physical world
execution and encounter-lifecycle re-observation.

The supervisor remains policy-light.  A caller supplies a replanner callback that may
use any explicit candidate preference/arbitration policy.  The supervisor only:

1. rejects stale or unsafe plans;
2. applies both commands at the current world time;
3. advances to the next explicit review event (never a hidden controller tick);
4. updates the persisted encounter lifecycle from the new physical states; and
5. asks the caller to replan while responsibility remains active.

When the lifecycle clears, the result exposes ``handoff_ready=True`` so route recovery
or another upstream navigator can resume control in a separate layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .coupled_avoidance import CoupledAvoidanceEvaluation
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .responsibility import ResponsibilityContext
from .world import World2D, WorldSnapshot

_EPS = 1e-10


class AvoidanceExecutionError(RuntimeError):
    """Base class for fail-closed avoidance execution errors."""


class StaleAvoidancePlan(AvoidanceExecutionError):
    """Raised when a plan is applied after the world has advanced from its planning time."""


class UnsafeAvoidancePlan(AvoidanceExecutionError):
    """Raised when a rejected coupled evaluation is submitted for execution."""


class NoAvoidanceExecutionPlan(AvoidanceExecutionError):
    """Raised when responsibility is active but the external replanner returns no plan."""


@dataclass(frozen=True, slots=True)
class AvoidanceExecutionConfig:
    review_interval_min: float
    max_reviews: int = 64

    def __post_init__(self) -> None:
        if self.review_interval_min <= 0.0:
            raise ValueError("review_interval_min must be positive")
        if self.max_reviews < 1:
            raise ValueError("max_reviews must be at least 1")


@dataclass(frozen=True, slots=True)
class PairExecutionPlan:
    first_id: str
    second_id: str
    planned_at_time: float
    evaluation: CoupledAvoidanceEvaluation

    def __post_init__(self) -> None:
        if not self.first_id or not self.second_id or self.first_id == self.second_id:
            raise ValueError("execution plan requires two distinct non-empty vessel ids")
        if self.planned_at_time < 0.0:
            raise ValueError("planned_at_time must be non-negative")


class AvoidanceStopReason(str, Enum):
    RESPONSIBILITY_CLEAR = "responsibility_clear"
    END_TIME = "end_time"


@dataclass(frozen=True, slots=True)
class AvoidanceExecutionStep:
    start_time: float
    review_time: float
    plan: PairExecutionPlan
    decision_before: EncounterLifecycleDecision
    decision_after: EncounterLifecycleDecision
    snapshot_after: WorldSnapshot


@dataclass(frozen=True, slots=True)
class AvoidanceExecutionResult:
    snapshot: WorldSnapshot
    decision: EncounterLifecycleDecision
    steps: tuple[AvoidanceExecutionStep, ...]
    stop_reason: AvoidanceStopReason

    @property
    def handoff_ready(self) -> bool:
        return self.stop_reason is AvoidanceStopReason.RESPONSIBILITY_CLEAR


Replanner = Callable[[WorldSnapshot, EncounterLifecycleDecision], PairExecutionPlan | None]


class AvoidanceExecutionSupervisor:
    """Execute accepted pair plans and re-observe at explicit review events."""

    def __init__(
        self,
        world: World2D,
        lifecycle: EncounterLifecycle,
        *,
        config: AvoidanceExecutionConfig,
    ) -> None:
        self.world = world
        self.lifecycle = lifecycle
        self.config = config
        # Force early validation that both lifecycle vessels actually exist in the world.
        snapshot = world.snapshot()
        snapshot.by_id(lifecycle.first_id)
        snapshot.by_id(lifecycle.second_id)

    def _validate_decision_now(self, decision: EncounterLifecycleDecision) -> None:
        if abs(decision.observation.time_min - self.world.time) > _EPS:
            raise StaleAvoidancePlan(
                "initial/review lifecycle decision time does not match current world time"
            )
        if decision.observation.first_id != self.lifecycle.first_id:
            raise ValueError("decision first_id does not match supervisor lifecycle")
        if decision.observation.second_id != self.lifecycle.second_id:
            raise ValueError("decision second_id does not match supervisor lifecycle")

    def apply_plan(self, plan: PairExecutionPlan) -> None:
        """Apply one accepted pair plan immediately at the current world time."""
        if abs(plan.planned_at_time - self.world.time) > _EPS:
            raise StaleAvoidancePlan(
                f"plan was built at {plan.planned_at_time}, world is at {self.world.time}"
            )
        if plan.first_id != self.lifecycle.first_id or plan.second_id != self.lifecycle.second_id:
            raise ValueError("execution plan vessel order does not match supervisor lifecycle")
        if not plan.evaluation.accepted:
            raise UnsafeAvoidancePlan("coupled avoidance evaluation is not accepted")

        self.world.schedule_command(
            at=self.world.time,
            vessel_id=self.lifecycle.first_id,
            command=plan.evaluation.first_candidate.command,
        )
        self.world.schedule_command(
            at=self.world.time,
            vessel_id=self.lifecycle.second_id,
            command=plan.evaluation.second_candidate.command,
        )
        # Process same-time command events now.  No physical time elapses here.
        self.world.run_until(self.world.time)

    def _observe_now(
        self,
        *,
        uncertain: bool,
        responsibility_context: ResponsibilityContext | None,
    ) -> EncounterLifecycleDecision:
        snapshot = self.world.snapshot()
        first = snapshot.by_id(self.lifecycle.first_id)
        second = snapshot.by_id(self.lifecycle.second_id)
        return self.lifecycle.update(
            first,
            second,
            time_min=self.world.time,
            uncertain=uncertain,
            responsibility_context=responsibility_context,
        )

    def run_until_clear(
        self,
        initial_decision: EncounterLifecycleDecision,
        *,
        replanner: Replanner,
        end_time: float,
        uncertain: bool = False,
        responsibility_context: ResponsibilityContext | None = None,
    ) -> AvoidanceExecutionResult:
        """Execute/replan at explicit review events until clear or ``end_time``.

        ``initial_decision`` must already be the lifecycle decision for the world's
        current time.  This avoids a hidden duplicate observation at the same timestamp
        (which the lifecycle correctly forbids).
        """
        if end_time < self.world.time - _EPS:
            raise ValueError("end_time must not move backwards")
        self._validate_decision_now(initial_decision)
        decision = initial_decision
        steps: list[AvoidanceExecutionStep] = []

        for _ in range(self.config.max_reviews):
            if not decision.responsibility_active:
                return AvoidanceExecutionResult(
                    snapshot=self.world.snapshot(),
                    decision=decision,
                    steps=tuple(steps),
                    stop_reason=AvoidanceStopReason.RESPONSIBILITY_CLEAR,
                )
            if self.world.time >= end_time - _EPS:
                return AvoidanceExecutionResult(
                    snapshot=self.world.snapshot(),
                    decision=decision,
                    steps=tuple(steps),
                    stop_reason=AvoidanceStopReason.END_TIME,
                )

            snapshot_before = self.world.snapshot()
            plan = replanner(snapshot_before, decision)
            if plan is None:
                raise NoAvoidanceExecutionPlan(
                    "responsibility remains active but replanner returned no accepted plan"
                )
            self.apply_plan(plan)

            start_time = self.world.time
            review_time = min(end_time, start_time + self.config.review_interval_min)
            if review_time <= start_time + _EPS:
                return AvoidanceExecutionResult(
                    snapshot=self.world.snapshot(),
                    decision=decision,
                    steps=tuple(steps),
                    stop_reason=AvoidanceStopReason.END_TIME,
                )

            self.world.run_until(review_time)
            decision_after = self._observe_now(
                uncertain=uncertain,
                responsibility_context=responsibility_context,
            )
            snapshot_after = self.world.snapshot()
            steps.append(
                AvoidanceExecutionStep(
                    start_time=start_time,
                    review_time=review_time,
                    plan=plan,
                    decision_before=decision,
                    decision_after=decision_after,
                    snapshot_after=snapshot_after,
                )
            )
            decision = decision_after
        if not decision.responsibility_active:
            return AvoidanceExecutionResult(
                snapshot=self.world.snapshot(),
                decision=decision,
                steps=tuple(steps),
                stop_reason=AvoidanceStopReason.RESPONSIBILITY_CLEAR,
            )
        if self.world.time >= end_time - _EPS:
            return AvoidanceExecutionResult(
                snapshot=self.world.snapshot(),
                decision=decision,
                steps=tuple(steps),
                stop_reason=AvoidanceStopReason.END_TIME,
            )
        raise AvoidanceExecutionError("avoidance execution exceeded max_reviews")
