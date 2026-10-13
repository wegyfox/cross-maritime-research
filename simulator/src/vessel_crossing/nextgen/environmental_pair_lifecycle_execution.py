"""Assimilate encounter lifecycle evidence at a93 execution-observation boundaries.

A94 composes two existing state machines without adding replanning policy:
- a93 owns physical advancement and execution-boundary records;
- the existing ``EncounterLifecycle`` owns encounter/risk/responsibility persistence.

The caller must supply the lifecycle decision that already exists at the a92/a93 apply
timestamp.  A94 never re-observes the a93 ``APPLIED`` record, preventing a duplicate
same-time lifecycle assimilation.  Only subsequent a93 records are converted to
environment-aware lifecycle evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .encounter_classification import EncounterClassificationConfig
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_lifecycle import update_environmental_lifecycle
from .environmental_pair_execution_session import (
    EnvironmentalPairExecutionObservation,
    EnvironmentalPairExecutionSession,
    EnvironmentalPairExecutionSessionResult,
)
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .responsibility import ResponsibilityContext

_EPS = 1e-10


class EnvironmentalPairLifecycleExecutionError(RuntimeError):
    """Base error for a94 lifecycle/execution composition failures."""


class EnvironmentalPairLifecycleExecutionMismatch(EnvironmentalPairLifecycleExecutionError):
    """Raised when lifecycle history does not match the supplied execution session."""


@dataclass(frozen=True, slots=True)
class EnvironmentalPairLifecycleExecutionRecord:
    execution: EnvironmentalPairExecutionObservation
    decision: EncounterLifecycleDecision


@dataclass(frozen=True, slots=True)
class EnvironmentalPairLifecycleExecutionResult:
    execution: EnvironmentalPairExecutionSessionResult
    decision: EncounterLifecycleDecision
    records: tuple[EnvironmentalPairLifecycleExecutionRecord, ...]

    @property
    def responsibility_active(self) -> bool:
        return self.decision.responsibility_active


class EnvironmentalPairLifecycleExecutionSession:
    """Update one existing lifecycle at a93 boundaries without replanning commands."""

    def __init__(
        self,
        execution_session: EnvironmentalPairExecutionSession,
        lifecycle: EncounterLifecycle,
        initial_decision: EncounterLifecycleDecision,
        *,
        uncertain: bool = False,
        responsibility_context: ResponsibilityContext | None = None,
        classification_config: EncounterClassificationConfig | None = None,
        direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    ) -> None:
        if not isinstance(execution_session, EnvironmentalPairExecutionSession):
            raise TypeError("execution_session must be EnvironmentalPairExecutionSession")
        if not isinstance(lifecycle, EncounterLifecycle):
            raise TypeError("lifecycle must be EncounterLifecycle")
        if not isinstance(initial_decision, EncounterLifecycleDecision):
            raise TypeError("initial_decision must be EncounterLifecycleDecision")
        if responsibility_context is not None and not isinstance(responsibility_context, ResponsibilityContext):
            raise TypeError("responsibility_context must be ResponsibilityContext or None")
        if len(execution_session.history) != 1:
            raise EnvironmentalPairLifecycleExecutionMismatch(
                "a94 must attach before a93 has produced post-apply execution records"
            )

        pair = {execution_session.first_id, execution_session.second_id}
        if pair != {lifecycle.first_id, lifecycle.second_id}:
            raise EnvironmentalPairLifecycleExecutionMismatch(
                "execution-session pair does not match lifecycle pair"
            )
        if {initial_decision.observation.first_id, initial_decision.observation.second_id} != pair:
            raise EnvironmentalPairLifecycleExecutionMismatch(
                "initial lifecycle decision does not describe execution-session pair"
            )
        applied_time = execution_session.applied.applied_at_time
        if abs(initial_decision.observation.time_min - applied_time) > _EPS:
            raise EnvironmentalPairLifecycleExecutionMismatch(
                "initial lifecycle decision time does not match applied-plan timestamp"
            )
        if not lifecycle.observations or lifecycle.observations[-1] != initial_decision.observation:
            raise EnvironmentalPairLifecycleExecutionMismatch(
                "initial decision is not the lifecycle's already-assimilated latest observation"
            )

        self.execution_session = execution_session
        self.lifecycle = lifecycle
        self.initial_decision = initial_decision
        self.uncertain = bool(uncertain)
        self.responsibility_context = responsibility_context
        self.classification_config = classification_config
        self.direction_basis = direction_basis
        self._latest_decision = initial_decision
        self._history: list[EnvironmentalPairLifecycleExecutionRecord] = []

        context = execution_session.applied.plan.planning_context
        self._first_input = context.for_vessel(execution_session.first_id)
        self._second_input = context.for_vessel(execution_session.second_id)

    @property
    def latest_decision(self) -> EncounterLifecycleDecision:
        return self._latest_decision

    @property
    def history(self) -> tuple[EnvironmentalPairLifecycleExecutionRecord, ...]:
        return tuple(self._history)

    def _assimilate(self, record: EnvironmentalPairExecutionObservation) -> EncounterLifecycleDecision:
        return update_environmental_lifecycle(
            self.lifecycle,
            record.first_state,
            self._first_input.environmental_drift_velocity,
            record.second_state,
            self._second_input.environmental_drift_velocity,
            time_min=record.time_min,
            uncertain=self.uncertain,
            responsibility_context=self.responsibility_context,
            classification_config=self.classification_config,
            direction_basis=self.direction_basis,
            first_domain=self._first_input.domain,
            second_domain=self._second_input.domain,
        )

    def run_until(self, end_time: float) -> EnvironmentalPairLifecycleExecutionResult:
        execution_result = self.execution_session.run_until(end_time)
        new_records: list[EnvironmentalPairLifecycleExecutionRecord] = []
        for execution_record in execution_result.records:
            decision = self._assimilate(execution_record)
            combined = EnvironmentalPairLifecycleExecutionRecord(execution_record, decision)
            self._history.append(combined)
            new_records.append(combined)
            self._latest_decision = decision
        return EnvironmentalPairLifecycleExecutionResult(
            execution=execution_result,
            decision=self._latest_decision,
            records=tuple(new_records),
        )
