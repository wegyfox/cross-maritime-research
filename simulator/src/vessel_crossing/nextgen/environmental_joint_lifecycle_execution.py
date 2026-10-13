"""Assimilate component pair lifecycles at true a108 JOINT execution boundaries.

A109 composes the persistent a108 execution observer with independent local copies of
all pair lifecycles that formed the selected JOINT component.  The already-assimilated
apply-time lifecycle decisions are supplied explicitly, so the a108 ``APPLIED`` record
is *not* written into those lifecycles a second time.

Every later a108 record is one component observation boundary.  At that timestamp each
original active pair lifecycle is assimilated exactly once from the actual
water-relative hull states plus the a83 environmental drift/domain evidence.  This
layer still does not choose a control action, regenerate candidates, rerank tuples,
mutate commands, or recapture changed fleet topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .encounter_classification import EncounterClassificationConfig
from .encounter_registry import PairKey, canonical_pair_key
from .environmental_encounter_classification import EnvironmentalEncounterDirectionBasis
from .environmental_joint_execution_session import (
    EnvironmentalJointExecutionObservation,
    EnvironmentalJointExecutionSession,
    EnvironmentalJointExecutionSessionResult,
)
from .environmental_lifecycle import update_environmental_lifecycle
from .lifecycle import EncounterLifecycle, EncounterLifecycleDecision
from .responsibility import ResponsibilityContext

_EPS = 1e-10


class EnvironmentalJointLifecycleExecutionError(RuntimeError):
    """Base error for a109 JOINT lifecycle/execution composition failures."""


class EnvironmentalJointLifecycleExecutionMismatch(EnvironmentalJointLifecycleExecutionError):
    """Raised when supplied pair lifecycle state is not bound to the a108 session."""


@dataclass(frozen=True, slots=True)
class EnvironmentalJointPairLifecycleDecision:
    key: PairKey
    decision: EncounterLifecycleDecision

    def __post_init__(self) -> None:
        if self.key != canonical_pair_key(*self.key):
            raise ValueError("key must be a canonical pair key")
        observed = canonical_pair_key(
            self.decision.observation.first_id,
            self.decision.observation.second_id,
        )
        if observed != self.key:
            raise ValueError("decision observation does not match key")


@dataclass(frozen=True, slots=True)
class EnvironmentalJointLifecycleExecutionRecord:
    execution: EnvironmentalJointExecutionObservation
    pair_decisions: tuple[EnvironmentalJointPairLifecycleDecision, ...]

    def decision_for(self, first_id: str, second_id: str) -> EncounterLifecycleDecision:
        key = canonical_pair_key(first_id, second_id)
        for item in self.pair_decisions:
            if item.key == key:
                return item.decision
        raise KeyError(key)

    @property
    def responsibility_active_pair_keys(self) -> tuple[PairKey, ...]:
        return tuple(item.key for item in self.pair_decisions if item.decision.responsibility_active)

    @property
    def all_responsibility_clear(self) -> bool:
        return not self.responsibility_active_pair_keys


@dataclass(frozen=True, slots=True)
class EnvironmentalJointLifecycleExecutionResult:
    execution: EnvironmentalJointExecutionSessionResult
    pair_decisions: tuple[EnvironmentalJointPairLifecycleDecision, ...]
    records: tuple[EnvironmentalJointLifecycleExecutionRecord, ...]

    def decision_for(self, first_id: str, second_id: str) -> EncounterLifecycleDecision:
        key = canonical_pair_key(first_id, second_id)
        for item in self.pair_decisions:
            if item.key == key:
                return item.decision
        raise KeyError(key)

    @property
    def responsibility_active_pair_keys(self) -> tuple[PairKey, ...]:
        return tuple(item.key for item in self.pair_decisions if item.decision.responsibility_active)

    @property
    def all_responsibility_clear(self) -> bool:
        return not self.responsibility_active_pair_keys


class EnvironmentalJointLifecycleExecutionSession:
    """Update all original component pair lifecycles once per post-apply a108 record."""

    def __init__(
        self,
        execution_session: EnvironmentalJointExecutionSession,
        lifecycles: Mapping[PairKey, EncounterLifecycle],
        initial_decisions: Mapping[PairKey, EncounterLifecycleDecision],
        *,
        uncertain_pairs: Iterable[PairKey] = (),
        responsibility_context: ResponsibilityContext | None = None,
        classification_config: EncounterClassificationConfig | None = None,
        direction_basis: EnvironmentalEncounterDirectionBasis = EnvironmentalEncounterDirectionBasis.HULL_ASPECT,
    ) -> None:
        if not isinstance(execution_session, EnvironmentalJointExecutionSession):
            raise TypeError("execution_session must be EnvironmentalJointExecutionSession")
        if responsibility_context is not None and not isinstance(
            responsibility_context, ResponsibilityContext
        ):
            raise TypeError("responsibility_context must be ResponsibilityContext or None")
        if len(execution_session.history) != 1:
            raise EnvironmentalJointLifecycleExecutionMismatch(
                "a109 must attach before a108 has produced post-apply execution records"
            )

        selected = execution_session.applied.plan.arbitration.selected.evaluation
        expected_pair_keys = tuple(
            sorted(
                (item.first_id, item.second_id)
                for item in selected.pair_semantics
                if item.pair_intents is not None
            )
        )
        if not expected_pair_keys:
            raise EnvironmentalJointLifecycleExecutionMismatch(
                "selected JOINT tuple contains no active semantic component pairs"
            )

        lifecycle_by_key = self._canonical_mapping(lifecycles, "lifecycles")
        decision_by_key = self._canonical_mapping(initial_decisions, "initial_decisions")
        if tuple(sorted(lifecycle_by_key)) != expected_pair_keys:
            raise EnvironmentalJointLifecycleExecutionMismatch(
                "lifecycle pair set does not match selected JOINT component pairs"
            )
        if tuple(sorted(decision_by_key)) != expected_pair_keys:
            raise EnvironmentalJointLifecycleExecutionMismatch(
                "initial-decision pair set does not match selected JOINT component pairs"
            )

        uncertain = {canonical_pair_key(*key) for key in uncertain_pairs}
        unknown_uncertain = uncertain - set(expected_pair_keys)
        if unknown_uncertain:
            raise EnvironmentalJointLifecycleExecutionMismatch(
                f"uncertain_pairs contains pairs outside selected JOINT component: {sorted(unknown_uncertain)}"
            )

        applied_time = execution_session.applied.applied_at_time
        vessel_set = set(execution_session.vessel_ids)
        latest: list[EnvironmentalJointPairLifecycleDecision] = []
        for key in expected_pair_keys:
            lifecycle = lifecycle_by_key[key]
            decision = decision_by_key[key]
            if not isinstance(lifecycle, EncounterLifecycle):
                raise TypeError(f"lifecycle for {key!r} must be EncounterLifecycle")
            if not isinstance(decision, EncounterLifecycleDecision):
                raise TypeError(f"initial decision for {key!r} must be EncounterLifecycleDecision")
            if set(key) - vessel_set:
                raise EnvironmentalJointLifecycleExecutionMismatch(
                    f"component pair {key!r} contains vessel outside a108 execution session"
                )
            if {lifecycle.first_id, lifecycle.second_id} != set(key):
                raise EnvironmentalJointLifecycleExecutionMismatch(
                    f"lifecycle ids do not match component pair {key!r}"
                )
            observed = canonical_pair_key(
                decision.observation.first_id,
                decision.observation.second_id,
            )
            if observed != key:
                raise EnvironmentalJointLifecycleExecutionMismatch(
                    f"initial decision does not describe component pair {key!r}"
                )
            if abs(decision.observation.time_min - applied_time) > _EPS:
                raise EnvironmentalJointLifecycleExecutionMismatch(
                    f"initial decision time for {key!r} does not match applied-plan timestamp"
                )
            if not lifecycle.observations or lifecycle.observations[-1] != decision.observation:
                raise EnvironmentalJointLifecycleExecutionMismatch(
                    f"initial decision for {key!r} is not the lifecycle's latest assimilated observation"
                )
            latest.append(EnvironmentalJointPairLifecycleDecision(key, decision))

        self.execution_session = execution_session
        self.pair_keys = expected_pair_keys
        self.lifecycles = lifecycle_by_key
        self.uncertain_pairs = frozenset(uncertain)
        self.responsibility_context = responsibility_context
        self.classification_config = classification_config
        self.direction_basis = direction_basis
        self._latest = tuple(latest)
        self._history: list[EnvironmentalJointLifecycleExecutionRecord] = []

        context = execution_session.applied.plan.planning_context
        self._inputs = {vessel_id: context.for_vessel(vessel_id) for vessel_id in context.vessel_ids}

    @staticmethod
    def _canonical_mapping(mapping: Mapping, label: str) -> dict[PairKey, object]:
        result: dict[PairKey, object] = {}
        for raw_key, value in mapping.items():
            try:
                key = canonical_pair_key(*raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} contains invalid pair key {raw_key!r}") from exc
            if key in result:
                raise ValueError(f"{label} contains duplicate canonical pair key {key!r}")
            result[key] = value
        return result

    @property
    def latest_decisions(self) -> tuple[EnvironmentalJointPairLifecycleDecision, ...]:
        return self._latest

    @property
    def history(self) -> tuple[EnvironmentalJointLifecycleExecutionRecord, ...]:
        return tuple(self._history)

    def latest_decision_for(self, first_id: str, second_id: str) -> EncounterLifecycleDecision:
        key = canonical_pair_key(first_id, second_id)
        for item in self._latest:
            if item.key == key:
                return item.decision
        raise KeyError(key)

    def _assimilate_pair(
        self,
        record: EnvironmentalJointExecutionObservation,
        key: PairKey,
    ) -> EncounterLifecycleDecision:
        first_id, second_id = key
        first = record.for_vessel(first_id).state
        second = record.for_vessel(second_id).state
        first_input = self._inputs[first_id]
        second_input = self._inputs[second_id]
        return update_environmental_lifecycle(
            self.lifecycles[key],
            first,
            first_input.environmental_drift_velocity,
            second,
            second_input.environmental_drift_velocity,
            time_min=record.time_min,
            uncertain=key in self.uncertain_pairs,
            responsibility_context=self.responsibility_context,
            classification_config=self.classification_config,
            direction_basis=self.direction_basis,
            first_domain=first_input.domain,
            second_domain=second_input.domain,
        )

    def _assimilate_record(
        self,
        execution_record: EnvironmentalJointExecutionObservation,
    ) -> EnvironmentalJointLifecycleExecutionRecord:
        pair_decisions = tuple(
            EnvironmentalJointPairLifecycleDecision(
                key,
                self._assimilate_pair(execution_record, key),
            )
            for key in self.pair_keys
        )
        combined = EnvironmentalJointLifecycleExecutionRecord(execution_record, pair_decisions)
        self._latest = pair_decisions
        self._history.append(combined)
        return combined

    def run_until(self, end_time: float) -> EnvironmentalJointLifecycleExecutionResult:
        execution_result = self.execution_session.run_until(end_time)
        new_records = tuple(self._assimilate_record(record) for record in execution_result.records)
        return EnvironmentalJointLifecycleExecutionResult(
            execution=execution_result,
            pair_decisions=self._latest,
            records=new_records,
        )
