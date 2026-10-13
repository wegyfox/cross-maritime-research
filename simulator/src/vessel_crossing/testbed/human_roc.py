"""C19 human / Remote Operations Centre (ROC) response layer.

The layer models human availability, workload and response latency as explicit
experimental evidence.  It does not expose GroundTruthWorld and does not command the
plant.  A completed human response can be adapted into C9's remote-takeover signal,
where authority transfer remains a separate state-machine decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Iterable, Mapping

from .operational_state import OperationalSignals
from .scope import TestbedFeature


class HumanOperatorRole(str, Enum):
    ROC_OPERATOR = "ROC_OPERATOR"
    ROC_SUPERVISOR = "ROC_SUPERVISOR"
    MASTER = "MASTER"
    SHORE_SPECIALIST = "SHORE_SPECIALIST"


class HumanInformationBasis(str, Enum):
    OBSERVED_WORLD = "OBSERVED_WORLD"
    COORDINATION_CHANNEL = "COORDINATION_CHANNEL"
    SYSTEM_ALARM = "SYSTEM_ALARM"
    OPERATOR_INPUT = "OPERATOR_INPUT"
    DOCUMENTED_ASSUMPTION = "DOCUMENTED_ASSUMPTION"


class HumanTakeoverOutcome(str, Enum):
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    CAPACITY_EXHAUSTED = "CAPACITY_EXHAUSTED"
    DECLINED = "DECLINED"
    DEADLINE_MISSED = "DEADLINE_MISSED"


class HumanTakeoverStatus(str, Enum):
    REQUESTED = "REQUESTED"
    ASSIGNED = "ASSIGNED"
    NOTICED = "NOTICED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    CAPACITY_EXHAUSTED = "CAPACITY_EXHAUSTED"
    DECLINED = "DECLINED"
    TIMED_OUT = "TIMED_OUT"


class HumanAssignmentPolicy(str, Enum):
    EARLIEST_READY = "EARLIEST_READY"
    LOWEST_WORKLOAD = "LOWEST_WORKLOAD"


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _unit_interval(identity: str, namespace: str) -> float:
    raw = hashlib.sha256(f"{namespace}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") / float(2**64)


def _finite_nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class HumanTakeoverRequest:
    request_id: str
    vessel_id: str
    created_at_min: float
    required_by_min: float | None
    information_basis: HumanInformationBasis
    information_sha256: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id or not self.vessel_id:
            raise ValueError("request_id and vessel_id must be non-empty")
        _finite_nonnegative("created_at_min", self.created_at_min)
        if self.required_by_min is not None:
            if not isfinite(self.required_by_min) or self.required_by_min < self.created_at_min:
                raise ValueError("required_by_min must be >= created_at_min")
        if len(self.information_sha256) != 64:
            raise ValueError("information_sha256 must be a 64-character SHA-256 hex digest")
        try:
            int(self.information_sha256, 16)
        except ValueError as exc:
            raise ValueError("information_sha256 must be hexadecimal") from exc
        if self.evidence_ids != tuple(sorted(set(self.evidence_ids))):
            raise ValueError("evidence_ids must be unique and sorted")

    def identity_sha256(self) -> str:
        return _sha({
            "request_id": self.request_id,
            "vessel_id": self.vessel_id,
            "created_at_min": self.created_at_min,
            "required_by_min": self.required_by_min,
            "information_basis": self.information_basis.value,
            "information_sha256": self.information_sha256,
            "evidence_ids": list(self.evidence_ids),
        })


@dataclass(frozen=True, slots=True)
class HumanOperatorSpec:
    operator_id: str
    role: HumanOperatorRole = HumanOperatorRole.ROC_OPERATOR
    max_concurrent_assignments: int = 1
    notice_latency_min: float = 0.25
    acknowledgement_latency_min: float = 0.25
    preparation_latency_min: float = 0.5
    workload_latency_gain: float = 1.0
    acceptance_probability: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.operator_id:
            raise ValueError("operator_id must be non-empty")
        if self.max_concurrent_assignments < 1:
            raise ValueError("max_concurrent_assignments must be >= 1")
        for name in (
            "notice_latency_min",
            "acknowledgement_latency_min",
            "preparation_latency_min",
            "workload_latency_gain",
        ):
            _finite_nonnegative(name, float(getattr(self, name)))
        if not isfinite(self.acceptance_probability) or not 0.0 <= self.acceptance_probability <= 1.0:
            raise ValueError("acceptance_probability must be in [0, 1]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def identity_sha256(self) -> str:
        return _sha({
            "operator_id": self.operator_id,
            "role": self.role.value,
            "max_concurrent_assignments": self.max_concurrent_assignments,
            "notice_latency_min": self.notice_latency_min,
            "acknowledgement_latency_min": self.acknowledgement_latency_min,
            "preparation_latency_min": self.preparation_latency_min,
            "workload_latency_gain": self.workload_latency_gain,
            "acceptance_probability": self.acceptance_probability,
            "seed": self.seed,
        })


@dataclass(frozen=True, slots=True)
class HumanOperatorSnapshot:
    operator_id: str
    time_min: float
    available: bool = True
    connected: bool = True
    active_assignments: int = 0
    workload_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.operator_id:
            raise ValueError("operator_id must be non-empty")
        _finite_nonnegative("time_min", self.time_min)
        if self.active_assignments < 0:
            raise ValueError("active_assignments must be non-negative")
        if not isfinite(self.workload_fraction) or not 0.0 <= self.workload_fraction <= 1.0:
            raise ValueError("workload_fraction must be in [0, 1]")

    def identity_sha256(self) -> str:
        return _sha({
            "operator_id": self.operator_id,
            "time_min": self.time_min,
            "available": self.available,
            "connected": self.connected,
            "active_assignments": self.active_assignments,
            "workload_fraction": self.workload_fraction,
        })


@dataclass(frozen=True, slots=True)
class HumanROCCentreSpec:
    centre_id: str
    centre_version: str
    operators: tuple[HumanOperatorSpec, ...]
    assignment_policy: HumanAssignmentPolicy = HumanAssignmentPolicy.EARLIEST_READY

    @classmethod
    def create(
        cls,
        *,
        centre_id: str,
        centre_version: str,
        operators: Iterable[HumanOperatorSpec],
        assignment_policy: HumanAssignmentPolicy = HumanAssignmentPolicy.EARLIEST_READY,
    ) -> "HumanROCCentreSpec":
        ordered = tuple(sorted(tuple(operators), key=lambda item: item.operator_id))
        return cls(centre_id, centre_version, ordered, assignment_policy)

    def __post_init__(self) -> None:
        if not self.centre_id or not self.centre_version:
            raise ValueError("centre_id and centre_version must be non-empty")
        if not self.operators:
            raise ValueError("operators must be non-empty")
        ids = [operator.operator_id for operator in self.operators]
        if len(ids) != len(set(ids)):
            raise ValueError("operator ids must be unique")
        if tuple(ids) != tuple(sorted(ids)):
            raise ValueError("operators must be sorted by operator_id")

    def identity_sha256(self) -> str:
        return _sha({
            "centre_id": self.centre_id,
            "centre_version": self.centre_version,
            "assignment_policy": self.assignment_policy.value,
            "operator_sha256": [operator.identity_sha256() for operator in self.operators],
        })


@dataclass(frozen=True, slots=True)
class HumanOperatorCandidateEvidence:
    operator_id: str
    snapshot_sha256: str
    eligible: bool
    capacity_available: bool
    acceptance_draw: float
    accepted: bool
    latency_multiplier: float
    assigned_at_min: float | None
    noticed_at_min: float | None
    acknowledged_at_min: float | None
    ready_at_min: float | None
    reason: str

    def identity_sha256(self) -> str:
        return _sha({
            "operator_id": self.operator_id,
            "snapshot_sha256": self.snapshot_sha256,
            "eligible": self.eligible,
            "capacity_available": self.capacity_available,
            "acceptance_draw": self.acceptance_draw,
            "accepted": self.accepted,
            "latency_multiplier": self.latency_multiplier,
            "assigned_at_min": self.assigned_at_min,
            "noticed_at_min": self.noticed_at_min,
            "acknowledged_at_min": self.acknowledged_at_min,
            "ready_at_min": self.ready_at_min,
            "reason": self.reason,
        })


@dataclass(frozen=True, slots=True)
class HumanTakeoverAssessment:
    request_sha256: str
    centre_sha256: str
    outcome: HumanTakeoverOutcome
    assigned_operator_id: str | None
    assigned_at_min: float | None
    noticed_at_min: float | None
    acknowledged_at_min: float | None
    ready_at_min: float | None
    candidate_evidence: tuple[HumanOperatorCandidateEvidence, ...]
    component_id: TestbedFeature = TestbedFeature.C19_HUMAN_ROC

    def status_at(self, time_min: float) -> HumanTakeoverStatus:
        _finite_nonnegative("time_min", time_min)
        if self.outcome is HumanTakeoverOutcome.UNAVAILABLE:
            return HumanTakeoverStatus.UNAVAILABLE
        if self.outcome is HumanTakeoverOutcome.CAPACITY_EXHAUSTED:
            return HumanTakeoverStatus.CAPACITY_EXHAUSTED
        if self.outcome is HumanTakeoverOutcome.DECLINED:
            return HumanTakeoverStatus.DECLINED
        if self.outcome is HumanTakeoverOutcome.DEADLINE_MISSED:
            return HumanTakeoverStatus.TIMED_OUT
        assert self.outcome is HumanTakeoverOutcome.READY
        assert self.assigned_at_min is not None
        assert self.noticed_at_min is not None
        assert self.acknowledged_at_min is not None
        assert self.ready_at_min is not None
        if time_min < self.assigned_at_min:
            return HumanTakeoverStatus.REQUESTED
        if time_min < self.noticed_at_min:
            return HumanTakeoverStatus.ASSIGNED
        if time_min < self.acknowledged_at_min:
            return HumanTakeoverStatus.NOTICED
        if time_min < self.ready_at_min:
            return HumanTakeoverStatus.ACKNOWLEDGED
        return HumanTakeoverStatus.READY

    def identity_sha256(self) -> str:
        return _sha({
            "request_sha256": self.request_sha256,
            "centre_sha256": self.centre_sha256,
            "outcome": self.outcome.value,
            "assigned_operator_id": self.assigned_operator_id,
            "assigned_at_min": self.assigned_at_min,
            "noticed_at_min": self.noticed_at_min,
            "acknowledged_at_min": self.acknowledged_at_min,
            "ready_at_min": self.ready_at_min,
            "candidate_evidence_sha256": [item.identity_sha256() for item in self.candidate_evidence],
        })


class HumanROCEvaluator:
    component_id = TestbedFeature.C19_HUMAN_ROC

    def __init__(self, spec: HumanROCCentreSpec) -> None:
        self.spec = spec
        self._operators = {operator.operator_id: operator for operator in spec.operators}

    def evaluate(
        self,
        request: HumanTakeoverRequest,
        *,
        operator_snapshots: Iterable[HumanOperatorSnapshot],
    ) -> HumanTakeoverAssessment:
        snapshots = tuple(sorted(tuple(operator_snapshots), key=lambda item: item.operator_id))
        snapshot_ids = [snapshot.operator_id for snapshot in snapshots]
        if len(snapshot_ids) != len(set(snapshot_ids)):
            raise ValueError("operator snapshot ids must be unique")
        if set(snapshot_ids) != set(self._operators):
            missing = sorted(set(self._operators) - set(snapshot_ids))
            extra = sorted(set(snapshot_ids) - set(self._operators))
            raise ValueError(f"operator snapshots must exactly match roster; missing={missing}, extra={extra}")
        for snapshot in snapshots:
            if abs(snapshot.time_min - request.created_at_min) > 1e-12:
                raise ValueError("operator snapshots must be taken at request.created_at_min")

        candidates: list[HumanOperatorCandidateEvidence] = []
        eligible_count = 0
        capacity_count = 0
        accepted_candidates: list[HumanOperatorCandidateEvidence] = []
        request_sha = request.identity_sha256()

        for snapshot in snapshots:
            operator = self._operators[snapshot.operator_id]
            eligible = snapshot.available and snapshot.connected
            if eligible:
                eligible_count += 1
            capacity_available = eligible and snapshot.active_assignments < operator.max_concurrent_assignments
            if capacity_available:
                capacity_count += 1
            draw = _unit_interval(request_sha, f"human-accept:{operator.operator_id}:{operator.seed}")
            accepted = capacity_available and draw < operator.acceptance_probability
            multiplier = 1.0 + operator.workload_latency_gain * snapshot.workload_fraction
            assigned_at = request.created_at_min if accepted else None
            noticed_at = assigned_at + operator.notice_latency_min * multiplier if accepted else None
            ack_at = noticed_at + operator.acknowledgement_latency_min * multiplier if accepted else None
            ready_at = ack_at + operator.preparation_latency_min * multiplier if accepted else None
            if not eligible:
                reason = "operator-unavailable-or-disconnected"
            elif not capacity_available:
                reason = "operator-capacity-exhausted"
            elif not accepted:
                reason = "operator-declined"
            elif request.required_by_min is not None and ready_at is not None and ready_at > request.required_by_min:
                reason = "projected-ready-after-deadline"
            else:
                reason = "candidate-ready"
            evidence = HumanOperatorCandidateEvidence(
                operator_id=operator.operator_id,
                snapshot_sha256=snapshot.identity_sha256(),
                eligible=eligible,
                capacity_available=capacity_available,
                acceptance_draw=draw,
                accepted=accepted,
                latency_multiplier=multiplier,
                assigned_at_min=assigned_at,
                noticed_at_min=noticed_at,
                acknowledged_at_min=ack_at,
                ready_at_min=ready_at,
                reason=reason,
            )
            candidates.append(evidence)
            if accepted:
                accepted_candidates.append(evidence)

        chosen: HumanOperatorCandidateEvidence | None = None
        if accepted_candidates:
            if self.spec.assignment_policy is HumanAssignmentPolicy.EARLIEST_READY:
                chosen = min(accepted_candidates, key=lambda item: (item.ready_at_min, item.operator_id))
            else:
                snapshot_by_id = {item.operator_id: item for item in snapshots}
                chosen = min(
                    accepted_candidates,
                    key=lambda item: (snapshot_by_id[item.operator_id].workload_fraction, item.ready_at_min, item.operator_id),
                )

        if chosen is not None:
            if request.required_by_min is not None and chosen.ready_at_min is not None and chosen.ready_at_min > request.required_by_min:
                outcome = HumanTakeoverOutcome.DEADLINE_MISSED
                chosen_for_output = chosen
            else:
                outcome = HumanTakeoverOutcome.READY
                chosen_for_output = chosen
        elif eligible_count == 0:
            outcome = HumanTakeoverOutcome.UNAVAILABLE
            chosen_for_output = None
        elif capacity_count == 0:
            outcome = HumanTakeoverOutcome.CAPACITY_EXHAUSTED
            chosen_for_output = None
        else:
            outcome = HumanTakeoverOutcome.DECLINED
            chosen_for_output = None

        return HumanTakeoverAssessment(
            request_sha256=request_sha,
            centre_sha256=self.spec.identity_sha256(),
            outcome=outcome,
            assigned_operator_id=None if chosen_for_output is None else chosen_for_output.operator_id,
            assigned_at_min=None if chosen_for_output is None else chosen_for_output.assigned_at_min,
            noticed_at_min=None if chosen_for_output is None else chosen_for_output.noticed_at_min,
            acknowledged_at_min=None if chosen_for_output is None else chosen_for_output.acknowledged_at_min,
            ready_at_min=None if chosen_for_output is None else chosen_for_output.ready_at_min,
            candidate_evidence=tuple(candidates),
        )


class NoOpHumanROCEvaluator:
    component_id = TestbedFeature.C19_HUMAN_ROC

    def evaluate(self, request: HumanTakeoverRequest, *, operator_snapshots: Iterable[HumanOperatorSnapshot]) -> None:
        return None


def operational_signals_with_human_response(
    signals: OperationalSignals,
    assessment: HumanTakeoverAssessment | None,
    *,
    time_min: float | None = None,
) -> OperationalSignals:
    """Adapt C19 response evidence into the C9 remote-takeover signal.

    No-op assessment returns the exact input object.  For a real assessment, the C9
    acknowledgement signal is asserted only when the human is READY for transfer, not
    merely when the request was noticed or verbally acknowledged.  C9 may still apply
    its own technical authority-transfer delay after this point.
    """

    if assessment is None:
        return signals
    effective_time = signals.time_min if time_min is None else float(time_min)
    if abs(effective_time - signals.time_min) > 1e-12:
        raise ValueError("time_min must equal OperationalSignals.time_min")
    ready = assessment.status_at(effective_time) is HumanTakeoverStatus.READY
    evidence_ids = tuple(sorted(set(signals.evidence_ids) | {f"human-roc:{assessment.identity_sha256()}"}))
    return OperationalSignals(
        time_min=signals.time_min,
        odd_status=signals.odd_status,
        degradation_active=signals.degradation_active,
        fallback_requested=signals.fallback_requested,
        remote_takeover_acknowledged=signals.remote_takeover_acknowledged or ready,
        admissible_action_available=signals.admissible_action_available,
        safe_state_reached=signals.safe_state_reached,
        evidence_ids=evidence_ids,
    )
