"""C10 physical recoverability evaluation.

This layer measures whether at least one maneuver in an explicitly modeled action
space remains physically reachable, admissible and safe.  It is deliberately
separate from controller search/ranking: a controller may fail while the modeled
safe feasible set is still non-empty.

C10 is measurement infrastructure, not a new controller, safety guarantee,
certification claim, or proof that the modeled action space is complete.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from math import isfinite, isnan
from typing import Any, Iterable

from .scope import TestbedFeature

_EPS = 1e-12


class RecoverabilityOutcome(str, Enum):
    CONTROLLER_FOUND_RECOVERY = "CONTROLLER_FOUND_RECOVERY"
    CONTROLLER_FAILED_BUT_RECOVERABLE = "CONTROLLER_FAILED_BUT_RECOVERABLE"
    UNRECOVERABLE_IN_MODELED_ACTION_SPACE = "UNRECOVERABLE_IN_MODELED_ACTION_SPACE"
    NO_PHYSICALLY_REACHABLE_CANDIDATE = "NO_PHYSICALLY_REACHABLE_CANDIDATE"

    @property
    def recoverable(self) -> bool:
        return self in {
            RecoverabilityOutcome.CONTROLLER_FOUND_RECOVERY,
            RecoverabilityOutcome.CONTROLLER_FAILED_BUT_RECOVERABLE,
        }


@dataclass(frozen=True, slots=True)
class RecoverabilityCandidateEvidence:
    candidate_id: str
    physically_reachable: bool
    admissible: bool
    safe: bool
    safety_margin_nm: float | None = None
    sea_room_margin_nm: float | None = None
    maneuver_time_margin_min: float | None = None
    rejection_reasons: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        for name, value in (
            ("safety_margin_nm", self.safety_margin_nm),
            ("sea_room_margin_nm", self.sea_room_margin_nm),
            ("maneuver_time_margin_min", self.maneuver_time_margin_min),
        ):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite when supplied")
        reasons = tuple(self.rejection_reasons)
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError("rejection_reasons must be unique and sorted")
        ids = tuple(self.evidence_ids)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("evidence_ids must be unique and sorted")

    @property
    def feasible_safe_recovery(self) -> bool:
        if not (self.physically_reachable and self.admissible and self.safe):
            return False
        for margin in (
            self.safety_margin_nm,
            self.sea_room_margin_nm,
            self.maneuver_time_margin_min,
        ):
            if margin is not None and margin < -_EPS:
                return False
        return True

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "candidate_id": self.candidate_id,
                "physically_reachable": self.physically_reachable,
                "admissible": self.admissible,
                "safe": self.safe,
                "safety_margin_nm": self.safety_margin_nm,
                "sea_room_margin_nm": self.sea_room_margin_nm,
                "maneuver_time_margin_min": self.maneuver_time_margin_min,
                "rejection_reasons": list(self.rejection_reasons),
                "evidence_ids": list(self.evidence_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class RecoverabilityEvaluation:
    time_min: float
    outcome: RecoverabilityOutcome
    candidates: tuple[RecoverabilityCandidateEvidence, ...]
    controller_candidate_id: str | None
    physically_reachable_candidate_ids: tuple[str, ...]
    admissible_candidate_ids: tuple[str, ...]
    safe_recovery_candidate_ids: tuple[str, ...]
    best_safety_margin_nm: float | None
    best_sea_room_margin_nm: float | None
    best_maneuver_time_margin_min: float | None
    action_space_id: str
    component_id: TestbedFeature = TestbedFeature.C10_RECOVERABILITY

    @property
    def recoverable(self) -> bool:
        return self.outcome.recoverable

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "time_min": self.time_min,
                "outcome": self.outcome.value,
                "controller_candidate_id": self.controller_candidate_id,
                "action_space_id": self.action_space_id,
                "candidate_identities": [candidate.identity_sha256() for candidate in self.candidates],
            }
        )


@dataclass(frozen=True, slots=True)
class FeasibleSetCollapseInterval:
    """Bracket for an observed recoverable->unrecoverable transition.

    It intentionally does not pretend that the exact collapse instant is known between
    two sampled evaluations.
    """

    last_recoverable_time_min: float
    first_unrecoverable_time_min: float
    last_recoverable_evaluation_sha256: str
    first_unrecoverable_evaluation_sha256: str

    def __post_init__(self) -> None:
        if not (
            isfinite(self.last_recoverable_time_min)
            and isfinite(self.first_unrecoverable_time_min)
        ):
            raise ValueError("collapse interval times must be finite")
        if self.last_recoverable_time_min > self.first_unrecoverable_time_min + _EPS:
            raise ValueError("collapse interval must be time ordered")

    @property
    def width_min(self) -> float:
        return self.first_unrecoverable_time_min - self.last_recoverable_time_min


class PhysicalRecoverabilityEvaluator:
    component_id = TestbedFeature.C10_RECOVERABILITY

    def __init__(self, *, action_space_id: str = "cross-c10/modeled-action-space/v1") -> None:
        if not action_space_id:
            raise ValueError("action_space_id must be non-empty")
        self.action_space_id = action_space_id

    def evaluate(
        self,
        *,
        time_min: float,
        candidates: Iterable[RecoverabilityCandidateEvidence],
        controller_candidate_id: str | None = None,
    ) -> RecoverabilityEvaluation:
        if not isfinite(time_min) or time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        ordered = tuple(sorted(tuple(candidates), key=lambda item: item.candidate_id))
        ids = [candidate.candidate_id for candidate in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        if controller_candidate_id is not None and controller_candidate_id not in set(ids):
            raise ValueError("controller_candidate_id must identify one supplied candidate")

        reachable = tuple(c.candidate_id for c in ordered if c.physically_reachable)
        admissible = tuple(
            c.candidate_id for c in ordered if c.physically_reachable and c.admissible
        )
        safe = tuple(c.candidate_id for c in ordered if c.feasible_safe_recovery)

        if not reachable:
            outcome = RecoverabilityOutcome.NO_PHYSICALLY_REACHABLE_CANDIDATE
        elif not safe:
            outcome = RecoverabilityOutcome.UNRECOVERABLE_IN_MODELED_ACTION_SPACE
        elif controller_candidate_id in set(safe):
            outcome = RecoverabilityOutcome.CONTROLLER_FOUND_RECOVERY
        else:
            outcome = RecoverabilityOutcome.CONTROLLER_FAILED_BUT_RECOVERABLE

        feasible = [c for c in ordered if c.feasible_safe_recovery]
        return RecoverabilityEvaluation(
            time_min=float(time_min),
            outcome=outcome,
            candidates=ordered,
            controller_candidate_id=controller_candidate_id,
            physically_reachable_candidate_ids=reachable,
            admissible_candidate_ids=admissible,
            safe_recovery_candidate_ids=safe,
            best_safety_margin_nm=_max_optional(feasible, "safety_margin_nm"),
            best_sea_room_margin_nm=_max_optional(feasible, "sea_room_margin_nm"),
            best_maneuver_time_margin_min=_max_optional(feasible, "maneuver_time_margin_min"),
            action_space_id=self.action_space_id,
        )


class NoOpRecoverabilityEvaluator:
    """Compatibility mode: pre-C10 execution produces no recoverability verdict."""

    component_id = TestbedFeature.C10_RECOVERABILITY

    def evaluate(self, *args: Any, **kwargs: Any) -> None:
        return None


class RecoverabilityTimeline:
    """Monotonic sampled recoverability evidence and collapse bracketing."""

    component_id = TestbedFeature.C10_RECOVERABILITY

    def __init__(self) -> None:
        self._evaluations: list[RecoverabilityEvaluation] = []

    @property
    def evaluations(self) -> tuple[RecoverabilityEvaluation, ...]:
        return tuple(self._evaluations)

    def append(self, evaluation: RecoverabilityEvaluation) -> None:
        if self._evaluations and evaluation.time_min + _EPS < self._evaluations[-1].time_min:
            raise ValueError("recoverability timeline requires monotonic time")
        self._evaluations.append(evaluation)

    def latest_irreversible_collapse_interval(self) -> FeasibleSetCollapseInterval | None:
        """Return the final sampled feasible-set collapse bracket, if observed.

        A later recovery means an earlier loss was not an irreversible collapse in the
        sampled evidence.  Therefore the last recoverable sample is used and the first
        later unrecoverable sample brackets the latest intervention boundary.
        """

        if not self._evaluations:
            return None
        recoverable_indexes = [
            index for index, evaluation in enumerate(self._evaluations) if evaluation.recoverable
        ]
        if not recoverable_indexes:
            return None
        last_recoverable_index = recoverable_indexes[-1]
        if last_recoverable_index >= len(self._evaluations) - 1:
            return None
        later = self._evaluations[last_recoverable_index + 1 :]
        first_unrecoverable = next((item for item in later if not item.recoverable), None)
        if first_unrecoverable is None:
            return None
        last_recoverable = self._evaluations[last_recoverable_index]
        return FeasibleSetCollapseInterval(
            last_recoverable_time_min=last_recoverable.time_min,
            first_unrecoverable_time_min=first_unrecoverable.time_min,
            last_recoverable_evaluation_sha256=last_recoverable.identity_sha256(),
            first_unrecoverable_evaluation_sha256=first_unrecoverable.identity_sha256(),
        )


def recoverability_candidate_from_environmental_pair_evaluation(
    evaluation: Any,
    *,
    candidate_id: str,
    sea_room_margin_nm: float | None = None,
    maneuver_time_margin_min: float | None = None,
    extra_evidence_ids: Iterable[str] = (),
) -> RecoverabilityCandidateEvidence:
    """Adapt an existing NextGen PAIR candidate evaluation into C10 evidence.

    The adapter reuses the existing physical reachability and clearance screening.  It
    does not reimplement candidate dynamics.  Sea-room and intervention-time margins are
    optional orthogonal evidence because the PAIR evaluator itself does not model shore
    boundaries or a response deadline.
    """

    from ..nextgen.environmental_pair_candidate import EnvironmentalPairCandidateRejection

    physically_reachable = bool(
        evaluation.first_transition.reachable and evaluation.second_transition.reachable
    )
    first_reasons = tuple(evaluation.first_rejection_reasons)
    second_reasons = tuple(evaluation.second_rejection_reasons)
    pair_reasons = tuple(evaluation.pair_rejection_reasons)
    all_reasons = first_reasons + second_reasons + pair_reasons

    physical_reasons = {EnvironmentalPairCandidateRejection.PHYSICALLY_UNREACHABLE}
    safety_reasons = {
        EnvironmentalPairCandidateRejection.DOMAIN_EVIDENCE_INCOMPLETE,
        EnvironmentalPairCandidateRejection.TRANSITION_CLEARANCE,
        EnvironmentalPairCandidateRejection.POST_TRANSITION_CLEARANCE,
    }
    semantic_reasons = set(all_reasons) - physical_reasons - safety_reasons
    admissible = physically_reachable and not semantic_reasons
    safe = physically_reachable and not (set(all_reasons) & safety_reasons)

    safety_margin = _pair_safety_margin_nm(evaluation)
    if safety_margin is None and safe:
        # Missing physical clearance evidence must never be promoted to a safe verdict.
        safe = False

    evidence_ids = {
        f"nextgen-pair-evaluation:{_sha256_payload(_jsonable(evaluation))}",
        *tuple(extra_evidence_ids),
    }
    return RecoverabilityCandidateEvidence(
        candidate_id=candidate_id,
        physically_reachable=physically_reachable,
        admissible=admissible,
        safe=safe,
        safety_margin_nm=safety_margin,
        sea_room_margin_nm=sea_room_margin_nm,
        maneuver_time_margin_min=maneuver_time_margin_min,
        rejection_reasons=tuple(sorted({getattr(reason, "value", str(reason)) for reason in all_reasons})),
        evidence_ids=tuple(sorted(evidence_ids)),
    )


def _pair_safety_margin_nm(evaluation: Any) -> float | None:
    clearance = evaluation.transition_clearance
    if clearance is None:
        return None
    margins: list[float] = []
    if clearance.domain is not None:
        margins.append(float(clearance.domain.conservative_lower_bound_nm))
    else:
        margins.append(
            float(clearance.centre.conservative_lower_bound_nm)
            - float(clearance.centre.required_min_separation_nm)
        )
    if evaluation.post_transition_domain_clearance is not None:
        margins.append(float(evaluation.post_transition_domain_clearance.conservative_lower_bound_nm))
    elif evaluation.post_transition_cpa is not None:
        margins.append(
            float(evaluation.post_transition_cpa.dcpa)
            - float(clearance.centre.required_min_separation_nm)
        )
    return min(margins) if margins else None


def _max_optional(items: Iterable[RecoverabilityCandidateEvidence], attr: str) -> float | None:
    values = [getattr(item, attr) for item in items if getattr(item, attr) is not None]
    return max(values) if values else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if isnan(value):
            return {"__nonfinite_float__": "nan"}
        if value == float("inf"):
            return {"__nonfinite_float__": "+inf"}
        if value == float("-inf"):
            return {"__nonfinite_float__": "-inf"}
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
