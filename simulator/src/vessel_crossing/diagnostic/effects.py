"""Matched adjacent-level effect estimation for diagnostic evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import sqrt
import statistics

from scipy.stats import t as student_t

from ..nextgen.reference_envelope import ReferenceEnvelopeLevel
from .evidence import DiagnosticEvidenceError, DiagnosticLevelEvidence, DiagnosticStudyEvidence

_EPS = 1e-12
_EFFECTS_FORMAT = "cross-diagnostic-effects-v1"


def diagnostic_effects_json_object(
    effects: tuple["DiagnosticEffectEstimate", ...],
) -> dict[str, object]:
    return {
        "format": _EFFECTS_FORMAT,
        "effects": [item.to_json_object() for item in effects],
    }


def diagnostic_effects_json_bytes(
    effects: tuple["DiagnosticEffectEstimate", ...],
) -> bytes:
    return (
        json.dumps(
            diagnostic_effects_json_object(effects),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def diagnostic_effects_sha256(
    effects: tuple["DiagnosticEffectEstimate", ...],
) -> str:
    return sha256(diagnostic_effects_json_bytes(effects)).hexdigest()


@dataclass(frozen=True, slots=True)
class DiagnosticEffectEstimate:
    from_level: ReferenceEnvelopeLevel
    to_level: ReferenceEnvelopeLevel
    intervention_id: str | None
    target_factor: str | None
    causal: bool
    paired: bool
    mean_effect: float
    ci_low: float | None
    ci_high: float | None
    positive_count: int | None
    zero_count: int | None
    negative_count: int | None
    replication_count: int | None

    @property
    def qualified_recoverable(self) -> bool:
        return (
            self.causal
            and self.mean_effect > _EPS
            and self.ci_low is not None
            and self.ci_low > _EPS
        )

    def to_json_object(self) -> dict[str, object]:
        return {
            "from_level": self.from_level.value,
            "to_level": self.to_level.value,
            "intervention_id": self.intervention_id,
            "target_factor": self.target_factor,
            "causal": self.causal,
            "paired": self.paired,
            "mean_effect": self.mean_effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "positive_count": self.positive_count,
            "zero_count": self.zero_count,
            "negative_count": self.negative_count,
            "replication_count": self.replication_count,
            "qualified_recoverable": self.qualified_recoverable,
        }


def _paired_differences(
    previous: DiagnosticLevelEvidence,
    current: DiagnosticLevelEvidence,
) -> list[float]:
    left = {item.replication_id: item.primary_kpi for item in previous.replications}
    right = {item.replication_id: item.primary_kpi for item in current.replications}
    if not left or set(left) != set(right):
        raise DiagnosticEvidenceError(
            f"paired replication identity mismatch: {previous.level.value}->{current.level.value}"
        )
    return [right[key] - left[key] for key in sorted(left)]


def _paired_estimate(
    previous: DiagnosticLevelEvidence,
    current: DiagnosticLevelEvidence,
    *,
    confidence_level: float,
) -> DiagnosticEffectEstimate:
    differences = _paired_differences(previous, current)
    mean = statistics.mean(differences)
    if len(differences) >= 2:
        sd = statistics.stdev(differences)
        alpha = 1.0 - confidence_level
        half = float(student_t.ppf(1.0 - alpha / 2.0, len(differences) - 1)) * sd / sqrt(
            len(differences)
        )
        ci_low, ci_high = mean - half, mean + half
    else:
        ci_low = ci_high = None
    positive = sum(value > _EPS for value in differences)
    negative = sum(value < -_EPS for value in differences)
    zero = len(differences) - positive - negative
    return DiagnosticEffectEstimate(
        previous.level,
        current.level,
        current.intervention_id,
        current.target_factor,
        True,
        True,
        mean,
        ci_low,
        ci_high,
        positive,
        zero,
        negative,
        len(differences),
    )


def estimate_ladder_effects(
    study: DiagnosticStudyEvidence,
    *,
    confidence_level: float = 0.95,
) -> tuple[DiagnosticEffectEstimate, ...]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be within (0, 1)")
    effects: list[DiagnosticEffectEstimate] = []
    for previous, current in zip(study.levels, study.levels[1:]):
        if current.identity_of is not None:
            effects.append(
                DiagnosticEffectEstimate(
                    previous.level,
                    current.level,
                    current.intervention_id,
                    current.target_factor,
                    False,
                    False,
                    0.0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        if current.causal:
            effects.append(
                _paired_estimate(previous, current, confidence_level=confidence_level)
            )
            continue
        effects.append(
            DiagnosticEffectEstimate(
                previous.level,
                current.level,
                current.intervention_id,
                current.target_factor,
                False,
                False,
                current.primary_kpi - previous.primary_kpi,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        )
    return tuple(effects)
