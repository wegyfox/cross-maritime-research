"""Conservative evidence-based bottleneck classification."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping

from .effects import DiagnosticEffectEstimate

_DEFAULT_TAXONOMY: Mapping[str, str] = MappingProxyType(
    {
        "external_corridor_intrusion": "EXTERNAL_CORRIDOR_INTRUSION",
        "source_admission": "SOURCE_ADMISSION_ORGANIZATION",
        "plant_speed_heterogeneity": "SPEED_PLANT_HETEROGENEITY",
        "global_slot_conflict": "GLOBAL_SLOT_COORDINATION_CONFLICT",
        "queue_order_structure": "TYPE_ORDER_STRUCTURE",
        "cross_stream_physical_separation": "PHYSICAL_REFERENCE_RESIDUAL",
    }
)


@dataclass(frozen=True, slots=True)
class DiagnosticBottleneckClassification:
    classification: str
    intervention_id: str | None
    target_factor: str | None
    mean_effect: float | None
    ci_low: float | None
    ci_high: float | None

    def to_json_object(self) -> dict[str, object]:
        return {
            "format": "cross-diagnostic-bottleneck-classification-v1",
            "classification": self.classification,
            "intervention_id": self.intervention_id,
            "target_factor": self.target_factor,
            "mean_effect": self.mean_effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_json_object(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @property
    def classification_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


def classify_diagnostic_bottleneck(
    effects: tuple[DiagnosticEffectEstimate, ...],
    *,
    taxonomy: Mapping[str, str] = _DEFAULT_TAXONOMY,
) -> DiagnosticBottleneckClassification:
    candidates = [item for item in effects if item.qualified_recoverable]
    if not candidates:
        return DiagnosticBottleneckClassification(
            "NO_QUALIFIED_RECOVERABLE_BOTTLENECK_IDENTIFIED",
            None,
            None,
            None,
            None,
            None,
        )
    selected = max(
        candidates,
        key=lambda item: (item.mean_effect, item.to_level.value, item.intervention_id or ""),
    )
    classification = taxonomy.get(selected.target_factor or "")
    if classification is None:
        classification = "UNMODELED_OR_CALIBRATION_LIMITED"
    return DiagnosticBottleneckClassification(
        classification,
        selected.intervention_id,
        selected.target_factor,
        selected.mean_effect,
        selected.ci_low,
        selected.ci_high,
    )
