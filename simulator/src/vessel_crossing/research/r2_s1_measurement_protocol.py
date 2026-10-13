"""Separate unresolved source measurement semantics from CROSS-owned protocol choices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .r2_s1_evidence import (
    ClaimExactness,
    EvidenceClass,
    EvidenceOwner,
    EvidenceProvenance,
)
from .r2_s1_pair_semantics import UNRESOLVED

CROSS_LONGITUDINAL_REFERENCE_LINE_PASS_V1: Final = "CROSS_LONGITUDINAL_REFERENCE_LINE_PASS_V1"
CONFLICT_CENTER_PASS: Final = "CONFLICT_CENTER_PASS"


@dataclass(frozen=True, slots=True)
class MeasurementSemanticBinding:
    source_term: str
    source_measurement_event: str
    cross_protocol_id: str
    runtime_event: str
    cross_mapping_provenance: EvidenceProvenance
    corroboration_provenance: EvidenceProvenance

    def __post_init__(self) -> None:
        for name in ("source_term", "source_measurement_event", "cross_protocol_id", "runtime_event"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.cross_mapping_provenance.evidence_class is not EvidenceClass.CROSS_HYPOTHESIS:
            raise ValueError("runtime mapping must be explicitly CROSS_HYPOTHESIS")
        if self.corroboration_provenance.evidence_class is not EvidenceClass.STRONG_CORROBORATION:
            raise ValueError("measurement corroboration must be STRONG_CORROBORATION")

    @property
    def guo_exact_measurement_ready(self) -> bool:
        return self.source_measurement_event != UNRESOLVED


R2_S1_MEASUREMENT_BINDING: Final = MeasurementSemanticBinding(
    source_term="safety time interval",
    source_measurement_event=UNRESOLVED,
    cross_protocol_id=CROSS_LONGITUDINAL_REFERENCE_LINE_PASS_V1,
    runtime_event=CONFLICT_CENTER_PASS,
    cross_mapping_provenance=EvidenceProvenance(
        EvidenceClass.CROSS_HYPOTHESIS,
        EvidenceOwner.CROSS,
        ClaimExactness.HYPOTHETICAL,
        "CROSS operationalization: controlled same-route center pass is used as the chosen reference-line passage event.",
    ),
    corroboration_provenance=EvidenceProvenance(
        EvidenceClass.STRONG_CORROBORATION,
        EvidenceOwner.SOURCE,
        ClaimExactness.INFERRED,
        "Same-author 2025 work supports fixed-observation-point headway semantics but is not literal Guo-2024 event binding.",
    ),
)
