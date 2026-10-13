"""Evidence provenance primitives for the R2-S1 scientific model.

R2-S1 separates source authority from CROSS-owned derivations and operational
choices.  These types deliberately do not reuse the older Stage-C readiness
strings, where source resolution and a CROSS operationalization could be easy
to conflate.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceClass(str, Enum):
    DIRECT_VERIFIED = "DIRECT_VERIFIED"
    STRONG_CORROBORATION = "STRONG_CORROBORATION"
    COMPOSITE_INFERENCE = "COMPOSITE_INFERENCE"
    CROSS_DERIVATION = "CROSS_DERIVATION"
    CROSS_HYPOTHESIS = "CROSS_HYPOTHESIS"


class EvidenceOwner(str, Enum):
    SOURCE = "SOURCE"
    CROSS = "CROSS"


class ClaimExactness(str, Enum):
    EXACT_FOR_CLAIM_SCOPE = "EXACT_FOR_CLAIM_SCOPE"
    DERIVED = "DERIVED"
    INFERRED = "INFERRED"
    HYPOTHETICAL = "HYPOTHETICAL"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    evidence_class: EvidenceClass
    owner: EvidenceOwner
    exactness: ClaimExactness
    note: str

    def __post_init__(self) -> None:
        if not self.note:
            raise ValueError("note must be non-empty")
        if self.evidence_class in {EvidenceClass.CROSS_DERIVATION, EvidenceClass.CROSS_HYPOTHESIS}:
            if self.owner is not EvidenceOwner.CROSS:
                raise ValueError("CROSS derivations/hypotheses must be CROSS-owned")
        elif self.owner is not EvidenceOwner.SOURCE:
            raise ValueError("source evidence classes must be SOURCE-owned")
        if self.evidence_class is EvidenceClass.CROSS_DERIVATION and self.exactness is not ClaimExactness.DERIVED:
            raise ValueError("CROSS_DERIVATION must use DERIVED exactness")
        if self.evidence_class is EvidenceClass.CROSS_HYPOTHESIS and self.exactness is not ClaimExactness.HYPOTHETICAL:
            raise ValueError("CROSS_HYPOTHESIS must use HYPOTHETICAL exactness")
        if self.evidence_class is EvidenceClass.COMPOSITE_INFERENCE and self.exactness is not ClaimExactness.INFERRED:
            raise ValueError("COMPOSITE_INFERENCE must use INFERRED exactness")


DIRECT_GUO_RUNNING_EXAMPLE = EvidenceProvenance(
    EvidenceClass.DIRECT_VERIFIED,
    EvidenceOwner.SOURCE,
    ClaimExactness.EXACT_FOR_CLAIM_SCOPE,
    "Guo 2024 Running Example value/relationship directly verified for the stated claim scope.",
)

CROSS_DERIVED_MATH = EvidenceProvenance(
    EvidenceClass.CROSS_DERIVATION,
    EvidenceOwner.CROSS,
    ClaimExactness.DERIVED,
    "Mathematics derived by CROSS from explicitly identified inputs; not a recovered source equation.",
)
