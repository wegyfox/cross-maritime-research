"""Fail-closed trust/conservative semantics for R2-S1."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Final

from .r2_s1_evidence import (
    ClaimExactness,
    EvidenceClass,
    EvidenceOwner,
    EvidenceProvenance,
)
from .r2_s1_pair_semantics import AS, HS


class TrustTransformStatus(str, Enum):
    UNBOUND = "UNBOUND"
    BOUND_CROSS_HYPOTHESIS = "BOUND_CROSS_HYPOTHESIS"
    BOUND_DIRECT_VERIFIED = "BOUND_DIRECT_VERIFIED"


class TrustTransformKind(str, Enum):
    ADD_MINUTES = "ADD_MINUTES"
    SCALE = "SCALE"
    OVERRIDE_INTERVAL_MIN = "OVERRIDE_INTERVAL_MIN"


@dataclass(frozen=True, slots=True)
class TrustTransformHypothesis:
    """Explicit non-source transform for exploratory CROSS scenarios only."""

    hypothesis_id: str
    follower_type: str
    leader_type: str
    transform_kind: TrustTransformKind
    parameter: float
    provenance: EvidenceProvenance
    calibration_note: str
    output_semantic_status: str = "NOT_GUO_EXACT"

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("CROSS trust hypothesis requires hypothesis_id")
        if (self.follower_type, self.leader_type) != (HS, AS):
            raise ValueError("R2-S1 v1 trust hypothesis may target only the candidate HS-follower/AS-leader pair")
        if not isfinite(self.parameter):
            raise ValueError("transform parameter must be finite")
        if self.transform_kind is TrustTransformKind.SCALE and self.parameter <= 0.0:
            raise ValueError("SCALE parameter must be positive")
        if self.transform_kind is TrustTransformKind.OVERRIDE_INTERVAL_MIN and self.parameter <= 0.0:
            raise ValueError("OVERRIDE_INTERVAL_MIN parameter must be positive")
        if self.provenance.evidence_class is not EvidenceClass.CROSS_HYPOTHESIS:
            raise ValueError("exploratory trust transform provenance must be CROSS_HYPOTHESIS")
        if self.provenance.owner is not EvidenceOwner.CROSS:
            raise ValueError("exploratory trust transform must be CROSS-owned")
        if self.provenance.exactness is not ClaimExactness.HYPOTHETICAL:
            raise ValueError("exploratory trust transform must be HYPOTHETICAL")
        if not self.calibration_note:
            raise ValueError("calibration_note must be non-empty")
        if self.output_semantic_status != "NOT_GUO_EXACT":
            raise ValueError("CROSS trust hypothesis output must remain NOT_GUO_EXACT")


@dataclass(frozen=True, slots=True)
class TrustBehaviorBinding:
    candidate_sensitive_follower_type: str
    candidate_sensitive_leader_type: str
    candidate_pair_provenance: EvidenceProvenance
    transform_status: TrustTransformStatus
    hypothesis: TrustTransformHypothesis | None = None

    def __post_init__(self) -> None:
        if (self.candidate_sensitive_follower_type, self.candidate_sensitive_leader_type) != (HS, AS):
            raise ValueError("R2-S1 v1 candidate sensitive pair must be HS follower behind AS leader")
        if self.candidate_pair_provenance.evidence_class is not EvidenceClass.COMPOSITE_INFERENCE:
            raise ValueError("candidate pair must remain COMPOSITE_INFERENCE")
        if self.transform_status is TrustTransformStatus.UNBOUND and self.hypothesis is not None:
            raise ValueError("unbound transform cannot carry a hypothesis")
        if self.transform_status is TrustTransformStatus.BOUND_CROSS_HYPOTHESIS and self.hypothesis is None:
            raise ValueError("BOUND_CROSS_HYPOTHESIS requires an explicit hypothesis")
        if self.transform_status is TrustTransformStatus.BOUND_DIRECT_VERIFIED and self.hypothesis is not None:
            raise ValueError("direct-verified transform cannot be represented by a CROSS hypothesis")

    @property
    def guo_exact_transform_ready(self) -> bool:
        return self.transform_status is TrustTransformStatus.BOUND_DIRECT_VERIFIED

    def require_guo_exact_transform(self) -> None:
        if not self.guo_exact_transform_ready:
            raise RuntimeError(
                "Guo 2024 trust/conservative transform is unbound or CROSS-hypothetical; "
                "exact-Guo evaluation is not authorized"
            )


R2_S1_TRUST_BINDING: Final = TrustBehaviorBinding(
    candidate_sensitive_follower_type=HS,
    candidate_sensitive_leader_type=AS,
    candidate_pair_provenance=EvidenceProvenance(
        EvidenceClass.COMPOSITE_INFERENCE,
        EvidenceOwner.SOURCE,
        ClaimExactness.INFERRED,
        "Multiple independent lines support HS follower behind AS leader as the trust-sensitive pair, but Guo's exact equation remains unrecovered.",
    ),
    transform_status=TrustTransformStatus.UNBOUND,
)
