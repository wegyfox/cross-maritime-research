"""Role-safe pair semantics for R2-S1.

The canonical source contract is follower -> leader.  Runtime code that is
leader-first must call the explicit adapter instead of re-keying raw cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Final

from .r2_s1_evidence import DIRECT_GUO_RUNNING_EXAMPLE, EvidenceProvenance

AS: Final = "AS"
HS: Final = "HS"
_ALLOWED_TYPES: Final = frozenset({AS, HS})
UNRESOLVED: Final = "UNRESOLVED"


def _validate_type(value: str, *, field: str) -> None:
    if value not in _ALLOWED_TYPES:
        raise ValueError(f"{field} must be AS or HS")


@dataclass(frozen=True, slots=True)
class SourcePairInterval:
    follower_type: str
    leader_type: str
    interval_min: float
    provenance: EvidenceProvenance
    source_location: str

    def __post_init__(self) -> None:
        _validate_type(self.follower_type, field="follower_type")
        _validate_type(self.leader_type, field="leader_type")
        if not isfinite(self.interval_min) or self.interval_min <= 0.0:
            raise ValueError("interval_min must be finite and positive")
        if not self.source_location:
            raise ValueError("source_location must be non-empty")


@dataclass(frozen=True, slots=True)
class SourcePairIntervalMatrix:
    reference_id: str
    source_title: str
    doi: str
    source_term: str
    unit: str
    source_measurement_event: str
    intervals: tuple[SourcePairInterval, ...]

    def __post_init__(self) -> None:
        for name in ("reference_id", "source_title", "doi", "source_term", "unit", "source_measurement_event"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(self.intervals, tuple):
            raise TypeError("intervals must be an immutable tuple")
        keys = [(x.follower_type, x.leader_type) for x in self.intervals]
        expected = {(a, b) for a in _ALLOWED_TYPES for b in _ALLOWED_TYPES}
        if len(keys) != 4 or set(keys) != expected:
            raise ValueError("intervals must contain each follower/leader pair exactly once")

    def interval_for_source_roles(self, *, follower_type: str, leader_type: str) -> float:
        _validate_type(follower_type, field="follower_type")
        _validate_type(leader_type, field="leader_type")
        for row in self.intervals:
            if row.follower_type == follower_type and row.leader_type == leader_type:
                return row.interval_min
        raise AssertionError("validated matrix is missing a pair")

    def interval_for_runtime_roles(self, *, leader_type: str, follower_type: str) -> float:
        """Translate a leader-first runtime call into the follower-first source contract."""
        return self.interval_for_source_roles(follower_type=follower_type, leader_type=leader_type)

    @property
    def source_measurement_event_resolved(self) -> bool:
        return self.source_measurement_event != UNRESOLVED

    @property
    def identity_sha256(self) -> str:
        payload = {
            "reference_id": self.reference_id,
            "source_title": self.source_title,
            "doi": self.doi,
            "source_term": self.source_term,
            "unit": self.unit,
            "source_pair_orientation": "FOLLOWER_TO_LEADER",
            "source_measurement_event": self.source_measurement_event,
            "intervals": [
                {
                    "follower_type": row.follower_type,
                    "leader_type": row.leader_type,
                    "interval_min": row.interval_min,
                    "evidence_class": row.provenance.evidence_class.value,
                    "owner": row.provenance.owner.value,
                    "exactness": row.provenance.exactness.value,
                    "source_location": row.source_location,
                }
                for row in sorted(self.intervals, key=lambda x: (x.follower_type, x.leader_type))
            ],
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


GUO_2024_RUNNING_EXAMPLE_S1: Final = SourcePairIntervalMatrix(
    reference_id="CROSS_R2_S1_GUO_2024_RUNNING_EXAMPLE_SOURCE_BINDING_V1",
    source_title=(
        "Mixed traffic conditions of autonomous and human-driven ships: "
        "Assessing channel traffic capacity bounds and optimizing channel management"
    ),
    doi="10.1016/j.oceaneng.2024.119734",
    source_term="safety time interval",
    unit="min",
    source_measurement_event=UNRESOLVED,
    intervals=(
        SourcePairInterval(AS, AS, 7.50, DIRECT_GUO_RUNNING_EXAMPLE, "Introduction, Running example"),
        SourcePairInterval(AS, HS, 9.00, DIRECT_GUO_RUNNING_EXAMPLE, "Introduction, Running example"),
        SourcePairInterval(HS, AS, 11.25, DIRECT_GUO_RUNNING_EXAMPLE, "Introduction, Running example"),
        SourcePairInterval(HS, HS, 11.25, DIRECT_GUO_RUNNING_EXAMPLE, "Introduction, Running example"),
    ),
)
