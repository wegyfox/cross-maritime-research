"""Portable evidence objects for the CROSS Diagnostic Product.

Evidence is deliberately upstream of diagnosis: these objects may contain modeled
measurements, activation/binding telemetry and provenance, but never a bottleneck or
recommended intervention answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from ..nextgen.reference_envelope import ReferenceEnvelopeLevel

_FORMAT = "cross-diagnostic-evidence-v1"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class DiagnosticEvidenceError(ValueError):
    """Raised when portable diagnostic evidence is incomplete or inconsistent."""


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DiagnosticEvidenceError(f"{name} must be JSON portable") from exc


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticEvidenceError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping_or_empty(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DiagnosticEvidenceError(f"{name} must be an object")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticEvidenceError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise DiagnosticEvidenceError(f"{name} must be finite")
    return result


def _counts(value: Mapping[str, int], *, name: str) -> Mapping[str, int]:
    out: dict[str, int] = {}
    for key, count in dict(value).items():
        key = _nonempty(key, name=f"{name} key")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise DiagnosticEvidenceError(f"{name}[{key!r}] must be a non-negative integer")
        out[key] = count
    return MappingProxyType(dict(sorted(out.items())))


@dataclass(frozen=True, slots=True)
class DiagnosticReplicationEvidence:
    replication_id: str
    primary_kpi: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    activation_counts: Mapping[str, int] = field(default_factory=dict)
    binding_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "replication_id", _nonempty(self.replication_id, name="replication_id")
        )
        object.__setattr__(self, "primary_kpi", _finite(self.primary_kpi, name="primary_kpi"))
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(_json_copy(dict(self.metrics), name="replication.metrics")),
        )
        object.__setattr__(
            self,
            "activation_counts",
            _counts(self.activation_counts, name="replication.activation_counts"),
        )
        object.__setattr__(
            self,
            "binding_counts",
            _counts(self.binding_counts, name="replication.binding_counts"),
        )

    @classmethod
    def from_json_object(cls, payload: Mapping[str, Any]) -> "DiagnosticReplicationEvidence":
        if not isinstance(payload, Mapping):
            raise DiagnosticEvidenceError("replication evidence must be an object")
        return cls(
            payload.get("replication_id"),
            payload.get("primary_kpi"),
            metrics=_mapping_or_empty(payload.get("metrics"), name="replication.metrics"),
            activation_counts=_mapping_or_empty(
                payload.get("activation_counts"), name="replication.activation_counts"
            ),
            binding_counts=_mapping_or_empty(
                payload.get("binding_counts"), name="replication.binding_counts"
            ),
        )

    def to_json_object(self) -> dict[str, Any]:
        return {
            "replication_id": self.replication_id,
            "primary_kpi": self.primary_kpi,
            "metrics": dict(self.metrics),
            "activation_counts": dict(self.activation_counts),
            "binding_counts": dict(self.binding_counts),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticLevelEvidence:
    level: ReferenceEnvelopeLevel
    aggregate_metrics: Mapping[str, Any]
    replications: tuple[DiagnosticReplicationEvidence, ...] = ()
    intervention_id: str | None = None
    target_factor: str | None = None
    causal: bool = False
    identity_of: ReferenceEnvelopeLevel | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, ReferenceEnvelopeLevel):
            raise TypeError("level must be ReferenceEnvelopeLevel")
        metrics = _json_copy(dict(self.aggregate_metrics), name=f"{self.level.value}.aggregate_metrics")
        forbidden = {"dominant_bottleneck", "recommended_next_counterfactual", "classification"}
        overlap = forbidden & set(metrics)
        if overlap:
            raise DiagnosticEvidenceError(
                f"evidence must not contain diagnosis fields: {sorted(overlap)}"
            )
        object.__setattr__(self, "aggregate_metrics", MappingProxyType(metrics))
        reps = tuple(self.replications)
        if any(not isinstance(item, DiagnosticReplicationEvidence) for item in reps):
            raise TypeError("replications must contain DiagnosticReplicationEvidence")
        ids = [item.replication_id for item in reps]
        if len(ids) != len(set(ids)):
            raise DiagnosticEvidenceError(f"duplicate replication ids at {self.level.value}")
        object.__setattr__(self, "replications", reps)
        if self.intervention_id is not None:
            object.__setattr__(
                self,
                "intervention_id",
                _nonempty(self.intervention_id, name=f"{self.level.value}.intervention_id"),
            )
        if self.target_factor is not None:
            object.__setattr__(
                self,
                "target_factor",
                _nonempty(self.target_factor, name=f"{self.level.value}.target_factor"),
            )
        if not isinstance(self.causal, bool):
            raise DiagnosticEvidenceError("causal must be boolean")
        if self.identity_of is not None and self.causal:
            raise DiagnosticEvidenceError("identity evidence cannot be causal")
        if self.identity_of is not None and not isinstance(self.identity_of, ReferenceEnvelopeLevel):
            raise TypeError("identity_of must be ReferenceEnvelopeLevel")
        if self.reason is not None:
            object.__setattr__(self, "reason", _nonempty(self.reason, name="reason"))

    @property
    def primary_kpi(self) -> float:
        for key in (
            "service_yield_per_hour",
            "mean_service_yield_per_hour",
            "fixed_window_service_yield_per_hour",
        ):
            if key in self.aggregate_metrics:
                return _finite(self.aggregate_metrics[key], name=f"{self.level.value}.{key}")
        if self.replications:
            return sum(item.primary_kpi for item in self.replications) / len(self.replications)
        raise DiagnosticEvidenceError(f"{self.level.value} has no primary KPI")

    @classmethod
    def from_json_object(cls, payload: Mapping[str, Any]) -> "DiagnosticLevelEvidence":
        if not isinstance(payload, Mapping):
            raise DiagnosticEvidenceError("level evidence must be an object")
        try:
            level = ReferenceEnvelopeLevel(payload.get("level"))
        except ValueError as exc:
            raise DiagnosticEvidenceError("unknown reference-envelope level") from exc
        identity_raw = payload.get("identity_of")
        try:
            identity = (
                ReferenceEnvelopeLevel(identity_raw) if identity_raw is not None else None
            )
        except ValueError as exc:
            raise DiagnosticEvidenceError("unknown identity_of level") from exc
        raw_reps = payload.get("replications", [])
        if not isinstance(raw_reps, list):
            raise DiagnosticEvidenceError("replications must be an array")
        return cls(
            level,
            _mapping_or_empty(payload.get("aggregate_metrics"), name="aggregate_metrics"),
            tuple(DiagnosticReplicationEvidence.from_json_object(item) for item in raw_reps),
            intervention_id=payload.get("intervention_id"),
            target_factor=payload.get("target_factor"),
            causal=payload.get("causal", False),
            identity_of=identity,
            reason=payload.get("reason"),
        )

    def to_json_object(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "intervention_id": self.intervention_id,
            "target_factor": self.target_factor,
            "causal": self.causal,
            "identity_of": self.identity_of.value if self.identity_of is not None else None,
            "reason": self.reason,
            "aggregate_metrics": dict(self.aggregate_metrics),
            "replications": [item.to_json_object() for item in self.replications],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(
            {
                "format": "cross-diagnostic-level-evidence-v1",
                **self.to_json_object(),
            }
        )

    @property
    def evidence_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class DiagnosticStudyEvidence:
    scenario_id: str
    levels: tuple[DiagnosticLevelEvidence, ...]
    primary_metric: str = "service_yield_per_hour"
    claim_scope: Mapping[str, bool] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _nonempty(self.scenario_id, name="scenario_id"))
        object.__setattr__(
            self, "primary_metric", _nonempty(self.primary_metric, name="primary_metric")
        )
        levels = tuple(self.levels)
        expected = tuple(ReferenceEnvelopeLevel)
        if tuple(item.level for item in levels) != expected:
            raise DiagnosticEvidenceError("levels must contain B0..B6 exactly in canonical order")
        object.__setattr__(self, "levels", levels)
        scope: dict[str, bool] = {}
        for key, value in dict(self.claim_scope).items():
            key = _nonempty(key, name="claim_scope key")
            if not isinstance(value, bool):
                raise DiagnosticEvidenceError(f"claim_scope[{key!r}] must be boolean")
            scope[key] = value
        object.__setattr__(self, "claim_scope", MappingProxyType(dict(sorted(scope.items()))))
        provenance = _json_copy(dict(self.provenance), name="provenance")
        forbidden = {"dominant_bottleneck", "recommended_next_counterfactual", "classification"}
        if forbidden & set(provenance):
            raise DiagnosticEvidenceError("provenance must not contain diagnosis answers")
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    def level(self, level: ReferenceEnvelopeLevel) -> DiagnosticLevelEvidence:
        return self.levels[tuple(ReferenceEnvelopeLevel).index(level)]

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "scenario_id": self.scenario_id,
            "primary_metric": self.primary_metric,
            "claim_scope": dict(self.claim_scope),
            "provenance": dict(self.provenance),
            "levels": [item.to_json_object() for item in self.levels],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def evidence_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()
