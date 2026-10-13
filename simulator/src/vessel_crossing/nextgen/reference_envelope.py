"""Portable actual-to-ideal reference-envelope reporting contract.

This module is evaluation-only.  It does not alter simulation physics, control, traffic,
or safety semantics.  It freezes the semantic shape of the CROSS report ladder so later
CLI/API/GUI surfaces cannot silently regress to a raw-throughput-only result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

_FORMAT = "vessel-crossing-nextgen-reference-envelope"
_SCHEMA_VERSION = 1


class ReferenceEnvelopeError(ValueError):
    """Raised when an ideal/reference-envelope report is incomplete or invalid."""


class ReferenceEnvelopeLevel(str, Enum):
    B0_ACTUAL_MIXED_NETWORK = "B0_ACTUAL_MIXED_NETWORK"
    B1_DEDICATED_AUTONOMOUS_CORRIDOR = "B1_DEDICATED_AUTONOMOUS_CORRIDOR"
    B2_CONTROLLED_ENTRY = "B2_CONTROLLED_ENTRY"
    B3_SPEED_TRAJECTORY_HARMONIZATION = "B3_SPEED_TRAJECTORY_HARMONIZATION"
    B4_SLOT_VTS_COORDINATION = "B4_SLOT_VTS_COORDINATION"
    B5_ORGANIZED_PLATOON_STRUCTURE = "B5_ORGANIZED_PLATOON_STRUCTURE"
    B6_PHYSICAL_UPPER_ENVELOPE = "B6_PHYSICAL_UPPER_ENVELOPE"


_ORDER = tuple(ReferenceEnvelopeLevel)


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _safe_metrics(metrics: Mapping[str, float | int | None]) -> MappingProxyType:
    out: dict[str, float | None] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric names must be non-empty strings")
        if value is None:
            out[key] = None
        else:
            out[key] = _finite_number(value, name=f"metrics[{key!r}]")
    return MappingProxyType(out)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class ReferenceEnvelopePoint:
    """One frozen B0..B6 counterfactual outcome using the same KPI vocabulary."""

    level: ReferenceEnvelopeLevel
    metrics: Mapping[str, float | int | None]
    model_status: str = "MODEL_DERIVED"
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, ReferenceEnvelopeLevel):
            raise TypeError("level must be ReferenceEnvelopeLevel")
        object.__setattr__(self, "metrics", _safe_metrics(self.metrics))
        if not isinstance(self.model_status, str) or not self.model_status.strip():
            raise ValueError("model_status must be non-empty")
        object.__setattr__(self, "model_status", self.model_status.strip())
        if self.notes is not None:
            if not isinstance(self.notes, str) or not self.notes.strip():
                raise ValueError("notes must be non-empty when supplied")
            object.__setattr__(self, "notes", self.notes.strip())

    def to_json_object(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "metrics": dict(self.metrics),
            "model_status": self.model_status,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ReferenceEnvelopeReport:
    """Mandatory actual -> managed corridor -> physical-upper reporting surface.

    The ladder is deliberately ordered but no monotonic-improvement assumption is made.
    A coordination/platoon intervention may make a KPI worse; that negative increment is
    preserved rather than repaired or hidden.
    """

    points: tuple[ReferenceEnvelopePoint, ...]
    primary_metric: str
    dominant_bottleneck: str
    recommended_next_counterfactual: str
    field_calibrated: bool = False
    deployment_recommendation: bool = False
    uncertainty_summary: Mapping[str, Any] = field(default_factory=dict)
    safety_feasibility_summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if tuple(point.level for point in points) != _ORDER:
            raise ReferenceEnvelopeError(
                "points must contain exactly B0..B6 in canonical order"
            )
        if not isinstance(self.primary_metric, str) or not self.primary_metric.strip():
            raise ValueError("primary_metric must be non-empty")
        primary = self.primary_metric.strip()
        for point in points:
            if primary not in point.metrics or point.metrics[primary] is None:
                raise ReferenceEnvelopeError(
                    f"primary metric {primary!r} missing at {point.level.value}"
                )
        for name, value in (
            ("dominant_bottleneck", self.dominant_bottleneck),
            ("recommended_next_counterfactual", self.recommended_next_counterfactual),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "primary_metric", primary)
        object.__setattr__(self, "points", points)
        # Round-trip through JSON to guarantee a portable evidence payload.
        for attr in ("uncertainty_summary", "safety_feasibility_summary"):
            value = getattr(self, attr)
            try:
                copied = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{attr} must be JSON portable") from exc
            object.__setattr__(self, attr, MappingProxyType(copied))

    def _value(self, level: ReferenceEnvelopeLevel) -> float:
        point = self.points[_ORDER.index(level)]
        value = point.metrics[self.primary_metric]
        assert value is not None
        return float(value)

    @property
    def actual(self) -> float:
        return self._value(ReferenceEnvelopeLevel.B0_ACTUAL_MIXED_NETWORK)

    @property
    def managed_corridor(self) -> float:
        return self._value(ReferenceEnvelopeLevel.B5_ORGANIZED_PLATOON_STRUCTURE)

    @property
    def physical_upper(self) -> float:
        return self._value(ReferenceEnvelopeLevel.B6_PHYSICAL_UPPER_ENVELOPE)

    @property
    def efficiency_vs_managed_corridor(self) -> float | None:
        return None if self.managed_corridor == 0.0 else self.actual / self.managed_corridor

    @property
    def efficiency_vs_physical_upper(self) -> float | None:
        return None if self.physical_upper == 0.0 else self.actual / self.physical_upper

    @property
    def incremental_primary_metric_changes(self) -> MappingProxyType:
        values = [self._value(level) for level in _ORDER]
        return MappingProxyType(
            {
                f"{_ORDER[i-1].value}->{_ORDER[i].value}": values[i] - values[i - 1]
                for i in range(1, len(_ORDER))
            }
        )

    def to_json_object(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "primary_metric": self.primary_metric,
            "actual": self.actual,
            "managed_corridor": self.managed_corridor,
            "physical_upper": self.physical_upper,
            "efficiency": {
                "vs_managed_corridor": self.efficiency_vs_managed_corridor,
                "vs_physical_upper": self.efficiency_vs_physical_upper,
                "recoverable_gap_to_managed_corridor": self.managed_corridor - self.actual,
                "recoverable_gap_to_physical_upper": self.physical_upper - self.actual,
            },
            "reference_ladder": [point.to_json_object() for point in self.points],
            "incremental_primary_metric_changes": dict(self.incremental_primary_metric_changes),
            "dominant_bottleneck": self.dominant_bottleneck,
            "recommended_next_counterfactual": self.recommended_next_counterfactual,
            "uncertainty_summary": dict(self.uncertainty_summary),
            "safety_feasibility_summary": dict(self.safety_feasibility_summary),
            "claim_scope": {
                "field_calibrated": self.field_calibrated,
                "deployment_recommendation": self.deployment_recommendation,
            },
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_json_object())

    @property
    def result_sha256(self) -> str:
        return sha256(self.to_json_bytes()).hexdigest()
