"""C11 orthogonal scenario-factor composition.

C11 freezes experiment-factor definitions and design points without creating one
scenario class per paper.  It is a configuration/provenance layer: translating a
factor payload into C2/C3/C5/C6/C7/etc. runtime objects remains the responsibility of
explicit protocol adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools
import json
from math import isfinite
from typing import Any, Iterable, Mapping

from .scope import TestbedFeature


class FactorAxisKind(str, Enum):
    DEMAND_TRAFFIC_DENSITY = "DEMAND_TRAFFIC_DENSITY"
    AUTONOMY_PENETRATION = "AUTONOMY_PENETRATION"
    VESSEL_MIX_MANEUVERABILITY = "VESSEL_MIX_MANEUVERABILITY"
    BEHAVIOR_MIX = "BEHAVIOR_MIX"
    AIS_VISIBILITY_NON_AIS_FRACTION = "AIS_VISIBILITY_NON_AIS_FRACTION"
    PERCEPTION_CONTINUITY_DROPOUT_LATENCY = "PERCEPTION_CONTINUITY_DROPOUT_LATENCY"
    FISHING_DENSITY_GEAR_CONSTRAINTS = "FISHING_DENSITY_GEAR_CONSTRAINTS"
    ENVIRONMENT_FORCING_ANOMALY = "ENVIRONMENT_FORCING_ANOMALY"
    SEA_ROOM_CHANNEL_GEOMETRY = "SEA_ROOM_CHANNEL_GEOMETRY"
    FAULT_TIMING_SEVERITY = "FAULT_TIMING_SEVERITY"
    COORDINATION_AVAILABILITY_LATENCY = "COORDINATION_AVAILABILITY_LATENCY"
    CUSTOM = "CUSTOM"


class FactorDesignKind(str, Enum):
    FULL_FACTORIAL = "FULL_FACTORIAL"
    FRACTIONAL_FACTORIAL = "FRACTIONAL_FACTORIAL"
    CUSTOM = "CUSTOM"
    NOOP_BASELINE = "NOOP_BASELINE"


@dataclass(frozen=True, slots=True)
class FrozenJSONMap:
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ScenarioFactorLevel:
    level_id: str
    frozen_value: Any

    @classmethod
    def from_value(cls, level_id: str, value: Any) -> "ScenarioFactorLevel":
        if not level_id:
            raise ValueError("level_id must be non-empty")
        return cls(level_id=level_id, frozen_value=_freeze_json(value))

    @property
    def value(self) -> Any:
        return _thaw_json(self.frozen_value)

    def identity_sha256(self) -> str:
        return _sha256_payload({"level_id": self.level_id, "value": _jsonable_frozen(self.frozen_value)})


@dataclass(frozen=True, slots=True)
class ScenarioFactorAxis:
    axis_id: str
    kind: FactorAxisKind
    levels: tuple[ScenarioFactorLevel, ...]

    @classmethod
    def from_unsorted(
        cls,
        *,
        axis_id: str,
        kind: FactorAxisKind,
        levels: Iterable[ScenarioFactorLevel],
    ) -> "ScenarioFactorAxis":
        if not axis_id:
            raise ValueError("axis_id must be non-empty")
        ordered = tuple(sorted(tuple(levels), key=lambda level: level.level_id))
        if not ordered:
            raise ValueError("factor axis must contain at least one level")
        ids = [level.level_id for level in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("factor level_id values must be unique within an axis")
        return cls(axis_id=axis_id, kind=kind, levels=ordered)

    def level(self, level_id: str) -> ScenarioFactorLevel:
        for level in self.levels:
            if level.level_id == level_id:
                return level
        raise KeyError(level_id)

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "axis_id": self.axis_id,
                "kind": self.kind.value,
                "levels": [
                    {"level_id": level.level_id, "value": _jsonable_frozen(level.frozen_value)}
                    for level in self.levels
                ],
            }
        )


@dataclass(frozen=True, slots=True, order=True)
class FactorAssignment:
    axis_id: str
    level_id: str


@dataclass(frozen=True, slots=True)
class ScenarioFactorPoint:
    assignments: tuple[FactorAssignment, ...]
    point_id: str
    point_seed: int

    def level_id(self, axis_id: str) -> str:
        for assignment in self.assignments:
            if assignment.axis_id == axis_id:
                return assignment.level_id
        raise KeyError(axis_id)


@dataclass(frozen=True, slots=True)
class ScenarioFactorDesign:
    design_id: str
    design_version: str
    kind: FactorDesignKind
    axes: tuple[ScenarioFactorAxis, ...]
    points: tuple[ScenarioFactorPoint, ...]
    base_seed: int
    design_note: str = ""
    component_id: TestbedFeature = TestbedFeature.C11_FACTOR_COMPOSITION

    @classmethod
    def full_factorial(
        cls,
        *,
        design_id: str,
        design_version: str,
        axes: Iterable[ScenarioFactorAxis],
        base_seed: int,
        design_note: str = "",
    ) -> "ScenarioFactorDesign":
        ordered_axes = _normalize_axes(axes)
        rows = [
            tuple(FactorAssignment(axis.axis_id, level.level_id) for axis, level in zip(ordered_axes, levels))
            for levels in itertools.product(*(axis.levels for axis in ordered_axes))
        ]
        if not ordered_axes:
            rows = [tuple()]
        return cls._build(
            design_id=design_id,
            design_version=design_version,
            kind=FactorDesignKind.FULL_FACTORIAL,
            axes=ordered_axes,
            rows=rows,
            base_seed=base_seed,
            design_note=design_note,
        )

    @classmethod
    def fractional_factorial(
        cls,
        *,
        design_id: str,
        design_version: str,
        axes: Iterable[ScenarioFactorAxis],
        selected_rows: Iterable[Mapping[str, str]],
        base_seed: int,
        fraction_label: str,
    ) -> "ScenarioFactorDesign":
        if not fraction_label:
            raise ValueError("fraction_label must be non-empty")
        ordered_axes = _normalize_axes(axes)
        if not ordered_axes:
            raise ValueError("fractional factorial requires at least one axis")
        rows = [_normalize_row(row, ordered_axes) for row in selected_rows]
        if not rows:
            raise ValueError("fractional factorial requires at least one selected row")
        full_count = 1
        for axis in ordered_axes:
            full_count *= len(axis.levels)
        if len(rows) >= full_count:
            raise ValueError("fractional factorial must contain fewer rows than the full Cartesian design")
        return cls._build(
            design_id=design_id,
            design_version=design_version,
            kind=FactorDesignKind.FRACTIONAL_FACTORIAL,
            axes=ordered_axes,
            rows=rows,
            base_seed=base_seed,
            design_note=fraction_label,
        )

    @classmethod
    def custom(
        cls,
        *,
        design_id: str,
        design_version: str,
        axes: Iterable[ScenarioFactorAxis],
        rows: Iterable[Mapping[str, str]],
        base_seed: int,
        design_note: str = "",
    ) -> "ScenarioFactorDesign":
        ordered_axes = _normalize_axes(axes)
        normalized_rows = [_normalize_row(row, ordered_axes) for row in rows]
        if not normalized_rows:
            raise ValueError("custom design requires at least one row")
        return cls._build(
            design_id=design_id,
            design_version=design_version,
            kind=FactorDesignKind.CUSTOM,
            axes=ordered_axes,
            rows=normalized_rows,
            base_seed=base_seed,
            design_note=design_note,
        )

    @classmethod
    def noop_baseline(cls, *, base_seed: int = 0) -> "ScenarioFactorDesign":
        return cls._build(
            design_id="cross-c11/noop-baseline",
            design_version="1",
            kind=FactorDesignKind.NOOP_BASELINE,
            axes=(),
            rows=(tuple(),),
            base_seed=base_seed,
            design_note="pre-C11 single-scenario compatibility",
        )

    @classmethod
    def _build(
        cls,
        *,
        design_id: str,
        design_version: str,
        kind: FactorDesignKind,
        axes: tuple[ScenarioFactorAxis, ...],
        rows: Iterable[tuple[FactorAssignment, ...]],
        base_seed: int,
        design_note: str,
    ) -> "ScenarioFactorDesign":
        if not design_id or not design_version:
            raise ValueError("design_id and design_version must be non-empty")
        _validate_seed(base_seed)
        canonical_rows = tuple(sorted(set(tuple(row) for row in rows), key=_row_sort_key))
        raw_rows = tuple(rows) if not isinstance(rows, tuple) else rows
        # A duplicate row is an experimental ambiguity rather than harmless repetition.
        if len(canonical_rows) != len(raw_rows):
            raise ValueError("factor design rows must be unique")
        point_payloads = []
        for row in canonical_rows:
            row_payload = {
                "design_id": design_id,
                "design_version": design_version,
                "kind": kind.value,
                "axis_identities": [axis.identity_sha256() for axis in axes],
                "assignments": [[a.axis_id, a.level_id] for a in row],
            }
            point_hash = _sha256_payload(row_payload)
            point_id = f"pt-{point_hash[:20]}"
            point_seed = _derive_seed(base_seed, "factor-point", point_hash)
            point_payloads.append(ScenarioFactorPoint(row, point_id, point_seed))
        return cls(
            design_id=design_id,
            design_version=design_version,
            kind=kind,
            axes=axes,
            points=tuple(point_payloads),
            base_seed=int(base_seed),
            design_note=str(design_note),
        )

    def axis(self, axis_id: str) -> ScenarioFactorAxis:
        for axis in self.axes:
            if axis.axis_id == axis_id:
                return axis
        raise KeyError(axis_id)

    def resolved_values(self, point: ScenarioFactorPoint) -> Mapping[str, Any]:
        if point.point_id not in {item.point_id for item in self.points}:
            raise ValueError("point does not belong to this design")
        return {
            assignment.axis_id: self.axis(assignment.axis_id).level(assignment.level_id).value
            for assignment in point.assignments
        }

    def shared_replication_seed(self, replication_index: int) -> int:
        """CRN-compatible seed shared by every point for one replication index."""
        if not isinstance(replication_index, int) or isinstance(replication_index, bool) or replication_index < 0:
            raise ValueError("replication_index must be a non-negative integer")
        return _derive_seed(self.base_seed, "shared-replication", str(replication_index))

    def point_namespace_seed(self, point: ScenarioFactorPoint, namespace: str) -> int:
        """Point-specific seed for stochastic factor realization, separate from CRN seed."""
        if not namespace:
            raise ValueError("namespace must be non-empty")
        if point.point_id not in {item.point_id for item in self.points}:
            raise ValueError("point does not belong to this design")
        return _derive_seed(self.base_seed, "point-namespace", point.point_id, namespace)

    def configuration_sha256(self) -> str:
        return _sha256_payload(
            {
                "design_id": self.design_id,
                "design_version": self.design_version,
                "kind": self.kind.value,
                "base_seed": self.base_seed,
                "design_note": self.design_note,
                "axes": [
                    {
                        "axis_id": axis.axis_id,
                        "kind": axis.kind.value,
                        "levels": [
                            {"level_id": level.level_id, "value": _jsonable_frozen(level.frozen_value)}
                            for level in axis.levels
                        ],
                    }
                    for axis in self.axes
                ],
                "points": [
                    [[assignment.axis_id, assignment.level_id] for assignment in point.assignments]
                    for point in self.points
                ],
            }
        )


class NoOpScenarioFactorComposer:
    component_id = TestbedFeature.C11_FACTOR_COMPOSITION

    def compose(self, *, base_seed: int = 0) -> ScenarioFactorDesign:
        return ScenarioFactorDesign.noop_baseline(base_seed=base_seed)


def _normalize_axes(axes: Iterable[ScenarioFactorAxis]) -> tuple[ScenarioFactorAxis, ...]:
    ordered = tuple(sorted(tuple(axes), key=lambda axis: axis.axis_id))
    ids = [axis.axis_id for axis in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("factor axis_id values must be unique")
    return ordered


def _normalize_row(
    row: Mapping[str, str], axes: tuple[ScenarioFactorAxis, ...]
) -> tuple[FactorAssignment, ...]:
    expected = {axis.axis_id for axis in axes}
    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"factor row must cover every axis exactly once; missing={missing}, extra={extra}")
    assignments = []
    for axis in axes:
        level_id = str(row[axis.axis_id])
        axis.level(level_id)  # validate membership
        assignments.append(FactorAssignment(axis.axis_id, level_id))
    return tuple(assignments)


def _row_sort_key(row: tuple[FactorAssignment, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((assignment.axis_id, assignment.level_id) for assignment in row)


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**63:
        raise ValueError("base_seed must be an integer in [0, 2**63)")


def _derive_seed(base_seed: int, *parts: str) -> int:
    payload = "\x00".join((str(base_seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("factor values must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("factor mapping keys must be strings")
            items.append((key, _freeze_json(item)))
        return FrozenJSONMap(tuple(sorted(items)))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"unsupported factor value type: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, FrozenJSONMap):
        return {key: _thaw_json(item) for key, item in value.items}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _jsonable_frozen(value: Any) -> Any:
    if isinstance(value, FrozenJSONMap):
        return {key: _jsonable_frozen(item) for key, item in value.items}
    if isinstance(value, tuple):
        return [_jsonable_frozen(item) for item in value]
    return value


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
