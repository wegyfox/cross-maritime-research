"""C8 declarative Operational Design Domain / Operational Envelope predicates.

This module is an evaluation layer only.  It never changes controller commands, plant
state, observations, or fault schedules.  An ODD/OE envelope is a frozen set of
predicates evaluated against an immutable metric context.  Each result reports a
signed margin to its declared limit so later C9/C10 logic can distinguish comfortably
inside, near a boundary, and outside.

The terms here are generic research-testbed terminology.  This module does *not* claim
implementation of ISO 25930 or any other standard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import hypot, isfinite
from typing import Iterable, Mapping, Protocol

from .environment_belief import EnvironmentalBeliefSnapshot, EnvironmentalFieldSample
from .observation import ObservedWorldSnapshot
from .scope import TestbedFeature

_EPS = 1e-12


class ODDCategory(str, Enum):
    ENVIRONMENT = "ENVIRONMENT"
    PERCEPTION = "PERCEPTION"
    TRAFFIC = "TRAFFIC"
    SEA_ROOM = "SEA_ROOM"
    SYSTEM_HEALTH = "SYSTEM_HEALTH"
    COMMUNICATION = "COMMUNICATION"


class ODDStatus(str, Enum):
    INSIDE = "INSIDE"
    MARGIN = "MARGIN"
    OUTSIDE = "OUTSIDE"


class ODDMetricSource(str, Enum):
    PHYSICAL_TRUTH = "PHYSICAL_TRUTH"
    OBSERVED_WORLD = "OBSERVED_WORLD"
    ENVIRONMENT_BELIEF = "ENVIRONMENT_BELIEF"
    SYSTEM = "SYSTEM"
    COMMUNICATION = "COMMUNICATION"
    DECLARED = "DECLARED"
    DERIVED = "DERIVED"


class ODDComparison(str, Enum):
    AT_MOST = "AT_MOST"
    AT_LEAST = "AT_LEAST"
    MUST_EQUAL = "MUST_EQUAL"


class ODDResultReason(str, Enum):
    WITHIN_LIMIT = "WITHIN_LIMIT"
    NEAR_LIMIT = "NEAR_LIMIT"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    MISSING_METRIC = "MISSING_METRIC"


@dataclass(frozen=True, slots=True)
class ODDMetricValue:
    metric_id: str
    value: float | bool
    unit: str
    source: ODDMetricSource

    def __post_init__(self) -> None:
        if not self.metric_id or not self.unit:
            raise ValueError("metric_id and unit must be non-empty")
        if isinstance(self.value, bool):
            return
        value = float(self.value)
        if not isfinite(value):
            raise ValueError("numeric ODD metric values must be finite")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class ODDContext:
    """Immutable values available to ODD/OE predicates at one instant."""

    time_min: float
    metrics: tuple[ODDMetricValue, ...]
    context_id: str = "cross-c8/odd-context/v1"

    def __post_init__(self) -> None:
        if not isfinite(self.time_min) or self.time_min < 0.0:
            raise ValueError("time_min must be finite and non-negative")
        if not self.context_id:
            raise ValueError("context_id must be non-empty")
        ids = [metric.metric_id for metric in self.metrics]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("ODD metrics must be unique and sorted by metric_id")

    @classmethod
    def from_unsorted(
        cls,
        *,
        time_min: float,
        metrics: Iterable[ODDMetricValue],
        context_id: str = "cross-c8/odd-context/v1",
    ) -> "ODDContext":
        return cls(
            time_min=float(time_min),
            metrics=tuple(sorted(tuple(metrics), key=lambda metric: metric.metric_id)),
            context_id=context_id,
        )

    def get(self, metric_id: str) -> ODDMetricValue | None:
        for metric in self.metrics:
            if metric.metric_id == metric_id:
                return metric
        return None

    def require(self, metric_id: str) -> ODDMetricValue:
        metric = self.get(metric_id)
        if metric is None:
            raise KeyError(metric_id)
        return metric

    def identity_sha256(self) -> str:
        return _sha256_payload(
            {
                "context_id": self.context_id,
                "time_min": self.time_min,
                "metrics": [
                    {
                        "metric_id": metric.metric_id,
                        "value": metric.value,
                        "unit": metric.unit,
                        "source": metric.source.value,
                    }
                    for metric in self.metrics
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class ODDPredicateResult:
    predicate_id: str
    category: ODDCategory
    metric_id: str
    status: ODDStatus
    reason: ODDResultReason
    actual: float | bool | None
    limit: float | bool
    comparison: ODDComparison
    unit: str
    signed_margin: float | None
    margin_band: float | None
    metric_source: ODDMetricSource | None

    def __post_init__(self) -> None:
        if not self.predicate_id or not self.metric_id or not self.unit:
            raise ValueError("predicate_id, metric_id and unit must be non-empty")
        if self.signed_margin is not None and not isfinite(self.signed_margin):
            raise ValueError("signed_margin must be finite when present")
        if self.margin_band is not None and (not isfinite(self.margin_band) or self.margin_band < 0.0):
            raise ValueError("margin_band must be finite and non-negative")


class ODDPredicate(Protocol):
    predicate_id: str
    category: ODDCategory
    metric_id: str

    def evaluate(self, context: ODDContext) -> ODDPredicateResult:
        ...

    def canonical_payload(self) -> Mapping[str, object]:
        ...


@dataclass(frozen=True, slots=True)
class NumericODDPredicate:
    """Upper/lower numeric ODD limit with an optional near-boundary band.

    ``signed_margin`` is positive inside, zero on the hard boundary, and negative
    outside.  ``margin_band`` is measured in the same unit as the metric.
    """

    predicate_id: str
    category: ODDCategory
    metric_id: str
    comparison: ODDComparison
    limit: float
    unit: str
    margin_band: float = 0.0

    def __post_init__(self) -> None:
        if not self.predicate_id or not self.metric_id or not self.unit:
            raise ValueError("predicate_id, metric_id and unit must be non-empty")
        if self.comparison not in {ODDComparison.AT_MOST, ODDComparison.AT_LEAST}:
            raise ValueError("numeric predicates support AT_MOST or AT_LEAST")
        if not isfinite(self.limit):
            raise ValueError("limit must be finite")
        if not isfinite(self.margin_band) or self.margin_band < 0.0:
            raise ValueError("margin_band must be finite and non-negative")

    def evaluate(self, context: ODDContext) -> ODDPredicateResult:
        metric = context.get(self.metric_id)
        if metric is None:
            return ODDPredicateResult(
                predicate_id=self.predicate_id,
                category=self.category,
                metric_id=self.metric_id,
                status=ODDStatus.OUTSIDE,
                reason=ODDResultReason.MISSING_METRIC,
                actual=None,
                limit=self.limit,
                comparison=self.comparison,
                unit=self.unit,
                signed_margin=None,
                margin_band=self.margin_band,
                metric_source=None,
            )
        if isinstance(metric.value, bool):
            raise TypeError(f"metric {self.metric_id!r} must be numeric")
        if metric.unit != self.unit:
            raise ValueError(
                f"metric {self.metric_id!r} unit {metric.unit!r} does not match predicate unit {self.unit!r}"
            )
        actual = float(metric.value)
        signed_margin = self.limit - actual if self.comparison is ODDComparison.AT_MOST else actual - self.limit
        if signed_margin < -_EPS:
            status = ODDStatus.OUTSIDE
            reason = ODDResultReason.LIMIT_EXCEEDED
        elif signed_margin <= self.margin_band + _EPS:
            status = ODDStatus.MARGIN
            reason = ODDResultReason.NEAR_LIMIT
        else:
            status = ODDStatus.INSIDE
            reason = ODDResultReason.WITHIN_LIMIT
        return ODDPredicateResult(
            predicate_id=self.predicate_id,
            category=self.category,
            metric_id=self.metric_id,
            status=status,
            reason=reason,
            actual=actual,
            limit=self.limit,
            comparison=self.comparison,
            unit=self.unit,
            signed_margin=signed_margin,
            margin_band=self.margin_band,
            metric_source=metric.source,
        )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "type": "numeric",
            "predicate_id": self.predicate_id,
            "category": self.category.value,
            "metric_id": self.metric_id,
            "comparison": self.comparison.value,
            "limit": self.limit,
            "unit": self.unit,
            "margin_band": self.margin_band,
        }


@dataclass(frozen=True, slots=True)
class BooleanODDPredicate:
    """Boolean operational-envelope requirement.

    A satisfied boolean requirement uses signed margin ``+1`` and a failed requirement
    uses ``-1``.  This keeps pass/fail direction explicit without pretending the value
    has a physical distance-to-limit scale.
    """

    predicate_id: str
    category: ODDCategory
    metric_id: str
    required: bool = True
    unit: str = "boolean"

    def __post_init__(self) -> None:
        if not self.predicate_id or not self.metric_id or not self.unit:
            raise ValueError("predicate_id, metric_id and unit must be non-empty")

    def evaluate(self, context: ODDContext) -> ODDPredicateResult:
        metric = context.get(self.metric_id)
        if metric is None:
            return ODDPredicateResult(
                predicate_id=self.predicate_id,
                category=self.category,
                metric_id=self.metric_id,
                status=ODDStatus.OUTSIDE,
                reason=ODDResultReason.MISSING_METRIC,
                actual=None,
                limit=self.required,
                comparison=ODDComparison.MUST_EQUAL,
                unit=self.unit,
                signed_margin=None,
                margin_band=None,
                metric_source=None,
            )
        if not isinstance(metric.value, bool):
            raise TypeError(f"metric {self.metric_id!r} must be boolean")
        if metric.unit != self.unit:
            raise ValueError(
                f"metric {self.metric_id!r} unit {metric.unit!r} does not match predicate unit {self.unit!r}"
            )
        passed = metric.value is self.required
        return ODDPredicateResult(
            predicate_id=self.predicate_id,
            category=self.category,
            metric_id=self.metric_id,
            status=ODDStatus.INSIDE if passed else ODDStatus.OUTSIDE,
            reason=ODDResultReason.WITHIN_LIMIT if passed else ODDResultReason.LIMIT_EXCEEDED,
            actual=metric.value,
            limit=self.required,
            comparison=ODDComparison.MUST_EQUAL,
            unit=self.unit,
            signed_margin=1.0 if passed else -1.0,
            margin_band=None,
            metric_source=metric.source,
        )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "type": "boolean",
            "predicate_id": self.predicate_id,
            "category": self.category.value,
            "metric_id": self.metric_id,
            "required": self.required,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class ODDEvaluation:
    envelope_id: str
    envelope_version: str
    status: ODDStatus
    results: tuple[ODDPredicateResult, ...]
    context_sha256: str
    envelope_sha256: str
    component_id: TestbedFeature = TestbedFeature.C8_ODD_PREDICATES

    @property
    def outside_predicate_ids(self) -> tuple[str, ...]:
        return tuple(result.predicate_id for result in self.results if result.status is ODDStatus.OUTSIDE)

    @property
    def margin_predicate_ids(self) -> tuple[str, ...]:
        return tuple(result.predicate_id for result in self.results if result.status is ODDStatus.MARGIN)

    @property
    def minimum_signed_margin(self) -> float | None:
        margins = [result.signed_margin for result in self.results if result.signed_margin is not None]
        return min(margins) if margins else None


@dataclass(frozen=True, slots=True)
class ODDEnvelope:
    envelope_id: str
    envelope_version: str
    predicates: tuple[ODDPredicate, ...]
    description: str = ""
    component_id = TestbedFeature.C8_ODD_PREDICATES

    def __post_init__(self) -> None:
        if not self.envelope_id or not self.envelope_version:
            raise ValueError("envelope_id and envelope_version must be non-empty")
        ids = [predicate.predicate_id for predicate in self.predicates]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("ODD predicates must be unique and sorted by predicate_id")

    @classmethod
    def from_unsorted(
        cls,
        *,
        envelope_id: str,
        envelope_version: str,
        predicates: Iterable[ODDPredicate],
        description: str = "",
    ) -> "ODDEnvelope":
        return cls(
            envelope_id=envelope_id,
            envelope_version=envelope_version,
            predicates=tuple(sorted(tuple(predicates), key=lambda predicate: predicate.predicate_id)),
            description=description,
        )

    def configuration_sha256(self) -> str:
        return _sha256_payload(
            {
                "envelope_id": self.envelope_id,
                "envelope_version": self.envelope_version,
                "description": self.description,
                "predicates": [dict(predicate.canonical_payload()) for predicate in self.predicates],
            }
        )

    def evaluate(self, context: ODDContext) -> ODDEvaluation:
        results = tuple(predicate.evaluate(context) for predicate in self.predicates)
        if any(result.status is ODDStatus.OUTSIDE for result in results):
            status = ODDStatus.OUTSIDE
        elif any(result.status is ODDStatus.MARGIN for result in results):
            status = ODDStatus.MARGIN
        else:
            status = ODDStatus.INSIDE
        return ODDEvaluation(
            envelope_id=self.envelope_id,
            envelope_version=self.envelope_version,
            status=status,
            results=results,
            context_sha256=context.identity_sha256(),
            envelope_sha256=self.configuration_sha256(),
        )


@dataclass(frozen=True, slots=True)
class NoOpODDEvaluator:
    """Compatibility mode for pre-C8 runs: never constrains prior semantics."""

    evaluator_id: str = "cross-c8/noop-odd/v1"
    component_id = TestbedFeature.C8_ODD_PREDICATES

    def evaluate(self, context: ODDContext) -> ODDEvaluation:
        return ODDEvaluation(
            envelope_id=self.evaluator_id,
            envelope_version="1",
            status=ODDStatus.INSIDE,
            results=(),
            context_sha256=context.identity_sha256(),
            envelope_sha256=_sha256_payload({"evaluator_id": self.evaluator_id, "mode": "NOOP"}),
        )


# Canonical metric names used by the generic runtime adapter.  Research protocols may
# add their own metric IDs without changing this module.
ENV_CURRENT_SPEED_KNOTS = "environment.current_speed_knots"
ENV_WIND_DRIFT_EQUIVALENT_KNOTS = "environment.wind_drift_equivalent_knots"
ENV_BELIEF_AGE_MIN = "environment.belief_age_min"
PERCEPTION_MAX_TRACK_AGE_MIN = "perception.max_track_age_min"
PERCEPTION_OBSERVED_TARGET_COUNT = "perception.observed_target_count"
PERCEPTION_CONTINUITY_FRACTION = "perception.continuity_fraction"
TRAFFIC_TARGET_COUNT = "traffic.target_count"
TRAFFIC_SMALL_CRAFT_COUNT = "traffic.small_craft_count"
TRAFFIC_FISHING_CRAFT_COUNT = "traffic.fishing_craft_count"
TRAFFIC_NON_AIS_FRACTION = "traffic.non_ais_fraction"
SEA_ROOM_MIN_CLEARANCE_NM = "sea_room.min_clearance_nm"
SYSTEM_HEALTH_FRACTION = "system.health_fraction"
COMMUNICATION_AVAILABLE = "communication.available"
COMMUNICATION_LATENCY_MIN = "communication.latency_min"


def odd_context_from_components(
    *,
    time_min: float,
    environment_truth: EnvironmentalFieldSample | None = None,
    environment_belief: EnvironmentalBeliefSnapshot | None = None,
    observed_world: ObservedWorldSnapshot | None = None,
    observer_vessel_id: str | None = None,
    perception_continuity_fraction: float | None = None,
    traffic_target_count: float | None = None,
    small_craft_count: float | None = None,
    fishing_craft_count: float | None = None,
    non_ais_fraction: float | None = None,
    sea_room_min_clearance_nm: float | None = None,
    system_health_fraction: float | None = None,
    communication_available: bool | None = None,
    communication_latency_min: float | None = None,
    extra_metrics: Iterable[ODDMetricValue] = (),
    context_id: str = "cross-c8/runtime-odd-context/v1",
) -> ODDContext:
    """Build a metric context without granting C8 authority over other layers.

    Values that cannot be inferred generically (for example fishing-craft count or
    minimum sea room) remain explicit protocol/runtime inputs.  Missing values stay
    missing and therefore fail closed when a corresponding predicate is declared.
    """

    metrics: list[ODDMetricValue] = list(extra_metrics)
    if environment_truth is not None:
        metrics.extend(
            (
                ODDMetricValue(
                    ENV_CURRENT_SPEED_KNOTS,
                    60.0 * hypot(*environment_truth.current_velocity_nm_per_min),
                    "knots",
                    ODDMetricSource.PHYSICAL_TRUTH,
                ),
                ODDMetricValue(
                    ENV_WIND_DRIFT_EQUIVALENT_KNOTS,
                    60.0 * hypot(*environment_truth.wind_velocity_nm_per_min),
                    "knots",
                    ODDMetricSource.PHYSICAL_TRUTH,
                ),
            )
        )
    if environment_belief is not None:
        metrics.append(
            ODDMetricValue(
                ENV_BELIEF_AGE_MIN,
                environment_belief.age_min,
                "min",
                ODDMetricSource.ENVIRONMENT_BELIEF,
            )
        )
    if observed_world is not None:
        entity_ages = [
            observed_world.observation_time_min - vessel.source_time_min
            for vessel in observed_world.vessels
            if vessel.vessel_id != observer_vessel_id
        ] + [
            observed_world.observation_time_min - obj.source_time_min for obj in observed_world.objects
        ]
        metrics.extend(
            (
                ODDMetricValue(
                    PERCEPTION_MAX_TRACK_AGE_MIN,
                    max(entity_ages, default=0.0),
                    "min",
                    ODDMetricSource.OBSERVED_WORLD,
                ),
                ODDMetricValue(
                    PERCEPTION_OBSERVED_TARGET_COUNT,
                    float(
                        sum(1 for vessel in observed_world.vessels if vessel.vessel_id != observer_vessel_id)
                        + len(observed_world.objects)
                    ),
                    "count",
                    ODDMetricSource.OBSERVED_WORLD,
                ),
            )
        )
    _append_optional_numeric(
        metrics, PERCEPTION_CONTINUITY_FRACTION, perception_continuity_fraction, "fraction", ODDMetricSource.DERIVED
    )
    _append_optional_numeric(metrics, TRAFFIC_TARGET_COUNT, traffic_target_count, "count", ODDMetricSource.DECLARED)
    _append_optional_numeric(metrics, TRAFFIC_SMALL_CRAFT_COUNT, small_craft_count, "count", ODDMetricSource.DECLARED)
    _append_optional_numeric(metrics, TRAFFIC_FISHING_CRAFT_COUNT, fishing_craft_count, "count", ODDMetricSource.DECLARED)
    _append_optional_numeric(metrics, TRAFFIC_NON_AIS_FRACTION, non_ais_fraction, "fraction", ODDMetricSource.DECLARED)
    _append_optional_numeric(metrics, SEA_ROOM_MIN_CLEARANCE_NM, sea_room_min_clearance_nm, "nm", ODDMetricSource.DERIVED)
    _append_optional_numeric(metrics, SYSTEM_HEALTH_FRACTION, system_health_fraction, "fraction", ODDMetricSource.SYSTEM)
    if communication_available is not None:
        metrics.append(
            ODDMetricValue(
                COMMUNICATION_AVAILABLE,
                bool(communication_available),
                "boolean",
                ODDMetricSource.COMMUNICATION,
            )
        )
    _append_optional_numeric(
        metrics, COMMUNICATION_LATENCY_MIN, communication_latency_min, "min", ODDMetricSource.COMMUNICATION
    )
    return ODDContext.from_unsorted(time_min=time_min, metrics=metrics, context_id=context_id)


def _append_optional_numeric(
    metrics: list[ODDMetricValue],
    metric_id: str,
    value: float | None,
    unit: str,
    source: ODDMetricSource,
) -> None:
    if value is not None:
        metrics.append(ODDMetricValue(metric_id, float(value), unit, source))


def _sha256_payload(payload: Mapping[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
