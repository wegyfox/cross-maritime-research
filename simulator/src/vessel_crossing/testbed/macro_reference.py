"""C14 independent macro reference models for micro–macro validity checks.

The macro model consumes declared demand/type-mix/headway assumptions, not calibrated
microsimulation outputs.  It is therefore suitable as an independent reference whose
disagreement with the microsimulator is evidence rather than something fitted away.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Iterable, Mapping

from .evaluation_oracles import OracleMetricSource, OracleMetricValue
from .scope import TestbedFeature


class MacroModelFamily(str, Enum):
    FIXED_HEADWAY_MD1 = "FIXED_HEADWAY_MD1"
    PAIR_HEADWAY_MG1 = "PAIR_HEADWAY_MG1"


@dataclass(frozen=True, slots=True, order=True)
class MacroTrafficClassFraction:
    class_id: str
    fraction: float

    def __post_init__(self) -> None:
        if not self.class_id:
            raise ValueError("class_id must be non-empty")
        if not isfinite(self.fraction) or self.fraction < 0.0 or self.fraction > 1.0:
            raise ValueError("class fraction must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True, order=True)
class PairHeadway:
    previous_class_id: str
    current_class_id: str
    headway_min: float

    def __post_init__(self) -> None:
        if not self.previous_class_id or not self.current_class_id:
            raise ValueError("pair-headway class ids must be non-empty")
        if not isfinite(self.headway_min) or self.headway_min <= 0.0:
            raise ValueError("headway_min must be finite and positive")


@dataclass(frozen=True, slots=True)
class MacroReferenceSpec:
    model_id: str
    model_version: str
    family: MacroModelFamily
    demand_rate_per_hour: float
    class_mix: tuple[MacroTrafficClassFraction, ...]
    pair_headways: tuple[PairHeadway, ...]
    assumptions: tuple[str, ...]
    component_id: TestbedFeature = TestbedFeature.C14_MACRO_REFERENCE_MODEL

    @classmethod
    def fixed_headway_md1(
        cls,
        *,
        model_id: str,
        model_version: str,
        demand_rate_per_hour: float,
        headway_min: float,
        assumptions: Iterable[str] = (),
    ) -> "MacroReferenceSpec":
        if not isfinite(headway_min) or headway_min <= 0.0:
            raise ValueError("headway_min must be finite and positive")
        return cls._validated(
            model_id=model_id,
            model_version=model_version,
            family=MacroModelFamily.FIXED_HEADWAY_MD1,
            demand_rate_per_hour=demand_rate_per_hour,
            class_mix=(MacroTrafficClassFraction("ALL", 1.0),),
            pair_headways=(PairHeadway("ALL", "ALL", headway_min),),
            assumptions=assumptions,
        )

    @classmethod
    def pair_headway_mg1(
        cls,
        *,
        model_id: str,
        model_version: str,
        demand_rate_per_hour: float,
        class_mix: Iterable[MacroTrafficClassFraction],
        pair_headways: Iterable[PairHeadway],
        assumptions: Iterable[str] = (),
    ) -> "MacroReferenceSpec":
        return cls._validated(
            model_id=model_id,
            model_version=model_version,
            family=MacroModelFamily.PAIR_HEADWAY_MG1,
            demand_rate_per_hour=demand_rate_per_hour,
            class_mix=class_mix,
            pair_headways=pair_headways,
            assumptions=assumptions,
        )

    @classmethod
    def _validated(
        cls,
        *,
        model_id: str,
        model_version: str,
        family: MacroModelFamily,
        demand_rate_per_hour: float,
        class_mix: Iterable[MacroTrafficClassFraction],
        pair_headways: Iterable[PairHeadway],
        assumptions: Iterable[str],
    ) -> "MacroReferenceSpec":
        if not model_id or not model_version:
            raise ValueError("model_id and model_version must be non-empty")
        if not isfinite(demand_rate_per_hour) or demand_rate_per_hour < 0.0:
            raise ValueError("demand_rate_per_hour must be finite and non-negative")
        mix = tuple(sorted(tuple(class_mix), key=lambda item: item.class_id))
        if not mix:
            raise ValueError("class_mix must be non-empty")
        ids = [item.class_id for item in mix]
        if len(ids) != len(set(ids)):
            raise ValueError("class_mix class_id values must be unique")
        if abs(sum(item.fraction for item in mix) - 1.0) > 1e-12:
            raise ValueError("class_mix fractions must sum to 1")
        headways = tuple(sorted(tuple(pair_headways), key=lambda h: (h.previous_class_id, h.current_class_id)))
        keys = [(h.previous_class_id, h.current_class_id) for h in headways]
        if len(keys) != len(set(keys)):
            raise ValueError("pair headway keys must be unique")
        expected = {(a, b) for a in ids for b in ids}
        supplied = set(keys)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"pair-headway matrix must be complete; missing={missing}, extra={extra}")
        return cls(
            model_id=model_id,
            model_version=model_version,
            family=family,
            demand_rate_per_hour=float(demand_rate_per_hour),
            class_mix=mix,
            pair_headways=headways,
            assumptions=tuple(sorted(set(assumptions))),
        )

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "model_id": self.model_id,
            "model_version": self.model_version,
            "family": self.family.value,
            "demand_rate_per_hour": self.demand_rate_per_hour,
            "class_mix": [{"class_id": x.class_id, "fraction": x.fraction} for x in self.class_mix],
            "pair_headways": [
                {"previous": h.previous_class_id, "current": h.current_class_id, "headway_min": h.headway_min}
                for h in self.pair_headways
            ],
            "assumptions": list(self.assumptions),
        })


@dataclass(frozen=True, slots=True)
class MacroReferencePrediction:
    model_spec_sha256: str
    mean_headway_min: float
    second_moment_headway_min2: float
    capacity_per_hour: float
    demand_rate_per_hour: float
    utilization: float
    stable: bool
    mean_queue_wait_min: float | None
    predicted_throughput_per_hour: float
    backlog_growth_per_hour: float
    component_id: TestbedFeature = TestbedFeature.C14_MACRO_REFERENCE_MODEL

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "model_spec_sha256": self.model_spec_sha256,
            "mean_headway_min": self.mean_headway_min,
            "second_moment_headway_min2": self.second_moment_headway_min2,
            "capacity_per_hour": self.capacity_per_hour,
            "demand_rate_per_hour": self.demand_rate_per_hour,
            "utilization": self.utilization,
            "stable": self.stable,
            "mean_queue_wait_min": self.mean_queue_wait_min,
            "predicted_throughput_per_hour": self.predicted_throughput_per_hour,
            "backlog_growth_per_hour": self.backlog_growth_per_hour,
        })


class AnalyticalMacroReferenceModel:
    component_id = TestbedFeature.C14_MACRO_REFERENCE_MODEL

    def predict(self, spec: MacroReferenceSpec) -> MacroReferencePrediction:
        probabilities = {item.class_id: item.fraction for item in spec.class_mix}
        headway_by_pair = {(h.previous_class_id, h.current_class_id): h.headway_min for h in spec.pair_headways}
        mean = 0.0
        second = 0.0
        for previous, p_previous in probabilities.items():
            for current, p_current in probabilities.items():
                probability = p_previous * p_current
                headway = headway_by_pair[(previous, current)]
                mean += probability * headway
                second += probability * headway * headway
        capacity = 60.0 / mean
        demand_per_min = spec.demand_rate_per_hour / 60.0
        utilization = demand_per_min * mean
        stable = utilization < 1.0
        if spec.demand_rate_per_hour == 0.0:
            wait = 0.0
        elif stable:
            # Pollaczek–Khinchine M/G/1 queue waiting time.  Fixed headway is the M/D/1 special case.
            wait = demand_per_min * second / (2.0 * (1.0 - utilization))
        else:
            wait = None
        throughput = min(spec.demand_rate_per_hour, capacity)
        backlog_growth = max(0.0, spec.demand_rate_per_hour - capacity)
        return MacroReferencePrediction(
            model_spec_sha256=spec.identity_sha256(),
            mean_headway_min=mean,
            second_moment_headway_min2=second,
            capacity_per_hour=capacity,
            demand_rate_per_hour=spec.demand_rate_per_hour,
            utilization=utilization,
            stable=stable,
            mean_queue_wait_min=wait,
            predicted_throughput_per_hour=throughput,
            backlog_growth_per_hour=backlog_growth,
        )


class NoOpMacroReferenceModel:
    component_id = TestbedFeature.C14_MACRO_REFERENCE_MODEL

    def predict(self, spec: MacroReferenceSpec) -> None:
        return None


@dataclass(frozen=True, slots=True)
class MicroAggregateObservation:
    evidence_id: str
    throughput_per_hour: float
    mean_queue_wait_min: float | None = None
    mean_realized_headway_min: float | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        for name in ("throughput_per_hour", "mean_queue_wait_min", "mean_realized_headway_min"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when provided")


@dataclass(frozen=True, slots=True)
class MicroMacroComparison:
    prediction_sha256: str
    micro_evidence_id: str
    throughput_difference_per_hour: float
    throughput_relative_error: float | None
    queue_wait_difference_min: float | None
    headway_difference_min: float | None
    component_id: TestbedFeature = TestbedFeature.C14_MACRO_REFERENCE_MODEL

    def identity_sha256(self) -> str:
        return _sha256_payload({
            "prediction_sha256": self.prediction_sha256,
            "micro_evidence_id": self.micro_evidence_id,
            "throughput_difference_per_hour": self.throughput_difference_per_hour,
            "throughput_relative_error": self.throughput_relative_error,
            "queue_wait_difference_min": self.queue_wait_difference_min,
            "headway_difference_min": self.headway_difference_min,
        })


def compare_micro_to_macro(*, micro: MicroAggregateObservation, macro: MacroReferencePrediction) -> MicroMacroComparison:
    throughput_diff = micro.throughput_per_hour - macro.predicted_throughput_per_hour
    throughput_rel = None if macro.predicted_throughput_per_hour == 0.0 else throughput_diff / macro.predicted_throughput_per_hour
    wait_diff = None
    if micro.mean_queue_wait_min is not None and macro.mean_queue_wait_min is not None:
        wait_diff = micro.mean_queue_wait_min - macro.mean_queue_wait_min
    headway_diff = None
    if micro.mean_realized_headway_min is not None:
        headway_diff = micro.mean_realized_headway_min - macro.mean_headway_min
    return MicroMacroComparison(
        prediction_sha256=macro.identity_sha256(),
        micro_evidence_id=micro.evidence_id,
        throughput_difference_per_hour=throughput_diff,
        throughput_relative_error=throughput_rel,
        queue_wait_difference_min=wait_diff,
        headway_difference_min=headway_diff,
    )


MACRO_THROUGHPUT_RELATIVE_ERROR = "macro.throughput_relative_error"
MACRO_QUEUE_WAIT_DIFFERENCE_MIN = "macro.queue_wait_difference_min"
MACRO_HEADWAY_DIFFERENCE_MIN = "macro.headway_difference_min"


def oracle_metrics_from_micro_macro_comparison(comparison: MicroMacroComparison) -> tuple[OracleMetricValue, ...]:
    metrics: list[OracleMetricValue] = []
    evidence = comparison.identity_sha256()
    if comparison.throughput_relative_error is not None:
        metrics.append(OracleMetricValue(
            MACRO_THROUGHPUT_RELATIVE_ERROR,
            comparison.throughput_relative_error,
            "fraction",
            OracleMetricSource.DECLARED,
            evidence,
        ))
    if comparison.queue_wait_difference_min is not None:
        metrics.append(OracleMetricValue(
            MACRO_QUEUE_WAIT_DIFFERENCE_MIN,
            comparison.queue_wait_difference_min,
            "min",
            OracleMetricSource.DECLARED,
            evidence,
        ))
    if comparison.headway_difference_min is not None:
        metrics.append(OracleMetricValue(
            MACRO_HEADWAY_DIFFERENCE_MIN,
            comparison.headway_difference_min,
            "min",
            OracleMetricSource.DECLARED,
            evidence,
        ))
    return tuple(sorted(metrics, key=lambda metric: metric.metric_id))


def _sha256_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
